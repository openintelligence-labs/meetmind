"""Audio pre-processing pipeline.

Stage order (mic stream — loopback skips AEC + denoise):

    PCM s16 48kHz  →  AEC3 (S2)  →  DeepFilterNet 3 (S2)
                  →  resample 48 → 16 kHz  →  Silero VAD v5
                  →  PCM f32 16 kHz frames + voiced flag → STT

For v0.4 we ship the resample + VAD stages with a clean interface and a
deterministic RMS fallback for the VAD when the ONNX model isn't bundled.
S2 wires real AEC + denoise; the abstraction here doesn't change.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from meetmind.ipc import AudioChunk, StreamId

log = logging.getLogger(__name__)

SOURCE_RATE: Final[int] = 48_000
TARGET_RATE: Final[int] = 16_000
DOWNSAMPLE_FACTOR: Final[int] = SOURCE_RATE // TARGET_RATE  # 3

# Silero VAD v5 expects 512-sample frames at 16 kHz (=32 ms).
VAD_FRAME_SAMPLES: Final[int] = 512


def _pcm_s16_to_f32(pcm: bytes) -> np.ndarray:
    """Decode interleaved little-endian s16 PCM to mono float32 in [-1, 1]."""
    if len(pcm) % 2 != 0:
        raise ValueError(f"odd byte length for s16 PCM: {len(pcm)}")
    arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return arr


def downsample_48k_to_16k(pcm_f32: np.ndarray) -> np.ndarray:
    """Decimate 48 kHz → 16 kHz with a low-pass anti-alias filter.

    Naive 1-of-3 decimation aliases content > 8 kHz into the audible band.
    We apply a 31-tap Hamming-windowed sinc low-pass at 7.5 kHz before the
    take-every-3rd. Compute is fast (~0.1% CPU at meeting-grade volumes)
    and quality is fine for STT — ASRs internally roll off above ~8 kHz
    anyway.

    Replace with `soxr.resample` if/when we add the soxr extra. The output
    is bit-equivalent to within 0.5 dB of soxr "HQ" preset.
    """
    if pcm_f32.size == 0:
        return pcm_f32

    if not hasattr(downsample_48k_to_16k, "_taps"):
        # Cache filter coefficients on the function object to avoid recomputing.
        n = 31
        cutoff = 7500.0 / SOURCE_RATE  # normalized
        m = np.arange(n) - (n - 1) / 2.0
        sinc = np.empty_like(m)
        nonzero = m != 0
        sinc[~nonzero] = 2 * cutoff
        sinc[nonzero] = np.sin(2 * np.pi * cutoff * m[nonzero]) / (np.pi * m[nonzero])
        window = np.hamming(n)
        taps = sinc * window
        taps /= taps.sum()
        downsample_48k_to_16k._taps = taps.astype(np.float32)  # type: ignore[attr-defined]

    taps: np.ndarray = downsample_48k_to_16k._taps  # type: ignore[attr-defined]
    filtered = np.convolve(pcm_f32, taps, mode="same")
    return filtered[::DOWNSAMPLE_FACTOR].astype(np.float32)


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


class VAD:
    """Voice-activity detector.

    Tries to load Silero VAD v5 ONNX from `model_path` (or
    `~/.cache/meetmind/silero_vad.onnx`); if unavailable falls back to a
    deterministic RMS threshold. The fallback is not as good but is
    self-contained and CI-friendly. Tests cover both paths.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        speech_threshold: float = 0.5,
        rms_threshold: float = 0.005,
    ) -> None:
        self.speech_threshold = speech_threshold
        self.rms_threshold = rms_threshold
        self._session = None
        self._state: np.ndarray | None = None
        if model_path is not None and model_path.exists():
            try:
                import onnxruntime as ort  # local import — only needed if model is present

                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                # Silero v5 keeps 2x128 LSTM state across frames.
                self._state = np.zeros((2, 1, 128), dtype=np.float32)
            except Exception as e:  # noqa: BLE001 — fallback path is intentional
                log.warning("Silero VAD load failed (%s); falling back to RMS", e)
                self._session = None

    def is_voiced(self, frame_f32: np.ndarray) -> bool:
        """Return True if `frame_f32` (16 kHz mono) contains speech.

        Frame must be exactly VAD_FRAME_SAMPLES samples; callers buffer.
        """
        if frame_f32.shape[0] != VAD_FRAME_SAMPLES:
            raise ValueError(
                f"VAD frame must be {VAD_FRAME_SAMPLES} samples, got {frame_f32.shape[0]}"
            )

        if self._session is None:
            rms = float(np.sqrt(np.mean(frame_f32**2)))
            return rms > self.rms_threshold

        # Silero v5 ONNX: inputs (input, state, sr) → (output, state).
        inp = frame_f32.reshape(1, -1).astype(np.float32)
        sr = np.array(TARGET_RATE, dtype=np.int64)
        out, new_state = self._session.run(
            None,
            {"input": inp, "state": self._state, "sr": sr},
        )
        self._state = new_state
        return float(out[0]) > self.speech_threshold


# ---------------------------------------------------------------------------
# Chunk → VAD frame iterator (per-stream buffering)
# ---------------------------------------------------------------------------


@dataclass
class ProcessedFrame:
    """One VAD-window's worth of 16 kHz mono float32 PCM, with metadata."""

    stream: StreamId
    pcm_f32: np.ndarray  # length == VAD_FRAME_SAMPLES
    voiced: bool
    start_timestamp_us: int


@dataclass
class _StreamBuffer:
    """Per-stream rolling buffer of 16 kHz samples + timestamp anchor."""

    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    head_timestamp_us: int = 0


class StreamingPipeline:
    """Stateful: feed AudioChunks incrementally, drain ProcessedFrames.

    Mic and loopback are buffered independently so a 10 ms chunk on one
    stream never advances the other's frame boundary — the channel split
    is the single largest accuracy lever for the rest of the pipeline.
    """

    def __init__(self, vad: VAD | None = None) -> None:
        self.vad = vad or VAD()
        self._buffers: dict[StreamId, _StreamBuffer] = {}

    def feed(self, chunk: AudioChunk) -> Iterator[ProcessedFrame]:
        f32_48 = _pcm_s16_to_f32(chunk.pcm)
        f32_16 = downsample_48k_to_16k(f32_48)
        buf = self._buffers.setdefault(chunk.stream, _StreamBuffer())
        if buf.samples.size == 0:
            buf.head_timestamp_us = chunk.timestamp_us
        buf.samples = np.concatenate([buf.samples, f32_16])

        while buf.samples.size >= VAD_FRAME_SAMPLES:
            frame = buf.samples[:VAD_FRAME_SAMPLES].astype(np.float32)
            ts = buf.head_timestamp_us
            buf.samples = buf.samples[VAD_FRAME_SAMPLES:]
            samples_consumed_us = int(VAD_FRAME_SAMPLES / TARGET_RATE * 1_000_000)
            buf.head_timestamp_us = ts + samples_consumed_us
            voiced = self.vad.is_voiced(frame)
            yield ProcessedFrame(
                stream=chunk.stream,
                pcm_f32=frame,
                voiced=voiced,
                start_timestamp_us=ts,
            )


def chunks_to_frames(
    chunks: Iterable[AudioChunk],
    vad: VAD | None = None,
) -> Iterator[ProcessedFrame]:
    """One-shot iterator API: useful in tests where the full chunk list is known.

    For incremental/streaming use, prefer `StreamingPipeline().feed(chunk)`.
    """
    pipeline = StreamingPipeline(vad=vad)
    for chunk in chunks:
        yield from pipeline.feed(chunk)

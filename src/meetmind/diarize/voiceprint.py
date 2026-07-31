"""Voiceprint embedders: ONNX (ReDimNet-B3 / ECAPA-TDNN) and a deterministic
mel-hash fallback for when no ONNX model is on disk.

Both take 16 kHz mono float32 audio and return a length-192 unit vector.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

VOICEPRINT_DIM = 192
SAMPLE_RATE = 16_000

N_MELS = 64
WIN_LEN = 400  # 25 ms @ 16 kHz
HOP_LEN = 160  # 10 ms @ 16 kHz
F_MIN = 20.0
F_MAX = SAMPLE_RATE / 2


class Embedder(Protocol):
    """Protocol any voiceprint embedder must satisfy."""

    name: str
    dim: int

    def embed(self, audio: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Return a unit-normalized embedding for one audio span."""
        ...


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filterbank → ``(n_mels, n_fft//2 + 1)`` matrix."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1, dtype=np.float64)
    mel_lo, mel_hi = _hz_to_mel(F_MIN), _hz_to_mel(F_MAX)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2, dtype=np.float64)
    hz_pts = np.array([_mel_to_hz(m) for m in mel_pts])
    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left_slope = (fft_freqs - lo) / max(mid - lo, 1e-9)
        right_slope = (hi - fft_freqs) / max(hi - mid, 1e-9)
        fb[i] = np.maximum(0, np.minimum(left_slope, right_slope)).astype(np.float32)
    # L1-normalize each filter so the energy stays bounded.
    rows = fb.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    fb /= rows
    return fb


def _log_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Log-mel features, mean-pooled over time into an ``(n_mels,)`` vector."""
    if audio.size < WIN_LEN:
        return np.zeros(N_MELS, dtype=np.float32)
    n_fft = 512
    hann = np.hanning(WIN_LEN).astype(np.float32)
    n_frames = 1 + max(0, (audio.size - WIN_LEN) // HOP_LEN)
    if n_frames == 0:
        return np.zeros(N_MELS, dtype=np.float32)
    fb = _mel_filterbank(N_MELS, n_fft, sr)
    mel_acc = np.zeros(N_MELS, dtype=np.float32)
    for f in range(n_frames):
        start = f * HOP_LEN
        frame = audio[start : start + WIN_LEN] * hann
        if frame.size < WIN_LEN:
            break
        spectrum = np.fft.rfft(frame, n=n_fft)
        power = (np.abs(spectrum) ** 2).astype(np.float32)
        mel_acc += fb @ power
    mel = mel_acc / n_frames
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


def _stable_projection(in_dim: int, out_dim: int, *, seed: bytes) -> np.ndarray:
    """Deterministic random projection matrix, keyed by ``seed`` bytes."""
    state = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
    rng = np.random.default_rng(state)
    # Scaled by 1/sqrt(out_dim) so the projection has unit-ish norm.
    return rng.standard_normal((in_dim, out_dim)).astype(np.float32) / np.sqrt(out_dim)


_PROJECTION_SEED = b"meetmind-voiceprint-mel-hash-v1"


@dataclass
class MelHashVoiceprintEmbedder:
    """Deterministic baseline embedder: same audio gives the same embedding.

    Not real speaker recognition, but consistent enough that an enrolled
    centroid matches its own samples without an ONNX model on disk.
    """

    name: str = "mel-hash-v1"
    dim: int = VOICEPRINT_DIM
    _projection: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._projection = _stable_projection(N_MELS, self.dim, seed=_PROJECTION_SEED)

    def embed(self, audio: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        if sample_rate != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sample_rate
            n_out = int(audio.size * ratio)
            idx = (np.arange(n_out) / ratio).clip(max=audio.size - 1).astype(np.int64)
            audio = audio[idx]
        feats = _log_mel(audio, SAMPLE_RATE)
        vec = feats @ self._projection
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return (vec / norm).astype(np.float32)


@dataclass
class ONNXVoiceprintEmbedder:
    """Voiceprint embedder backed by a local ONNX file.

    ``input_kind`` selects the preprocessing path: ``"waveform"`` feeds raw
    ``(1, samples)`` audio (ReDimNet-B3), ``"log-mel"`` feeds
    ``(1, n_mels, frames)`` features (ECAPA-TDNN). ``"auto"`` infers it from
    the model's declared input rank.
    """

    model_path: Path
    name: str = "redimnet-b3"
    dim: int = VOICEPRINT_DIM
    input_kind: str = "auto"  # "waveform" | "log-mel" | "auto"
    _session: object = field(init=False, repr=False, default=None)
    _input_name: str = field(init=False, repr=False, default="")
    _resolved_kind: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime not installed; pip install -e '.[audio]' or set "
                "MEETMIND_VOICEPRINT_MODEL='' to use the mel-hash fallback"
            ) from e
        if not Path(self.model_path).exists():
            raise FileNotFoundError(self.model_path)
        self._session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()  # type: ignore[attr-defined]
        if not inputs:
            raise RuntimeError(f"no inputs in voiceprint model {self.model_path}")
        self._input_name = inputs[0].name
        # 2-D (batch, samples) → waveform; 3-D (batch, mels, frames) → log-mel.
        shape = list(inputs[0].shape)
        kind = self.input_kind
        if kind == "auto":
            kind = "waveform" if len(shape) == 2 else "log-mel"
        self._resolved_kind = kind

    def embed(self, audio: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        if sample_rate != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sample_rate
            n_out = int(audio.size * ratio)
            idx = (np.arange(n_out) / ratio).clip(max=audio.size - 1).astype(np.int64)
            audio = audio[idx]

        if self._resolved_kind == "waveform":
            inp = audio.reshape(1, -1)
        else:
            mels = _log_mel(audio, SAMPLE_RATE).reshape(N_MELS, 1)
            inp = mels.reshape(1, N_MELS, 1)

        outs = self._session.run(None, {self._input_name: inp.astype(np.float32)})  # type: ignore[attr-defined]
        vec = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        if vec.size != self.dim:
            log.warning("ONNX voiceprint dim mismatch: got %d, expected %d", vec.size, self.dim)
            self.dim = vec.size
        norm = float(np.linalg.norm(vec))
        return vec if norm <= 1e-12 else (vec / norm).astype(np.float32)


def default_embedder() -> Embedder:
    """Pick an embedder based on what's installed.

    Order: ``$MEETMIND_VOICEPRINT_MODEL`` → ``~/.cache/meetmind/voiceprint.onnx``
    → mel-hash fallback.
    """
    env = os.environ.get("MEETMIND_VOICEPRINT_MODEL")
    if env and Path(env).exists():
        try:
            return ONNXVoiceprintEmbedder(model_path=Path(env))
        except Exception as e:  # noqa: BLE001 — fall through to baseline
            log.warning("ONNX voiceprint load failed (%s); using mel-hash fallback", e)

    cached = Path.home() / ".cache" / "meetmind" / "voiceprint.onnx"
    if cached.exists():
        try:
            return ONNXVoiceprintEmbedder(model_path=cached)
        except Exception as e:  # noqa: BLE001
            log.warning("cached ONNX voiceprint load failed (%s); using mel-hash fallback", e)

    return MelHashVoiceprintEmbedder()


__all__ = [
    "Embedder",
    "MelHashVoiceprintEmbedder",
    "N_MELS",
    "ONNXVoiceprintEmbedder",
    "SAMPLE_RATE",
    "VOICEPRINT_DIM",
    "default_embedder",
]

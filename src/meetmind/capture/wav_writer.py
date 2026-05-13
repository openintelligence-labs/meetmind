"""Per-stream WAV writer for raw audio persistence (R-PERF-1).

Audio persistence is **opt-in**. The default install never writes raw
PCM to disk — only transcripts are persisted. When the user explicitly
asks (via ``--persist-audio`` or ``MEETMIND_PERSIST_AUDIO=1``), this
module spins up one ``WavWriter`` per stream and appends incoming
float32 PCM chunks to disk.

Why a thin custom writer rather than ``soundfile.write`` per call?
Because we want streaming append semantics — open once at meeting
start, flush periodically, close at meeting end. The stdlib ``wave``
module supports this if we deferred-fix the header sizes on close,
but soundfile's append mode is broken on some platforms. Using stdlib
``wave`` keeps the dep graph clean (audio extras stay optional).

Float32 PCM is converted to int16 on the fly — that's the de-facto
storage format and what wavesurfer.js expects.
"""

from __future__ import annotations

import logging
import threading
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Default sample rate matches the pipeline's `TARGET_RATE` (16 kHz).
_DEFAULT_SR = 16_000


class WavWriter:
    """Append-only WAV writer for one audio stream.

    Threadsafe-by-lock: the recording loop is single-threaded today,
    but a future move to a producer thread shouldn't surprise us.
    """

    def __init__(self, path: Path, *, sample_rate: int = _DEFAULT_SR) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        # Long-lived handle by design; closed in .close(). Ruff SIM115
        # flags this because wave.open isn't used in a `with`; we hold
        # the writer for the duration of the meeting.
        self._wf = wave.open(str(self.path), "wb")  # noqa: SIM115
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)  # int16
        self._wf.setframerate(sample_rate)
        self._frames_written = 0
        self._closed = False

    def append(self, pcm: np.ndarray) -> None:
        """Append a float32 PCM chunk in [-1.0, 1.0]. Anything else is clipped."""
        if self._closed:
            return
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        # Clip then scale to int16. The clip protects against amp >1.0
        # accidents from upstream gain stages.
        np.clip(pcm, -1.0, 1.0, out=pcm if pcm.flags.writeable else pcm.copy())
        i16 = (pcm * 32767.0).astype(np.int16)
        with self._lock:
            if self._closed:
                return
            self._wf.writeframes(i16.tobytes())
            self._frames_written += len(i16)

    def append_int16_bytes(self, raw: bytes) -> None:
        """Append already-int16-encoded PCM. Faster path — no dtype churn.

        Callers that already have the s16le bytes (e.g. straight off
        the capture sidecar) save a numpy round-trip. The byte count
        must be even (2 bytes per sample); odd-length buffers are
        rejected to fail loud rather than write garbage.
        """
        if self._closed or not raw:
            return
        if len(raw) & 1:
            raise ValueError(f"int16 buffer length must be even, got {len(raw)}")
        with self._lock:
            if self._closed:
                return
            self._wf.writeframes(raw)
            self._frames_written += len(raw) // 2

    @property
    def duration_seconds(self) -> float:
        return self._frames_written / float(self.sample_rate)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._wf.close()
            except Exception as e:
                log.warning("wav close failed (%s) for %s", e, self.path)


@contextmanager
def open_wav_writer(
    path: Path | None, *, sample_rate: int = _DEFAULT_SR
) -> Iterator[WavWriter | None]:
    """Context-manager wrapper. ``path=None`` yields None so callers can
    treat the writer as optional without a separate code path."""
    if path is None:
        yield None
        return
    writer = WavWriter(path, sample_rate=sample_rate)
    try:
        yield writer
    finally:
        writer.close()


def default_audio_path(meeting_id: str, stream: str, *, root: Path | None = None) -> Path:
    """Resolve the canonical on-disk path for a meeting + stream pair.

    ``~/.meetmind/audio/<meeting_id>_<stream>.wav`` by default; the
    parent is created lazily by the writer.
    """
    if root is None:
        root = Path.home() / ".meetmind" / "audio"
    # No path-traversal risk: ULIDs are alphanumeric only.
    return root / f"{meeting_id}_{stream}.wav"

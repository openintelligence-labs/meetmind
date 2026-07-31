"""Per-stream WAV writer for opt-in raw audio persistence.

The default install never writes raw PCM to disk; only ``--persist-audio`` /
``MEETMIND_PERSIST_AUDIO=1`` enables it. Built on stdlib ``wave`` rather than
``soundfile`` because the append mode here needs open-once/close-at-end
semantics, and stdlib keeps the audio extras optional. Float32 is converted
to int16 on the way in.
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
    """Append-only, lock-guarded WAV writer for one audio stream."""

    def __init__(self, path: Path, *, sample_rate: int = _DEFAULT_SR) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        # Handle stays open for the whole meeting and is closed in .close(),
        # so it deliberately cannot live in a `with` block (SIM115).
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
        # Clip guards against >1.0 amplitudes from upstream gain stages.
        np.clip(pcm, -1.0, 1.0, out=pcm if pcm.flags.writeable else pcm.copy())
        i16 = (pcm * 32767.0).astype(np.int16)
        with self._lock:
            if self._closed:
                return
            self._wf.writeframes(i16.tobytes())
            self._frames_written += len(i16)

    def append_int16_bytes(self, raw: bytes) -> None:
        """Append already-int16-encoded PCM, skipping the numpy round-trip.

        Raises ValueError on an odd-length buffer rather than writing garbage.
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
    """Context-manager wrapper; ``path=None`` yields None so callers can treat
    the writer as optional without a separate code path."""
    if path is None:
        yield None
        return
    writer = WavWriter(path, sample_rate=sample_rate)
    try:
        yield writer
    finally:
        writer.close()


def default_audio_path(meeting_id: str, stream: str, *, root: Path | None = None) -> Path:
    """Resolve the on-disk path for a meeting and stream pair.

    Defaults to ``~/.meetmind/audio/<meeting_id>_<stream>.wav``; the parent
    directory is created lazily by the writer.
    """
    if root is None:
        root = Path.home() / ".meetmind" / "audio"
    # No path-traversal risk: ULIDs are alphanumeric only.
    return root / f"{meeting_id}_{stream}.wav"

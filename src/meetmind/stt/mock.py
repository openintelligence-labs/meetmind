"""Deterministic STT backend for tests, CI, and platforms without a native
backend. Emits a scripted sequence of partials and finals driven by RMS
energy, without loading a model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

import numpy as np

from meetmind.stt.base import Final, Partial


@dataclass
class MockSTTBackend:
    """Emit a fixed transcript phrase whenever cumulative RMS exceeds threshold."""

    name: str = "mock"
    phrase: str = "the quick brown fox jumps over the lazy dog"
    rms_threshold: float = 0.005
    finalize_after_seconds: float = 2.0
    sample_rate: int = 16_000

    _energy_acc: float = field(default=0.0, init=False)
    _energy_n: int = field(default=0, init=False)
    _open_start_ms: int | None = field(default=None, init=False)
    _open_text_chars: int = field(default=0, init=False)
    _t_ms: int = field(default=0, init=False)

    async def stream(
        self,
        frames: AsyncIterator[np.ndarray],
        sample_rate: int = 16_000,
    ) -> AsyncIterator[Partial | Final]:
        words = self.phrase.split()
        async for f in frames:
            n = f.shape[0]
            duration_ms = int(round(1000 * n / sample_rate))
            rms = float(np.sqrt(np.mean(f**2))) if n else 0.0
            if rms > self.rms_threshold:
                if self._open_start_ms is None:
                    self._open_start_ms = self._t_ms
                self._energy_acc += rms * n
                self._energy_n += n
                # Reveal one word per ~200 ms of voiced audio.
                target_chars = min(
                    len(self.phrase),
                    self._open_text_chars + max(1, duration_ms // 50),
                )
                if target_chars > self._open_text_chars:
                    self._open_text_chars = target_chars
                    visible = " ".join(
                        w for w in (self.phrase[: self._open_text_chars].strip().split())
                    )
                    if visible:
                        yield Partial(
                            text=visible,
                            start_ms=self._open_start_ms,
                            end_ms=self._t_ms + duration_ms,
                            confidence=0.6,
                        )
            else:
                if self._open_start_ms is not None:
                    open_dur_s = (self._t_ms - self._open_start_ms) / 1000.0
                    if open_dur_s >= self.finalize_after_seconds * 0.25:
                        text = " ".join(words[: max(1, self._open_text_chars // 5)])
                        yield Final(
                            text=text or self.phrase,
                            start_ms=self._open_start_ms,
                            end_ms=self._t_ms,
                            confidence=0.85,
                        )
                    self._open_start_ms = None
                    self._open_text_chars = 0
            self._t_ms += duration_ms

        if self._open_start_ms is not None:
            yield Final(
                text=self.phrase,
                start_ms=self._open_start_ms,
                end_ms=self._t_ms,
                confidence=0.9,
            )

    async def transcribe(
        self,
        audio: np.ndarray | Iterable[np.ndarray],
        sample_rate: int = 16_000,
    ) -> Final:
        arr = audio if isinstance(audio, np.ndarray) else np.concatenate(list(audio))
        n = arr.shape[0]
        return Final(
            text=self.phrase,
            start_ms=0,
            end_ms=int(1000 * n / sample_rate),
            confidence=0.95,
        )

    async def aclose(self) -> None:
        return None

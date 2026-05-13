"""STT backend protocol.

Two operating modes:

* **Streaming** — the live captioning tier. Backend consumes 16 kHz
  mono float32 frames as they're produced by the capture pipeline and
  emits `Partial`s (incremental hypothesis) and `Final`s (committed,
  punctuated text spans). Used for live captions, coaching, assist.
* **Batch** — the polish tier. Backend consumes a complete utterance
  (or a whole meeting) and returns a single high-quality transcript.
  Used for the post-meeting Whisper / Canary polish pass.

Implementations live alongside this file — `mock.py` for tests, and
`parakeet_v3.py` (S1.6) for the production FluidAudio path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Partial:
    """An incremental, possibly-revisable hypothesis from a streaming STT.

    `text` is the full hypothesis since the last Final, not just the
    delta — this matches Whisper / Parakeet streaming conventions and
    makes overlay rendering trivial (just replace last line).
    """

    text: str
    start_ms: int
    end_ms: int
    confidence: float = 0.0


@dataclass(frozen=True)
class Final:
    """A committed transcript span. No further revision."""

    text: str
    start_ms: int
    end_ms: int
    confidence: float = 0.0
    language: str = "en"


@runtime_checkable
class STTBackend(Protocol):
    """The contract every STT implementation must satisfy."""

    name: str

    async def stream(
        self,
        frames: AsyncIterator[np.ndarray],
        sample_rate: int = 16_000,
    ) -> AsyncIterator[Partial | Final]: ...

    async def transcribe(
        self,
        audio: np.ndarray | Iterable[np.ndarray],
        sample_rate: int = 16_000,
    ) -> Final: ...

    async def aclose(self) -> None: ...

"""STT backend protocol, covering two operating modes.

Streaming backends consume 16 kHz mono float32 frames from the capture
pipeline and emit `Partial`s and `Final`s for live captioning. Batch backends
consume a complete utterance or meeting and return one high-quality transcript
for the post-meeting polish pass.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Partial:
    """An incremental, possibly-revisable hypothesis from a streaming STT.

    `text` is the full hypothesis since the last Final, not the delta, so
    an overlay renders by replacing its last line.
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

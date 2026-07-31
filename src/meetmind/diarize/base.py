"""Diarization base types and backend protocol.

Output is deliberately minimal: spans with an opaque cluster id. Resolving a
cluster to a `Speaker` is a separate step in `diarize/voiceprint.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from meetmind.ipc import StreamId


@dataclass(frozen=True)
class DiarSegment:
    """One contiguous span attributed to a single speaker cluster."""

    start_ms: int
    end_ms: int
    cluster_id: str
    confidence: float = 0.0
    channel: StreamId | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@runtime_checkable
class DiarBackend(Protocol):
    """Streaming diarizer.

    Consumes per-frame audio with stream and start_ms metadata, emitting
    `DiarSegment`s. Implementations may emit overlapping segments and let the
    stitcher reconcile them, or emit only finalized ones.
    """

    name: str

    async def stream(
        self,
        frames: AsyncIterator[tuple[StreamId, np.ndarray, int]],
        sample_rate: int = 16_000,
    ) -> AsyncIterator[DiarSegment]: ...

    async def aclose(self) -> None: ...

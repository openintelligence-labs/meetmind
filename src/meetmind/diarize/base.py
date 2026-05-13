"""Diarization base types + protocol.

Diarization output is intentionally minimal: just `(start_ms, end_ms,
cluster_id, confidence, channel)` tuples. Identity resolution (cluster
→ Speaker) is a separate step that lives in `diarize/voiceprint.py`.

Cluster IDs are opaque strings — usually short identifiers like "A",
"B", "remote-2". The pipeline downstream treats them as labels.

`channel` is the originating capture stream. The channel-prior gate
overrides cluster IDs to "self" / "remote" when the mic vs loopback
split makes the answer trivial.
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

    Consumes per-frame audio (with stream + start_ms metadata) and emits
    `DiarSegment`s. Segments may be revised — implementations are free
    to emit overlapping segments and rely on the stitcher to reconcile,
    or to emit only finalized segments. The mock + production
    Sortformer adapter both emit only finalized.
    """

    name: str

    async def stream(
        self,
        frames: AsyncIterator[tuple[StreamId, np.ndarray, int]],
        sample_rate: int = 16_000,
    ) -> AsyncIterator[DiarSegment]: ...

    async def aclose(self) -> None: ...

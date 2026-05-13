"""Pure-Python diarization mock for tests + CI.

Emits one DiarSegment per voiced run, alternating between two cluster
ids (`A` and `B`). RMS-thresholded so it produces deterministic output
on synthetic inputs. Used by tests + the CLI smoke harness.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

from meetmind.diarize.base import DiarSegment
from meetmind.ipc import StreamId


@dataclass
class MockDiarBackend:
    """Two-cluster RMS-gated diarizer."""

    name: str = "mock-diar"
    rms_threshold: float = 0.005
    silence_close_ms: int = 200

    _open: dict[StreamId, dict] = field(default_factory=dict, init=False)
    _last_cluster: dict[StreamId, str] = field(default_factory=dict, init=False)

    async def stream(
        self,
        frames: AsyncIterator[tuple[StreamId, np.ndarray, int]],
        sample_rate: int = 16_000,
    ) -> AsyncIterator[DiarSegment]:
        async for stream, pcm, start_ms in frames:
            duration_ms = int(round(1000 * pcm.shape[0] / sample_rate))
            rms = float(np.sqrt(np.mean(pcm**2))) if pcm.size else 0.0
            o = self._open.setdefault(stream, {"start_ms": None, "last_voiced_ms": None})
            if rms > self.rms_threshold:
                if o["start_ms"] is None:
                    last = self._last_cluster.get(stream, "B")
                    next_cluster = "A" if last == "B" else "B"
                    self._last_cluster[stream] = next_cluster
                    o["start_ms"] = start_ms
                    o["cluster"] = next_cluster
                o["last_voiced_ms"] = start_ms + duration_ms
            else:
                if (
                    o["start_ms"] is not None
                    and o["last_voiced_ms"] is not None
                    and start_ms + duration_ms - o["last_voiced_ms"] >= self.silence_close_ms
                ):
                    yield DiarSegment(
                        start_ms=o["start_ms"],
                        end_ms=o["last_voiced_ms"],
                        cluster_id=o["cluster"],
                        confidence=0.8,
                        channel=stream,
                    )
                    o["start_ms"] = None
                    o["last_voiced_ms"] = None

        # Flush any open segments.
        for stream, o in self._open.items():
            if o["start_ms"] is not None and o["last_voiced_ms"] is not None:
                yield DiarSegment(
                    start_ms=o["start_ms"],
                    end_ms=o["last_voiced_ms"],
                    cluster_id=o["cluster"],
                    confidence=0.85,
                    channel=stream,
                )

    async def aclose(self) -> None:
        return None

"""Tests for the diarization mock backend + base types."""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from meetmind.diarize.base import DiarBackend, DiarSegment
from meetmind.diarize.mock import MockDiarBackend
from meetmind.ipc import StreamId


def _silence(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


def _tone(samples: int, freq: float = 600.0) -> np.ndarray:
    t = np.arange(samples) / 16_000
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


async def _frames(
    *triples: tuple[StreamId, np.ndarray, int],
) -> AsyncIterator[tuple[StreamId, np.ndarray, int]]:
    for x in triples:
        yield x


def test_diar_segment_duration_property():
    seg = DiarSegment(start_ms=100, end_ms=350, cluster_id="A")
    assert seg.duration_ms == 250


def test_mock_backend_satisfies_protocol():
    backend = MockDiarBackend()
    assert isinstance(backend, DiarBackend)
    assert backend.name == "mock-diar"


async def test_mock_diarizer_emits_two_segments_separated_by_silence():
    frames = []
    t = 0
    for _ in range(7):
        frames.append((StreamId.MIC, _tone(512), t))
        t += 32
    for _ in range(8):
        frames.append((StreamId.MIC, _silence(512), t))
        t += 32
    for _ in range(7):
        frames.append((StreamId.MIC, _tone(512), t))
        t += 32

    segments = []
    async for seg in MockDiarBackend(rms_threshold=0.001).stream(_frames(*frames)):
        segments.append(seg)

    assert len(segments) == 2
    assert {s.cluster_id for s in segments} == {"A", "B"}
    assert all(s.channel is StreamId.MIC for s in segments)
    assert segments[0].end_ms < segments[1].start_ms


async def test_mock_diarizer_keeps_streams_independent():
    frames = []
    t = 0
    for _ in range(5):
        frames.append((StreamId.MIC, _tone(512), t))
        frames.append((StreamId.LOOPBACK, _tone(512, freq=900.0), t))
        t += 32

    segments = []
    async for seg in MockDiarBackend(rms_threshold=0.001).stream(_frames(*frames)):
        segments.append(seg)

    by_stream = {s.channel: s for s in segments}
    assert StreamId.MIC in by_stream
    assert StreamId.LOOPBACK in by_stream

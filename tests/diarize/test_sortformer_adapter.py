"""Integration tests for SortformerSidecarBackend against the mock diar sidecar."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from meetmind.diarize.base import DiarSegment
from meetmind.diarize.live import SortformerSidecarBackend
from meetmind.ipc import StreamId

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mock_diar_sidecar.py"


@pytest.fixture
def mock_diar_binary() -> Path:
    wrapper = FIXTURE.parent / "_mock_diar_launcher.sh"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{FIXTURE}" "$@"\n')
    wrapper.chmod(0o755)
    return wrapper


def _tone(samples: int, freq: float = 600.0) -> np.ndarray:
    t = np.arange(samples) / 16000
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silent(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


async def _frames(
    *triples: tuple[StreamId, np.ndarray, int],
) -> AsyncIterator[tuple[StreamId, np.ndarray, int]]:
    for x in triples:
        yield x


async def test_sortformer_emits_at_least_one_segment(mock_diar_binary):
    backend = SortformerSidecarBackend(binary=mock_diar_binary)
    async with backend:
        triples = []
        t = 0
        for _ in range(10):
            triples.append((StreamId.MIC, _tone(512), t))
            t += 32
        for _ in range(10):
            triples.append((StreamId.MIC, _silent(512), t))
            t += 32

        segments = []
        async for seg in backend.stream(_frames(*triples)):
            segments.append(seg)
        assert len(segments) >= 1
        assert all(isinstance(s, DiarSegment) for s in segments)
        assert segments[0].channel is StreamId.MIC


async def test_sortformer_handles_dual_streams(mock_diar_binary):
    backend = SortformerSidecarBackend(binary=mock_diar_binary)
    async with backend:
        triples = []
        t = 0
        for _ in range(15):
            triples.append((StreamId.MIC, _tone(512), t))
            triples.append((StreamId.LOOPBACK, _tone(512, freq=900.0), t))
            t += 32

        segments = []
        async for seg in backend.stream(_frames(*triples)):
            segments.append(seg)

        channels = {s.channel for s in segments}
        assert StreamId.MIC in channels
        assert StreamId.LOOPBACK in channels

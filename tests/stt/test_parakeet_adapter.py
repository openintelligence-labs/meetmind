"""Tests for ParakeetSidecarBackend against the mock STT sidecar."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from meetmind.ipc import StreamId
from meetmind.stt.base import Final, Partial
from meetmind.stt.parakeet_v3 import ParakeetSidecarBackend

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mock_stt_sidecar.py"


@pytest.fixture
def mock_stt_binary() -> Path:
    wrapper = FIXTURE.parent / "_mock_stt_launcher.sh"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{FIXTURE}" "$@"\n')
    wrapper.chmod(0o755)
    return wrapper


def _tone_frames(num_frames: int, samples: int = 512, freq: float = 600.0):
    t = np.arange(samples) / 16000
    tone = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    async def _gen() -> AsyncIterator[np.ndarray]:
        for _ in range(num_frames):
            yield tone

    return _gen()


async def test_adapter_yields_partials_then_final_for_voiced_input(mock_stt_binary):
    backend = ParakeetSidecarBackend(binary=mock_stt_binary)
    async with backend:
        events = []
        async for evt in backend.stream(_tone_frames(20), stream_id=StreamId.MIC):
            events.append(evt)
        partials = [e for e in events if isinstance(e, Partial)]
        finals = [e for e in events if isinstance(e, Final)]
        assert len(partials) >= 1
        assert len(finals) >= 1
        assert finals[-1].text  # non-empty
        assert finals[-1].language == "en"
        lengths = [len(p.text) for p in partials]
        assert lengths == sorted(lengths)


async def test_adapter_transcribe_returns_final(mock_stt_binary):
    backend = ParakeetSidecarBackend(binary=mock_stt_binary)
    async with backend:
        n = 16000 * 2
        t = np.arange(n) / 16000
        audio = (0.4 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
        final = await backend.transcribe(audio)
        assert isinstance(final, Final)
        assert final.text
        assert final.end_ms > 0

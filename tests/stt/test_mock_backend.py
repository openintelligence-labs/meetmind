"""Tests for the MockSTTBackend + STTBackend protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from meetmind.stt.base import Final, Partial, STTBackend
from meetmind.stt.mock import MockSTTBackend


async def _frames_from(*arrays: np.ndarray) -> AsyncIterator[np.ndarray]:
    for a in arrays:
        yield a


def _silence(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


def _tone(samples: int, freq: float = 600.0, sr: int = 16_000) -> np.ndarray:
    t = np.arange(samples) / sr
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mock_backend_satisfies_protocol():
    backend = MockSTTBackend()
    # Runtime-checkable Protocol, so this tracks changes to base.py.
    assert isinstance(backend, STTBackend)
    assert backend.name == "mock"


async def test_mock_streams_partials_for_voiced_audio():
    backend = MockSTTBackend(rms_threshold=0.001)
    # 4 × 512-sample frames of voiced audio = ~128 ms.
    voiced = [_tone(512) for _ in range(4)]
    out = []
    async for evt in backend.stream(_frames_from(*voiced)):
        out.append(evt)
    partials = [e for e in out if isinstance(e, Partial)]
    assert len(partials) >= 1
    # Hypothesis text grows monotonically.
    lengths = [len(p.text) for p in partials]
    assert lengths == sorted(lengths)


async def test_mock_emits_final_after_silence():
    # Mock finalizes when an open utterance is followed by silence AND
    # the open span exceeds finalize_after_seconds * 0.25.
    backend = MockSTTBackend(rms_threshold=0.001, finalize_after_seconds=0.4)
    voiced = [_tone(512) for _ in range(20)]  # ~640 ms (above 100 ms threshold)
    silent = [_silence(512) for _ in range(3)]  # ~96 ms silence
    out = []
    async for evt in backend.stream(_frames_from(*voiced, *silent)):
        out.append(evt)
    finals = [e for e in out if isinstance(e, Final)]
    assert len(finals) >= 1
    assert finals[-1].text  # non-empty
    assert finals[-1].confidence > 0.5
    assert finals[-1].end_ms > finals[-1].start_ms


async def test_mock_transcribe_returns_final_proportional_to_length():
    backend = MockSTTBackend()
    audio = _tone(16_000)  # 1 second
    final = await backend.transcribe(audio)
    assert isinstance(final, Final)
    assert final.start_ms == 0
    assert final.end_ms == 1000
    assert final.text


async def test_mock_aclose_is_idempotent():
    backend = MockSTTBackend()
    await backend.aclose()
    await backend.aclose()  # second call must not raise

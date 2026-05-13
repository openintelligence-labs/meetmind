"""Tests for the voiceprint embedder (mel-hash fallback)."""

from __future__ import annotations

import numpy as np
import pytest

from meetmind.diarize.voiceprint import (
    SAMPLE_RATE,
    VOICEPRINT_DIM,
    MelHashVoiceprintEmbedder,
    default_embedder,
)


def _tone(seconds: float, freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _formant_vowel(seconds: float, formants: list[float], sr: int = SAMPLE_RATE) -> np.ndarray:
    audio = np.zeros(int(seconds * sr), dtype=np.float32)
    for f in formants:
        audio += _tone(seconds, f, sr)
    audio /= max(1.0, np.abs(audio).max())
    return audio


def test_embedder_outputs_unit_normalized_vector():
    emb = MelHashVoiceprintEmbedder()
    vec = emb.embed(_tone(1.0, 440))
    assert vec.shape == (VOICEPRINT_DIM,)
    assert pytest.approx(float(np.linalg.norm(vec)), rel=1e-5) == 1.0


def test_embedder_is_deterministic():
    emb = MelHashVoiceprintEmbedder()
    a = emb.embed(_tone(1.0, 440))
    b = emb.embed(_tone(1.0, 440))
    assert np.allclose(a, b)


def test_embedder_separates_distinct_tones():
    emb = MelHashVoiceprintEmbedder()
    sam = emb.embed(_formant_vowel(1.0, [220, 700, 1200]))
    priya = emb.embed(_formant_vowel(1.0, [330, 900, 2400]))
    cosine = float(np.dot(sam, priya))
    assert abs(cosine) < 0.99


def test_embedder_handles_short_audio_without_crashing():
    emb = MelHashVoiceprintEmbedder()
    short = np.zeros(100, dtype=np.float32)
    vec = emb.embed(short)
    assert vec.shape == (VOICEPRINT_DIM,)


def test_embedder_handles_silence():
    emb = MelHashVoiceprintEmbedder()
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    vec = emb.embed(silence)
    assert vec.shape == (VOICEPRINT_DIM,)


def test_embedder_resamples_non_16k_input():
    emb = MelHashVoiceprintEmbedder()
    audio_48k = _tone(1.0, 440, sr=48_000)
    vec_48k = emb.embed(audio_48k, sample_rate=48_000)
    vec_16k = emb.embed(_tone(1.0, 440, sr=16_000))
    assert vec_48k.shape == vec_16k.shape == (VOICEPRINT_DIM,)
    cosine = float(np.dot(vec_48k, vec_16k))
    assert cosine > 0.5


def test_default_embedder_returns_something_usable(monkeypatch):
    monkeypatch.delenv("MEETMIND_VOICEPRINT_MODEL", raising=False)
    emb = default_embedder()
    assert emb.dim == VOICEPRINT_DIM
    assert hasattr(emb, "embed")
    vec = emb.embed(_tone(1.0, 440))
    assert vec.shape == (VOICEPRINT_DIM,)


def test_embedder_protocol_is_satisfied():
    emb = MelHashVoiceprintEmbedder()
    assert hasattr(emb, "name") and isinstance(emb.name, str)
    assert hasattr(emb, "dim") and isinstance(emb.dim, int)
    assert callable(emb.embed)

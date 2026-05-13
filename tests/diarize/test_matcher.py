"""Tests for the cosine voiceprint matcher."""

from __future__ import annotations

import numpy as np
import pytest

from meetmind.diarize.matcher import (
    Matcher,
    MatcherConfig,
    _decode_centroid,
    _decode_ring,
    _to_unit,
    cosine,
)
from meetmind.models import Speaker


def _embed(seed: int, dim: int = 192, drift: float = 0.0) -> np.ndarray:
    """Deterministic pseudo-embedding seeded by `seed`."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim).astype(np.float32)
    if drift:
        base += drift * rng.standard_normal(dim).astype(np.float32)
    return base


def _enroll(seed: int, *, name: str, dim: int = 192) -> Speaker:
    """Build a Speaker with a centroid embedded from `seed`."""
    sp = Speaker(id=f"01{name.upper()}", display_name=name)
    return Matcher().update_centroid(sp, _embed(seed, dim))


def test_to_unit_normalizes_to_length_one():
    v = np.array([3.0, 4.0], dtype=np.float32)
    u = _to_unit(v)
    assert pytest.approx(np.linalg.norm(u), rel=1e-6) == 1.0


def test_to_unit_handles_zero_vector():
    v = np.zeros(8, dtype=np.float32)
    u = _to_unit(v)
    assert np.all(u == 0)


def test_cosine_identical_vectors_is_one():
    v = _embed(1)
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine(a, b) == pytest.approx(0.0, abs=1e-6)


def test_match_returns_speaker_when_embedding_matches():
    sam = _enroll(1, name="sam")
    priya = _enroll(2, name="priya")
    matcher = Matcher()
    # Re-embed Sam exactly → should accept.
    decision = matcher.match(_embed(1), [sam, priya])
    assert decision.speaker_id == sam.id
    assert decision.band == "accept"
    assert decision.cosine > 0.95


def test_match_returns_uncertain_for_mid_band_cosine():
    """Construct an embedding 60° off Sam's centroid (cos ≈ 0.5).

    A 60-degree mix of two random unit vectors lands in [0.45, 0.58]
    most of the time. We pick a deterministic seed that does.
    """
    centroid = _to_unit(_embed(1))
    other = _to_unit(_embed(99))
    # cos ≈ 0.5 by construction
    mixed = _to_unit(np.cos(1.0472) * centroid + np.sin(1.0472) * other)
    sam = Speaker(
        id="01SAM",
        display_name="Sam",
        voiceprint_centroid=centroid.tobytes(),
    )
    matcher = Matcher()
    decision = matcher.match(mixed, [sam])
    assert decision.band == "uncertain"
    assert decision.speaker_id is None  # don't auto-attribute when uncertain


def test_match_rejects_orthogonal_embedding():
    sam = _enroll(1, name="sam")
    matcher = Matcher()
    far = _embed(99)
    decision = matcher.match(far, [sam])
    # Random embeddings in 192-d are nearly orthogonal — cos < 0.45 → reject.
    assert decision.band == "reject"
    assert decision.speaker_id is None


def test_match_with_no_enrolled_speakers_is_a_reject():
    matcher = Matcher()
    decision = matcher.match(_embed(1), [])
    assert decision.band == "reject"
    assert decision.speaker_id is None


def test_match_skips_speakers_with_no_centroid():
    """A Speaker row without a stored centroid is invisible to the matcher."""
    sam = Speaker(id="01SAM", display_name="Sam")  # no centroid
    matcher = Matcher()
    decision = matcher.match(_embed(1), [sam])
    assert decision.band == "reject"
    assert decision.speaker_id is None


def test_priors_bias_match_when_cosines_are_close():
    """When two speakers tie on cosine, calendar prior breaks the tie."""
    centroid = _to_unit(_embed(1))
    sam = Speaker(id="01SAM", display_name="Sam", voiceprint_centroid=centroid.tobytes())
    priya = Speaker(id="01PRIYA", display_name="Priya", voiceprint_centroid=centroid.tobytes())
    # Both speakers have the SAME centroid; cos to either is identical.
    matcher = Matcher()
    biased = matcher.match(
        _embed(1),
        [sam, priya],
        priors={"01SAM": 0.9, "01PRIYA": 0.05},
    )
    assert biased.speaker_id == "01SAM"


def test_should_update_gates_on_quality_signals():
    matcher = Matcher()
    # All signals good → should update.
    assert matcher.should_update(cos=0.7, snr_db=20.0, duration_seconds=3.0) is True
    # Cos too low → no update.
    assert matcher.should_update(cos=0.6, snr_db=20.0, duration_seconds=3.0) is False
    # Duration too short → no update.
    assert matcher.should_update(cos=0.7, snr_db=20.0, duration_seconds=2.0) is False
    # Noisy → no update.
    assert matcher.should_update(cos=0.7, snr_db=5.0, duration_seconds=3.0) is False
    # SNR not measured → no SNR gate.
    assert matcher.should_update(cos=0.7, snr_db=None, duration_seconds=3.0) is True


def test_update_centroid_starts_fresh_when_speaker_has_none():
    matcher = Matcher()
    sp = Speaker(id="01NEW", display_name="New")
    updated = matcher.update_centroid(sp, _embed(1))
    centroid = _decode_centroid(updated)
    assert centroid is not None
    assert pytest.approx(np.linalg.norm(centroid), rel=1e-5) == 1.0
    ring = _decode_ring(updated)
    assert len(ring) == 1


def test_update_centroid_ema_drifts_towards_new_sample():
    matcher = Matcher()
    sp = _enroll(1, name="sam")
    before = _decode_centroid(sp)
    assert before is not None
    drifted = matcher.update_centroid(sp, _embed(2))
    after = _decode_centroid(drifted)
    assert after is not None
    # Centroid moved toward the new sample but not all the way (α=0.05).
    delta = float(np.linalg.norm(after - before))
    assert 0.0 < delta < 0.3


def test_update_centroid_ring_caps_at_capacity():
    matcher = Matcher(MatcherConfig(ring_capacity=4))
    sp = Speaker(id="01R", display_name="Ring")
    for i in range(7):
        sp = matcher.update_centroid(sp, _embed(i + 10))
    ring = _decode_ring(sp)
    assert len(ring) == 4


def test_recentroid_recomputes_from_ring():
    matcher = Matcher()
    sp = Speaker(id="01R", display_name="Ring")
    for i in range(5):
        sp = matcher.update_centroid(sp, _embed(i + 100))
    # Manually corrupt centroid; recentroid should fix it.
    sp = sp.model_copy(update={"voiceprint_centroid": np.zeros(192, np.float32).tobytes()})
    fixed = matcher.recentroid(sp)
    centroid = _decode_centroid(fixed)
    assert centroid is not None
    assert pytest.approx(np.linalg.norm(centroid), rel=1e-5) == 1.0

"""Tests for calendar-prior Bayesian fusion."""

from __future__ import annotations

import numpy as np
import pytest

from meetmind.diarize.calendar_prior import CalendarPrior, attendee_overlap, bayesian_priors
from meetmind.diarize.matcher import Matcher
from meetmind.models import Speaker


def _enroll(speaker_id: str, name: str, seed: int) -> Speaker:
    rng = np.random.default_rng(seed)
    centroid = rng.standard_normal(192).astype(np.float32)
    sp = Speaker(id=speaker_id, display_name=name)
    return Matcher().update_centroid(sp, centroid)


def test_no_calendar_returns_empty_priors():
    speakers = [_enroll("01SAM", "Sam", 1), _enroll("01PRIYA", "Priya", 2)]
    assert bayesian_priors(speakers, calendar_attendee_ids=None) == {}
    assert bayesian_priors(speakers, calendar_attendee_ids=[]) == {}


def test_calendar_priors_split_remaining_mass():
    speakers = [
        _enroll("01SAM", "Sam", 1),
        _enroll("01PRIYA", "Priya", 2),
        _enroll("01BOB", "Bob", 3),
    ]
    priors = bayesian_priors(speakers, calendar_attendee_ids=["01SAM", "01PRIYA"])
    assert priors["01SAM"] == pytest.approx(0.45, abs=1e-6)
    assert priors["01PRIYA"] == pytest.approx(0.45, abs=1e-6)
    assert "01BOB" not in priors


def test_non_attendee_residual_distributes_mass():
    speakers = [
        _enroll("01SAM", "Sam", 1),
        _enroll("01PRIYA", "Priya", 2),
        _enroll("01BOB", "Bob", 3),
        _enroll("01EVE", "Eve", 4),
    ]
    priors = bayesian_priors(
        speakers,
        calendar_attendee_ids=["01SAM"],
        config=CalendarPrior(unknown_alpha=0.10, non_attendee_residual=0.30),
    )
    assert priors["01SAM"] == pytest.approx(0.60, abs=1e-6)
    for sid in ("01PRIYA", "01BOB", "01EVE"):
        assert priors[sid] == pytest.approx(0.10, abs=1e-6)


def test_unenrolled_calendar_attendees_dont_break_priors():
    speakers = [_enroll("01SAM", "Sam", 1)]
    priors = bayesian_priors(speakers, calendar_attendee_ids=["01SAM", "01STRANGER"])
    assert "01SAM" in priors
    assert "01STRANGER" not in priors
    assert priors["01SAM"] == pytest.approx(0.90, abs=1e-6)


def test_attendee_overlap_reports_known_and_unknown():
    speakers = [_enroll("01SAM", "Sam", 1), _enroll("01PRIYA", "Priya", 2)]
    known, unknown = attendee_overlap(speakers, calendar_attendee_ids=["01SAM", "01STRANGER"])
    assert known == {"01SAM"}
    assert unknown == {"01STRANGER"}


def test_calendar_prior_breaks_cosine_ties_in_matcher():
    """Two speakers with identical centroids: without a prior, the
    matcher correctly refuses to pick (50/50 → uncertain band). With a
    calendar prior favoring Sam, the matcher confidently chooses him.
    This is the "free 30%" wedge from combining self/remote splits
    with calendar-attendee priors."""
    rng = np.random.default_rng(42)
    centroid = rng.standard_normal(192).astype(np.float32)
    sam = Speaker(id="01SAM", display_name="Sam", voiceprint_centroid=centroid.tobytes())
    priya = Speaker(id="01PRIYA", display_name="Priya", voiceprint_centroid=centroid.tobytes())
    matcher = Matcher()

    # Without a prior: 50/50 ambiguity → matcher refuses to attribute.
    decision_uniform = matcher.match(centroid, [sam, priya])
    assert decision_uniform.speaker_id is None
    assert decision_uniform.band == "uncertain"

    # With calendar prior on Sam: posterior crosses the threshold.
    priors = bayesian_priors([sam, priya], calendar_attendee_ids=["01SAM"])
    decision = matcher.match(centroid, [sam, priya], priors=priors)
    assert decision.speaker_id == "01SAM"
    assert decision.band == "accept"
    assert decision.posterior > decision_uniform.posterior

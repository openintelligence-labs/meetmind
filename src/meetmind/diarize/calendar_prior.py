"""Calendar-prior Bayesian fusion.

When the calendar tells us a meeting has N specific attendees, the
matcher can do better than uniform-prior cosine ranking. We bias
posterior(speaker | embedding) toward the calendar set while reserving
some mass for ``UNKNOWN`` (someone joined unexpectedly).

This module is purely functional: it builds a ``priors`` dict that
``Matcher.match()`` accepts. No state, no I/O, no model.

The fusion is the documented log-linear form:

    posterior(c) ∝ exp(cos(emb, μ_c) / τ) × prior(c)
    prior(c)     = (1 - α) / N      for c in calendar_attendees
    prior(other) = (residual / |others|) for non-attendees with centroids
    prior(UNKNOWN) = α               (synthetic candidate at accept_threshold)

Decide identity only when ``posterior(best) ≥ threshold AND
cos(best) ≥ accept_threshold``. The ``Matcher`` already enforces both
gates; this module just shapes the priors.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from meetmind.models import Speaker


@dataclass(frozen=True)
class CalendarPrior:
    """Configuration knobs for calendar-attendee biasing."""

    unknown_alpha: float = 0.10  # mass reserved for "not in the calendar"
    non_attendee_residual: float = 0.0  # mass shared across non-attendees
    """0.0 means non-attendee speakers get zero weight in the prior; the
    only way they can win is if posterior ≥ threshold without a prior
    boost (i.e. their cosine is overwhelmingly higher than the
    attendees'). Set to e.g. 0.05 to allow a fall-through when calendar
    metadata is wrong."""


def bayesian_priors(
    speakers: Iterable[Speaker],
    *,
    calendar_attendee_ids: Iterable[str] | None = None,
    config: CalendarPrior | None = None,
) -> dict[str, float]:
    """Build a ``{speaker_id: prior}`` map for ``Matcher.match(priors=…)``.

    If ``calendar_attendee_ids`` is None or empty, returns an empty dict
    (the matcher falls back to a uniform prior).
    """
    cfg = config or CalendarPrior()
    attendees = set(calendar_attendee_ids or [])
    if not attendees:
        return {}

    speakers = list(speakers)
    speaker_ids = {s.id for s in speakers}

    in_calendar = [sid for sid in attendees if sid in speaker_ids]
    out_of_calendar = [sid for sid in speaker_ids if sid not in attendees]

    priors: dict[str, float] = {}
    if in_calendar:
        per_attendee = (1.0 - cfg.unknown_alpha - cfg.non_attendee_residual) / len(in_calendar)
        for sid in in_calendar:
            priors[sid] = per_attendee

    if out_of_calendar and cfg.non_attendee_residual > 0:
        per_other = cfg.non_attendee_residual / len(out_of_calendar)
        for sid in out_of_calendar:
            priors[sid] = per_other

    # Anyone in attendees who isn't enrolled yet contributes to UNKNOWN
    # mass implicitly — Matcher.match() treats UNKNOWN as a synthetic
    # candidate at config.unknown_prior. This module doesn't need to
    # touch UNKNOWN explicitly.

    return priors


def attendee_overlap(
    speakers: Iterable[Speaker],
    calendar_attendee_ids: Iterable[str] | None,
) -> tuple[set[str], set[str]]:
    """Diagnostic helper: ``(known_attendees, unknown_attendees)``.

    ``known_attendees`` are calendar IDs we already have voiceprints for;
    ``unknown_attendees`` are calendar IDs without enrollment — those
    show up as ``cluster_id="unknown"`` until the user attributes them.
    """
    attendees = set(calendar_attendee_ids or [])
    known_ids = {s.id for s in speakers}
    return (attendees & known_ids, attendees - known_ids)

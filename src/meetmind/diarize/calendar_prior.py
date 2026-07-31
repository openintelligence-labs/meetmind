"""Calendar-attendee priors for the voiceprint matcher.

Builds the ``priors`` dict ``Matcher.match()`` accepts, biasing toward known
calendar attendees while reserving mass for someone who joined unexpectedly:

    prior(c)       = (1 - α - residual) / N  for c in calendar_attendees
    prior(other)   = residual / |others|     for enrolled non-attendees
    prior(UNKNOWN) = α

Purely functional; the matcher enforces the posterior and cosine gates.
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
    """At 0.0 a non-attendee can only win on an overwhelmingly higher cosine.
    Raise it (e.g. 0.05) to leave a fall-through path for wrong calendar
    metadata."""


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

    # Unenrolled attendees need no entry: Matcher.match() already carries
    # UNKNOWN as a synthetic candidate at config.unknown_prior.
    return priors


def attendee_overlap(
    speakers: Iterable[Speaker],
    calendar_attendee_ids: Iterable[str] | None,
) -> tuple[set[str], set[str]]:
    """Split calendar IDs into ``(enrolled, not_enrolled)``.

    Unenrolled attendees show up as ``cluster_id="unknown"`` until the user
    attributes them.
    """
    attendees = set(calendar_attendee_ids or [])
    known_ids = {s.id for s in speakers}
    return (attendees & known_ids, attendees - known_ids)

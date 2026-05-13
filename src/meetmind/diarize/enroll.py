"""Voiceprint enrollment + consent flow.

GDPR/BIPA/CUBI rules require:
  • **Explicit, written consent** per speaker before storing any
    voiceprint (biometric special-category data).
  • An immutable **audit log** of every enrollment / revocation /
    deletion, signed with a key the user controls.
  • A **deletion cascade** that removes the centroid + ring buffer +
    related segments' speaker_id pointers, while **retaining** the
    consent log itself for accountability.

This module orchestrates the workflow:

    enroll(name, embedding, store, identity, disclosure_version=…)
        → Speaker (with centroid populated) and a signed ConsentEvent

    revoke(speaker_id, store, identity, …)
        → flips Speaker centroid → None and writes a `revoke` ConsentEvent.
        Re-enrollment requires a fresh consent capture.

    forget(speaker_id, store, identity, …)
        → cascading delete of the speaker row + downstream attribution
        in transcript_segments. Consent log is retained.

The actual UX (CLI prompt today, Tauri modal later) is in `cli.py`.
This module is pure logic and has no I/O beyond the store + identity
parameters that the caller supplies.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal, Protocol

import numpy as np

from meetmind.crypto.canonicalize import canonical_json
from meetmind.crypto.identity import Identity
from meetmind.diarize.matcher import Matcher
from meetmind.models import ConsentEvent, Speaker


# Structural typing for the bits of `meetmind.memory.store.Store` we
# need. Keeps `diarize/` from importing `memory/` (the import-linter
# contract forbids it); the CLI passes the concrete Store at call time.
class _StoreLike(Protocol):
    def get_speaker(self, speaker_id: str) -> Speaker | None: ...
    def upsert_speaker(self, sp: Speaker) -> None: ...
    def append_consent_event(self, event: ConsentEvent) -> None: ...
    def forget_speaker(self, speaker_id: str) -> None: ...


# Each enrollment-flow change bumps this. The disclosure text shown to
# the user, the data we store, or the retention default — any of those
# changes flips the version. ConsentEvent rows record the version that
# was current at enrollment time so we can prove what the user agreed
# to even if we later evolve the policy.
CURRENT_DISCLOSURE_VERSION = "2026-05-v1"


# Default retention windows, picked to satisfy the strictest of BIPA /
# CUBI / MHMDA: BIPA requires destruction within 3 years of last
# interaction; CUBI within 1 year of "purpose served". 1 year matches
# both.
DEFAULT_RETENTION_DAYS = 365


@dataclass(frozen=True)
class EnrollmentResult:
    speaker: Speaker
    event: ConsentEvent


def _consent_payload(
    *,
    actor_speaker_id: str,
    action: Literal["enroll", "revoke", "delete", "export"],
    disclosure_version: str,
    ts: datetime,
) -> bytes:
    """Canonical bytes signed by the install identity for this event.

    The signature commits to (action, actor_speaker_id, version, ts).
    Anyone with the install's public key can verify the audit log
    wasn't tampered with after the fact.
    """
    return canonical_json(
        {
            "action": action,
            "actor_speaker_id": actor_speaker_id,
            "disclosure_version": disclosure_version,
            "ts": ts.isoformat(),
        }
    )


def _sign_event(
    identity: Identity,
    *,
    actor_speaker_id: str,
    action: Literal["enroll", "revoke", "delete", "export"],
    disclosure_version: str,
    ts: datetime,
) -> ConsentEvent:
    payload = _consent_payload(
        actor_speaker_id=actor_speaker_id,
        action=action,
        disclosure_version=disclosure_version,
        ts=ts,
    )
    sig = identity.sign(payload)
    return ConsentEvent(
        ts=ts,
        actor_speaker_id=actor_speaker_id,
        action=action,
        disclosure_version=disclosure_version,
        signature=sig,
    )


def verify_consent_event(
    event: ConsentEvent,
    identity: Identity,
) -> bool:
    """Verify an event's signature against the install's identity."""
    if event.signature is None:
        return False
    payload = _consent_payload(
        actor_speaker_id=event.actor_speaker_id,
        action=event.action,
        disclosure_version=event.disclosure_version,
        ts=event.ts,
    )
    return identity.verify(event.signature, payload)


# ---------------------------------------------------------------------------
# Enroll
# ---------------------------------------------------------------------------


def enroll(
    *,
    name: str,
    embedding: np.ndarray,
    store: _StoreLike,
    identity: Identity,
    matcher: Matcher | None = None,
    disclosure_version: str = CURRENT_DISCLOSURE_VERSION,
    aliases: Iterable[str] | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    speaker_id: str | None = None,
) -> EnrollmentResult:
    """Enroll a new speaker. Caller has already obtained explicit consent.

    Returns the persisted ``Speaker`` and the ``ConsentEvent`` written
    to the audit log. Both have stable ULIDs.
    """
    matcher = matcher or Matcher()
    now = datetime.now(UTC)
    expiry = now.date() + timedelta(days=retention_days)

    sp = Speaker(
        id=speaker_id or _new_speaker_id(),
        display_name=name,
        consent_ts=now,
        consent_disclosure_version=disclosure_version,
        aliases=list(aliases or []),
        retention_until=expiry,
        confidence=1.0,  # explicit human-confirmed enrollment
    )
    sp = matcher.update_centroid(sp, embedding)
    store.upsert_speaker(sp)

    event = _sign_event(
        identity,
        actor_speaker_id=sp.id,
        action="enroll",
        disclosure_version=disclosure_version,
        ts=now,
    )
    store.append_consent_event(event)
    return EnrollmentResult(speaker=sp, event=event)


def revoke(
    *,
    speaker_id: str,
    store: _StoreLike,
    identity: Identity,
    disclosure_version: str = CURRENT_DISCLOSURE_VERSION,
) -> ConsentEvent:
    """Revoke consent for ``speaker_id`` (clears centroid + ring).

    The Speaker row stays in place — segment attributions remain valid
    historically — but the voiceprint can no longer be used for
    future identification. Re-enrollment requires fresh consent.
    """
    sp = store.get_speaker(speaker_id)
    if sp is None:
        raise ValueError(f"no speaker {speaker_id} to revoke")

    now = datetime.now(UTC)
    cleared = sp.model_copy(
        update={
            "voiceprint_centroid": None,
            "voiceprint_ring": [],
            "consent_ts": None,
            "confidence": 0.0,
        }
    )
    store.upsert_speaker(cleared)

    event = _sign_event(
        identity,
        actor_speaker_id=speaker_id,
        action="revoke",
        disclosure_version=disclosure_version,
        ts=now,
    )
    store.append_consent_event(event)
    return event


def forget(
    *,
    speaker_id: str,
    store: _StoreLike,
    identity: Identity,
    disclosure_version: str = CURRENT_DISCLOSURE_VERSION,
) -> ConsentEvent:
    """Permanently delete a speaker.

    Cascades:
      • DELETE FROM speakers (FK ON DELETE SET NULL on transcript_segments
        keeps the segments but nulls the speaker_id).
      • The consent log is **retained** — auditors need to see that the
        deletion happened, including its signed timestamp.
    """
    sp = store.get_speaker(speaker_id)
    if sp is None:
        raise ValueError(f"no speaker {speaker_id} to forget")
    now = datetime.now(UTC)
    event = _sign_event(
        identity,
        actor_speaker_id=speaker_id,
        action="delete",
        disclosure_version=disclosure_version,
        ts=now,
    )
    store.append_consent_event(event)  # log first, then delete
    store.forget_speaker(speaker_id)
    return event


def export_event(
    *,
    speaker_id: str,
    store: _StoreLike,
    identity: Identity,
    disclosure_version: str = CURRENT_DISCLOSURE_VERSION,
) -> ConsentEvent:
    """Log that a voiceprint export was performed for ``speaker_id``.

    Voiceprint export is opt-in and audit-logged: the consent event is
    signed and retained even after the speaker row is deleted.
    """
    now = datetime.now(UTC)
    event = _sign_event(
        identity,
        actor_speaker_id=speaker_id,
        action="export",
        disclosure_version=disclosure_version,
        ts=now,
    )
    store.append_consent_event(event)
    return event


# ---------------------------------------------------------------------------
# Disclosure text
# ---------------------------------------------------------------------------


DISCLOSURE_TEXT = {
    "2026-05-v1": (
        "MeetMind voiceprint enrollment\n"
        "==============================\n"
        "We will create a voice signature ('voiceprint') from a short\n"
        "audio sample of you speaking. This voiceprint is stored on\n"
        "this device only, encrypted at rest. It is biometric data\n"
        "under GDPR Art. 9 / Illinois BIPA / Texas CUBI / Washington\n"
        "MHMDA — your explicit consent is required.\n"
        "\n"
        "  Purpose: Identify you across meetings so transcripts label\n"
        "    your spoken segments correctly.\n"
        "  Retention: 1 year after your last enrolled meeting (you can\n"
        "    revoke or delete sooner).\n"
        "  Sharing: Never. The voiceprint never leaves this device.\n"
        "\n"
        "By proceeding you confirm you have read this disclosure and\n"
        "consent to the creation and storage of your voiceprint."
    )
}


def _new_speaker_id() -> str:
    from ulid import ULID

    return str(ULID())


# Re-export for callers that prefer a single import.
__all__ = [
    "CURRENT_DISCLOSURE_VERSION",
    "DEFAULT_RETENTION_DAYS",
    "DISCLOSURE_TEXT",
    "EnrollmentResult",
    "enroll",
    "export_event",
    "forget",
    "revoke",
    "verify_consent_event",
]


# Tiny helper for `meetmind speakers` / dump / debug — JSON-able view.


def speaker_to_summary(speaker: Speaker) -> dict:
    return {
        "id": speaker.id,
        "display_name": speaker.display_name,
        "consent_ts": speaker.consent_ts.isoformat() if speaker.consent_ts else None,
        "consent_disclosure_version": speaker.consent_disclosure_version,
        "has_voiceprint": bool(speaker.voiceprint_centroid),
        "ring_size": len(speaker.voiceprint_ring or []),
        "aliases": list(speaker.aliases or []),
        "retention_until": speaker.retention_until.isoformat()
        if isinstance(speaker.retention_until, date)
        else speaker.retention_until,
    }


def consent_event_to_summary(event: ConsentEvent) -> dict:
    return {
        "id": event.id,
        "ts": event.ts.isoformat(),
        "actor_speaker_id": event.actor_speaker_id,
        "action": event.action,
        "disclosure_version": event.disclosure_version,
        "signed": event.signature is not None,
    }


# Compatibility — sometimes callers want the raw signature payload as JSON
# bytes for offline auditing.


def signing_payload_for(event: ConsentEvent) -> bytes:
    return _consent_payload(
        actor_speaker_id=event.actor_speaker_id,
        action=event.action,
        disclosure_version=event.disclosure_version,
        ts=event.ts,
    )


# Re-export json for callers that want to dump consent events directly.
__all__ += ["consent_event_to_summary", "signing_payload_for", "speaker_to_summary"]


_ = json  # keep the import; consent payloads are canonicalized via crypto.canonicalize

"""Voiceprint enrollment, revocation and deletion, with a signed consent log.

Pure logic: all I/O goes through the ``store`` and ``identity`` the caller
supplies. Consent capture UX lives in ``cli.py``.
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


# Structural typing keeps `diarize/` from importing `memory/`, which the
# import-linter contract forbids; the CLI passes the concrete Store at runtime.
class _StoreLike(Protocol):
    def get_speaker(self, speaker_id: str) -> Speaker | None: ...
    def upsert_speaker(self, sp: Speaker) -> None: ...
    def append_consent_event(self, event: ConsentEvent) -> None: ...
    def forget_speaker(self, speaker_id: str) -> None: ...


# Changing the disclosure text, the data stored, or the retention default
# requires a NEW version string here. ConsentEvent rows record the version
# current at enrollment time, which is what proves what a user agreed to
# after the policy later evolves.
CURRENT_DISCLOSURE_VERSION = "2026-05-v1"


# Satisfies the strictest of BIPA (destroy within 3 years of last
# interaction) and CUBI (within 1 year of "purpose served").
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
    """Canonical bytes signed by the install identity for this event."""
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
    """Revoke consent for ``speaker_id``, clearing its centroid and ring.

    The Speaker row stays so historical segment attributions remain valid.
    Re-enrollment requires fresh consent.
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

    Transcript segments survive with a null speaker_id (FK ON DELETE SET
    NULL). The consent log is retained so auditors can see the deletion.
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

    The signed event is retained even after the speaker row is deleted.
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


def signing_payload_for(event: ConsentEvent) -> bytes:
    return _consent_payload(
        actor_speaker_id=event.actor_speaker_id,
        action=event.action,
        disclosure_version=event.disclosure_version,
        ts=event.ts,
    )


__all__ += ["consent_event_to_summary", "signing_payload_for", "speaker_to_summary"]


_ = json  # re-exported for callers; payloads themselves use crypto.canonicalize

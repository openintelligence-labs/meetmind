"""Retention TTL enforcement (S14.4).

Idempotent sweep that deletes meetings older than ``meetings_ttl_days``
and speakers (incl. their voiceprints) older than ``voiceprint_ttl_days``.
Run on demand via ``meetmind compliance retention-sweep`` or wire into
cron / launchd.

The default TTLs are intentionally conservative:
  * meetings: 1095 days (3y) — BIPA upper bound for biometric records;
    operational records can usually live longer but we apply the
    biometric bound to the whole meeting because transcripts often
    contain identifiable utterances.
  * voiceprints: 365 days (1y) — CUBI mandatory upper bound when
    purpose has expired.

Override per-org via ``MEETMIND_RETENTION_MEETINGS_DAYS`` and
``MEETMIND_RETENTION_VOICEPRINT_DAYS`` env vars.

Consent events are NEVER deleted by this sweep — they're the proof-of-
consent retention requirement under GDPR Art. 7(1) and stay even after
the speaker they reference has been forgotten (tombstoned).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from meetmind.memory.store import Store


@dataclass
class RetentionPolicy:
    meetings_ttl_days: int = 1095
    voiceprint_ttl_days: int = 365

    @classmethod
    def from_env(cls) -> RetentionPolicy:
        return cls(
            meetings_ttl_days=int(
                os.environ.get("MEETMIND_RETENTION_MEETINGS_DAYS", cls.meetings_ttl_days)
            ),
            voiceprint_ttl_days=int(
                os.environ.get("MEETMIND_RETENTION_VOICEPRINT_DAYS", cls.voiceprint_ttl_days)
            ),
        )


@dataclass
class RetentionReport:
    meetings_deleted: list[str]
    speakers_deleted: list[str]
    consent_events_retained: int

    def as_lines(self) -> list[str]:
        out = [
            f"meetings deleted    : {len(self.meetings_deleted)}",
            f"speakers deleted    : {len(self.speakers_deleted)}",
            f"consent events kept : {self.consent_events_retained}",
        ]
        if self.meetings_deleted:
            out.append("  meetings:")
            out.extend(f"    - {mid}" for mid in self.meetings_deleted)
        if self.speakers_deleted:
            out.append("  speakers:")
            out.extend(f"    - {sid}" for sid in self.speakers_deleted)
        return out


def sweep(
    db_path: Path,
    *,
    policy: RetentionPolicy | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
) -> RetentionReport:
    """Apply retention policy to the store at ``db_path``.

    ``dry_run=True`` returns the report without mutating the store.
    """
    policy = policy or RetentionPolicy.from_env()
    now = now or datetime.now(UTC)
    meetings_cutoff = now - timedelta(days=policy.meetings_ttl_days)
    voiceprint_cutoff = now - timedelta(days=policy.voiceprint_ttl_days)

    deleted_meetings: list[str] = []
    deleted_speakers: list[str] = []

    with Store.open(db_path) as store:
        # Meetings: anything ended_at (or started_at, or created_at) older
        # than the cutoff is past its TTL.
        for m in store.list_meetings(limit=100000):
            ts = m.ended_at or m.started_at or m.created_at
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts < meetings_cutoff:
                deleted_meetings.append(m.id)
                if not dry_run:
                    store.forget_meeting(m.id)

        # Speakers: any voiceprint whose `consent_ts` is older than the
        # voiceprint TTL is purged. The ConsentEvent tombstone remains.
        rows = store.conn.execute(
            "SELECT id, consent_ts FROM speakers WHERE voiceprint_centroid IS NOT NULL"
        ).fetchall()
        for row in rows:
            sid = row["id"]
            ts_raw = row["consent_ts"]
            if ts_raw is None:
                continue
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts < voiceprint_cutoff:
                deleted_speakers.append(sid)
                if not dry_run:
                    store.forget_speaker(sid)

        consent_kept = int(store.conn.execute("SELECT COUNT(*) FROM consent_events").fetchone()[0])

    return RetentionReport(
        meetings_deleted=deleted_meetings,
        speakers_deleted=deleted_speakers,
        consent_events_retained=consent_kept,
    )

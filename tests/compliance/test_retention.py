"""Tests for compliance.retention (S14.4) + erasure cascade (S14.3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from meetmind.compliance.retention import RetentionPolicy, sweep
from meetmind.memory.store import Store
from meetmind.models import (
    ChannelKind,
    ConsentEvent,
    Meeting,
    Speaker,
    TranscriptSegment,
)


def _seed_store(db: Path) -> Store:
    store = Store.open(db)
    return store


def test_sweep_deletes_old_meetings_keeps_recent(tmp_path: Path):
    db = tmp_path / "store.db"
    now = datetime(2026, 5, 6, tzinfo=UTC)
    with _seed_store(db) as store:
        # Old meeting (5y ago)
        old = Meeting(
            id="01OLD",
            title="ancient",
            started_at=now - timedelta(days=365 * 5),
            ended_at=now - timedelta(days=365 * 5),
        )
        recent = Meeting(
            id="01NEW",
            title="recent",
            started_at=now - timedelta(days=10),
            ended_at=now - timedelta(days=10),
        )
        store.upsert_meeting(old)
        store.upsert_meeting(recent)

    report = sweep(db, policy=RetentionPolicy(meetings_ttl_days=365), now=now)
    assert "01OLD" in report.meetings_deleted
    assert "01NEW" not in report.meetings_deleted

    with Store.open(db) as store:
        assert store.get_meeting("01OLD") is None
        assert store.get_meeting("01NEW") is not None


def test_sweep_dry_run_makes_no_changes(tmp_path: Path):
    db = tmp_path / "store.db"
    now = datetime(2026, 5, 6, tzinfo=UTC)
    with _seed_store(db) as store:
        store.upsert_meeting(
            Meeting(
                id="01ANCIENT",
                title="ancient",
                started_at=now - timedelta(days=999),
                ended_at=now - timedelta(days=999),
            )
        )
    report = sweep(db, policy=RetentionPolicy(meetings_ttl_days=30), now=now, dry_run=True)
    assert "01ANCIENT" in report.meetings_deleted
    with Store.open(db) as store:
        assert store.get_meeting("01ANCIENT") is not None  # not actually deleted


def test_sweep_deletes_old_voiceprints_keeps_consent_event(tmp_path: Path):
    """S14.3 — erasure cascades through speakers; consent tombstone retained."""
    db = tmp_path / "store.db"
    now = datetime(2026, 5, 6, tzinfo=UTC)
    old_consent = now - timedelta(days=400)

    with _seed_store(db) as store:
        sp = Speaker(
            id="01SPK",
            display_name="Alice",
            consent_ts=old_consent,
            consent_disclosure_version="v1",
            voiceprint_centroid=b"\x00" * 192 * 4,  # plausible-shape blob
        )
        store.upsert_speaker(sp)
        store.append_consent_event(
            ConsentEvent(
                actor_speaker_id=sp.id,
                action="enroll",
                disclosure_version="v1",
                signature=b"sig",
            )
        )

    report = sweep(db, policy=RetentionPolicy(voiceprint_ttl_days=365), now=now)
    assert "01SPK" in report.speakers_deleted

    with Store.open(db) as store:
        assert store.get_speaker("01SPK") is None
        # ConsentEvent retained as tombstone.
        events = store.list_consent_events("01SPK")
        assert len(events) >= 1
        assert events[0].actor_speaker_id == "01SPK"


def test_forget_meeting_cascade_deletes_segments(tmp_path: Path):
    """S14.3 sanity: deleting a meeting wipes its child segments via FK cascade."""
    db = tmp_path / "store.db"
    with _seed_store(db) as store:
        m = Meeting(id="01CASCADE", title="kickoff")
        store.upsert_meeting(m)
        store.append_segment(
            m.id,
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="hello",
                channel=ChannelKind.MIC,
            ),
        )
        assert len(store.list_segments(m.id)) == 1
        store.forget_meeting(m.id)
        assert store.get_meeting(m.id) is None
        assert len(store.list_segments(m.id)) == 0


def test_policy_from_env(monkeypatch):
    monkeypatch.setenv("MEETMIND_RETENTION_MEETINGS_DAYS", "42")
    monkeypatch.setenv("MEETMIND_RETENTION_VOICEPRINT_DAYS", "7")
    p = RetentionPolicy.from_env()
    assert p.meetings_ttl_days == 42
    assert p.voiceprint_ttl_days == 7

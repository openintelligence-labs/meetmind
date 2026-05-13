"""Tests for the persistent store DAL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from meetmind.memory.store import Store
from meetmind.models import (
    ActionItem,
    ChannelKind,
    ConsentEvent,
    Decision,
    Meeting,
    MeetingTemplate,
    Speaker,
    TranscriptSegment,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.open(tmp_path / "test.db")


def test_open_applies_schema_idempotently(tmp_path: Path):
    path = tmp_path / "idempotent.db"
    a = Store.open(path)
    a.close()
    b = Store.open(path)
    b.close()


def test_meeting_round_trip(store: Store):
    m = Meeting(
        title="Product weekly",
        template=MeetingTemplate.STANDUP,
        started_at=datetime.now(UTC),
        audio_path_mic=Path("/tmp/mic.opus.enc"),
    )
    store.upsert_meeting(m)
    out = store.get_meeting(m.id)
    assert out is not None
    assert out.title == "Product weekly"
    assert out.template == MeetingTemplate.STANDUP
    assert out.audio_path_mic == Path("/tmp/mic.opus.enc")


def test_segments_round_trip_in_order(store: Store):
    m = Meeting(title="Test")
    store.upsert_meeting(m)
    segs = [
        TranscriptSegment(
            start_seconds=0.0,
            end_seconds=1.5,
            text="Hello",
            channel=ChannelKind.MIC,
            speaker_id="01SELF",
            confidence=0.92,
        ),
        TranscriptSegment(
            start_seconds=1.5,
            end_seconds=3.0,
            text="World",
            channel=ChannelKind.LOOPBACK,
            speaker_id="01REMOTE",
            confidence=0.88,
        ),
    ]
    for s in segs:
        store.append_segment(m.id, s)
    out = store.list_segments(m.id)
    assert len(out) == 2
    assert [s.text for s in out] == ["Hello", "World"]
    assert out[0].channel == ChannelKind.MIC
    assert out[1].channel == ChannelKind.LOOPBACK


def test_forget_meeting_cascades_to_segments(store: Store):
    m = Meeting(title="Doomed")
    store.upsert_meeting(m)
    store.append_segment(m.id, TranscriptSegment(start_seconds=0.0, end_seconds=1.0, text="bye"))
    assert len(store.list_segments(m.id)) == 1
    store.forget_meeting(m.id)
    assert store.get_meeting(m.id) is None
    assert store.list_segments(m.id) == []


def test_speaker_with_consent_round_trip(store: Store):
    speaker = Speaker(
        id="01SAM",
        display_name="Sam Chen",
        consent_ts=datetime.now(UTC),
        consent_disclosure_version="2026-05-v1",
        voiceprint_centroid=b"\x00" * 768,
        voiceprint_ring=[b"\x01" * 768, b"\x02" * 768],
        aliases=["sam@example.com"],
        retention_until=(datetime.now(UTC) + timedelta(days=365)).date(),
        confidence=0.95,
    )
    store.upsert_speaker(speaker)
    out = store.get_speaker("01SAM")
    assert out is not None
    assert out.display_name == "Sam Chen"
    assert out.confidence == 0.95
    assert out.aliases == ["sam@example.com"]
    assert len(out.voiceprint_ring) == 2
    assert out.voiceprint_ring[0] == b"\x01" * 768


def test_consent_events_audit_log_round_trip(store: Store):
    evt = ConsentEvent(
        actor_speaker_id="01SAM",
        action="enroll",
        disclosure_version="2026-05-v1",
    )
    store.append_consent_event(evt)
    events = store.list_consent_events("01SAM")
    assert len(events) == 1
    assert events[0].action == "enroll"


def test_forget_speaker_keeps_consent_events(store: Store):
    speaker = Speaker(id="01PRIYA", display_name="Priya")
    store.upsert_speaker(speaker)
    store.append_consent_event(
        ConsentEvent(actor_speaker_id="01PRIYA", action="enroll", disclosure_version="v1")
    )
    store.append_consent_event(
        ConsentEvent(actor_speaker_id="01PRIYA", action="delete", disclosure_version="v1")
    )
    store.forget_speaker("01PRIYA")
    assert store.get_speaker("01PRIYA") is None
    assert len(store.list_consent_events("01PRIYA")) == 2


def test_action_items_filter_by_status_and_meeting(store: Store):
    m1 = Meeting(title="One")
    m2 = Meeting(title="Two")
    store.upsert_meeting(m1)
    store.upsert_meeting(m2)
    store.upsert_action_item(
        m1.id,
        ActionItem(
            description="Send deck",
            owner=None,
            evidence_quote="I'll send the deck Friday",
            status="open",
        ),
    )
    store.upsert_action_item(
        m1.id,
        ActionItem(description="Follow up", evidence_quote="will follow up", status="done"),
    )
    store.upsert_action_item(m2.id, ActionItem(description="Other meeting item", status="open"))

    open_in_m1 = store.list_action_items(status="open", meeting_id=m1.id)
    assert len(open_in_m1) == 1
    assert open_in_m1[0].description == "Send deck"
    assert open_in_m1[0].evidence_quote == "I'll send the deck Friday"


def test_decisions_round_trip(store: Store):
    m = Meeting(title="ADR")
    store.upsert_meeting(m)
    dec = Decision(
        decision="Adopt LanceDB",
        rationale="Beats sqlite-vec at 1M+ vectors",
        dissenters=["01SAM", "01PRIYA"],
        source_segment_ids=[12, 13, 14],
    )
    store.upsert_decision(m.id, dec)
    out = store.list_decisions(m.id)
    assert len(out) == 1
    assert out[0].decision == "Adopt LanceDB"
    assert out[0].dissenters == ["01SAM", "01PRIYA"]
    assert out[0].source_segment_ids == [12, 13, 14]


def test_transaction_rolls_back_on_error(store: Store):
    m = Meeting(title="Survives")
    store.upsert_meeting(m)
    with pytest.raises(RuntimeError), store.transaction():
        store.upsert_meeting(Meeting(id="01OTHER", title="Phantom"))
        raise RuntimeError("whoops")
    assert store.get_meeting("01OTHER") is None
    assert store.get_meeting(m.id) is not None

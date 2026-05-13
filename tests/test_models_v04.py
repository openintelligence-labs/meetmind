"""Tests for the v0.4 data model additions.

Covers: ChannelKind, MeetingTemplate, ms-helper conversions, Speaker,
ActionItem evidence-quote/status, Decision, ConsentEvent, Meeting
audio paths and signing fields.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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


def test_channel_kind_values():
    assert ChannelKind.MIC == "mic"
    assert ChannelKind.LOOPBACK == "loopback"
    # Enum is exhaustive — adding a new channel must be a deliberate change.
    assert {c.value for c in ChannelKind} == {"mic", "loopback"}


def test_meeting_template_includes_assist():
    """v1.1 assist sessions persist as Meeting rows when --archive is set."""
    assert MeetingTemplate.ASSIST == "assist"
    assert MeetingTemplate.STANDUP == "standup"


def test_segment_ms_helpers_are_int_and_consistent():
    seg = TranscriptSegment(
        start_seconds=1.234,
        end_seconds=2.567,
        text="hi",
        channel=ChannelKind.LOOPBACK,
    )
    assert isinstance(seg.start_ms, int)
    assert isinstance(seg.end_ms, int)
    assert seg.start_ms == 1234
    assert seg.end_ms == 2567


def test_segment_carries_channel_and_speaker_id():
    seg = TranscriptSegment(
        start_seconds=0,
        end_seconds=1,
        text="hello",
        channel=ChannelKind.MIC,
        speaker_id="01ABCDEF",
        confidence=0.95,
    )
    assert seg.channel == ChannelKind.MIC
    assert seg.speaker_id == "01ABCDEF"
    assert seg.confidence == 0.95
    assert seg.language == "en"  # default


def test_action_item_defaults_to_open_with_ulid():
    item = ActionItem(description="Send the deck", owner="alice")
    assert item.status == "open"
    assert item.id  # auto-populated ULID
    assert len(item.id) == 26
    assert item.evidence_quote is None
    assert item.closed_in_meeting_id is None


def test_action_item_carries_evidence_quote():
    """Verbatim citation guard lives on the model."""
    item = ActionItem(
        description="Update the migration plan",
        owner="bob",
        evidence_quote="I'll update the migration plan by Friday.",
        source_segment_id=42,
    )
    assert item.evidence_quote.startswith("I'll update")
    assert item.source_segment_id == 42


def test_decision_with_dissenters():
    d = Decision(
        decision="Adopt LanceDB for the vector store",
        rationale="Better at 1M+ vectors than sqlite-vec",
        dissenters=["01SAM", "01PRIYA"],
        source_segment_ids=[12, 13, 14],
    )
    assert "01SAM" in d.dissenters
    assert d.id  # auto ULID


def test_speaker_voiceprint_is_optional_and_consent_logged():
    """Voiceprints are opt-in. A Speaker with no consent is a diarization
    label, not a biometric record."""
    s = Speaker(display_name="Sam Chen")
    assert s.voiceprint_centroid is None
    assert s.consent_ts is None
    assert s.confidence == 0.0
    assert s.id


def test_speaker_with_consent_event_round_trip():
    speaker_id = "01SAM"
    enrollment_ts = datetime.now(UTC)
    s = Speaker(
        id=speaker_id,
        display_name="Sam",
        consent_ts=enrollment_ts,
        consent_disclosure_version="2026-05-v1",
        voiceprint_centroid=b"\x00" * 768,  # 192-d float32 placeholder
    )
    event = ConsentEvent(
        actor_speaker_id=speaker_id,
        action="enroll",
        disclosure_version="2026-05-v1",
    )
    assert s.consent_ts == enrollment_ts
    assert event.actor_speaker_id == speaker_id
    assert event.action == "enroll"
    assert event.signature is None  # signed lazily by crypto module


def test_consent_event_action_is_constrained():
    with pytest.raises(ValueError):
        ConsentEvent(actor_speaker_id="01X", action="bogus", disclosure_version="v1")


def test_speaker_retention_until_in_future():
    """BIPA retention ≤ 3y; CUBI ≤ 1y after last interaction."""
    s = Speaker(
        display_name="Priya",
        retention_until=(datetime.now(UTC) + timedelta(days=365)).date(),
    )
    assert s.retention_until > datetime.now(UTC).date()


def test_meeting_carries_dual_audio_paths():
    """Mic and loopback streams are kept separate end-to-end."""
    m = Meeting(
        title="1:1 with Sam",
        template=MeetingTemplate.ONE_ON_ONE,
        audio_path_mic=Path("/tmp/mic.opus.enc"),
        audio_path_loopback=Path("/tmp/loop.opus.enc"),
    )
    assert m.audio_path_mic != m.audio_path_loopback
    assert m.template == MeetingTemplate.ONE_ON_ONE


def test_meeting_signing_fields_default_unset():
    """Legal-mode bundle (§5.7) is opt-in; default Meeting is unsigned."""
    m = Meeting(title="Quick standup")
    assert m.transcript_hash is None
    assert m.signature is None

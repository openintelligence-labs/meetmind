from __future__ import annotations

from meetmind.models import ActionItem, Meeting, Summary, Transcript, TranscriptSegment


def test_transcript_full_text_joins_segments():
    t = Transcript(
        segments=[
            TranscriptSegment(start_seconds=0, end_seconds=1, text="Hello"),
            TranscriptSegment(start_seconds=1, end_seconds=2, text="world"),
        ]
    )
    assert t.full_text == "Hello world"


def test_transcript_duration_is_last_end():
    t = Transcript(
        segments=[
            TranscriptSegment(start_seconds=0, end_seconds=1.5, text="a"),
            TranscriptSegment(start_seconds=1.5, end_seconds=3.25, text="b"),
        ]
    )
    assert t.duration_seconds == 3.25


def test_empty_transcript_has_zero_duration():
    assert Transcript().duration_seconds == 0.0


def test_summary_with_action_items():
    s = Summary(
        tl_dr="We decided X.",
        key_decisions=["X"],
        action_items=[ActionItem(description="Do Y", owner="alice")],
    )
    assert s.action_items[0].owner == "alice"


def test_meeting_defaults():
    m = Meeting(id="01ABC", title="Standup")
    assert m.summary is None
    assert m.transcript.duration_seconds == 0.0

"""Tests for the STT × diarization stitcher."""

from __future__ import annotations

from meetmind.diarize.base import DiarSegment
from meetmind.diarize.stitch import SpeakerSegment, stitch
from meetmind.ipc import StreamId
from meetmind.stt.base import Final


def _final(text: str, start_ms: int, end_ms: int, conf: float = 0.9) -> Final:
    return Final(text=text, start_ms=start_ms, end_ms=end_ms, confidence=conf, language="en")


def _diar(
    cluster: str,
    start_ms: int,
    end_ms: int,
    channel: StreamId | None = None,
) -> DiarSegment:
    return DiarSegment(
        start_ms=start_ms, end_ms=end_ms, cluster_id=cluster, confidence=0.8, channel=channel
    )


def test_single_final_single_diar_aligns_trivially():
    finals = [_final("hello world", 0, 1000)]
    diars = [_diar("self", 0, 1000, StreamId.MIC)]
    out = stitch(finals, diars)
    assert len(out) == 1
    assert isinstance(out[0], SpeakerSegment)
    assert out[0].text == "hello world"
    assert out[0].cluster_id == "self"
    assert out[0].channel is StreamId.MIC


def test_final_with_no_overlapping_diar_is_unknown():
    finals = [_final("orphan", 0, 1000)]
    diars = [_diar("self", 2000, 3000)]
    out = stitch(finals, diars)
    assert len(out) == 1
    assert out[0].cluster_id == "unknown"


def test_final_split_at_diar_boundary():
    finals = [_final("hello there friend", 0, 600)]
    diars = [
        _diar("self", 0, 300),
        _diar("remote-B", 300, 600),
    ]
    out = stitch(finals, diars)
    assert len(out) == 2
    combined_chars = sum(len(s.text) for s in out)
    assert combined_chars >= len("hello there friend") - 4
    assert {s.cluster_id for s in out} == {"self", "remote-B"}
    total = sum(s.end_ms - s.start_ms for s in out)
    assert total == 600


def test_max_overlap_assignment_when_one_diar_dominates():
    finals = [_final("one big chunk of speech here", 0, 1000)]
    diars = [
        _diar("self", 0, 950, StreamId.MIC),
        _diar("remote-A", 950, 1000, StreamId.LOOPBACK),
    ]
    out = stitch(finals, diars)
    self_chars = sum(len(s.text) for s in out if s.cluster_id == "self")
    remote_chars = sum(len(s.text) for s in out if s.cluster_id == "remote-A")
    assert self_chars > remote_chars * 5


def test_outputs_are_sorted_by_start_ms():
    finals = [
        _final("second", 1000, 2000),
        _final("first", 0, 500),
    ]
    diars = [
        _diar("self", 0, 500),
        _diar("remote-A", 1000, 2000),
    ]
    out = stitch(finals, diars)
    starts = [s.start_ms for s in out]
    assert starts == sorted(starts)


def test_empty_inputs_return_empty():
    assert stitch([], []) == []
    out = stitch([_final("hi", 0, 100)], [])
    assert len(out) == 1
    assert out[0].cluster_id == "unknown"
    assert out[0].text == "hi"


def test_empty_text_finals_skipped():
    finals = [_final("", 0, 500), _final("real content", 1000, 2000)]
    diars = [_diar("self", 0, 2000)]
    out = stitch(finals, diars)
    assert len(out) == 1
    assert out[0].text == "real content"


def test_confidence_is_min_of_stt_and_diar():
    finals = [_final("hello", 0, 1000, conf=0.9)]
    diars = [DiarSegment(start_ms=0, end_ms=1000, cluster_id="self", confidence=0.5)]
    out = stitch(finals, diars)
    assert out[0].confidence == 0.5

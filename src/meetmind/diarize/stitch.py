"""Align STT `Final` spans with diarization segments into a speaker-attributed
transcript.

Splitting across a diarization boundary is character-proportional rather than
duration-proportional: word boundaries inside a span are not known at this
stage, and characters track speech density better than wall-clock time, which
long pauses inflate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from meetmind.diarize.base import DiarSegment
from meetmind.ipc import StreamId
from meetmind.stt.base import Final


@dataclass(frozen=True)
class SpeakerSegment:
    """One contiguous span of speech by a single speaker.

    `cluster_id` is the diarization label, post-channel-prior: typically
    `self`, `remote-A` or `unknown`.
    """

    start_ms: int
    end_ms: int
    text: str
    cluster_id: str
    speaker_id: str | None = None
    channel: StreamId | None = None
    confidence: float = 0.0
    language: str = "en"

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def stitch(
    finals: Iterable[Final],
    diars: Iterable[DiarSegment],
) -> list[SpeakerSegment]:
    """Align Finals with DiarSegments, splitting where boundaries cross.

    Both inputs are eagerly materialized. Output is sorted by `start_ms`.
    """
    finals_list = sorted(finals, key=lambda f: f.start_ms)
    diars_list = sorted(diars, key=lambda d: d.start_ms)

    out: list[SpeakerSegment] = []

    for fin in finals_list:
        if not fin.text:
            continue

        overlapping = [
            d for d in diars_list if _overlap_ms(fin.start_ms, fin.end_ms, d.start_ms, d.end_ms) > 0
        ]

        if not overlapping:
            out.append(_seg_from_final(fin, "unknown", None))
            continue

        if len(overlapping) == 1:
            d = overlapping[0]
            out.append(_seg_from_final(fin, d.cluster_id, d.channel, d.confidence))
            continue

        total_overlap = sum(
            _overlap_ms(fin.start_ms, fin.end_ms, d.start_ms, d.end_ms) for d in overlapping
        )
        if total_overlap <= 0:
            out.append(_seg_from_final(fin, "unknown", None))
            continue

        accum_chars = 0
        accum_ms = 0
        text = fin.text
        text_len = len(text)
        for i, d in enumerate(overlapping):
            ov = _overlap_ms(fin.start_ms, fin.end_ms, d.start_ms, d.end_ms)
            share = ov / total_overlap
            if i == len(overlapping) - 1:
                # Last slice takes the remainder, so rounding never drops chars.
                slice_chars = text_len - accum_chars
                slice_ms = (fin.end_ms - fin.start_ms) - accum_ms
            else:
                slice_chars = max(0, round(share * text_len))
                slice_ms = max(0, round(share * (fin.end_ms - fin.start_ms)))

            slice_text = text[accum_chars : accum_chars + slice_chars]
            slice_start = fin.start_ms + accum_ms
            slice_end = slice_start + slice_ms

            slice_conf = min(fin.confidence, d.confidence) if d.confidence else fin.confidence
            if slice_text:
                out.append(
                    SpeakerSegment(
                        start_ms=slice_start,
                        end_ms=slice_end,
                        text=slice_text,
                        cluster_id=d.cluster_id,
                        speaker_id=None,
                        channel=d.channel,
                        confidence=slice_conf,
                        language=fin.language,
                    )
                )

            accum_chars += slice_chars
            accum_ms += slice_ms

    return out


def _seg_from_final(
    fin: Final,
    cluster_id: str,
    channel: StreamId | None,
    diar_confidence: float | None = None,
) -> SpeakerSegment:
    conf = min(fin.confidence, diar_confidence) if diar_confidence is not None else fin.confidence
    return SpeakerSegment(
        start_ms=fin.start_ms,
        end_ms=fin.end_ms,
        text=fin.text,
        cluster_id=cluster_id,
        speaker_id=None,
        channel=channel,
        confidence=conf,
        language=fin.language,
    )

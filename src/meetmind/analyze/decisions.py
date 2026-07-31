"""Decision extraction with the same substring guard as `actions.py`.

A decision is an explicit choice made in a meeting, with optional rationale
and dissenters. Dissenter labels are stored raw; voiceprint resolution is
downstream.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from meetmind.analyze.actions import LLMCallable
from meetmind.models import Decision

log = logging.getLogger(__name__)


class ExtractedDecision(BaseModel):
    """LLM output schema for one decision."""

    decision: str = Field(min_length=1)
    rationale: str | None = None
    dissenters: list[str] = Field(default_factory=list)
    evidence_quote: str = Field(min_length=1)


class DecisionExtractionPayload(BaseModel):
    decisions: list[ExtractedDecision] = Field(default_factory=list)


@dataclass
class DecisionExtractionResult:
    accepted: list[Decision]
    rejected: list[tuple[ExtractedDecision, str]]


SYSTEM_PROMPT = """You extract explicit decisions from a meeting transcript.

Output JSON matching:
  { "decisions": [
      { "decision": str, "rationale": str|null, "dissenters": [str],
        "evidence_quote": str } ] }

Rules:
  - A decision is a CHOICE that was made — not a question, not an opinion,
    not a possibility. "We will adopt X" / "we agreed on Y" / "the call is Z".
  - `decision` is the choice restated cleanly (≤ 200 chars).
  - `rationale` is the reasoning given in-meeting, if any (≤ 240 chars).
  - `dissenters` is a list of speaker labels who pushed back or voted against.
  - `evidence_quote` MUST be a verbatim substring of the transcript.
  - If no decisions were made, return { "decisions": [] }.
"""


def build_user_prompt(transcript_window: str) -> str:
    return f"<transcript>\n{transcript_window}\n</transcript>"


def extract_decisions(
    transcript_window: str,
    llm: LLMCallable,
    *,
    source_segment_ids: Iterable[int] | None = None,
) -> DecisionExtractionResult:
    """LLM extraction → validated `Decision` rows.

    `source_segment_ids` is attached to every accepted Decision.
    """
    prompt = SYSTEM_PROMPT + "\n" + build_user_prompt(transcript_window)
    raw = llm(prompt)
    payload = DecisionExtractionPayload.model_validate(raw)

    accepted: list[Decision] = []
    rejected: list[tuple[ExtractedDecision, str]] = []
    seg_ids = list(source_segment_ids or [])

    for item in payload.decisions:
        reason = _validate(item, transcript_window)
        if reason is not None:
            log.info("decision rejected: %s — %s", item.decision, reason)
            rejected.append((item, reason))
            continue
        accepted.append(
            Decision(
                decision=item.decision,
                rationale=item.rationale or "",
                dissenters=list(item.dissenters),
                source_segment_ids=seg_ids,
            )
        )
    return DecisionExtractionResult(accepted=accepted, rejected=rejected)


def _validate(item: ExtractedDecision, transcript_window: str) -> str | None:
    if not item.evidence_quote.strip():
        return "empty evidence_quote"
    if item.evidence_quote not in transcript_window:
        return "evidence_quote is not a substring of the transcript"
    if not item.decision.strip():
        return "empty decision"
    if len(item.decision) > 240:
        return f"decision too long ({len(item.decision)} chars > 240)"
    if item.rationale and len(item.rationale) > 480:
        return f"rationale too long ({len(item.rationale)} chars > 480)"
    return None


from meetmind.analyze.actions import MockLLM  # noqa: E402,F401  (re-exported)


def _example() -> dict[str, Any]:  # pragma: no cover — doc only
    return {
        "decisions": [
            {
                "decision": "Adopt LanceDB for the vector store",
                "rationale": "Beats sqlite-vec at 1M+ vectors",
                "dissenters": ["Sam"],
                "evidence_quote": "the call is to use LanceDB",
            }
        ]
    }

"""Chain-of-Density summarizer (Adams et al. 2023).

Drafts a sparse summary, then densifies it with missing entities at the same
length. Key decisions and action items come from `analyze.decisions` and
`analyze.actions`; they are not re-extracted here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from meetmind.analyze.actions import LLMCallable
from meetmind.models import Summary

log = logging.getLogger(__name__)


class _DraftPayload(BaseModel):
    tl_dr: str = Field(min_length=1)
    headline_topics: list[str] = Field(default_factory=list)


class _DensePayload(BaseModel):
    tl_dr: str = Field(min_length=1)
    headline_topics: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)


SYSTEM_DRAFT = """You write the first draft of a meeting summary.
Output JSON: { "tl_dr": str, "headline_topics": [str] }.

Rules:
  - tl_dr: 60-100 words, plain prose, no bullets, no markdown.
  - headline_topics: 3-6 short phrases (≤ 50 chars each).
  - Cover the meeting's PURPOSE and OUTCOMES, not chronological play-by-play.
"""

SYSTEM_DENSIFY = """You densify a meeting summary by adding 2-3 entities
that the previous draft missed.

Output JSON: { "tl_dr": str, "headline_topics": [str], "missing_entities": [str] }.

Rules:
  - tl_dr stays roughly the same length but is rewritten to incorporate
    2-3 entities (people, products, dates, numbers) that weren't in the
    previous draft.
  - missing_entities lists exactly the entities you added.
  - Do NOT introduce facts not present in the transcript.
"""


def build_draft_prompt(transcript_window: str) -> str:
    return SYSTEM_DRAFT + f"\n<transcript>\n{transcript_window}\n</transcript>"


def build_densify_prompt(transcript_window: str, draft: _DraftPayload) -> str:
    return (
        SYSTEM_DENSIFY
        + f"\n<previous_draft>\n{draft.tl_dr}\nTopics: {', '.join(draft.headline_topics)}\n"
        f"</previous_draft>\n<transcript>\n{transcript_window}\n</transcript>"
    )


@dataclass
class SummarizeResult:
    summary: Summary
    densify_passes: int
    headline_topics: list[str]


def summarize_meeting(
    transcript_window: str,
    llm: LLMCallable,
    *,
    densify_passes: int = 1,
    key_decisions: list[str] | None = None,
    action_items: list | None = None,
) -> SummarizeResult:
    """Run Chain-of-Density and return a `Summary` populated with prose.

    `key_decisions` and `action_items` are passed through from the upstream
    extractors, which carry their own citation guarantees.
    """
    draft_raw = llm(build_draft_prompt(transcript_window))
    draft = _DraftPayload.model_validate(draft_raw)

    densify_count = 0
    current = draft
    for _ in range(max(0, densify_passes)):
        dense_raw = llm(build_densify_prompt(transcript_window, current))
        try:
            dense = _DensePayload.model_validate(dense_raw)
        except Exception:  # noqa: BLE001 — fall back to draft if densify malforms
            log.info("densify pass returned malformed payload; keeping previous draft")
            break
        # No new entities means densification has converged.
        if len(dense.missing_entities) == 0:
            break
        current = _DraftPayload(tl_dr=dense.tl_dr, headline_topics=dense.headline_topics)
        densify_count += 1

    summary = Summary(
        tl_dr=current.tl_dr,
        key_decisions=list(key_decisions or []),
        action_items=list(action_items or []),
    )
    return SummarizeResult(
        summary=summary,
        densify_passes=densify_count,
        headline_topics=list(current.headline_topics),
    )


# Re-export MockLLM for tests.
from meetmind.analyze.actions import MockLLM  # noqa: E402,F401

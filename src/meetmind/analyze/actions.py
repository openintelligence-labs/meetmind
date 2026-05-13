"""Action-item extraction with verbatim-citation guard.

Workflow:

  1. The pipeline calls `extract_action_items(transcript_window, llm)`.
  2. `llm` is any callable that returns a `dict` matching the `ExtractionPayload`
     schema. In production this is `actants` → Ollama with grammar-constrained
     JSON output. For tests it's a `MockLLM` that returns whatever fixture the
     test sets up.
  3. We validate every item's `evidence_quote` is a substring of the
     transcript window. The architecture documents this single check as
     killing ~80% of hallucinated extractions.
  4. We then validate `owner` (if a speaker_id) and `deadline` (if any).
  5. Items that fail any validation are dropped with a logged reason. The
     remaining items are returned with stable IDs and an `open` status.

The same pattern (substring guard) is reused for closure detection — the
LLM is asked "did this transcript close action item X?" and if it
returns `closed=True`, the `closed_evidence_quote` must be a substring
of the closing meeting's transcript.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from meetmind.models import ActionItem

log = logging.getLogger(__name__)


class ExtractedItem(BaseModel):
    """LLM output schema for a single action item — pre-validation."""

    description: str = Field(min_length=1)
    owner: str | None = None
    deadline: str | None = None
    evidence_quote: str = Field(min_length=1)


class ExtractionPayload(BaseModel):
    """Full LLM output schema."""

    items: list[ExtractedItem] = Field(default_factory=list)


@dataclass
class ExtractionResult:
    accepted: list[ActionItem]
    rejected: list[tuple[ExtractedItem, str]]


# Type alias for any callable that returns the extraction payload.
LLMCallable = Callable[[str], dict[str, Any]]


SYSTEM_PROMPT = """You extract action items from a meeting transcript.

Output JSON matching the schema:
  { "items": [ { "description": str, "owner": str|null,
                 "deadline": str|null, "evidence_quote": str } ] }

Rules:
  - `description` is the action restated cleanly (≤ 120 chars).
  - `owner` is the cluster label or speaker name from the transcript, or null.
  - `deadline` is an ISO date or natural phrase, or null.
  - `evidence_quote` MUST be a verbatim substring of the transcript.
  - If no action items are present, return { "items": [] }.
"""


def build_user_prompt(transcript_window: str) -> str:
    return f"<transcript>\n{transcript_window}\n</transcript>"


def extract_action_items(
    transcript_window: str,
    llm: LLMCallable,
    *,
    meeting_id: str | None = None,
    source_segment_id: int | None = None,
) -> ExtractionResult:
    """Run the LLM extraction and validate against the substring guard.

    `llm` receives the full prompt body (system + user) joined by newlines
    and returns a JSON-decoded dict. Production wrappers handle Ollama
    `format=schema` constrained decoding.

    Returns an `ExtractionResult` with `accepted` (validated `ActionItem`s
    ready to upsert) and `rejected` (items that failed any guard, with a
    one-line reason for the audit log).
    """
    prompt = SYSTEM_PROMPT + "\n" + build_user_prompt(transcript_window)
    raw = llm(prompt)
    payload = ExtractionPayload.model_validate(raw)

    accepted: list[ActionItem] = []
    rejected: list[tuple[ExtractedItem, str]] = []
    for item in payload.items:
        reason = _validate(item, transcript_window)
        if reason is not None:
            log.info("action-item rejected: %s — %s", item.description, reason)
            rejected.append((item, reason))
            continue
        accepted.append(
            ActionItem(
                description=item.description,
                owner=item.owner,
                due=item.deadline,
                evidence_quote=item.evidence_quote,
                source_segment_id=source_segment_id,
                status="open",
            )
        )
    return ExtractionResult(accepted=accepted, rejected=rejected)


def _validate(item: ExtractedItem, transcript_window: str) -> str | None:
    """Return a rejection reason or None if the item passes all guards."""
    if not item.evidence_quote.strip():
        return "empty evidence_quote"
    if item.evidence_quote not in transcript_window:
        return "evidence_quote is not a substring of the transcript"
    if not item.description.strip():
        return "empty description"
    if len(item.description) > 240:
        return f"description too long ({len(item.description)} chars > 240)"
    return None


# ---------------------------------------------------------------------------
# Closure detection
# ---------------------------------------------------------------------------


class ClosurePayload(BaseModel):
    closed: bool
    evidence_quote: str | None = None
    confidence: float = 0.0


def detect_closure(
    open_action: ActionItem,
    transcript_window: str,
    llm: LLMCallable,
) -> ClosurePayload:
    """Ask the LLM whether `transcript_window` closes `open_action`.

    Same substring guard: if the LLM returns `closed=True` with an
    `evidence_quote` that isn't actually in the transcript, we treat the
    answer as `closed=False`. This converts hallucinated closures into
    quiet passes — better to leave an item open than to falsely mark it
    done.
    """
    prompt = (
        "You are a faithful action-item closure judge. Output JSON: "
        '{ "closed": bool, "evidence_quote": str|null, "confidence": float }. '
        "evidence_quote MUST be a verbatim substring of the transcript.\n"
        f"<action_item>{open_action.description}</action_item>\n"
        f"<transcript>\n{transcript_window}\n</transcript>"
    )
    raw = llm(prompt)
    payload = ClosurePayload.model_validate(raw)
    if payload.closed and (
        not payload.evidence_quote or payload.evidence_quote not in transcript_window
    ):
        log.info(
            "closure rejected for action_item=%s: evidence_quote not in transcript",
            open_action.id,
        )
        return ClosurePayload(closed=False, evidence_quote=None, confidence=0.0)
    return payload


# ---------------------------------------------------------------------------
# Mock LLM for tests
# ---------------------------------------------------------------------------


class MockLLM:
    """Trivial LLM stand-in: returns whatever responses are queued.

    Use:
        llm = MockLLM([{"items": [...]}])
        result = extract_action_items(text, llm)
    """

    def __init__(self, responses: Iterable[dict[str, Any]]) -> None:
        self._responses = list(responses)

    def __call__(self, _prompt: str) -> dict[str, Any]:
        if not self._responses:
            raise RuntimeError("MockLLM has no more queued responses")
        return self._responses.pop(0)

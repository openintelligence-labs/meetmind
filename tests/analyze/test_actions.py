"""Tests for action-item extraction + closure detection."""

from __future__ import annotations

from meetmind.analyze.actions import MockLLM, detect_closure, extract_action_items
from meetmind.models import ActionItem

SAMPLE_TRANSCRIPT = (
    "Sam: I'll send the deck on Friday. We also need to update the migration plan. "
    "Priya: I'll write up the migration plan by next Tuesday. "
    "Bob: Great. Can someone schedule the security review? "
    "Sam: I'll handle the security review."
)


def test_valid_extraction_round_trips_to_action_items():
    llm = MockLLM(
        [
            {
                "items": [
                    {
                        "description": "Send the deck",
                        "owner": "Sam",
                        "deadline": "Friday",
                        "evidence_quote": "I'll send the deck on Friday.",
                    },
                    {
                        "description": "Write the migration plan",
                        "owner": "Priya",
                        "deadline": "Tuesday",
                        "evidence_quote": "I'll write up the migration plan by next Tuesday.",
                    },
                ]
            }
        ]
    )
    result = extract_action_items(SAMPLE_TRANSCRIPT, llm, source_segment_id=42)
    assert len(result.accepted) == 2
    assert result.rejected == []
    assert all(isinstance(a, ActionItem) for a in result.accepted)
    assert result.accepted[0].evidence_quote in SAMPLE_TRANSCRIPT
    assert result.accepted[0].source_segment_id == 42
    assert result.accepted[0].status == "open"


def test_hallucinated_evidence_quote_is_rejected():
    llm = MockLLM(
        [
            {
                "items": [
                    {
                        "description": "Send the deck",
                        "owner": "Sam",
                        "deadline": "Friday",
                        "evidence_quote": "Sam said he'd ship the deck Tuesday and bake a cake",
                    }
                ]
            }
        ]
    )
    result = extract_action_items(SAMPLE_TRANSCRIPT, llm)
    assert result.accepted == []
    assert len(result.rejected) == 1
    assert "not a substring" in result.rejected[0][1]


def test_partial_hallucination_still_rejected():
    llm = MockLLM(
        [
            {
                "items": [
                    {
                        "description": "Schedule the security review",
                        "owner": "Sam",
                        "deadline": None,
                        "evidence_quote": "I'll handle the security review.",
                    },
                    {
                        "description": "Buy lunch for everyone",
                        "owner": "Bob",
                        "deadline": "tomorrow",
                        "evidence_quote": "Bob promised to buy lunch tomorrow",
                    },
                ]
            }
        ]
    )
    result = extract_action_items(SAMPLE_TRANSCRIPT, llm)
    assert len(result.accepted) == 1
    assert result.accepted[0].description == "Schedule the security review"
    assert len(result.rejected) == 1


def test_empty_evidence_quote_rejected():
    llm = MockLLM([{"items": [{"description": "Do thing", "evidence_quote": "   "}]}])
    result = extract_action_items(SAMPLE_TRANSCRIPT, llm)
    assert result.accepted == []
    assert "empty evidence_quote" in result.rejected[0][1]


def test_no_items_returned_is_clean():
    llm = MockLLM([{"items": []}])
    result = extract_action_items(SAMPLE_TRANSCRIPT, llm)
    assert result.accepted == []
    assert result.rejected == []


def test_overly_long_description_rejected():
    llm = MockLLM(
        [
            {
                "items": [
                    {
                        "description": "x" * 250,
                        "evidence_quote": "I'll send the deck on Friday.",
                    }
                ]
            }
        ]
    )
    result = extract_action_items(SAMPLE_TRANSCRIPT, llm)
    assert result.accepted == []
    assert "too long" in result.rejected[0][1]


CLOSURE_TRANSCRIPT = (
    "Sam: I just sent the deck — confirmed Priya got it. "
    "Priya: Got it, thanks. "
    "Bob: Great, that's done."
)


def test_closure_detected_with_valid_evidence():
    item = ActionItem(description="Send the deck", evidence_quote="I'll send the deck on Friday.")
    llm = MockLLM(
        [
            {
                "closed": True,
                "evidence_quote": "I just sent the deck — confirmed Priya got it.",
                "confidence": 0.95,
            }
        ]
    )
    result = detect_closure(item, CLOSURE_TRANSCRIPT, llm)
    assert result.closed is True
    assert result.confidence == 0.95


def test_hallucinated_closure_evidence_is_overridden():
    item = ActionItem(description="Send the deck", evidence_quote="I'll send the deck Friday.")
    llm = MockLLM(
        [
            {
                "closed": True,
                "evidence_quote": "Sam confirmed completion in the followup",
                "confidence": 0.9,
            }
        ]
    )
    result = detect_closure(item, CLOSURE_TRANSCRIPT, llm)
    assert result.closed is False
    assert result.evidence_quote is None
    assert result.confidence == 0.0


def test_closure_negative_passes_through():
    item = ActionItem(description="Send the deck")
    llm = MockLLM([{"closed": False, "confidence": 0.1}])
    result = detect_closure(item, CLOSURE_TRANSCRIPT, llm)
    assert result.closed is False

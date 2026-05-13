"""Tests for decision extraction."""

from __future__ import annotations

from meetmind.analyze.actions import MockLLM
from meetmind.analyze.decisions import extract_decisions
from meetmind.models import Decision

TRANSCRIPT = (
    "Sam: I think we should keep using sqlite-vec for now. "
    "Priya: I disagree — we hit performance issues at 200k vectors. "
    "Bob: After the discussion, the call is to use LanceDB going forward. "
    "Sam: Fine, but I'm noting my dissent. "
    "Bob: We'll also defer the Whisper polish pass until v0.8."
)


def test_valid_decisions_extracted():
    llm = MockLLM(
        [
            {
                "decisions": [
                    {
                        "decision": "Adopt LanceDB for the vector store",
                        "rationale": "Performance issues with sqlite-vec at 200k vectors",
                        "dissenters": ["Sam"],
                        "evidence_quote": "the call is to use LanceDB going forward",
                    },
                    {
                        "decision": "Defer Whisper polish pass to v0.8",
                        "rationale": None,
                        "dissenters": [],
                        "evidence_quote": "We'll also defer the Whisper polish pass until v0.8.",
                    },
                ]
            }
        ]
    )
    result = extract_decisions(TRANSCRIPT, llm, source_segment_ids=[10, 11, 12])
    assert len(result.accepted) == 2
    assert isinstance(result.accepted[0], Decision)
    assert result.accepted[0].decision.startswith("Adopt LanceDB")
    assert result.accepted[0].dissenters == ["Sam"]
    assert result.accepted[0].source_segment_ids == [10, 11, 12]


def test_hallucinated_evidence_quote_rejected():
    llm = MockLLM(
        [
            {
                "decisions": [
                    {
                        "decision": "Move office to Mars",
                        "rationale": "Cheaper rent",
                        "dissenters": [],
                        "evidence_quote": "We unanimously agreed to relocate to Mars",
                    }
                ]
            }
        ]
    )
    result = extract_decisions(TRANSCRIPT, llm)
    assert result.accepted == []
    assert len(result.rejected) == 1
    assert "not a substring" in result.rejected[0][1]


def test_overlong_decision_rejected():
    llm = MockLLM(
        [
            {
                "decisions": [
                    {
                        "decision": "A" * 300,
                        "rationale": None,
                        "dissenters": [],
                        "evidence_quote": "the call is to use LanceDB going forward",
                    }
                ]
            }
        ]
    )
    result = extract_decisions(TRANSCRIPT, llm)
    assert result.accepted == []
    assert "too long" in result.rejected[0][1]


def test_empty_decision_list_passes_through():
    llm = MockLLM([{"decisions": []}])
    result = extract_decisions(TRANSCRIPT, llm)
    assert result.accepted == []
    assert result.rejected == []


def test_rationale_overlong_rejected():
    llm = MockLLM(
        [
            {
                "decisions": [
                    {
                        "decision": "Adopt X",
                        "rationale": "B" * 600,
                        "dissenters": [],
                        "evidence_quote": "the call is to use LanceDB going forward",
                    }
                ]
            }
        ]
    )
    result = extract_decisions(TRANSCRIPT, llm)
    assert result.accepted == []
    assert "rationale too long" in result.rejected[0][1]


def test_dissenters_preserved_as_list():
    llm = MockLLM(
        [
            {
                "decisions": [
                    {
                        "decision": "Adopt LanceDB",
                        "rationale": None,
                        "dissenters": ["Sam", "Priya"],
                        "evidence_quote": "the call is to use LanceDB going forward",
                    }
                ]
            }
        ]
    )
    result = extract_decisions(TRANSCRIPT, llm)
    assert result.accepted[0].dissenters == ["Sam", "Priya"]

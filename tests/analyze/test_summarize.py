"""Tests for the Chain-of-Density summarizer."""

from __future__ import annotations

import os

import pytest

from meetmind.analyze.actions import MockLLM
from meetmind.analyze.summarize import summarize_meeting
from meetmind.models import ActionItem, Summary

TRANSCRIPT = (
    "Sam: Welcome everyone. We need to decide on the vector store today. "
    "Priya: I propose LanceDB — sqlite-vec falls over at 200k vectors. "
    "Bob: Agreed. Sam, can you write up the migration plan by Friday? "
    "Sam: Sure. We also need to defer the Whisper polish pass to v0.8."
)


def test_one_pass_density_round_trip():
    llm = MockLLM(
        [
            {
                "tl_dr": "The team chose LanceDB and lined up a migration plan.",
                "headline_topics": [
                    "vector store choice",
                    "migration plan",
                    "Whisper deferral",
                ],
            },
            {
                "tl_dr": (
                    "Sam, Priya, and Bob chose LanceDB over sqlite-vec because it "
                    "handles >200k vectors. Sam will deliver the migration plan by Friday."
                ),
                "headline_topics": ["LanceDB choice", "migration plan", "Friday deadline"],
                "missing_entities": ["Sam", "Priya", "Bob", "Friday", "200k"],
            },
        ]
    )
    result = summarize_meeting(TRANSCRIPT, llm, densify_passes=1)
    assert isinstance(result.summary, Summary)
    assert "LanceDB" in result.summary.tl_dr
    assert result.densify_passes == 1
    assert "LanceDB choice" in result.headline_topics


def test_zero_passes_returns_draft_only():
    llm = MockLLM(
        [
            {
                "tl_dr": "Quick standup on vector store.",
                "headline_topics": ["vector store"],
            }
        ]
    )
    result = summarize_meeting(TRANSCRIPT, llm, densify_passes=0)
    assert result.densify_passes == 0
    assert result.summary.tl_dr == "Quick standup on vector store."


def test_densify_with_no_missing_entities_stops_early():
    """If a densify pass returns no missing entities, we stop iterating."""
    llm = MockLLM(
        [
            {"tl_dr": "Initial summary.", "headline_topics": ["topic A"]},
            {
                "tl_dr": "Initial summary.",
                "headline_topics": ["topic A"],
                "missing_entities": [],
            },
        ]
    )
    result = summarize_meeting(TRANSCRIPT, llm, densify_passes=3)
    assert result.densify_passes == 0
    assert result.summary.tl_dr == "Initial summary."


def test_action_items_and_decisions_passed_through():
    llm = MockLLM([{"tl_dr": "Brief.", "headline_topics": ["t"]}])
    actions = [ActionItem(description="Send the deck", evidence_quote="the deck Friday")]
    decisions = ["Adopt LanceDB"]
    result = summarize_meeting(
        TRANSCRIPT,
        llm,
        densify_passes=0,
        key_decisions=decisions,
        action_items=actions,
    )
    assert result.summary.key_decisions == ["Adopt LanceDB"]
    assert len(result.summary.action_items) == 1
    assert result.summary.action_items[0].description == "Send the deck"


@pytest.mark.skipif(
    os.environ.get("OLLAMA_LIVE") != "1",
    reason="set OLLAMA_LIVE=1 for live model summarization",
)
def test_live_summarize_with_local_ollama():
    """Live: real Ollama call, two-shape extraction routed by prompt content.

    Constructs a fresh `actants.LLM` per call so the underlying httpx
    client doesn't capture a worker-thread loop that gets closed
    between calls. Production callers cache the LLM across the whole
    session; this is just a test-isolation concession.
    """
    from meetmind.analyze.llm import _run, get_default_llm
    from meetmind.analyze.summarize import _DensePayload, _DraftPayload

    def _route(prompt: str) -> dict:
        llm = get_default_llm()
        schema = _DensePayload if "previous_draft" in prompt else _DraftPayload
        return _run(llm.extract(prompt, schema)).model_dump(mode="python")

    result = summarize_meeting(TRANSCRIPT, _route, densify_passes=1)
    assert result.summary.tl_dr  # non-empty
    assert len(result.summary.tl_dr) > 20

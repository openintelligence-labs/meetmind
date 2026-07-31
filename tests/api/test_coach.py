"""Tests for the live coach loop."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from meetmind.api.bus import EventBus
from meetmind.api.coach import CoachConfig, CoachLoop
from meetmind.api.events import CoachTipEvent, FinalEvent, PartialEvent, SpeakerEvent


@dataclass
class _StubResult:
    content: str


@dataclass
class StubLLM:
    """Deterministic stand-in for `actants.LLM` — returns a fixed JSON body."""

    body: str = json.dumps(
        {
            "tip": "Ask Sam when the migration deck will be ready.",
            "category": "question",
            "confidence": 0.85,
        }
    )
    calls: list[str] = field(default_factory=list)

    async def complete(self, prompt: str, *, model: str | None = None) -> _StubResult:
        self.calls.append(prompt)
        return _StubResult(content=self.body)


def _final(text: str, start_ms: int, end_ms: int) -> FinalEvent:
    return FinalEvent(text=text, start_ms=start_ms, end_ms=end_ms, language="en")


def _speaker(text: str, start_ms: int, end_ms: int) -> SpeakerEvent:
    return SpeakerEvent(
        text=text,
        cluster_id="spk0",
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=0.9,
    )


def test_partial_events_are_ignored():
    coach = CoachLoop(llm=StubLLM())
    coach.ingest(PartialEvent(text="we should probably", start_ms=0, end_ms=500))
    assert len(coach._spans) == 0


def test_final_and_speaker_events_are_kept():
    coach = CoachLoop(llm=StubLLM())
    coach.ingest(_final("Sam mentioned the deck is Friday.", 0, 4000))
    coach.ingest(_speaker("Priya proposed adopting LanceDB.", 4000, 8000))
    assert len(coach._spans) == 2


def test_window_evicts_spans_outside_60s():
    coach = CoachLoop(llm=StubLLM(), config=CoachConfig(window_seconds=5.0))
    coach.ingest(_final("old", 0, 1000))
    coach.ingest(_final("middle", 4000, 5000))
    coach.ingest(_final("recent", 9000, 10000))
    # Window cutoff is end_ms_of_last - 5s = 5000ms; "old" (ended 1000) gone.
    assert [s.text for s in coach._spans] == ["middle", "recent"]


@pytest.mark.asyncio
async def test_emit_tip_publishes_to_bus():
    bus = EventBus()
    coach = CoachLoop(bus=bus, llm=StubLLM(), config=CoachConfig(min_text_chars=10))
    coach.ingest(
        _final("Sam said the deck will be ready by Friday for the migration review.", 0, 4000)
    )

    received: list = []
    async with bus.subscription() as queue:
        tip = await coach.emit_tip_now()
        try:
            while True:
                received.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            pass

    assert tip is not None
    assert isinstance(tip, CoachTipEvent)
    assert "migration" in tip.tip.lower() or "deck" in tip.tip.lower()
    assert tip.category == "question"
    assert tip.confidence == 0.85
    assert tip.window_start_ms == 0
    assert tip.window_end_ms == 4000
    assert any(isinstance(e, CoachTipEvent) for e in received)


@pytest.mark.asyncio
async def test_skip_when_below_min_text_chars():
    coach = CoachLoop(llm=StubLLM(), config=CoachConfig(min_text_chars=100))
    coach.ingest(_final("too short", 0, 1000))
    tip = await coach.emit_tip_now()
    assert tip is None


@pytest.mark.asyncio
async def test_skip_when_llm_returns_empty_tip():
    llm = StubLLM(body=json.dumps({"tip": "", "category": "other", "confidence": 0.0}))
    coach = CoachLoop(llm=llm, config=CoachConfig(min_text_chars=10))
    coach.ingest(_final("a" * 50 + " is what they said about the migration today", 0, 2000))
    tip = await coach.emit_tip_now()
    assert tip is None


@pytest.mark.asyncio
async def test_skip_when_confidence_too_low():
    llm = StubLLM(body=json.dumps({"tip": "Maybe ask?", "category": "question", "confidence": 0.1}))
    coach = CoachLoop(llm=llm, config=CoachConfig(min_text_chars=10, min_confidence=0.5))
    coach.ingest(_final("Sam mentioned the deck for the migration review tomorrow", 0, 4000))
    tip = await coach.emit_tip_now()
    assert tip is None


@pytest.mark.asyncio
async def test_does_not_re_emit_for_unchanged_window():
    coach = CoachLoop(llm=StubLLM(), config=CoachConfig(min_text_chars=10))
    coach.ingest(_final("Sam mentioned the deck for the migration review tomorrow", 0, 4000))
    tip1 = await coach.emit_tip_now()
    tip2 = await coach.emit_tip_now()
    assert tip1 is not None
    assert tip2 is None


@pytest.mark.asyncio
async def test_re_emits_after_new_span():
    coach = CoachLoop(llm=StubLLM(), config=CoachConfig(min_text_chars=10))
    coach.ingest(_final("Sam mentioned the deck for the migration review tomorrow", 0, 4000))
    tip1 = await coach.emit_tip_now()
    coach.ingest(_final("Priya proposed switching the vector store to LanceDB", 4000, 8000))
    tip2 = await coach.emit_tip_now()
    assert tip1 is not None
    assert tip2 is not None
    assert tip2.window_end_ms == 8000


@pytest.mark.asyncio
async def test_handles_llm_returning_garbage():
    llm = StubLLM(body="not json at all sorry")
    coach = CoachLoop(llm=llm, config=CoachConfig(min_text_chars=10))
    coach.ingest(_final("Sam mentioned the deck for the migration review tomorrow", 0, 4000))
    tip = await coach.emit_tip_now()
    assert tip is None  # gracefully drops malformed responses


@pytest.mark.asyncio
async def test_handles_llm_raising():
    class BoomLLM:
        async def complete(self, prompt, *, model=None):
            raise RuntimeError("ollama is down")

    coach = CoachLoop(llm=BoomLLM(), config=CoachConfig(min_text_chars=10))
    coach.ingest(_final("Sam mentioned the deck for the migration review tomorrow", 0, 4000))
    tip = await coach.emit_tip_now()
    assert tip is None  # never crashes the bus


@pytest.mark.asyncio
async def test_run_loop_consumes_published_events():
    bus = EventBus()
    coach = CoachLoop(
        bus=bus,
        llm=StubLLM(),
        config=CoachConfig(window_seconds=60.0, tick_seconds=0.05, min_text_chars=10),
    )
    stop = asyncio.Event()
    runner = asyncio.create_task(coach.run(stop=stop))
    await asyncio.sleep(0)  # let the subscription register
    await bus.publish(_final("Sam mentioned the deck for the migration review tomorrow", 0, 4000))

    # Wait for at least one tick to fire.
    deadline = asyncio.get_event_loop().time() + 1.0
    while asyncio.get_event_loop().time() < deadline:
        if len(coach._spans) >= 1:
            break
        await asyncio.sleep(0.02)

    stop.set()
    await asyncio.wait_for(runner, timeout=1.0)
    assert len(coach._spans) >= 1


@pytest.mark.asyncio
async def test_lenient_json_extraction_strips_code_fences():
    body = (
        "```json\n"
        + json.dumps(
            {"tip": "Ask about the migration timeline.", "category": "question", "confidence": 0.7}
        )
        + "\n```"
    )
    coach = CoachLoop(llm=StubLLM(body=body), config=CoachConfig(min_text_chars=10))
    coach.ingest(_final("Sam mentioned the deck for the migration review tomorrow", 0, 4000))
    tip = await coach.emit_tip_now()
    assert tip is not None
    assert "migration" in tip.tip.lower()


@pytest.mark.asyncio
async def test_unknown_category_normalized_to_other():
    body = json.dumps({"tip": "Listen more.", "category": "weird-thing", "confidence": 0.6})
    coach = CoachLoop(llm=StubLLM(body=body), config=CoachConfig(min_text_chars=10))
    coach.ingest(_final("Sam mentioned the deck for the migration review tomorrow", 0, 4000))
    tip = await coach.emit_tip_now()
    assert tip is not None
    assert tip.category == "other"

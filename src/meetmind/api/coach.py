"""Live coaching loop, off by default and enabled by ``meetmind record --coach``.

Holds a rolling window of recent transcript text and periodically asks the
configured LLM (``MEETMIND_COACH_MODEL``, else ``MEETMIND_LLM_MODEL``) for one
short tip, published back onto the bus as a ``CoachTipEvent``. Ticks are
dropped rather than queued so a slow model never blocks the transcript path.

Lives in ``meetmind.api`` rather than ``meetmind.analyze`` because it both
subscribes to and publishes onto the ``EventBus``, which the import-linter
contract forbids ``analyze`` from touching.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from meetmind.api.bus import EventBus, default_bus
from meetmind.api.events import CoachTipEvent, Event, FinalEvent, SpeakerEvent

log = logging.getLogger(__name__)


_DEFAULT_PROMPT = """You are a meeting coach watching a live transcript.
The user (the participant) wants ONE short, actionable tip about what
they should do RIGHT NOW given the last minute of conversation.

Rules:
- Output JSON only, with keys: tip (string, ≤ 25 words), category
  (one of: question, missed_point, follow_up, clarify, other),
  confidence (float 0.0-1.0).
- Be specific. Reference what was said. Do not be generic.
- If nothing is worth saying, set tip to an empty string and confidence to 0.0.
- Never speculate about content that isn't in the transcript.

Transcript window (most recent first):
{transcript}

JSON only, no preamble:"""


@dataclass
class CoachConfig:
    window_seconds: float = 60.0
    tick_seconds: float = 15.0
    min_text_chars: int = 80
    min_confidence: float = 0.3
    model: str | None = None
    prompt_template: str = _DEFAULT_PROMPT


@dataclass
class _Span:
    text: str
    start_ms: int
    end_ms: int


@dataclass
class CoachLoop:
    """Background task: subscribes to bus, emits CoachTipEvents."""

    bus: EventBus = field(default_factory=lambda: default_bus)
    config: CoachConfig = field(default_factory=CoachConfig)
    llm: Any | None = None  # actants.LLM-shaped; lazily constructed
    _spans: deque[_Span] = field(default_factory=deque)
    _last_tip_window: tuple[int, int] | None = None

    async def run(self, *, stop: asyncio.Event | None = None) -> None:
        """Run forever. Cancellable via ``stop`` event or task cancel."""
        stop = stop or asyncio.Event()
        async with self.bus.subscription() as queue:
            consumer = asyncio.create_task(self._consume(queue, stop))
            ticker = asyncio.create_task(self._tick(stop))
            try:
                await asyncio.wait(
                    {consumer, ticker, asyncio.create_task(stop.wait())},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                consumer.cancel()
                ticker.cancel()
                with _suppress_cancel():
                    await asyncio.gather(consumer, ticker, return_exceptions=True)

    def ingest(self, event: Event) -> None:
        """Add an event's text to the rolling window, if it carries any."""
        span = _event_to_span(event)
        if span is not None:
            self._spans.append(span)
            self._evict()

    async def emit_tip_now(self) -> CoachTipEvent | None:
        """Emit a tip for the current window now.

        Returns the published event, or None when the window lacks signal or
        the LLM declines.
        """
        return await self._maybe_emit()

    async def _consume(self, queue: asyncio.Queue[Event], stop: asyncio.Event) -> None:
        while not stop.is_set():
            event = await queue.get()
            self.ingest(event)

    async def _tick(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.tick_seconds)
                return
            except TimeoutError:
                pass
            try:
                await self._maybe_emit()
            except Exception:  # noqa: BLE001 — coach must never crash the bus
                log.exception("coach tick failed")

    def _evict(self) -> None:
        if not self._spans:
            return
        cutoff_ms = self._spans[-1].end_ms - int(self.config.window_seconds * 1000)
        while self._spans and self._spans[0].end_ms < cutoff_ms:
            self._spans.popleft()

    def _window_text(self) -> str:
        return " ".join(s.text for s in self._spans).strip()

    def _window_bounds(self) -> tuple[int, int] | None:
        if not self._spans:
            return None
        return self._spans[0].start_ms, self._spans[-1].end_ms

    async def _maybe_emit(self) -> CoachTipEvent | None:
        bounds = self._window_bounds()
        if bounds is None:
            return None
        text = self._window_text()
        if len(text) < self.config.min_text_chars:
            return None
        if self._last_tip_window == bounds:
            return None  # nothing new since the last tip

        prompt = self.config.prompt_template.format(transcript=text)
        result = await self._ask_llm(prompt)
        if result is None or not result.get("tip"):
            self._last_tip_window = bounds
            return None
        confidence = float(result.get("confidence", 0.0))
        if confidence < self.config.min_confidence:
            self._last_tip_window = bounds
            return None

        event = CoachTipEvent(
            tip=str(result["tip"]).strip(),
            category=_normalize_category(result.get("category")),
            window_start_ms=bounds[0],
            window_end_ms=bounds[1],
            model=self._resolve_model(),
            confidence=confidence,
        )
        await self.bus.publish(event)
        self._last_tip_window = bounds
        return event

    async def _ask_llm(self, prompt: str) -> dict[str, Any] | None:
        llm = self._ensure_llm()
        if llm is None:
            return None
        try:
            result = await llm.complete(prompt, model=self._resolve_model())
        except Exception as e:  # noqa: BLE001
            log.warning("coach LLM call failed: %s", e)
            return None
        body = getattr(result, "content", "") or ""
        return _extract_json(body)

    def _ensure_llm(self) -> Any | None:
        if self.llm is not None:
            return self.llm
        try:
            from meetmind.analyze.llm import get_default_llm  # noqa: PLC0415

            self.llm = get_default_llm()
            return self.llm
        except Exception as e:  # noqa: BLE001
            log.warning("coach LLM unavailable: %s", e)
            return None

    def _resolve_model(self) -> str | None:
        return (
            self.config.model
            or os.environ.get("MEETMIND_COACH_MODEL")
            or os.environ.get("MEETMIND_LLM_MODEL")
            or None
        )


def _event_to_span(event: Event) -> _Span | None:
    """Pluck committed text out of FinalEvent / SpeakerEvent, dropping the rest.

    `partial` events are ignored on purpose: they are revisable, so feeding
    them to the LLM would make the tip flip-flop on every revision.
    """
    if isinstance(event, SpeakerEvent):
        return _Span(text=event.text, start_ms=event.start_ms, end_ms=event.end_ms)
    if isinstance(event, FinalEvent):
        return _Span(text=event.text, start_ms=event.start_ms, end_ms=event.end_ms)
    return None


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(body: str) -> dict[str, Any] | None:
    """Parse JSON from a model response, tolerating surrounding prose or fences."""
    body = body.strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        match = _JSON_RE.search(body)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _normalize_category(value: Any) -> str:
    allowed = {"question", "missed_point", "follow_up", "clarify", "other"}
    if isinstance(value, str) and value.lower() in allowed:
        return value.lower()
    return "other"


class _suppress_cancel:
    """`contextlib.suppress(CancelledError)` shim that also handles BaseException."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is asyncio.CancelledError


def _now_ms() -> int:
    return int(time.monotonic() * 1000)

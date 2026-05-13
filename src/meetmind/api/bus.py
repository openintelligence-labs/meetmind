"""In-process pub/sub bus for live transcript events.

The pipeline (capture → STT → diarize → stitcher) calls
`bus.publish(event)`. The FastAPI SSE endpoint subscribes via
`async for event in bus.subscribe(): ...`.

Multiple subscribers each get their own queue; slow subscribers don't
block the publisher (we drop oldest events past the queue cap).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from meetmind.api.events import Event

log = logging.getLogger(__name__)

_DEFAULT_CAP = 1024


class EventBus:
    """Fan-out async event bus, one queue per subscriber.

    Slow subscribers don't block the publisher: when a subscriber queue
    is full we drop the oldest event and bump a per-bus counter
    (``dropped_events``). Production callers can poll it to detect when
    a UI tab or SSE client has stalled.
    """

    def __init__(self, queue_cap: int = _DEFAULT_CAP) -> None:
        self._queue_cap = queue_cap
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._lock = asyncio.Lock()
        self._dropped_events: int = 0
        self._slow_subscriber_warnings: int = 0

    async def publish(self, event: Event) -> None:
        async with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            if q.qsize() >= self._queue_cap:
                # Drop oldest; pipeline must keep producing.
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                self._dropped_events += 1
                # Warn at exponential-backoff intervals so we don't
                # flood the log on a persistently-stalled subscriber.
                if self._dropped_events & (self._dropped_events - 1) == 0:
                    self._slow_subscriber_warnings += 1
                    log.warning(
                        "EventBus dropped %d events to slow subscribers (cap=%d)",
                        self._dropped_events,
                        self._queue_cap,
                    )
            await q.put(event)

    @property
    def dropped_events(self) -> int:
        """Cumulative count of events dropped due to backpressure."""
        return self._dropped_events

    @asynccontextmanager
    async def subscription(self) -> AsyncIterator[asyncio.Queue[Event]]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_cap)
        async with self._lock:
            self._subscribers.add(q)
        try:
            yield q
        finally:
            async with self._lock:
                self._subscribers.discard(q)

    async def subscribe(self) -> AsyncIterator[Event]:
        """Convenience: yield events forever until the consumer cancels."""
        async with self.subscription() as q:
            while True:
                yield await q.get()


# Process-global default bus. The CLI publishes to this; the FastAPI app
# subscribes from this. Tests construct their own EventBus instances.
default_bus = EventBus()

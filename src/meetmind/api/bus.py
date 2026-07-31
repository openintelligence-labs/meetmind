"""In-process pub/sub bus for live transcript events.

The capture pipeline publishes; the SSE endpoint subscribes. Each subscriber
gets its own queue, and events past the queue cap are dropped rather than
blocking the publisher.
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

    A full subscriber queue drops its oldest event and increments
    ``dropped_events``, which callers can poll to detect a stalled client.
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
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                self._dropped_events += 1
                # Warn only on powers of two, so a persistently stalled
                # subscriber cannot flood the log.
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
        """Yield events until the consumer cancels."""
        async with self.subscription() as q:
            while True:
                yield await q.get()


# Process-global bus shared by the CLI publisher and the FastAPI app.
default_bus = EventBus()

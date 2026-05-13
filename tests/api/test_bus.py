"""Tests for the in-process event bus."""

from __future__ import annotations

import asyncio

from meetmind.api.bus import EventBus
from meetmind.api.events import FinalEvent, MetaEvent, PartialEvent


async def test_publish_with_no_subscribers_is_a_noop():
    bus = EventBus()
    await bus.publish(MetaEvent(event="session_started"))


async def test_subscriber_receives_published_events():
    bus = EventBus()
    received: list = []

    async def consume():
        async with bus.subscription() as q:
            for _ in range(2):
                received.append(await q.get())

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let consumer subscribe
    await bus.publish(PartialEvent(text="hi", start_ms=0, end_ms=100))
    await bus.publish(FinalEvent(text="hi there", start_ms=0, end_ms=200))
    await consumer
    assert isinstance(received[0], PartialEvent)
    assert received[1].text == "hi there"


async def test_multiple_subscribers_all_receive_events():
    bus = EventBus()
    counts = [0, 0]

    async def consume(idx: int) -> None:
        async with bus.subscription() as q:
            counts[idx] += 1 if (await q.get()).kind == "meta" else 0

    a = asyncio.create_task(consume(0))
    b = asyncio.create_task(consume(1))
    await asyncio.sleep(0)
    await bus.publish(MetaEvent(event="session_started"))
    await asyncio.gather(a, b)
    assert counts == [1, 1]


async def test_slow_subscriber_drops_oldest():
    bus = EventBus(queue_cap=3)
    async with bus.subscription() as q:
        for i in range(5):
            await bus.publish(PartialEvent(text=f"#{i}", start_ms=i, end_ms=i + 1))
        # Cap was 3, so oldest two were dropped.
        assert q.qsize() == 3
        first_in_queue = await q.get()
        assert first_in_queue.text == "#2"

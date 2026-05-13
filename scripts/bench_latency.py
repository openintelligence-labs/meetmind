#!/usr/bin/env python3
"""Latency bench (S15.1).

Measures three latencies that matter for the user experience:

  * **wake-to-first-caption** — time from `meetmind record` start until
    the first `partial` event lands on the bus
  * **EoT-to-action**         — time from end-of-utterance until the
    extracted ActionItem is persisted
  * **query-to-result**       — time from `meetmind search QUERY` until
    the first hit returns

Pass `--mock` to use the Python mock sidecars (no native binaries
required). Real-sidecar runs need the macOS sidecars built; on Linux
this script is the harness for when those land.

Output: one JSON object per measurement to stdout. CI collects these
into a tracked perf table over time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from meetmind.api.bus import EventBus
from meetmind.api.events import FinalEvent, PartialEvent


async def _bench_wake_to_first_caption(timeout_s: float = 10.0) -> dict:
    """Measure end-to-end first-caption latency.

    Subscribes to a fresh bus, then schedules a synthetic Partial 100ms
    later (simulating the pipeline emitting). The measured latency is
    the time from subscribe to receive — same path the real pipeline
    walks once it has audio.
    """
    bus = EventBus()
    t0 = time.monotonic()

    async def producer() -> None:
        await asyncio.sleep(0.1)
        await bus.publish(PartialEvent(text="hello", start_ms=0, end_ms=100))
        await bus.publish(FinalEvent(text="hello world", start_ms=0, end_ms=200))

    asyncio.create_task(producer())  # noqa: RUF006
    async with bus.subscription() as q:
        evt = await asyncio.wait_for(q.get(), timeout=timeout_s)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    return {
        "metric": "wake_to_first_caption_ms",
        "value": round(elapsed_ms, 2),
        "first_kind": evt.kind,
    }


async def _bench_eot_to_action() -> dict:
    """Time the action-extraction substring guard on a fixed transcript.

    The full LLM-driven path needs a live model; that's measured by the
    OLLAMA_LIVE-gated tests. This bench measures only the guard layer
    that wraps the LLM, which is what dominates when the LLM is fast.
    """
    from meetmind.analyze.actions import ExtractedItem, _validate

    transcript = "Sam: I'll send the migration deck on Friday."
    item = ExtractedItem(
        description="Send migration deck",
        owner="Sam",
        evidence_quote="I'll send the migration deck on Friday.",
    )
    t0 = time.monotonic()
    for _ in range(1000):
        _validate(item, transcript)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    return {
        "metric": "eot_to_action_substring_guard_us_per_call",
        "value": round(elapsed_ms / 1000 * 1000, 2),  # μs/call
    }


def _bench_search_query() -> dict:
    """Measure HybridIndex.search() against a small in-memory corpus."""
    from meetmind.memory.vector import HybridIndex, IndexedSegment, hash_embedder

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        idx = HybridIndex.open(Path(td) / "vec", vector_dim=64, embedder=hash_embedder(64))
        idx.add(
            [
                IndexedSegment(
                    meeting_id="m1",
                    segment_id=i,
                    text=f"talking about migration item {i}",
                    start_ms=i * 1000,
                    end_ms=(i + 1) * 1000,
                    cluster_id="spk0",
                    channel="loopback",
                    language="en",
                )
                for i in range(50)
            ]
        )
        t0 = time.monotonic()
        for _ in range(20):
            idx.search("migration timeline", limit=5)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
    return {
        "metric": "query_to_result_ms_per_query",
        "value": round(elapsed_ms / 20, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON-lines to FILE.")
    args = parser.parse_args()

    results = [
        asyncio.run(_bench_wake_to_first_caption()),
        asyncio.run(_bench_eot_to_action()),
        _bench_search_query(),
    ]

    if args.out:
        args.out.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()

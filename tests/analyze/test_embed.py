"""Tests for the actants embedder wrapper.

Live tests run only when OLLAMA_LIVE=1 and a local
`nomic-embed-text` model is installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from meetmind.analyze.embed import make_embedder, probe_embedder_dim
from meetmind.analyze.llm import list_local_models
from meetmind.memory.vector import HybridIndex, IndexedSegment


def _have_embed_model() -> bool:
    return any(m.get("name", "").startswith("nomic-embed-text") for m in list_local_models())


pytestmark = pytest.mark.skipif(
    os.environ.get("OLLAMA_LIVE") != "1" or not _have_embed_model(),
    reason="set OLLAMA_LIVE=1 with nomic-embed-text installed",
)


def test_probe_dim_returns_positive_integer():
    embedder = make_embedder()
    dim = probe_embedder_dim(embedder)
    assert dim > 0
    assert dim in (256, 512, 768, 1024)


def test_embedder_returns_consistent_dimension():
    embedder = make_embedder()
    a = embedder("hello world")
    b = embedder("good evening")
    assert len(a) == len(b)
    assert all(isinstance(x, float) for x in a[:5])


def test_real_embedder_drives_hybrid_index(tmp_path: Path):
    """Round-trip: real embedder + LanceDB store + retrieval works end-to-end."""
    embedder = make_embedder()
    dim = probe_embedder_dim(embedder)
    index = HybridIndex.open(tmp_path / "vec", vector_dim=dim, embedder=embedder)
    index.add(
        [
            IndexedSegment(
                meeting_id="M1",
                segment_id=1,
                text="Sam discussed the Snowflake migration timeline",
                start_ms=0,
                end_ms=4000,
                cluster_id="remote-A",
                channel="loopback",
                language="en",
            ),
            IndexedSegment(
                meeting_id="M1",
                segment_id=2,
                text="Priya mentioned the OKR review next month",
                start_ms=4000,
                end_ms=8000,
                cluster_id="remote-B",
                channel="loopback",
                language="en",
            ),
            IndexedSegment(
                meeting_id="M1",
                segment_id=3,
                text="Bob asked about the Q3 budget",
                start_ms=8000,
                end_ms=12000,
                cluster_id="remote-C",
                channel="loopback",
                language="en",
            ),
        ]
    )
    hits = index.search("data warehouse rollout date", limit=3)
    assert len(hits) >= 1
    top_ids = [h.segment.segment_id for h in hits]
    assert 1 in top_ids

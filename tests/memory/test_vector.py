"""Tests for the LanceDB hybrid vector index."""

from __future__ import annotations

from pathlib import Path

import pytest

from meetmind.memory.vector import (
    HybridIndex,
    IndexedSegment,
    _bm25_scores,
    _cosine,
    _existing_table_names,
    _rank,
    hash_embedder,
)

# ────────── LanceDB list_tables() shape regression guard ──────────


def test_existing_table_names_handles_flat_list() -> None:
    """Older LanceDB returned `list[str]` from list_tables()."""

    class _FakeDB:
        def list_tables(self):
            return ["segments", "other"]

    assert _existing_table_names(_FakeDB()) == ["segments", "other"]


def test_existing_table_names_handles_paginated_tuple() -> None:
    """LanceDB ≥0.16 returns `[("tables", [...]), ("page_token", None)]`.

    Earlier versions of our `open()` did `if table_name in existing:` on
    this shape and always missed — every `meetmind index` then tried to
    re-create the table and raised. Regression guard for that fix.
    """

    class _FakeDB:
        def list_tables(self):
            return [("tables", ["segments"]), ("page_token", None)]

    assert _existing_table_names(_FakeDB()) == ["segments"]


def test_existing_table_names_falls_back_to_legacy_table_names() -> None:
    class _FakeDB:
        # No list_tables — old release.
        def table_names(self):
            return ["segments"]

    assert _existing_table_names(_FakeDB()) == ["segments"]


def test_existing_table_names_unknown_shape_returns_empty(caplog) -> None:
    class _FakeDB:
        def list_tables(self):
            return [{"weird": "shape"}]

    with caplog.at_level("WARNING"):
        names = _existing_table_names(_FakeDB())
    assert names == []
    assert any("unrecognized" in r.message for r in caplog.records)


@pytest.fixture
def index(tmp_path: Path) -> HybridIndex:
    embed = hash_embedder(dim=64)
    return HybridIndex.open(tmp_path / "vec", vector_dim=64, embedder=embed)


def _seg(text: str, sid: int, mid: str = "M1", **kw) -> IndexedSegment:
    return IndexedSegment(
        meeting_id=mid,
        segment_id=sid,
        text=text,
        start_ms=sid * 1000,
        end_ms=(sid + 1) * 1000,
        cluster_id=kw.get("cluster_id", "self"),
        channel=kw.get("channel", "mic"),
        language="en",
    )


def test_bm25_higher_for_query_term_overlap():
    docs = [
        "the quick brown fox",
        "lazy dogs sleep all day",
        "Snowflake migration is on Friday",
    ]
    scores = _bm25_scores("snowflake migration", docs)
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]


def test_bm25_zero_when_no_overlap():
    docs = ["alpha beta gamma"]
    scores = _bm25_scores("delta", docs)
    assert scores[0] == 0.0


def test_cosine_identical_unit_vectors_is_one():
    v = [0.6, 0.8]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_rank_descending_handles_ties():
    assert _rank([3.0, 1.0, 3.0, 2.0], descending=True) == [0, 3, 0, 2]


def test_index_length_grows_on_add(index: HybridIndex):
    assert len(index) == 0
    n = index.add([_seg("hello world", 1), _seg("goodbye moon", 2)])
    assert n == 2
    assert len(index) == 2


def test_search_returns_relevant_results(index: HybridIndex):
    index.add(
        [
            _seg("Sam mentioned Snowflake migration timeline", 1),
            _seg("Bob asked about quarterly OKRs", 2),
            _seg("Priya proposed adopting LanceDB instead of sqlite-vec", 3),
            _seg("the team agreed on Friday for the deployment", 4),
        ]
    )
    hits = index.search("snowflake migration", limit=3)
    assert len(hits) >= 1
    assert hits[0].segment.segment_id == 1
    assert hits[0].score > 0


def test_search_filter_by_meeting_id(index: HybridIndex):
    index.add(
        [
            _seg("Snowflake topic in M1", 1, mid="M1"),
            _seg("Snowflake topic in M2", 2, mid="M2"),
        ]
    )
    hits_m1 = index.search("snowflake", meeting_id="M1")
    assert len(hits_m1) == 1
    assert hits_m1[0].segment.meeting_id == "M1"


def test_search_returns_empty_when_no_match(index: HybridIndex):
    index.add([_seg("alpha beta gamma", 1)])
    assert index.search("absolutely-no-such-term") == []


def test_add_rejects_wrong_dim_vector(tmp_path: Path):
    bad_embedder = lambda _t: [0.0] * 8  # noqa: E731
    idx = HybridIndex.open(tmp_path / "bad", vector_dim=64, embedder=bad_embedder)
    with pytest.raises(ValueError):
        idx.add([_seg("hello", 1)])


def test_search_uses_both_bm25_and_dense_signal(index: HybridIndex):
    index.add(
        [
            _seg("the migration plan was discussed", 1),
            _seg("Sam talked about the migration plan", 2),
            _seg("totally unrelated lorem ipsum text", 3),
        ]
    )
    hits = index.search("migration plan", limit=3)
    seg_ids = {h.segment.segment_id for h in hits}
    assert 1 in seg_ids
    assert 2 in seg_ids

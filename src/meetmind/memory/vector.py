"""LanceDB embedded vector store + hybrid search.

Stores transcript-segment embeddings keyed by `(meeting_id, segment_id)`
with their text, channel, speaker, and start/end timestamps. Hybrid
retrieval combines BM25 (lexical) + cosine (dense) via Reciprocal Rank
Fusion (RRF, k=60). ColBERT rerank lands in v0.10.x once we have a
named-entity heuristic.

This module is purely the persistence + retrieval layer. The embedder
is pluggable: pass any callable `embed(text: str) -> list[float]` to
`HybridIndex.add` / `HybridIndex.search`. Production uses
`nomic-embed-text v2`; tests use a deterministic hash-based stub.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

log = logging.getLogger(__name__)

DEFAULT_TABLE_NAME = "segments"
RRF_K = 60


Embedder = Callable[[str], list[float]]


@dataclass(frozen=True)
class IndexedSegment:
    """One row in the vector store. Wire-format-stable."""

    meeting_id: str
    segment_id: int
    text: str
    start_ms: int
    end_ms: int
    cluster_id: str | None
    channel: str | None
    language: str

    def to_record(self, vector: list[float]) -> dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "segment_id": self.segment_id,
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "cluster_id": self.cluster_id or "",
            "channel": self.channel or "",
            "language": self.language,
            "vector": vector,
        }


@dataclass(frozen=True)
class SearchHit:
    segment: IndexedSegment
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


def _existing_table_names(db) -> list[str]:
    """Return the list of table names known to a LanceDB connection.

    LanceDB ≥0.16 returns a paginated tuple from ``list_tables()``
    (shape: ``[("tables", [...]), ("page_token", None)]``) which is
    counter-intuitive — older releases returned a flat list of strings,
    and even older releases used ``table_names()``. This helper covers
    all three so we can keep the call site clean.
    """
    raw: Any
    try:
        raw = db.list_tables()  # type: ignore[attr-defined]
    except AttributeError:
        raw = db.table_names()  # type: ignore[attr-defined]
    # Already a plain list of strings? Return it.
    materialized = list(raw)
    if all(isinstance(x, str) for x in materialized):
        return materialized
    # Paginated tuple form: dict-style 2-tuples carrying the tables key.
    for item in materialized:
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "tables":
            inner = item[1]
            if isinstance(inner, list):
                return [s for s in inner if isinstance(s, str)]
    # Unrecognized shape — log + return empty so the caller will create.
    log.warning("unrecognized list_tables() shape: %r", materialized)
    return []


def _make_schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("meeting_id", pa.string()),
            pa.field("segment_id", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("start_ms", pa.int64()),
            pa.field("end_ms", pa.int64()),
            pa.field("cluster_id", pa.string()),
            pa.field("channel", pa.string()),
            pa.field("language", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def _row_to_segment(row: dict[str, Any]) -> IndexedSegment:
    return IndexedSegment(
        meeting_id=row["meeting_id"],
        segment_id=int(row["segment_id"]),
        text=row["text"],
        start_ms=int(row["start_ms"]),
        end_ms=int(row["end_ms"]),
        cluster_id=row.get("cluster_id") or None,
        channel=row.get("channel") or None,
        language=row.get("language") or "en",
    )


class HybridIndex:
    """Hybrid (BM25 + dense) retriever over transcript segments.

    Open with `HybridIndex.open(path, vector_dim, embedder)`; the LanceDB
    directory at `path` is created lazily on first add. Tests typically
    pass `tmp_path / "lancedb"`.
    """

    def __init__(
        self,
        db: lancedb.DBConnection,
        table: lancedb.table.Table,
        vector_dim: int,
        embedder: Embedder,
    ) -> None:
        self._db = db
        self._table = table
        self.vector_dim = vector_dim
        self.embedder = embedder

    @classmethod
    def open(
        cls,
        path: Path | str,
        *,
        vector_dim: int,
        embedder: Embedder,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> HybridIndex:
        db = lancedb.connect(str(path))
        existing = _existing_table_names(db)
        if table_name in existing:
            table = db.open_table(table_name)
        else:
            table = db.create_table(table_name, schema=_make_schema(vector_dim))
        return cls(db=db, table=table, vector_dim=vector_dim, embedder=embedder)

    def __len__(self) -> int:
        return self._table.count_rows()

    def add(self, segments: Iterable[IndexedSegment]) -> int:
        rows = []
        for seg in segments:
            vec = self.embedder(seg.text)
            if len(vec) != self.vector_dim:
                raise ValueError(
                    f"embedder returned {len(vec)}-d vector; expected {self.vector_dim}"
                )
            rows.append(seg.to_record(vec))
        if not rows:
            return 0
        self._table.add(rows)
        return len(rows)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        meeting_id: str | None = None,
    ) -> list[SearchHit]:
        """Hybrid BM25+dense → RRF fusion. Returns top-`limit` hits."""
        query_vec = self.embedder(query)
        if len(query_vec) != self.vector_dim:
            raise ValueError(
                f"embedder returned {len(query_vec)}-d vector; expected {self.vector_dim}"
            )

        # Pull candidate rows once; for the data sizes this index targets
        # (one user's meetings) loading and ranking in Python is fine.
        # Production paths swap to LanceDB native hybrid search once the
        # FTS index API stabilizes there. Keeping it simple for v0.10.
        rows: list[dict[str, Any]] = self._fetch_all(meeting_id=meeting_id)
        if not rows:
            return []

        bm25_scores = _bm25_scores(query, [r["text"] for r in rows])
        dense_scores = [_cosine(query_vec, r["vector"]) for r in rows]

        bm25_rank = _rank(bm25_scores, descending=True)
        dense_rank = _rank(dense_scores, descending=True)

        fused: list[tuple[float, int]] = []  # (score, idx)
        for i in range(len(rows)):
            score = 0.0
            if bm25_scores[i] > 0:
                score += 1.0 / (RRF_K + bm25_rank[i] + 1)
            if dense_scores[i] > 0:
                score += 1.0 / (RRF_K + dense_rank[i] + 1)
            fused.append((score, i))
        fused.sort(reverse=True)

        hits: list[SearchHit] = []
        for score, idx in fused[:limit]:
            if score == 0:
                break
            hits.append(
                SearchHit(
                    segment=_row_to_segment(rows[idx]),
                    score=score,
                    bm25_rank=bm25_rank[idx],
                    dense_rank=dense_rank[idx],
                )
            )
        return hits

    def _fetch_all(self, *, meeting_id: str | None) -> list[dict[str, Any]]:
        """Return matching rows as plain dicts. Pyarrow-table → list-of-dict."""
        if meeting_id is None:
            tbl = self._table.to_arrow()
        else:
            tbl = (
                self._table.search()  # type: ignore[no-untyped-call]
                .where(f"meeting_id = '{meeting_id}'")
                .limit(100_000)
                .to_arrow()
            )
        return [
            {col: tbl.column(col)[i].as_py() for col in tbl.column_names}
            for i in range(tbl.num_rows)
        ]


# ---------------------------------------------------------------------------
# Stand-alone scoring helpers (also exported for unit tests)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Plain BM25 implementation — sufficient for single-user corpora."""
    if not documents:
        return []
    docs_tokens = [_tokenize(d) for d in documents]
    avgdl = sum(len(d) for d in docs_tokens) / len(docs_tokens)
    n_docs = len(documents)

    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    q_terms = _tokenize(query)
    scores: list[float] = []
    for tokens in docs_tokens:
        score = 0.0
        dl = len(tokens) or 1
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf_num = n_docs - df.get(term, 0) + 0.5
            idf_den = df.get(term, 0) + 0.5
            idf = math.log((idf_num / idf_den) + 1.0)
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf * (f * (k1 + 1)) / denom
        scores.append(score)
    return scores


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _rank(values: list[float], *, descending: bool) -> list[int]:
    """Return the rank position of each value (0-based, ties get the same rank)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=descending)
    out = [0] * len(values)
    last_value = None
    last_rank = -1
    for new_rank, (i, v) in enumerate(indexed):
        if v != last_value:
            last_value = v
            last_rank = new_rank
        out[i] = last_rank
    return out


# ---------------------------------------------------------------------------
# Reference embedder for tests + low-resource fallback
# ---------------------------------------------------------------------------


def hash_embedder(dim: int = 64) -> Embedder:
    """Token-hash bag-of-words embedder. Deterministic, no model deps.

    Sufficient for unit tests of the indexing + ranking machinery; not
    intended for production retrieval quality. Production uses
    `nomic-embed-text v2` via actants → Ollama.
    """
    import hashlib

    def embed(text: str) -> list[float]:
        vec = [0.0] * dim
        for tok in _tokenize(text):
            h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big")
            vec[h % dim] += 1.0
        # L2-normalize so cosine = dot.
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    return embed

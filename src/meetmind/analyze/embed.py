"""Embedder via `actants.Embeddings`, defaulting to Ollama nomic-embed-text.

`make_embedder()` returns a `Callable[[str], list[float]]` compatible with
`meetmind.memory.vector.HybridIndex`; use `probe_embedder_dim()` to size the
LanceDB table.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from meetmind.memory.vector import Embedder

log = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL_ENV = "MEETMIND_EMBED_MODEL"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def _resolve_model(model: str | None) -> str:
    if model is not None:
        return model
    return os.environ.get(DEFAULT_EMBED_MODEL_ENV, DEFAULT_EMBED_MODEL)


def get_default_embeddings(model: str | None = None) -> Any:
    """Return an `actants.Embeddings` configured for local Ollama."""
    import actants  # noqa: PLC0415

    return actants.Embeddings(model=_resolve_model(model))


def make_embedder(model: str | None = None) -> Embedder:
    """Return a sync `Embedder` callable wrapping actants Embeddings.

    Embeds one string per call; see `embed_many` for lists.
    """
    embeddings = get_default_embeddings(model)

    def _call(text: str) -> list[float]:
        result = _run(embeddings.embed_one(text))
        # actants returns an `EmbeddingResult` with .vector; older shapes
        # duck-type as a plain sequence of floats.
        vec = getattr(result, "vector", None) or getattr(result, "embedding", None)
        if vec is None and isinstance(result, list | tuple):
            vec = list(result)
        if vec is None:
            raise RuntimeError(f"actants Embeddings returned unexpected shape: {type(result)}")
        return [float(x) for x in vec]

    return _call


def embed_many(model: str | None, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. One round-trip per text; actants has no batch API."""
    embedder = make_embedder(model)
    return [embedder(t) for t in texts]


def probe_embedder_dim(embedder: Embedder, *, probe: str = "dimension probe") -> int:
    """Call the embedder once with a stable string to discover vector dim."""
    return len(embedder(probe))


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return _await_in_thread(coro)


def _await_in_thread(coro: Any) -> Any:
    """Run `coro` on a fresh loop in a worker thread (safe inside a running loop)."""
    import concurrent.futures

    def _runner() -> Any:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_runner).result()

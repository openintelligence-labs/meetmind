"""Production embedder via `actants.Embeddings` → Ollama nomic-embed-text v2.

Default model: `nomic-embed-text v2` (137M MoE, 768-dim, Apache-2.0).
The Ollama image is `nomic-embed-text:latest`.

`make_embedder(model_name)` returns an `Embedder`-shape callable
(`Callable[[str], list[float]]`) compatible with
`meetmind.memory.vector.HybridIndex`.

The first `embed()` call probes the model's vector dimension and caches
it on the embedder instance. Callers can read `.dim` to size the
LanceDB table accordingly:

    embedder = make_embedder("nomic-embed-text")
    dim = probe_embedder_dim(embedder)
    index = HybridIndex.open(path, vector_dim=dim, embedder=embedder)
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

    The returned callable embeds one string at a time. Batch embedding
    (much cheaper for index builds) lives in `embed_many`.
    """
    embeddings = get_default_embeddings(model)

    def _call(text: str) -> list[float]:
        result = _run(embeddings.embed_one(text))
        # actants returns an `EmbeddingResult` with .vector; fall back to
        # treating the result itself as a list of floats if it duck-types.
        vec = getattr(result, "vector", None) or getattr(result, "embedding", None)
        if vec is None and isinstance(result, list | tuple):
            vec = list(result)
        if vec is None:
            raise RuntimeError(f"actants Embeddings returned unexpected shape: {type(result)}")
        return [float(x) for x in vec]

    return _call


def embed_many(model: str | None, texts: list[str]) -> list[list[float]]:
    """Batch helper. One Ollama round-trip per text — actants currently
    doesn't expose a true batch API, but reusing the connection keeps
    overhead low. Future actants versions may add `embed_batch`."""
    embedder = make_embedder(model)
    return [embedder(t) for t in texts]


def probe_embedder_dim(embedder: Embedder, *, probe: str = "dimension probe") -> int:
    """Call the embedder once with a stable string to discover vector dim."""
    return len(embedder(probe))


# ---------------------------------------------------------------------------
# Async helpers (mirror analyze/llm.py)
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return _await_in_thread(coro)


def _await_in_thread(coro: Any) -> Any:
    """Fresh loop per call in a worker thread — see analyze/llm.py for
    the rationale."""
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

"""Production LLM transport via `actants`.

MeetMind is **local-by-default** but **BYO-LLM**: ``actants`` already
speaks Ollama, OpenAI, Anthropic, Gemini, Groq, and Mistral. We forward
``MEETMIND_LLM_*`` env vars to the underlying ``ACTANTS_*`` settings so
users get a single, branded knob set:

  MEETMIND_LLM_PROVIDER   → ACTANTS_PROVIDER   (default: ollama)
  MEETMIND_LLM_MODEL      → ACTANTS_MODEL      (also: actants default)
  MEETMIND_LLM_BASE_URL   → ACTANTS_BASE_URL   (default: 127.0.0.1:11434)
  MEETMIND_LLM_API_KEY    → ACTANTS_API_KEY    (only for hosted providers)

Audio + transcripts never leave the device regardless of provider —
only the *prompted* text does, and only if the user picks a hosted
provider. Local Ollama remains the privacy-first default.

`analyze.actions` and `analyze.decisions` are written against an
LLMCallable contract: a callable taking a prompt and returning a JSON
dict. In production we don't want a dict — we want a typed Pydantic
model. `actants.LLM.extract(prompt, schema)` does exactly that, including
one-shot self-repair when the first response doesn't parse.

Module imports `actants` lazily so the optional dep stays optional in
test runs that don't touch the LLM path.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


_DEFAULT_MODEL_ENV = "MEETMIND_LLM_MODEL"

# MEETMIND_LLM_* → ACTANTS_* mapping. We forward on import so any
# `actants.LLM()` constructed downstream picks them up via pydantic-
# settings without us having to thread the values through.
_ENV_FORWARD = {
    "MEETMIND_LLM_PROVIDER": "ACTANTS_PROVIDER",
    "MEETMIND_LLM_MODEL": "ACTANTS_MODEL",
    "MEETMIND_LLM_BASE_URL": "ACTANTS_BASE_URL",
    "MEETMIND_LLM_API_KEY": "ACTANTS_API_KEY",
}


def _forward_env() -> None:
    """Mirror MEETMIND_LLM_* env vars into the ACTANTS_* namespace.

    User-set ACTANTS_* values win — we never overwrite. This lets power
    users still use the actants-native names if they prefer.
    """
    for src, dst in _ENV_FORWARD.items():
        val = os.environ.get(src)
        if val and not os.environ.get(dst):
            os.environ[dst] = val


# Forward on module import so the env config is in place by the time
# anyone constructs an LLM. Idempotent and cheap.
_forward_env()


def _resolve_model(model: str | None) -> str | None:
    if model is not None:
        return model
    return os.environ.get(_DEFAULT_MODEL_ENV) or None


def llm_config_summary() -> dict[str, str | None]:
    """What provider/model/base_url will the next LLM use? For `meetmind status`."""
    return {
        "provider": os.environ.get("MEETMIND_LLM_PROVIDER")
        or os.environ.get("ACTANTS_PROVIDER")
        or "ollama",
        "model": os.environ.get("MEETMIND_LLM_MODEL") or os.environ.get("ACTANTS_MODEL") or None,
        "base_url": os.environ.get("MEETMIND_LLM_BASE_URL")
        or os.environ.get("ACTANTS_BASE_URL")
        or "http://localhost:11434",
        "has_api_key": bool(
            os.environ.get("MEETMIND_LLM_API_KEY") or os.environ.get("ACTANTS_API_KEY")
        ),
    }


def get_default_llm() -> Any:
    """Construct an `actants.LLM` with our defaults.

    Defaults to local Ollama. Caller can override via env or by passing
    a pre-built LLM in. Lazy-imports `actants` to keep it optional.

    Resolution order:
      1. ``MEETMIND_LLM_MODEL`` env var (explicit user choice).
      2. ``best_available_local_model()`` — picks the best model that's
         actually installed (gemma4 > gemma3 > qwen3 > qwen2.5 > …).
      3. actants default — only hit when no local Ollama is reachable.
    """
    import actants  # noqa: PLC0415

    _forward_env()
    model = _resolve_model(None) or best_available_local_model()
    if model:
        log.info("meetmind LLM: model=%s", model)
        return actants.LLM(model=model)
    return actants.LLM()


def extract[Schema: BaseModel](
    llm: Any, prompt: str, schema: type[Schema], *, model: str | None = None
) -> Schema:
    """Synchronous wrapper around `llm.extract` for the analyze package."""
    coro = llm.extract(prompt, schema, model=_resolve_model(model))
    return _run(coro)


def make_callable(
    llm: Any | None = None,
    *,
    schema: type[BaseModel] | None = None,
    model: str | None = None,
) -> Callable[[str], dict[str, Any]]:
    """Adapt an `actants.LLM` to the LLMCallable contract used by analyze.

    `schema` lets the wrapper use `extract()` (typed) and emit a dict that
    matches the analyze module's expectations. If `schema` is None, falls
    back to `complete()` and the caller must JSON-decode the body.
    """
    llm = llm or get_default_llm()
    resolved_model = _resolve_model(model)

    def _call(prompt: str) -> dict[str, Any]:
        if schema is not None:
            obj = _run(llm.extract(prompt, schema, model=resolved_model))
            return obj.model_dump(mode="python")
        result = _run(llm.complete(prompt, model=resolved_model))
        # Caller must parse `result.content` themselves.
        return {"content": result.content}

    return _call


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run an async call from sync code, regardless of event-loop state."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return _await_in_thread(coro)


def _await_in_thread(coro: Any) -> Any:
    """Run coro on a fresh event loop in a worker thread.

    `actants.LLM.extract` is async; the analyze package is sync. When
    we're inside a running loop already, we cannot nest `asyncio.run`,
    so we hop to a worker thread with a brand-new loop. Each call gets
    its own loop so no state leaks between calls.
    """
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


def _await_in_loop(coro: Any, loop: asyncio.AbstractEventLoop | None = None) -> Any:
    """Backward-compat alias kept for the embed module."""
    return _await_in_thread(coro)


# ---------------------------------------------------------------------------
# Health check + model discovery
# ---------------------------------------------------------------------------


def list_local_models() -> list[dict[str, Any]]:
    """Return the model catalog reported by the configured Ollama daemon.

    Honors ``MEETMIND_LLM_BASE_URL`` / ``ACTANTS_BASE_URL`` so users running
    Ollama on a different host or port (or LM Studio's `/v1/models`) get
    their actual catalog. Returns ``[]`` for any non-Ollama provider —
    listing remote models requires hosted-API auth, out of scope for the
    `status` command. Useful for `meetmind status`.
    """
    import httpx  # already a dep

    cfg = llm_config_summary()
    if cfg["provider"] != "ollama":
        return []
    base = (cfg["base_url"] or "http://localhost:11434").rstrip("/")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=2.0)
        r.raise_for_status()
        return r.json().get("models", [])
    except Exception as e:  # noqa: BLE001 — daemon may simply be down
        log.debug("ollama not reachable at %s: %s", base, e)
        return []


def best_available_local_model(prefer: list[str] | None = None) -> str | None:
    """Pick a sensible local model for analyze paths.

    Strategy:
      1. If `MEETMIND_LLM_MODEL` is set, use it.
      2. Else if any of `prefer` are installed locally, use the first hit.
      3. Else first installed local (non-cloud) model.
      4. Else None — caller falls back to MockLLM or surfaces a hint.
    """
    env_model = _resolve_model(None)
    if env_model:
        return env_model

    catalog = list_local_models()
    installed = {m.get("name", ""): m for m in catalog}
    # Cloud-routed entries have name suffix `:cloud`. We prefer local-only
    # for the analyze path because LLM calls there can include private
    # transcript content.
    local_only = {n: m for n, m in installed.items() if not n.endswith(":cloud")}
    if not local_only:
        return None

    preferences = list(
        prefer
        or [
            # Preference ladder — first match wins. Gemma 4 is the
            # default chat model for analyze (best instruction
            # following + reliable JSON among small local models).
            "gemma4",
            "gemma3",
            "qwen3:30b-a3b",
            "qwen3:30b",
            "qwen3:8b",
            "qwen3:4b",
            "qwen2.5:7b",
            "phi-4",
            "phi-4-mini",
            "llama3.1",
            "llama2",
        ]
    )
    for name in preferences:
        for installed_name in local_only:
            if installed_name.startswith(name) or name in installed_name:
                return installed_name
    # Fall back to whatever's there.
    return next(iter(local_only))

"""LLM transport via `actants`, local-by-default and BYO-LLM.

``MEETMIND_LLM_*`` env vars are forwarded to the underlying ``ACTANTS_*``
settings so there is a single branded knob set. Imports `actants` lazily to
keep the optional dependency optional.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


_DEFAULT_MODEL_ENV = "MEETMIND_LLM_MODEL"

# Forwarded on import so any downstream `actants.LLM()` picks these up via
# pydantic-settings, without threading the values through every call site.
_ENV_FORWARD = {
    "MEETMIND_LLM_PROVIDER": "ACTANTS_PROVIDER",
    "MEETMIND_LLM_MODEL": "ACTANTS_MODEL",
    "MEETMIND_LLM_BASE_URL": "ACTANTS_BASE_URL",
    "MEETMIND_LLM_API_KEY": "ACTANTS_API_KEY",
}


def _forward_env() -> None:
    """Mirror MEETMIND_LLM_* env vars into the ACTANTS_* namespace.

    Existing ACTANTS_* values are never overwritten, so the actants-native
    names keep working.
    """
    for src, dst in _ENV_FORWARD.items():
        val = os.environ.get(src)
        if val and not os.environ.get(dst):
            os.environ[dst] = val


_forward_env()


def _resolve_model(model: str | None) -> str | None:
    if model is not None:
        return model
    return os.environ.get(_DEFAULT_MODEL_ENV) or None


def llm_config_summary() -> dict[str, str | None]:
    """Return the provider/model/base_url the next LLM will use."""
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
    """Construct an `actants.LLM`, defaulting to local Ollama.

    Model resolution: ``MEETMIND_LLM_MODEL``, then
    ``best_available_local_model()``, then the actants default (reached only
    when no local Ollama is available).
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

    With a `schema`, uses typed `extract()` and dumps the model to a dict.
    Without one, falls back to `complete()` and the caller decodes the body.
    """
    llm = llm or get_default_llm()
    resolved_model = _resolve_model(model)

    def _call(prompt: str) -> dict[str, Any]:
        if schema is not None:
            obj = _run(llm.extract(prompt, schema, model=resolved_model))
            return obj.model_dump(mode="python")
        result = _run(llm.complete(prompt, model=resolved_model))
        return {"content": result.content}

    return _call


_LLM_LOOP: asyncio.AbstractEventLoop | None = None
_LLM_LOOP_LOCK = threading.Lock()


def _persistent_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, starting it on first use.

    Every actants call from the sync analyze path runs on this one loop: a
    shared `actants.LLM` holds an httpx pool whose connections are bound to
    the loop that created them, so a per-call `asyncio.run()` loop would leave
    later calls reusing connections from a closed loop.
    """
    global _LLM_LOOP
    with _LLM_LOOP_LOCK:
        if _LLM_LOOP is None or _LLM_LOOP.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="meetmind-llm-loop", daemon=True
            )
            thread.start()
            _LLM_LOOP = loop
        return _LLM_LOOP


def _run(coro: Any) -> Any:
    """Run an async call from sync code, including from inside a running loop."""
    return asyncio.run_coroutine_threadsafe(coro, _persistent_loop()).result()


def _await_in_thread(coro: Any) -> Any:
    """Run coro on the shared background loop from any thread."""
    return asyncio.run_coroutine_threadsafe(coro, _persistent_loop()).result()


def _await_in_loop(coro: Any, loop: asyncio.AbstractEventLoop | None = None) -> Any:
    """Backward-compat alias kept for the embed module."""
    return _await_in_thread(coro)


def list_local_models() -> list[dict[str, Any]]:
    """Return the model catalog reported by the configured Ollama daemon.

    Honors ``MEETMIND_LLM_BASE_URL`` / ``ACTANTS_BASE_URL``. Returns ``[]``
    for non-Ollama providers, whose catalogs need hosted-API auth.
    """
    import httpx

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
    """Pick a local model for analyze paths.

    Order: `MEETMIND_LLM_MODEL`, then the first installed entry from `prefer`,
    then any installed local model. None when nothing local is installed.
    """
    env_model = _resolve_model(None)
    if env_model:
        return env_model

    catalog = list_local_models()
    installed = {m.get("name", ""): m for m in catalog}
    # Cloud-routed entries carry a `:cloud` suffix. Analyze prompts can contain
    # private transcript text, so only local-only models are eligible here.
    local_only = {n: m for n, m in installed.items() if not n.endswith(":cloud")}
    if not local_only:
        return None

    preferences = list(
        prefer
        or [
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

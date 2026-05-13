"""Tests for the actants LLM adapter."""

from __future__ import annotations

import importlib
import os

import pytest

from meetmind.analyze import llm as llm_mod
from meetmind.analyze.actions import ExtractionPayload, extract_action_items
from meetmind.analyze.llm import (
    best_available_local_model,
    list_local_models,
    llm_config_summary,
    make_callable,
)


def test_list_local_models_returns_list_or_empty():
    """Smoke: works whether or not Ollama is running."""
    out = list_local_models()
    assert isinstance(out, list)


def test_best_available_respects_env(monkeypatch):
    monkeypatch.setenv("MEETMIND_LLM_MODEL", "qwen3:4b")
    assert best_available_local_model() == "qwen3:4b"


def test_best_available_skips_cloud_models(monkeypatch):
    monkeypatch.delenv("MEETMIND_LLM_MODEL", raising=False)
    if not list_local_models():
        pytest.skip("ollama not running")
    pick = best_available_local_model()
    if pick is None:
        pytest.skip("no local-only model installed")
    assert not pick.endswith(":cloud")


def test_env_forward_meetmind_to_actants(monkeypatch):
    """MEETMIND_LLM_* should mirror into ACTANTS_* without overwriting user-set ACTANTS_*."""
    for k in (
        "MEETMIND_LLM_PROVIDER",
        "MEETMIND_LLM_MODEL",
        "MEETMIND_LLM_BASE_URL",
        "MEETMIND_LLM_API_KEY",
        "ACTANTS_PROVIDER",
        "ACTANTS_MODEL",
        "ACTANTS_BASE_URL",
        "ACTANTS_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("MEETMIND_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEETMIND_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("MEETMIND_LLM_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("MEETMIND_LLM_API_KEY", "sk-test")
    # User-set ACTANTS_* takes precedence — must not be overwritten.
    monkeypatch.setenv("ACTANTS_PROVIDER", "anthropic")

    importlib.reload(llm_mod)

    assert os.environ["ACTANTS_PROVIDER"] == "anthropic"  # not overwritten
    assert os.environ["ACTANTS_MODEL"] == "gpt-4o-mini"
    assert os.environ["ACTANTS_BASE_URL"] == "https://api.openai.com"
    assert os.environ["ACTANTS_API_KEY"] == "sk-test"


def test_llm_config_summary_reflects_overrides(monkeypatch):
    monkeypatch.setenv("MEETMIND_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEETMIND_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("MEETMIND_LLM_API_KEY", "sk-test")
    cfg = llm_config_summary()
    assert cfg["provider"] == "openai"
    assert cfg["model"] == "gpt-4o-mini"
    assert cfg["has_api_key"] is True


def test_llm_config_summary_defaults(monkeypatch):
    for k in (
        "MEETMIND_LLM_PROVIDER",
        "MEETMIND_LLM_MODEL",
        "MEETMIND_LLM_BASE_URL",
        "MEETMIND_LLM_API_KEY",
        "ACTANTS_PROVIDER",
        "ACTANTS_MODEL",
        "ACTANTS_BASE_URL",
        "ACTANTS_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = llm_config_summary()
    assert cfg["provider"] == "ollama"
    assert cfg["base_url"] == "http://localhost:11434"
    assert cfg["has_api_key"] is False


def test_list_local_models_returns_empty_for_non_ollama_provider(monkeypatch):
    monkeypatch.setenv("MEETMIND_LLM_PROVIDER", "openai")
    assert list_local_models() == []


@pytest.mark.skipif(
    os.environ.get("OLLAMA_LIVE") != "1",
    reason="set OLLAMA_LIVE=1 to run live Ollama tests",
)
def test_live_extract_parses_payload():
    """Live: actants.LLM.extract → ExtractionPayload round-trip with a real model."""
    transcript = (
        "Sam: I'll send the deck on Friday. "
        "Priya: I'll write up the migration plan by next Tuesday."
    )
    callable_llm = make_callable(schema=ExtractionPayload)
    result = extract_action_items(transcript, callable_llm)
    assert all(a.evidence_quote in transcript for a in result.accepted)

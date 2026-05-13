"""Tests for the GitHub Issues integration (S11.8).

We avoid touching the real ``gh`` binary by writing a tiny stub script
to ``tmp_path`` and pointing the integration at it via ``gh_binary=``.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from meetmind.integrations.github import (
    GhCliMissingError,
    _gh_path,
    export_action_items,
)
from meetmind.memory.store import Store
from meetmind.models import ActionItem, Meeting


def _write_stub_gh(tmp_path: Path, exit_code: int = 0, stdout: str = "") -> Path:
    stub = tmp_path / "gh-stub"
    stub.write_text(f'#!/usr/bin/env bash\necho "{stdout}"\nexit {exit_code}\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return stub


@pytest.fixture
def populated_store(tmp_path: Path) -> Store:
    store = Store.open(tmp_path / "store.db")
    m = Meeting(id="01GH", title="kickoff")
    store.upsert_meeting(m)
    store.upsert_action_item(
        m.id,
        ActionItem(
            description="Send the migration deck",
            owner="remote-A",
            due="2026-05-08",
            evidence_quote="I'll send the deck Friday.",
            status="open",
        ),
    )
    store.upsert_action_item(
        m.id,
        ActionItem(
            description="Schedule the follow-up review",
            owner="remote-B",
            status="open",
        ),
    )
    # A done item — should NOT be exported.
    store.upsert_action_item(
        m.id,
        ActionItem(description="already shipped", owner="remote-A", status="done"),
    )
    return store


def test_export_creates_one_issue_per_open_action(populated_store, tmp_path):
    stub = _write_stub_gh(tmp_path, stdout="https://github.com/acme/demo/issues/42")
    refs = export_action_items(populated_store, "01GH", repo="acme/demo", gh_binary=str(stub))
    assert len(refs) == 2  # only the two `open` ones
    assert refs[0].url.endswith("/42")
    assert refs[0].number == 42
    assert refs[0].repo == "acme/demo"


def test_dry_run_does_not_invoke_binary(populated_store, tmp_path):
    # Stub that would FAIL if invoked — ensures dry-run doesn't shell out.
    bad_stub = _write_stub_gh(tmp_path, exit_code=1, stdout="")
    refs = export_action_items(
        populated_store, "01GH", repo="acme/demo", dry_run=True, gh_binary=str(bad_stub)
    )
    assert len(refs) == 2
    assert all(r.number is None for r in refs)


def test_no_open_actions_returns_empty(tmp_path: Path):
    store = Store.open(tmp_path / "store.db")
    store.upsert_meeting(Meeting(id="01EMPTY", title="silence"))
    refs = export_action_items(store, "01EMPTY", repo="acme/demo", dry_run=True)
    assert refs == []
    store.close()


def test_missing_gh_raises(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(GhCliMissingError):
        _gh_path()

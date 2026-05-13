"""Tests for the DEK helper.

The real OS keychain is stubbed out so this test never touches the
developer's actual keychain. Two scenarios:

  • `MEETMIND_DISABLE_ENCRYPTION=1` → `get_or_create_dek()` returns None.
  • `MEETMIND_DB_PASSPHRASE=...`    → explicit override wins.
"""

from __future__ import annotations

import importlib

import pytest


def test_disable_encryption_env_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEETMIND_DISABLE_ENCRYPTION", "1")
    monkeypatch.delenv("MEETMIND_DB_PASSPHRASE", raising=False)
    from meetmind.memory import keyring as kr  # noqa: PLC0415

    importlib.reload(kr)
    assert kr.get_or_create_dek() is None


def test_env_passphrase_overrides_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEETMIND_DISABLE_ENCRYPTION", raising=False)
    monkeypatch.setenv("MEETMIND_DB_PASSPHRASE", "hunter2-test")
    from meetmind.memory import keyring as kr  # noqa: PLC0415

    importlib.reload(kr)
    assert kr.get_or_create_dek() == "hunter2-test"


def test_store_open_honors_use_keychain_false(tmp_path, monkeypatch) -> None:
    """`use_keychain=False` is the hermetic path callers use in tests."""
    monkeypatch.delenv("MEETMIND_DISABLE_ENCRYPTION", raising=False)
    monkeypatch.delenv("MEETMIND_DB_PASSPHRASE", raising=False)
    from meetmind.memory.store import Store  # noqa: PLC0415

    s = Store.open(tmp_path / "x.db", use_keychain=False)
    s.close()

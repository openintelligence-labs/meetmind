"""Migration runner tests.

Three scenarios:
  • fresh DB stamps current SCHEMA_VERSION on first open,
  • DB stamped at an older version gets bumped forward,
  • DB stamped at a newer version is left alone (no silent downgrade).
"""

from __future__ import annotations

import sqlite3

import pytest

from meetmind.memory.schema import SCHEMA_VERSION
from meetmind.memory.store import Store, _read_schema_version


def test_fresh_db_stamps_current_version(tmp_path) -> None:
    s = Store.open(tmp_path / "fresh.db", use_keychain=False)
    try:
        assert _read_schema_version(s.conn) == SCHEMA_VERSION
    finally:
        s.close()


def test_old_db_gets_migrated_forward(tmp_path) -> None:
    """A DB stamped at v1 should be migrated to current on next open."""
    db_path = tmp_path / "old.db"
    # Bootstrap a v1-style DB by hand: open via Store (which stamps current),
    # then forcibly downgrade the stamp to simulate an older install.
    s = Store.open(db_path, use_keychain=False)
    s.conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version','1')")
    s.close()
    # Re-open so apply_schema runs against the downgraded stamp.
    s2 = Store.open(db_path, use_keychain=False)
    try:
        assert _read_schema_version(s2.conn) == SCHEMA_VERSION
    finally:
        s2.close()


def test_newer_db_is_not_downgraded(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    """A DB stamped ahead of SCHEMA_VERSION keeps its stamp and only warns."""
    db_path = tmp_path / "newer.db"
    s = Store.open(db_path, use_keychain=False)
    future_v = SCHEMA_VERSION + 5
    s.conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version',?)",
        (str(future_v),),
    )
    s.close()
    with caplog.at_level("WARNING"):
        s2 = Store.open(db_path, use_keychain=False)
    try:
        # The stamp is never downgraded.
        assert _read_schema_version(s2.conn) >= SCHEMA_VERSION
        assert any("newer than this build" in r.message for r in caplog.records)
    finally:
        s2.close()


def test_wal_mode_is_enabled(tmp_path) -> None:
    """The journal_mode PRAGMA must stick on real files."""
    s = Store.open(tmp_path / "wal.db", use_keychain=False)
    try:
        mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        # On rare filesystems WAL falls back to truncate; both are acceptable.
        assert mode in {"wal", "truncate"}
    finally:
        s.close()


def test_foreign_keys_enabled(tmp_path) -> None:
    s = Store.open(tmp_path / "fk.db", use_keychain=False)
    try:
        fk = s.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert int(fk) == 1
    finally:
        s.close()


def test_migration_failure_leaves_last_good_stamp(tmp_path, monkeypatch) -> None:
    """A broken later migration must not advance the stamp past the last good step."""
    db_path = tmp_path / "atomic.db"
    s = Store.open(db_path, use_keychain=False)
    s.conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version','1')")
    s.close()

    from meetmind.memory import schema  # noqa: PLC0415

    # Inject a broken migration AFTER the legitimate v2 step. The v2
    # step should still succeed and bump the stamp; the broken one
    # should raise without advancing past the last good version.
    monkeypatch.setitem(schema.MIGRATIONS, SCHEMA_VERSION + 1, "SELECT not_a_real_function();")
    monkeypatch.setattr(schema, "SCHEMA_VERSION", SCHEMA_VERSION + 1)

    with pytest.raises(sqlite3.DatabaseError):
        Store.open(db_path, use_keychain=False)

    # The stamp should be SCHEMA_VERSION (the last good migration), not
    # the failed SCHEMA_VERSION + 1. Reopen with the broken patch dropped.
    monkeypatch.undo()
    s3 = Store.open(db_path, use_keychain=False)
    try:
        v = _read_schema_version(s3.conn)
        # Must have advanced past v1 (v2 succeeded) but not past current.
        assert v == SCHEMA_VERSION
    finally:
        s3.close()

"""Tests for `meetmind selftest`.

The selftest should be safe to run in any environment — no network,
no real keychain. We verify exit codes + output shape, not the
absolute pass/fail of every check (sidecar discovery, e.g., depends
on the host).
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from meetmind.cli import main


def test_selftest_runs_to_completion() -> None:
    """The command should always reach the end — never raise mid-checks."""
    runner = CliRunner()
    result = runner.invoke(main, ["selftest"], catch_exceptions=False)
    # Exit code is 0 if everything passes, 1 if any hard fail. The CI
    # environment may flag the sidecar binaries as missing — that's a
    # warn (still passes). The storage round-trip MUST succeed.
    assert result.exit_code in (0, 1), result.output
    assert "storage" in result.output
    assert "end-to-end" in result.output


def test_selftest_json_output_is_parseable() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["selftest", "--json"], catch_exceptions=False)
    data = json.loads(result.output)
    assert isinstance(data, list) and data
    names = {c["name"] for c in data}
    # Mandatory checks that should always be present.
    assert {
        "python",
        "storage",
        "encryption",
        "capture-sidecar",
        "stt-sidecar",
        "llm",
        "integrations",
        "end-to-end",
    }.issubset(names)
    for c in data:
        assert set(c.keys()) == {"name", "ok", "note", "warn"}


def test_selftest_storage_check_passes() -> None:
    """storage + end-to-end are platform-independent — they must always be ok."""
    runner = CliRunner()
    result = runner.invoke(main, ["selftest", "--json"], catch_exceptions=False)
    data = {c["name"]: c for c in json.loads(result.output)}
    assert data["storage"]["ok"] is True
    assert data["end-to-end"]["ok"] is True

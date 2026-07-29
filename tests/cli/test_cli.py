"""CLI smoke tests."""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

import meetmind
from meetmind.cli import main


def test_version_matches_package():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert meetmind.__version__ in result.output


def test_status_outputs_json_with_expected_keys():
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in (
        "version",
        "python",
        "platform",
        "machine",
        "capture_sidecar",
        "stt_sidecar",
    ):
        assert key in data
    assert data["version"] == meetmind.__version__


def test_help_lists_record_and_status():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "record" in result.output
    assert "status" in result.output


def test_record_help_documents_options():
    runner = CliRunner()
    result = runner.invoke(main, ["record", "--help"])
    assert result.exit_code == 0
    for needle in ("--duration", "--source", "--stt", "--mock", "--stream"):
        assert needle in result.output


@pytest.mark.timeout(30)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="mock sidecars are POSIX shell launchers; Windows support tracked in issue #3",
)
def test_record_with_mock_sidecars_emits_transcript():
    """End-to-end smoke: capture mock → pipeline → STT mock → stdout.

    Run as a subprocess (rather than CliRunner) because the CLI uses
    asyncio.run which conflicts with pytest-asyncio's event loop.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "meetmind",
            "record",
            "--mock",
            "--duration",
            "1.0",
            "--stream",
            "mic",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    combined = proc.stdout + proc.stderr
    assert any(
        marker in combined
        for marker in ("partial", "final", "the quick", "fox", "dog", "snowflake")
    ), combined

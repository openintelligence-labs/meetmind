"""Tests for the SidecarProcess watchdog hooks.

Covers the two properties the watchdog relies on rather than the full
`_run_record` loop: ``returncode`` becomes non-None once the sidecar exits,
and ``stderr_tail()`` carries the last lines it emitted.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from meetmind.ipc.protocol import SidecarProcess

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="test fixtures are POSIX shell launchers; Windows support tracked in issue #3",
)


def _make_crash_sidecar(tmp_path: Path) -> Path:
    """Write a tiny script that emits HELLO (so .start() succeeds) then exits
    with a non-zero code + some stderr."""
    # HELLO frame: type 0x01, length-prefixed JSON payload.
    py = sys.executable
    src = tmp_path / "crasher.py"
    src.write_text(
        f"""#!/usr/bin/env python3
import json, os, struct, sys, time
from pathlib import Path
_REPO_SRC = Path({str(Path(__file__).resolve().parents[2] / "src")!r})
sys.path.insert(0, str(_REPO_SRC))
from meetmind.ipc import FrameType, encode_frame

hello = {{
    "sidecar": "crasher", "version": "0.0", "protocol_version": "1.0.0",
    "platform": "test", "capabilities": [], "permissions": {{}}
}}
sys.stdout.buffer.write(encode_frame(FrameType.HELLO, json.dumps(hello).encode()))
sys.stdout.buffer.flush()
# Now panic on stderr and exit non-zero.
print("FATAL: simulated TCC denial", file=sys.stderr)
print("aborting capture loop", file=sys.stderr)
time.sleep(0.05)
sys.exit(2)
"""
    )
    launcher = tmp_path / "crasher.sh"
    launcher.write_text(f'#!/usr/bin/env bash\nexec "{py}" "{src}" "$@"\n')
    launcher.chmod(0o755)
    return launcher


async def test_returncode_becomes_non_none_after_crash(tmp_path: Path) -> None:
    sidecar = SidecarProcess(_make_crash_sidecar(tmp_path))
    await sidecar.start()  # consumes HELLO

    # Poll for the exit. The crasher sleeps 50 ms before exiting.
    for _ in range(50):
        await asyncio.sleep(0.05)
        if sidecar.returncode is not None:
            break
    assert sidecar.returncode == 2

    # Stderr ring should contain the FATAL line.
    tail = sidecar.stderr_tail()
    # Stderr is drained asynchronously, so it may lag briefly.
    for _ in range(20):
        if tail:
            break
        await asyncio.sleep(0.05)
        tail = sidecar.stderr_tail()
    assert "FATAL" in tail or "aborting" in tail

    await sidecar.stop()


async def test_returncode_none_for_unstarted_sidecar(tmp_path: Path) -> None:
    s = SidecarProcess(_make_crash_sidecar(tmp_path))
    assert s.returncode is None
    assert s.stderr_tail() == ""


async def test_stderr_tail_caps_to_byte_limit(tmp_path: Path) -> None:
    """The tail must trim to the requested byte budget."""
    s = SidecarProcess(_make_crash_sidecar(tmp_path))
    # Inject a ring directly rather than spawning a process.
    s._stderr_ring = ["A" * 100, "B" * 100, "C" * 100]  # 300+ bytes total
    tail = s.stderr_tail(max_bytes=128)
    assert len(tail) <= 128
    # Most recent lines should be the C-block tail.
    assert tail.endswith("CCC" * 5) or tail.endswith("C")

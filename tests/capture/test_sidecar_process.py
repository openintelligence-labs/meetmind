"""End-to-end tests against the mock sidecar fixture.

These exercise the full IPC dance: spawn → HELLO → start → AUDIO frames →
stop → BYE → exit 0. No native code; mock_sidecar is pure Python.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from meetmind.ipc import IPCError, SidecarProcess, StreamId

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mock_sidecar.py"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="mock sidecar launcher is a POSIX shell script; Windows sidecar is post-1.0 (ROADMAP)",
)


def _binary() -> Path:
    """Return a small wrapper script that runs the mock with this Python."""
    wrapper = FIXTURE.parent / "_mock_sidecar_launcher.sh"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{FIXTURE}" "$@"\n')
    wrapper.chmod(0o755)
    return wrapper


@pytest.fixture
def mock_binary() -> Path:
    return _binary()


async def test_hello_handshake(mock_binary: Path):
    sidecar = SidecarProcess(mock_binary)
    hello = await sidecar.start()
    try:
        assert hello.sidecar == "mock-sidecar"
        assert hello.protocol_version.startswith("1.")
        assert "mic" in hello.capabilities
        assert "loopback" in hello.capabilities
        assert hello.permissions["microphone"] == "granted"
    finally:
        await sidecar.stop()


async def test_start_emits_audio_for_both_streams(mock_binary: Path):
    env = {**os.environ, "MOCK_CHUNKS": "5"}
    sidecar = SidecarProcess(mock_binary, env=env)
    async with sidecar:
        await sidecar.send("start", {"streams": ["mic", "loopback"]})
        mic = 0
        loop = 0
        async for chunk in sidecar.audio():
            if chunk.stream is StreamId.MIC:
                mic += 1
            elif chunk.stream is StreamId.LOOPBACK:
                loop += 1
            if mic >= 5 and loop >= 5:
                break
        assert mic == 5
        assert loop == 5


async def test_segment_flags_reach_python(mock_binary: Path):
    env = {**os.environ, "MOCK_CHUNKS": "3"}
    sidecar = SidecarProcess(mock_binary, env=env)
    async with sidecar:
        await sidecar.send("start", {"streams": ["mic", "loopback"]})
        first_seen = False
        last_seen = False
        seen = 0
        async for chunk in sidecar.audio():
            if chunk.stream is StreamId.MIC:
                if chunk.is_segment_start:
                    first_seen = True
                if chunk.is_segment_end:
                    last_seen = True
                seen += 1
                if seen >= 3:
                    break
        assert first_seen
        assert last_seen


async def test_unknown_op_returns_ack_error(mock_binary: Path):
    sidecar = SidecarProcess(mock_binary)
    await sidecar.start()
    try:
        with pytest.raises(IPCError) as excinfo:
            await sidecar.send("definitely-not-real")
        assert "unknown op" in str(excinfo.value)
    finally:
        await sidecar.stop()


async def test_clean_shutdown_returns_zero(mock_binary: Path):
    sidecar = SidecarProcess(mock_binary)
    await sidecar.start()
    await sidecar.send("start", {"streams": ["mic"]})
    await sidecar.stop()
    assert sidecar._proc is not None
    assert sidecar._proc.returncode == 0


async def test_missing_binary_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        await SidecarProcess(tmp_path / "no-such-thing").start()

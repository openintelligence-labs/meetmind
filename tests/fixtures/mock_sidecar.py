"""Mock capture sidecar — emits the IPC wire format on stdout.

Used in tests to exercise SidecarProcess without OS permissions or native binaries.
Runs as a subprocess, so it imports the project's IPC codec via sys.path setup.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

# Make `meetmind` importable when spawned as a subprocess.
_THIS = Path(__file__).resolve()
_REPO_SRC = _THIS.parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from meetmind.ipc import (  # noqa: E402  (after path setup)
    FrameType,
    StreamId,
    encode_audio_payload,
    encode_frame,
)


def _w(b: bytes) -> None:
    sys.stdout.buffer.write(b)
    sys.stdout.buffer.flush()


def _read_command() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def main() -> int:
    # ---- HELLO ----
    hello = {
        "sidecar": "mock-sidecar",
        "version": "0.0.1",
        "protocol_version": "1.0.0",
        "platform": "test",
        "capabilities": ["mic", "loopback"],
        "permissions": {"microphone": "granted", "screen_audio": "granted"},
    }
    _w(encode_frame(FrameType.HELLO, json.dumps(hello).encode("utf-8")))

    # Number of audio chunks to emit per stream comes from env (test knob).
    chunks = int(os.environ.get("MOCK_CHUNKS", "10"))
    # Tone vs silence is also a test knob. Default for `meetmind record --mock`
    # is a 600 Hz tone so the downstream STT mock has voiced audio to detect.
    use_tone = os.environ.get("MOCK_TONE", "1") == "1"
    if use_tone:
        import math

        samples = 480  # 10 ms @ 48 kHz mono
        amp = 0.4
        freq = 600.0
        rate = 48000
        chunk_pcm = b"".join(
            int(amp * 32767 * math.sin(2 * math.pi * freq * (i / rate))).to_bytes(
                2, byteorder="little", signed=True
            )
            for i in range(samples)
        )
    else:
        chunk_pcm = b"\x00\x00" * 480

    while True:
        cmd = _read_command()
        if cmd is None:
            break
        op = cmd.get("op")
        cid = cmd.get("id", "")
        if op == "start":
            _w(
                encode_frame(
                    FrameType.CONTROL_ACK,
                    json.dumps({"id": cid, "ok": True}).encode("utf-8"),
                )
            )
            for stream in (StreamId.MIC, StreamId.LOOPBACK):
                _w(
                    encode_frame(
                        FrameType.READY,
                        json.dumps(
                            {
                                "stream_id": "mic" if stream is StreamId.MIC else "loopback",
                                "format": {
                                    "rate": 48000,
                                    "encoding": "s16le",
                                    "channels": 1,
                                },
                            }
                        ).encode("utf-8"),
                    )
                )
            base_ts = int(time.monotonic() * 1_000_000)
            for i in range(chunks):
                ts = base_ts + i * 10_000  # 10 ms steps
                for stream in (StreamId.MIC, StreamId.LOOPBACK):
                    payload = encode_audio_payload(
                        stream=stream,
                        timestamp_us=ts,
                        pcm=chunk_pcm,
                        is_segment_start=(i == 0),
                        is_segment_end=(i == chunks - 1),
                    )
                    _w(encode_frame(FrameType.AUDIO, payload))
        elif op == "stop":
            _w(
                encode_frame(
                    FrameType.CONTROL_ACK,
                    json.dumps({"id": cid, "ok": True}).encode("utf-8"),
                )
            )
            _w(
                encode_frame(
                    FrameType.BYE,
                    json.dumps(
                        {
                            "frames_mic": chunks,
                            "frames_loopback": chunks,
                            "duration_ms": chunks * 10,
                        }
                    ).encode("utf-8"),
                )
            )
            break
        else:
            _w(
                encode_frame(
                    FrameType.CONTROL_ACK,
                    json.dumps({"id": cid, "ok": False, "error": f"unknown op: {op}"}).encode(
                        "utf-8"
                    ),
                )
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())


# Convenience for tests: when imported (not run as a script), expose binary path
# of a tiny launcher script that re-execs this module.
def fixture_binary() -> Path:
    """Return a launcher that runs this script via the current Python."""
    launcher = _THIS.parent / "_mock_sidecar_launcher.sh"
    if not launcher.exists():
        launcher.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{_THIS}" "$@"\n')
        launcher.chmod(0o755)
    # Recreate it idempotently to track python interpreter changes.
    launcher.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{_THIS}" "$@"\n')
    launcher.chmod(0o755)
    return launcher


# Pin encode_frame's 4-byte length prefix against the documented format.
assert struct.calcsize("<BI") == 5

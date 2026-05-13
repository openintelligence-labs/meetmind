"""Mock diarization sidecar — emits DIAR_SEGMENT frames for voiced AUDIO_F32.

Same wire format as the STT mock; alternates two cluster ids (`A`, `B`)
across silence boundaries, per-stream independently. Used by tests +
the v0.5 CLI smoke harness.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_REPO_SRC = _THIS.parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from meetmind.diarize.live import DiarFrameType  # noqa: E402
from meetmind.ipc import (  # noqa: E402
    FrameType,
    encode_frame,
    read_frame,
)


def _w(b: bytes) -> None:
    sys.stdout.buffer.write(b)
    sys.stdout.buffer.flush()


def _ack(cid: str, ok: bool, error: str | None = None) -> None:
    obj: dict = {"id": cid, "ok": ok}
    if error:
        obj["error"] = error
    _w(encode_frame(FrameType.CONTROL_ACK, json.dumps(obj).encode("utf-8")))


def _emit_segment(cluster: str, stream: int, start_ms: int, end_ms: int) -> None:
    obj = {
        "cluster_id": cluster,
        "stream": "mic" if stream == 0 else "loopback",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "confidence": 0.85,
    }
    _w(encode_frame(FrameType(DiarFrameType.DIAR_SEGMENT), json.dumps(obj).encode("utf-8")))


def _new_state() -> dict:
    return {
        "open_start_ms": None,
        "last_voiced_ms": None,
        "last_cluster": "B",
        "t_ms": 0,
    }


def _flush_open(stream: int, st: dict) -> None:
    if st["open_start_ms"] is None or st["last_voiced_ms"] is None:
        return
    next_cluster = "A" if st["last_cluster"] == "B" else "B"
    st["last_cluster"] = next_cluster
    _emit_segment(next_cluster, stream, st["open_start_ms"], st["last_voiced_ms"])
    st["open_start_ms"] = None
    st["last_voiced_ms"] = None


def _process_audio(payload: bytes, state: dict[int, dict]) -> None:
    if len(payload) < 8:
        return
    stream_byte = payload[0]
    f32 = np.frombuffer(payload[8:], dtype="<f4")
    if f32.size == 0:
        return
    duration_ms = int(round(1000 * f32.size / 16000))
    rms = float(np.sqrt(np.mean(f32**2)))
    st = state.setdefault(stream_byte, _new_state())
    cur_ms = st["t_ms"]
    st["t_ms"] = cur_ms + duration_ms
    if rms > 0.01:
        if st["open_start_ms"] is None:
            st["open_start_ms"] = cur_ms
        st["last_voiced_ms"] = cur_ms + duration_ms
    else:
        if st["open_start_ms"] is not None and st["last_voiced_ms"] is not None:
            silence_ms = (cur_ms + duration_ms) - st["last_voiced_ms"]
            if silence_ms >= 200:
                _flush_open(stream_byte, st)


async def main() -> int:
    _w(
        encode_frame(
            FrameType.HELLO,
            json.dumps(
                {
                    "sidecar": "mock-diar",
                    "version": "0.0.1",
                    "protocol_version": "1.0.0",
                    "model": "mock-sortformer",
                    "max_speakers": 4,
                }
            ).encode("utf-8"),
        )
    )

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    state: dict[int, dict] = {0: _new_state(), 1: _new_state()}

    while True:
        try:
            frame = await read_frame(reader)
        except asyncio.IncompleteReadError:
            break
        if frame is None:
            break
        if frame.type.value == DiarFrameType.CONTROL.value:
            cmd = json.loads(frame.payload.decode("utf-8"))
            cid = cmd.get("id", "")
            op = cmd.get("op", "")
            if op == "start":
                _ack(cid, True)
            elif op == "flush":
                _ack(cid, True)
                for s, st in state.items():
                    _flush_open(s, st)
            elif op == "stop":
                _ack(cid, True)
                for s, st in state.items():
                    _flush_open(s, st)
                _w(encode_frame(FrameType.BYE, json.dumps({"ok": True}).encode("utf-8")))
                with contextlib.suppress(OSError):
                    sys.stdout.flush()
                return 0
            else:
                _ack(cid, False, error=f"unknown op: {op}")
        elif frame.type.value == DiarFrameType.AUDIO_F32.value:
            _process_audio(frame.payload, state)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Mock STT sidecar — emits PARTIAL/FINAL frames for voiced AUDIO_F32 input.

Implements the STT IPC end-to-end without any real ASR model. Both
directions use the framed wire format defined in
``meetmind.ipc.protocol``.
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

from meetmind.ipc import (  # noqa: E402
    FrameType,
    encode_frame,
    read_frame,
)
from meetmind.stt.parakeet_v3 import STTFrameType  # noqa: E402

# Bank of phrases the mock STT cycles through. The demo (real sidecars)
# never uses these — they're only for `meetmind demo --mock` / CI / dev
# fallback. Realistic meeting language so manual smoke runs feel right.
PHRASES = [
    "let's start with the snowflake migration timeline",
    "sam will send the deck on friday",
    "priya proposed adopting lancedb for the vector store",
    "we agreed to ship the prototype by end of next week",
    "ravi flagged a concern about the consent flow",
    "let's move the security review to thursday",
    "i'll write up the action items after this call",
    "any blockers from the platform team",
    "we'll circle back on the pricing question next week",
    "let's wrap up and send the recap to everyone",
]


def _w(b: bytes) -> None:
    sys.stdout.buffer.write(b)
    sys.stdout.buffer.flush()


def _ack(cid: str, ok: bool, error: str | None = None) -> None:
    obj: dict = {"id": cid, "ok": ok}
    if error:
        obj["error"] = error
    _w(encode_frame(FrameType.CONTROL_ACK, json.dumps(obj).encode("utf-8")))


def _partial(stream: int, text: str, start_ms: int, end_ms: int) -> None:
    obj = {
        "stream": "mic" if stream == 0 else "loopback",
        "text": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "confidence": 0.6,
    }
    _w(encode_frame(FrameType(STTFrameType.PARTIAL), json.dumps(obj).encode("utf-8")))


def _final(stream: int, text: str, start_ms: int, end_ms: int) -> None:
    obj = {
        "stream": "mic" if stream == 0 else "loopback",
        "text": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "confidence": 0.9,
        "language": "en",
    }
    _w(encode_frame(FrameType(STTFrameType.FINAL), json.dumps(obj).encode("utf-8")))


def _new_state() -> dict:
    return {"open_start_ms": None, "t_ms": 0, "chars_visible": 0, "phrase_idx": 0}


def _current_phrase(st: dict) -> str:
    return PHRASES[st["phrase_idx"] % len(PHRASES)]


def _emit_final_if_open(stream: int, st: dict) -> None:
    if st["open_start_ms"] is None:
        return
    phrase = _current_phrase(st)
    text = phrase[: st["chars_visible"]].rstrip() or phrase
    _final(stream, text, st["open_start_ms"], st["t_ms"])
    st["open_start_ms"] = None
    st["chars_visible"] = 0
    st["phrase_idx"] += 1


def _process_audio(payload: bytes, state: dict[int, dict]) -> None:
    if len(payload) < 8:
        return
    stream_byte = payload[0]
    f32 = np.frombuffer(payload[8:], dtype="<f4")
    if f32.size == 0:
        return
    duration_ms = int(round(1000 * f32.size / 16000))
    rms = float(np.sqrt(np.mean(f32**2)))
    st = state.get(stream_byte)
    if st is None:
        return
    if rms > 0.01:
        if st["open_start_ms"] is None:
            st["open_start_ms"] = st["t_ms"]
        phrase = _current_phrase(st)
        target = min(len(phrase), st["chars_visible"] + max(1, duration_ms // 50))
        if target > st["chars_visible"]:
            st["chars_visible"] = target
            visible = phrase[: st["chars_visible"]].rstrip()
            if visible:
                _partial(stream_byte, visible, st["open_start_ms"], st["t_ms"] + duration_ms)
        # Commit the phrase once fully revealed. Without this the mock never
        # finalizes, since a constant-tone capture has no silence transition.
        if st["chars_visible"] >= len(phrase):
            _emit_final_if_open(stream_byte, st)
    else:
        _emit_final_if_open(stream_byte, st)
    st["t_ms"] += duration_ms


async def main() -> int:
    _w(
        encode_frame(
            FrameType.HELLO,
            json.dumps(
                {
                    "sidecar": "mock-stt",
                    "version": "0.0.1",
                    "protocol_version": "1.0.0",
                    "model": "mock-parakeet",
                    "sample_rate": 16000,
                    "languages": ["en"],
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
        if frame.type.value == STTFrameType.CONTROL.value:
            cmd = json.loads(frame.payload.decode("utf-8"))
            cid = cmd.get("id", "")
            op = cmd.get("op", "")
            args = cmd.get("args") or {}
            if op == "start":
                _ack(cid, True)
            elif op == "flush":
                stream = 0 if args.get("stream", "mic") == "mic" else 1
                _ack(cid, True)
                _emit_final_if_open(stream, state[stream])
            elif op == "stop":
                _ack(cid, True)
                for s, st in state.items():
                    _emit_final_if_open(s, st)
                _w(encode_frame(FrameType.BYE, json.dumps({"ok": True}).encode("utf-8")))
                with contextlib.suppress(OSError):
                    sys.stdout.flush()
                return 0
            else:
                _ack(cid, False, error=f"unknown op: {op}")
        elif frame.type.value == STTFrameType.AUDIO_F32.value:
            _process_audio(frame.payload, state)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

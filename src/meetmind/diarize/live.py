"""Streaming Sortformer 4spk-v2 backend (FluidAudio sidecar).

Same IPC shape as the STT adapter in `stt/parakeet_v3.py`: the sidecar
consumes AUDIO_F32 frames over stdin and emits DIAR_SEGMENT frames over
stdout. Frame definitions live in `meetmind.capture.ipc`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from meetmind.diarize.base import DiarSegment
from meetmind.ipc import (
    FrameType,
    IPCError,
    StreamId,
    encode_frame,
    read_frame,
)

log = logging.getLogger(__name__)


class DiarFrameType(IntEnum):
    """Diar-specific aliases over the shared FrameType numbering."""

    CONTROL = int(FrameType.CONTROL)
    AUDIO_F32 = int(FrameType.AUDIO_F32)
    DIAR_SEGMENT = int(FrameType.DIAR_SEGMENT)


_END_SENTINEL: object = object()


@dataclass
class SortformerSidecarBackend:
    """DiarBackend that delegates to a native Sortformer Swift sidecar."""

    binary: Path
    model: str = "diar-streaming-sortformer-4spk-v2"
    name: str = "sortformer-v2"

    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)
    _ack_q: asyncio.Queue | None = field(default=None, init=False, repr=False)
    _event_q: asyncio.Queue | None = field(default=None, init=False, repr=False)
    _drain_task: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> SortformerSidecarBackend:
        await self._spawn()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def _spawn(self) -> None:
        if self._proc is not None:
            return
        if not self.binary.exists():
            raise FileNotFoundError(f"diar sidecar not found: {self.binary}")
        self._proc = await asyncio.create_subprocess_exec(
            str(self.binary),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._ack_q = asyncio.Queue()
        self._event_q = asyncio.Queue()
        assert self._proc.stdout is not None
        hello = await read_frame(self._proc.stdout)
        if hello is None or hello.type is not FrameType.HELLO:
            raise IPCError(f"diar sidecar did not say HELLO; got {hello}")
        self._drain_task = asyncio.create_task(self._drain_loop(), name="diar-drain")
        # First-run model download can take a while on a fresh install.
        await self._send_control("start", {"model": self.model}, timeout=300.0)

    async def _drain_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                frame = await read_frame(proc.stdout)
                if frame is None:
                    break
                if frame.type is FrameType.CONTROL_ACK:
                    assert self._ack_q is not None
                    await self._ack_q.put(frame.as_json())
                elif frame.type is FrameType.LOG:
                    log.info("diar: %s", frame.payload.decode("utf-8", "replace"))
                elif frame.type is FrameType.BYE:
                    break
                elif frame.type.value == DiarFrameType.DIAR_SEGMENT.value:
                    assert self._event_q is not None
                    await self._event_q.put(frame)
        except asyncio.IncompleteReadError:
            pass
        except asyncio.CancelledError:
            raise
        finally:
            if self._event_q is not None:
                await self._event_q.put(_END_SENTINEL)

    async def _send_control(
        self, op: str, args: dict | None = None, *, timeout: float = 5.0
    ) -> dict:
        proc = self._require_proc()
        assert proc.stdin is not None
        assert self._ack_q is not None
        self._next_id += 1
        cmd_id = f"{self._next_id}"
        body = json.dumps({"id": cmd_id, "op": op, "args": args or {}}).encode("utf-8")
        proc.stdin.write(encode_frame(FrameType(DiarFrameType.CONTROL), body))
        await proc.stdin.drain()
        while True:
            ack = await asyncio.wait_for(self._ack_q.get(), timeout=timeout)
            if ack.get("id") == cmd_id:
                if not ack.get("ok"):
                    raise IPCError(f"diar sidecar rejected {op}: {ack.get('error')}")
                return ack

    async def stream(
        self,
        frames: AsyncIterator[tuple[StreamId, np.ndarray, int]],
        sample_rate: int = 16_000,
    ) -> AsyncIterator[DiarSegment]:
        proc = self._require_proc()
        assert proc.stdin is not None
        assert self._event_q is not None

        feed_done = asyncio.Event()

        async def _feed() -> None:
            try:
                async for stream_id, f32, start_ms in frames:
                    if f32.dtype != np.float32:
                        f32 = f32.astype(np.float32)
                    payload = self._encode_audio_payload_f32(stream_id, start_ms * 1000, f32)
                    if proc.stdin is None or proc.stdin.is_closing():
                        return
                    proc.stdin.write(encode_frame(FrameType(DiarFrameType.AUDIO_F32), payload))
                    with contextlib.suppress(ConnectionError):
                        await proc.stdin.drain()
            finally:
                feed_done.set()
                with contextlib.suppress(IPCError, TimeoutError):
                    await self._send_control("flush")

        feeder = asyncio.create_task(_feed(), name="diar-feeder")
        try:
            while True:
                try:
                    item = await asyncio.wait_for(self._event_q.get(), timeout=0.3)
                except TimeoutError:
                    if feed_done.is_set():
                        return
                    continue
                if item is _END_SENTINEL:
                    return
                frame = item  # type: ignore[assignment]
                data = json.loads(frame.payload.decode("utf-8"))
                channel: StreamId | None = None
                stream_str = data.get("stream")
                if stream_str == "mic":
                    channel = StreamId.MIC
                elif stream_str == "loopback":
                    channel = StreamId.LOOPBACK
                yield DiarSegment(
                    start_ms=int(data["start_ms"]),
                    end_ms=int(data["end_ms"]),
                    cluster_id=str(data["cluster_id"]),
                    confidence=float(data.get("confidence", 0.0)),
                    channel=channel,
                )
                if feed_done.is_set() and self._event_q.empty():
                    await asyncio.sleep(0.05)
                    if self._event_q.empty():
                        return
        finally:
            if not feeder.done():
                feeder.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await feeder

    async def aclose(self) -> None:
        if self._proc is None:
            return
        with contextlib.suppress(IPCError, TimeoutError):
            await asyncio.wait_for(self._send_control("stop"), timeout=2.0)
        if self._proc.stdin is not None and not self._proc.stdin.is_closing():
            with contextlib.suppress(Exception):
                self._proc.stdin.close()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except TimeoutError:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=1.0)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
        self._proc = None

    @staticmethod
    def _encode_audio_payload_f32(
        stream: StreamId,
        timestamp_us: int,
        f32: np.ndarray,
    ) -> bytes:
        if timestamp_us < 0 or timestamp_us >= (1 << 48):
            raise IPCError("timestamp out of u48 range")
        head = struct.pack("<BB", int(stream), 0)
        ts_bytes = struct.pack("<Q", timestamp_us)[:6]
        if not f32.dtype.isnative or f32.dtype.byteorder == ">":
            f32 = f32.astype("<f4")
        return head + ts_bytes + f32.tobytes(order="C")

    def _require_proc(self) -> asyncio.subprocess.Process:
        if self._proc is None:
            raise IPCError("diar sidecar not started")
        if self._proc.returncode is not None:
            raise IPCError(f"diar sidecar exited with code {self._proc.returncode}")
        return self._proc


def find_diar_sidecar() -> Path | None:
    """Best-effort lookup for the diar sidecar binary."""
    import shutil

    env = os.environ.get("MEETMIND_DIAR_SIDECAR")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).resolve().parents[3]
    dev = here / "sidecars" / "macos" / ".build" / "release" / "meetmind-diar-macos"
    if dev.exists():
        return dev
    found = shutil.which("meetmind-diar-macos")
    return Path(found) if found else None

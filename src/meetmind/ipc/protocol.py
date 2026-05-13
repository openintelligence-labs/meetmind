"""Capture sidecar IPC — length-prefixed binary frames over stdio.

Pure-Python implementation: frame codec + async subprocess manager.
Native sidecars (Swift/C++/C) on the other end of the pipe speak the
same frame types defined in this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import struct
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0.0"

# Frame header sizes. Match SPEC_CAPTURE_IPC.md.
_FRAME_HEADER = struct.Struct("<BI")  # type:u8, length:u32 LE
_AUDIO_HEADER = struct.Struct("<BB")  # stream:u8, flags:u8 (timestamp follows as 6B)


class FrameType(IntEnum):
    """Shared frame types across capture + STT + diarization IPC.

    0x00–0x0F: control plane (HELLO, READY, CONTROL).
    0x10–0x1F: data plane (AUDIO from capture, AUDIO_F32 from STT, PARTIAL,
    FINAL, DIAR_SEGMENT).
    0x20–0x2F: lifecycle events.
    0x30–0x3F: logging.
    0x40–0x4F: control acks.
    0xFF: BYE.
    """

    HELLO = 0x01
    READY = 0x02
    CONTROL = 0x05  # Python → STT/diar sidecar (framed JSON command)
    AUDIO = 0x10  # capture: PCM s16
    AUDIO_F32 = 0x10  # alias — STT/diar: PCM f32; same hex slot
    PARTIAL = 0x11  # STT → Python
    FINAL = 0x12  # STT → Python
    DIAR_SEGMENT = 0x13  # diarizer → Python
    EVENT = 0x20
    LOG = 0x30
    CONTROL_ACK = 0x40
    BYE = 0xFF


class StreamId(IntEnum):
    MIC = 0x00
    LOOPBACK = 0x01


class IPCError(RuntimeError):
    """Protocol violation, framing error, or sidecar non-zero exit."""


class PermissionMissing(IPCError):
    """Sidecar exited 2 — a required OS permission is not granted."""


@dataclass
class AudioChunk:
    """One AUDIO frame's worth of PCM, decoded.

    `pcm` is raw bytes in the format declared by the matching READY frame
    (default s16le 48 kHz mono per stream). The pipeline downstream is
    responsible for interpreting it.
    """

    stream: StreamId
    timestamp_us: int
    pcm: bytes
    is_segment_start: bool = False
    is_segment_end: bool = False


@dataclass
class Frame:
    type: FrameType
    payload: bytes

    def as_audio(self) -> AudioChunk:
        if self.type is not FrameType.AUDIO:
            raise IPCError(f"not an AUDIO frame: {self.type!r}")
        if len(self.payload) < 8:
            raise IPCError("AUDIO frame payload < 8 bytes (header truncated)")
        stream_byte, flags = _AUDIO_HEADER.unpack(self.payload[:2])
        # u48 LE timestamp = 6 bytes after the 2-byte mini-header
        ts_bytes = self.payload[2:8] + b"\x00\x00"
        timestamp_us = struct.unpack("<Q", ts_bytes)[0]
        try:
            stream = StreamId(stream_byte)
        except ValueError as e:
            raise IPCError(f"unknown stream id {stream_byte:#x}") from e
        return AudioChunk(
            stream=stream,
            timestamp_us=timestamp_us,
            pcm=self.payload[8:],
            is_segment_start=bool(flags & 0b0000_0001),
            is_segment_end=bool(flags & 0b0000_0010),
        )

    def as_json(self) -> dict[str, Any]:
        if self.type in (
            FrameType.HELLO,
            FrameType.READY,
            FrameType.EVENT,
            FrameType.CONTROL_ACK,
            FrameType.BYE,
        ):
            return json.loads(self.payload.decode("utf-8"))
        raise IPCError(f"frame type {self.type!r} is not JSON-bearing")


def encode_frame(type_: FrameType, payload: bytes) -> bytes:
    """Encode a single frame (test/mock-side helper)."""
    if len(payload) > 0xFFFF_FFFF:
        raise IPCError("frame payload exceeds u32 length")
    return _FRAME_HEADER.pack(int(type_), len(payload)) + payload


def encode_audio_payload(
    stream: StreamId,
    timestamp_us: int,
    pcm: bytes,
    is_segment_start: bool = False,
    is_segment_end: bool = False,
) -> bytes:
    if timestamp_us < 0 or timestamp_us >= (1 << 48):
        raise IPCError("timestamp out of u48 range")
    flags = (0b01 if is_segment_start else 0) | (0b10 if is_segment_end else 0)
    ts_bytes = struct.pack("<Q", timestamp_us)[:6]
    return _AUDIO_HEADER.pack(int(stream), flags) + ts_bytes + pcm


async def read_frame(reader: asyncio.StreamReader) -> Frame | None:
    """Read one length-prefixed frame from `reader`. Returns None at EOF."""
    header = await reader.readexactly(_FRAME_HEADER.size) if not reader.at_eof() else b""
    if not header:
        return None
    try:
        type_byte, length = _FRAME_HEADER.unpack(header)
    except struct.error as e:
        raise IPCError(f"malformed frame header: {e}") from e
    try:
        type_ = FrameType(type_byte)
    except ValueError as e:
        raise IPCError(f"unknown frame type {type_byte:#x}") from e
    payload = await reader.readexactly(length) if length > 0 else b""
    return Frame(type=type_, payload=payload)


@dataclass
class SidecarFormat:
    sample_rate: int = 48000
    encoding: str = "s16le"
    channels: int = 1


@dataclass
class SidecarHello:
    sidecar: str
    version: str
    protocol_version: str
    platform: str
    capabilities: list[str] = field(default_factory=list)
    permissions: dict[str, str] = field(default_factory=dict)


class SidecarProcess:
    """Spawn a native capture sidecar, exchange the IPC protocol with it."""

    def __init__(
        self,
        binary: str | os.PathLike[str],
        *args: str,
        env: dict[str, str] | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.args = tuple(args)
        self.env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._hello: SidecarHello | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._next_id = 0
        # Ring buffer of recent stderr lines — used by the watchdog to
        # report what the sidecar said just before dying.
        self._stderr_ring: list[str] = []
        self._stderr_ring_max = 32

    @property
    def returncode(self) -> int | None:
        """None if running or never started; the exit code if exited."""
        if self._proc is None:
            return None
        return self._proc.returncode

    def stderr_tail(self, max_bytes: int = 512) -> str:
        """Most recent stderr lines joined, trimmed to ``max_bytes``."""
        joined = "\n".join(self._stderr_ring)
        if len(joined) <= max_bytes:
            return joined
        return joined[-max_bytes:]

    @property
    def hello(self) -> SidecarHello:
        if self._hello is None:
            raise IPCError("sidecar has not sent HELLO yet")
        return self._hello

    async def start(self) -> SidecarHello:
        """Spawn the sidecar and read its HELLO frame."""
        if self._proc is not None:
            raise IPCError("sidecar already started")
        if not self.binary.exists():
            raise FileNotFoundError(f"sidecar binary not found: {self.binary}")
        self._proc = await asyncio.create_subprocess_exec(
            str(self.binary),
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        assert self._proc.stdout is not None
        self._stderr_task = asyncio.create_task(self._drain_stderr(), name="sidecar-stderr")
        first = await read_frame(self._proc.stdout)
        if first is None:
            await self._raise_dead("sidecar exited before HELLO")
        if first.type is not FrameType.HELLO:
            raise IPCError(f"expected HELLO, got {first.type!r}")
        data = first.as_json()
        self._hello = SidecarHello(
            sidecar=data.get("sidecar", "unknown"),
            version=data.get("version", "0.0.0"),
            protocol_version=data.get("protocol_version", "0.0.0"),
            platform=data.get("platform", "unknown"),
            capabilities=list(data.get("capabilities", [])),
            permissions=dict(data.get("permissions", {})),
        )
        if not self._hello.protocol_version.startswith("1."):
            raise IPCError(f"unsupported sidecar protocol {self._hello.protocol_version}; need 1.x")
        return self._hello

    async def send(self, op: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a control command and await its CONTROL_ACK."""
        proc = self._require_proc()
        assert proc.stdin is not None
        self._next_id += 1
        cmd_id = f"{self._next_id}"
        line = json.dumps({"id": cmd_id, "op": op, "args": args or {}}) + "\n"
        proc.stdin.write(line.encode("utf-8"))
        await proc.stdin.drain()
        # Wait specifically for the matching CONTROL_ACK (skipping any in-flight
        # AUDIO/LOG frames). Bounded by a sensible timeout.
        async for frame in self.frames():
            if frame.type is FrameType.CONTROL_ACK:
                ack = frame.as_json()
                if ack.get("id") == cmd_id:
                    if not ack.get("ok"):
                        raise IPCError(f"sidecar rejected {op!r}: {ack.get('error')}")
                    return ack
            elif frame.type is FrameType.BYE:
                raise IPCError(f"sidecar said BYE before ACKing {op!r}")
        raise IPCError(f"sidecar closed stdout before ACKing {op!r}")

    async def frames(self) -> AsyncIterator[Frame]:
        proc = self._require_proc()
        assert proc.stdout is not None
        while True:
            try:
                frame = await read_frame(proc.stdout)
            except asyncio.IncompleteReadError as e:
                if e.partial:
                    raise IPCError("truncated frame at EOF") from e
                return
            if frame is None:
                return
            if frame.type is FrameType.LOG:
                log.info("sidecar: %s", frame.payload.decode("utf-8", "replace"))
                continue
            yield frame
            if frame.type is FrameType.BYE:
                return

    async def audio(self) -> AsyncIterator[AudioChunk]:
        async for frame in self.frames():
            if frame.type is FrameType.AUDIO:
                yield frame.as_audio()

    async def stop(self, timeout: float = 2.0) -> None:
        if self._proc is None:
            return
        # Best-effort graceful stop. Bound everything strictly — sidecars
        # speaking the protocol incorrectly must not deadlock teardown.
        with contextlib.suppress(IPCError, TimeoutError):
            await asyncio.wait_for(self.send("stop"), timeout=timeout)
        # Close stdin so the sidecar's `for cmd in stdin` loop exits even if
        # `stop` ACK was lost.
        if self._proc.stdin is not None and not self._proc.stdin.is_closing():
            with contextlib.suppress(Exception):
                self._proc.stdin.close()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=timeout)
        except TimeoutError:
            log.warning("sidecar did not exit in %.1fs; terminating", timeout)
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=timeout)
            except TimeoutError:
                log.warning("sidecar did not respond to SIGTERM; killing")
                self._proc.kill()
                await self._proc.wait()
        finally:
            if self._stderr_task is not None:
                self._stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._stderr_task

    async def __aenter__(self) -> SidecarProcess:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def _require_proc(self) -> asyncio.subprocess.Process:
        if self._proc is None:
            raise IPCError("sidecar not started")
        if self._proc.returncode is not None:
            raise IPCError(f"sidecar exited with code {self._proc.returncode}")
        return self._proc

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        async for line in proc.stderr:
            decoded = line.decode("utf-8", "replace").rstrip()
            log.debug("sidecar[stderr]: %s", decoded)
            # Maintain the ring so the watchdog can include it on death.
            self._stderr_ring.append(decoded)
            if len(self._stderr_ring) > self._stderr_ring_max:
                del self._stderr_ring[0]

    async def _raise_dead(self, msg: str) -> None:
        proc = self._proc
        rc = -1 if proc is None else (proc.returncode if proc.returncode is not None else -1)
        if rc == 2:
            raise PermissionMissing("sidecar reports missing OS permission")
        raise IPCError(f"{msg} (exit code {rc})")


@dataclass
class IpcId:
    """Helper for tests: generate a deterministic correlation ID."""

    prefix: str = "test"
    _counter: int = 0

    def next(self) -> str:
        self._counter += 1
        return f"{self.prefix}-{self._counter}-{uuid.uuid4().hex[:6]}"

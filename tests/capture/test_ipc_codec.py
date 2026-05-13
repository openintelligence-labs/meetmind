"""Pure codec tests for the capture IPC wire format.

These lock in the byte-for-byte representation; if any of these break
that means the wire format changed and the protocol-version major in
SPEC_CAPTURE_IPC.md must bump.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from meetmind.ipc import (
    PROTOCOL_VERSION,
    FrameType,
    IPCError,
    StreamId,
    encode_audio_payload,
    encode_frame,
    read_frame,
)


def test_protocol_version_pinned():
    assert PROTOCOL_VERSION == "1.0.0"


def test_frame_header_is_5_bytes():
    encoded = encode_frame(FrameType.LOG, b"hi")
    assert len(encoded) == 5 + 2
    assert encoded[0] == FrameType.LOG.value
    length = struct.unpack("<I", encoded[1:5])[0]
    assert length == 2
    assert encoded[5:] == b"hi"


def test_zero_length_payload_is_legal():
    encoded = encode_frame(FrameType.BYE, b"")
    assert encoded == bytes([FrameType.BYE.value, 0, 0, 0, 0])


def test_audio_payload_layout_and_decode():
    pcm = b"\x01\x02\x03\x04"
    raw = encode_audio_payload(
        stream=StreamId.LOOPBACK,
        timestamp_us=0x0123_4567_89AB,
        pcm=pcm,
        is_segment_start=True,
        is_segment_end=False,
    )
    # 1 byte stream + 1 byte flags + 6 byte ts + pcm
    assert len(raw) == 8 + len(pcm)
    assert raw[0] == StreamId.LOOPBACK.value
    assert raw[1] == 0b01  # segment_start, not segment_end
    # 0x0123_4567_89AB little-endian over 6 bytes:
    assert raw[2:8] == bytes([0xAB, 0x89, 0x67, 0x45, 0x23, 0x01])
    assert raw[8:] == pcm


def test_audio_payload_rejects_huge_timestamp():
    with pytest.raises(IPCError):
        encode_audio_payload(StreamId.MIC, 1 << 48, b"")


async def _frames_from_bytes(data: bytes) -> list:
    """Helper: feed bytes through asyncio.StreamReader and read frames."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    out = []
    while True:
        f = await read_frame(reader)
        if f is None:
            break
        out.append(f)
    return out


async def test_round_trip_two_frames():
    blob = encode_frame(FrameType.LOG, b"first") + encode_frame(FrameType.LOG, b"second")
    frames = await _frames_from_bytes(blob)
    assert [f.type for f in frames] == [FrameType.LOG, FrameType.LOG]
    assert [f.payload for f in frames] == [b"first", b"second"]


async def test_round_trip_audio_frame():
    payload = encode_audio_payload(
        stream=StreamId.MIC,
        timestamp_us=12345,
        pcm=b"\xff" * 32,
        is_segment_start=False,
        is_segment_end=True,
    )
    blob = encode_frame(FrameType.AUDIO, payload)
    frames = await _frames_from_bytes(blob)
    assert len(frames) == 1
    chunk = frames[0].as_audio()
    assert chunk.stream is StreamId.MIC
    assert chunk.timestamp_us == 12345
    assert chunk.is_segment_start is False
    assert chunk.is_segment_end is True
    assert chunk.pcm == b"\xff" * 32


async def test_unknown_frame_type_raises():
    bad = bytes([0x99, 0, 0, 0, 0])
    reader = asyncio.StreamReader()
    reader.feed_data(bad)
    reader.feed_eof()
    with pytest.raises(IPCError):
        await read_frame(reader)


async def test_truncated_header_raises():
    reader = asyncio.StreamReader()
    reader.feed_data(b"\x01\x00\x00")  # only 3 of 5 header bytes
    reader.feed_eof()
    with pytest.raises((asyncio.IncompleteReadError, IPCError)):
        await read_frame(reader)

"""Shared IPC protocol types.

The capture, STT, and diarization sidecars all speak the length-prefixed
binary frame format defined in ``meetmind.ipc.protocol``.

Module boundary: leaf — never imports project-internal modules.
"""

from meetmind.ipc.protocol import (
    PROTOCOL_VERSION,
    AudioChunk,
    Frame,
    FrameType,
    IPCError,
    PermissionMissing,
    SidecarFormat,
    SidecarHello,
    SidecarProcess,
    StreamId,
    encode_audio_payload,
    encode_frame,
    read_frame,
)

__all__ = [
    "PROTOCOL_VERSION",
    "AudioChunk",
    "Frame",
    "FrameType",
    "IPCError",
    "PermissionMissing",
    "SidecarFormat",
    "SidecarHello",
    "SidecarProcess",
    "StreamId",
    "encode_audio_payload",
    "encode_frame",
    "read_frame",
]

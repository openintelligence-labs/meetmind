"""Shared IPC protocol types.

The capture, STT, and diarization sidecars all speak the same
length-prefixed binary frame format defined in ``meetmind.ipc.protocol``.
The wire format is neutral infrastructure — neither capture-specific nor
diarize-specific — so it lives in this top-level module rather than under
any single subsystem.

Module boundary: leaf — never imports project-internal modules. Imported by
`capture/`, `stt/`, and `diarize/` adapters.
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

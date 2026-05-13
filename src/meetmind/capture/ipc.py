"""Backwards-compat shim — IPC protocol now lives in `meetmind.ipc`.

Importing from `meetmind.capture.ipc` continues to work. New code should
import from `meetmind.ipc` directly.
"""

from meetmind.ipc import (  # noqa: F401
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

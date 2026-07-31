"""Event types shipped over `/v1/transcripts/live`.

Each event is serialized as JSON in the SSE `data:` field, with the kind in
the `event:` field. A `partial` revises the previous hypothesis; a `final` is
committed and never revised.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PartialEvent(BaseModel):
    kind: Literal["partial"] = "partial"
    text: str
    start_ms: int
    end_ms: int
    confidence: float = 0.0
    stream: Literal["mic", "loopback"] | None = None


class FinalEvent(BaseModel):
    kind: Literal["final"] = "final"
    text: str
    start_ms: int
    end_ms: int
    confidence: float = 0.0
    language: str = "en"
    stream: Literal["mic", "loopback"] | None = None


class DiarEvent(BaseModel):
    kind: Literal["diar"] = "diar"
    cluster_id: str
    start_ms: int
    end_ms: int
    confidence: float = 0.0
    stream: Literal["mic", "loopback"] | None = None


class SpeakerEvent(BaseModel):
    kind: Literal["speaker"] = "speaker"
    text: str
    cluster_id: str
    speaker_id: str | None = None
    start_ms: int
    end_ms: int
    confidence: float = 0.0
    stream: Literal["mic", "loopback"] | None = None


class MetaEvent(BaseModel):
    kind: Literal["meta"] = "meta"
    event: Literal["session_started", "session_stopped", "error"]
    detail: str | None = None


class CoachTipEvent(BaseModel):
    """Live coaching suggestion from the rolling-window LLM.

    ``window_start_ms`` / ``window_end_ms`` bound the transcript span the tip
    is grounded in, so the overlay can highlight it.
    """

    kind: Literal["coach_tip"] = "coach_tip"
    tip: str
    category: Literal["question", "missed_point", "follow_up", "clarify", "other"] = "other"
    window_start_ms: int
    window_end_ms: int
    model: str | None = None
    confidence: float = 0.0


class SidecarEvent(BaseModel):
    """Sidecar lifecycle event, including mid-meeting deaths and restarts.

    ``returncode`` is None while the sidecar is still running; non-None means
    it exited.
    """

    kind: Literal["sidecar"] = "sidecar"
    sidecar: Literal["capture", "stt", "diar"]
    event: Literal["started", "ready", "died", "restarting", "gave_up"]
    returncode: int | None = None
    stderr_tail: str | None = None  # last ~512 bytes of stderr, trimmed
    attempt: int = 0


Event = (
    PartialEvent | FinalEvent | DiarEvent | SpeakerEvent | MetaEvent | CoachTipEvent | SidecarEvent
)

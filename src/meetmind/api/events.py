"""Event types for the local SSE bus.

Anything that gets shipped over `/v1/transcripts/live` is one of these:

  • `partial`   — incremental, revisable hypothesis (replace last line)
  • `final`     — committed transcript span (append, never revise)
  • `diar`      — speaker boundary (start_ms, end_ms, cluster_id)
  • `speaker`   — fully-stitched speaker-attributed segment
  • `meta`      — session lifecycle (started, stopped, error)

Each event is a small Pydantic model serialized as JSON in the SSE
`data:` field. The `event:` field carries the event kind.
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

    Emitted by the optional coach loop (off by default, opt-in via
    ``meetmind record --coach``). Tips reference a window of recent
    transcript spans by ``window_start_ms`` / ``window_end_ms`` so the
    overlay can highlight what the suggestion is grounded in.
    """

    kind: Literal["coach_tip"] = "coach_tip"
    tip: str
    category: Literal["question", "missed_point", "follow_up", "clarify", "other"] = "other"
    window_start_ms: int
    window_end_ms: int
    model: str | None = None
    confidence: float = 0.0


class SidecarEvent(BaseModel):
    """Sidecar lifecycle event — covers mid-meeting deaths + restart attempts.

    The watchdog in ``cli._run_record`` publishes this so the UI can show
    "recording interrupted" instead of going silent. ``returncode`` is
    None for a still-running sidecar (start/ready); non-None means the
    sidecar exited.
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

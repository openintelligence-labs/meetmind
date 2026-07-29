"""MeetMind — local-first meeting assistant."""

from meetmind.models import (
    ActionItem,
    ChannelKind,
    ConsentEvent,
    Decision,
    Meeting,
    MeetingTemplate,
    Speaker,
    Summary,
    Transcript,
    TranscriptSegment,
)

__version__ = "1.0.0"

__all__ = [
    "ActionItem",
    "ChannelKind",
    "ConsentEvent",
    "Decision",
    "Meeting",
    "MeetingTemplate",
    "Speaker",
    "Summary",
    "Transcript",
    "TranscriptSegment",
    "__version__",
]

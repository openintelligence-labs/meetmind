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

__version__ = "0.1.0.dev0"

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

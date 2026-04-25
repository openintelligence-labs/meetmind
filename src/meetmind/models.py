from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start_seconds: float
    end_seconds: float
    speaker: str | None = None
    text: str


class Transcript(BaseModel):
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def duration_seconds(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end_seconds


class ActionItem(BaseModel):
    description: str
    owner: str | None = None
    due: str | None = None


class Summary(BaseModel):
    tl_dr: str
    key_decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


class Meeting(BaseModel):
    id: str
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    transcript: Transcript = Field(default_factory=Transcript)
    summary: Summary | None = None

"""Pydantic data model for MeetMind.

Entities: ``Transcript``, ``TranscriptSegment``, ``Summary``,
``ActionItem``, ``Decision``, ``Meeting``, ``MeetingTemplate``,
``Speaker``, ``ConsentEvent``, ``ChannelKind``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_ulid() -> str:
    return str(ULID())


class ChannelKind(StrEnum):
    """Source channel of a captured audio segment.

    Mic = "self"; loopback = "remote(s)". Kept distinct end-to-end —
    the channel split is the single biggest accuracy lever in the
    diarization pipeline.
    """

    MIC = "mic"
    LOOPBACK = "loopback"


class MeetingTemplate(StrEnum):
    STANDUP = "standup"
    ONE_ON_ONE = "1on1"
    SALES_DISCOVERY = "sales_discovery"
    BRAINSTORM = "brainstorm"
    GENERIC = "generic"
    ASSIST = "assist"  # v1.1 — assist-mode sessions, when --archive is set


class TranscriptSegment(BaseModel):
    """A single contiguous span of speech.

    `start_seconds` / `end_seconds` are kept for backwards compatibility.
    `start_ms` / `end_ms` are the canonical wire format going forward.
    """

    model_config = ConfigDict(populate_by_name=True)

    start_seconds: float
    end_seconds: float
    speaker: str | None = None
    text: str

    # New in v0.4 — additive only.
    channel: ChannelKind | None = None
    speaker_id: str | None = None
    confidence: float | None = None
    language: str = "en"

    @property
    def start_ms(self) -> int:
        return int(self.start_seconds * 1000)

    @property
    def end_ms(self) -> int:
        return int(self.end_seconds * 1000)


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
    """An extracted commitment from a meeting.

    `evidence_quote` (v0.4+) is the verbatim substring from the source
    transcript that justified extraction. Substring-validation against
    the source segment kills ~80% of hallucinated closures (architecture
    §5.4).
    """

    description: str
    owner: str | None = None
    due: str | None = None

    # New in v0.4 — additive only.
    id: str = Field(default_factory=_new_ulid)
    source_segment_id: int | None = None
    evidence_quote: str | None = None
    status: Literal["open", "done", "cancelled", "follow_up_needed"] = "open"
    closed_in_meeting_id: str | None = None
    closed_evidence_quote: str | None = None


class Decision(BaseModel):
    """An explicit decision made in a meeting.

    Tracks dissenters by speaker_id so 1-on-1 prep / weekly review can
    surface "X pushed back on this in last week's meeting".
    """

    id: str = Field(default_factory=_new_ulid)
    decision: str
    rationale: str = ""
    dissenters: list[str] = Field(default_factory=list)
    source_segment_ids: list[int] = Field(default_factory=list)


class Summary(BaseModel):
    tl_dr: str
    key_decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


class Speaker(BaseModel):
    """A persistent, opt-in voiceprint identity.

    Voiceprints are biometric special-category data under GDPR Art. 9
    and BIPA / CUBI in the US. Storage requires explicit per-speaker
    consent, which is logged immutably as a `ConsentEvent`.

    `voiceprint_centroid` is the EMA-updated mean of L2-normalized
    embeddings. Stored as raw bytes here; the encryption envelope
    lives in `meetmind.crypto`.
    """

    id: str = Field(default_factory=_new_ulid)
    display_name: str | None = None
    consent_ts: datetime | None = None
    consent_disclosure_version: str | None = None
    voiceprint_centroid: bytes | None = None
    voiceprint_ring: list[bytes] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    enrolled_at: datetime = Field(default_factory=_utcnow)
    confidence: float = 0.0
    retention_until: date | None = None


class ConsentEvent(BaseModel):
    """Immutable audit-log entry for biometric-data lifecycle events.

    Required for BIPA "written consent" via ESIGN and for GDPR
    accountability. Never deleted; on speaker erasure the event is
    retained but the `actor_speaker_id` becomes a tombstone reference.
    """

    id: str = Field(default_factory=_new_ulid)
    ts: datetime = Field(default_factory=_utcnow)
    actor_speaker_id: str
    action: Literal["enroll", "revoke", "delete", "export"]
    disclosure_version: str
    signature: bytes | None = None  # Ed25519 over canonical fields


class Meeting(BaseModel):
    """A single recording session.

    `template = ASSIST` marks an opt-in archived assist-mode session
    (v1.1). The `audio_path_*` fields point to encrypted Ogg-Opus files;
    decrypted content never touches disk outside the SQLCipher-protected
    directory.

    `transcript_hash` + `signature` enable the "legal-mode signed
    transcript bundle" export feature.
    """

    id: str = Field(default_factory=_new_ulid)
    title: str
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    template: MeetingTemplate | None = None
    calendar_event_id: str | None = None
    audio_path_mic: Path | None = None
    audio_path_loopback: Path | None = None
    transcript: Transcript = Field(default_factory=Transcript)
    summary: Summary | None = None
    decisions: list[Decision] = Field(default_factory=list)
    transcript_hash: bytes | None = None
    signature: bytes | None = None
    cost_usd: float = 0.0

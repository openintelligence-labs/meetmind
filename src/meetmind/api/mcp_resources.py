"""MCP `resources/*` primitives for MeetMind.

Resources are addressable, read-only chunks of meeting state that an MCP
client (Claude / Cursor / Windsurf) can fetch by URI. Unlike tools (which
are RPC-style), resources are listable and pull-able like files.

URI scheme: ``meetmind://`` — opaque, local, never leaves the device.

  meetmind://meetings                       — index of recent meetings
  meetmind://meeting/{id}                   — full meeting record (JSON)
  meetmind://meeting/{id}/transcript        — transcript as Markdown
  meetmind://meeting/{id}/summary           — Chain-of-Density summary
  meetmind://meeting/{id}/decisions         — decisions as Markdown
  meetmind://meeting/{id}/actions           — action items as Markdown

The MCP wire payload for `resources/read` is `{contents: [{uri, mimeType,
text}]}`. We always return a single content item per read; multi-resource
reads fan out at the JSON-RPC layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from meetmind.memory.store import Store
from meetmind.models import ActionItem, Decision, Meeting, TranscriptSegment


@dataclass(frozen=True)
class ResourceDescriptor:
    """One row in `resources/list`."""

    uri: str
    name: str
    description: str
    mime_type: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass(frozen=True)
class ResourceContent:
    """Payload of `resources/read`."""

    uri: str
    mime_type: str
    text: str

    def to_wire(self) -> dict[str, Any]:
        return {"uri": self.uri, "mimeType": self.mime_type, "text": self.text}


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

_MEETING_URI_RE = re.compile(
    r"^meetmind://meeting/(?P<id>[A-Za-z0-9._-]+)(?:/(?P<sub>transcript|summary|decisions|actions))?$"
)

_PERSON_URI_RE = re.compile(r"^meetmind://person/(?P<id>[A-Za-z0-9._-]+)(?:/(?P<sub>profile))?$")


def list_resources(store: Store, *, limit: int = 50) -> list[ResourceDescriptor]:
    """Enumerate the meeting index plus per-meeting subresources.

    The first item is always the meetings index. After that we emit one
    descriptor per meeting (the meeting record itself) — the per-meeting
    subresources (transcript/summary/decisions/actions) are advertised
    via the ``meetings`` index document so clients can pull them on
    demand without us blowing up the list with N×4 items.
    """
    out: list[ResourceDescriptor] = [
        ResourceDescriptor(
            uri="meetmind://meetings",
            name="All meetings",
            description="Index of all stored meetings (most recent first).",
            mime_type="application/json",
        ),
        ResourceDescriptor(
            uri="meetmind://people",
            name="All speakers",
            description="Index of enrolled speakers with profile links.",
            mime_type="application/json",
        ),
    ]
    for m in store.list_meetings(limit=limit):
        title = m.title or m.id
        out.append(
            ResourceDescriptor(
                uri=f"meetmind://meeting/{m.id}",
                name=title,
                description=_meeting_blurb(m),
                mime_type="application/json",
            )
        )
    return out


def read_resource(store: Store, uri: str) -> ResourceContent:
    """Return the content addressed by ``uri``.

    Raises ``KeyError`` for unknown URIs and ``LookupError`` for
    well-formed URIs whose target is missing (e.g. unknown meeting id).
    """
    if uri == "meetmind://meetings":
        return _read_meetings_index(store)
    if uri == "meetmind://people":
        return _read_people_index(store)

    person_match = _PERSON_URI_RE.match(uri)
    if person_match is not None:
        return _read_person_profile(store, person_match.group("id"))

    match = _MEETING_URI_RE.match(uri)
    if match is None:
        raise KeyError(f"unknown resource URI: {uri}")
    meeting_id = match.group("id")
    sub = match.group("sub")

    meeting = store.get_meeting(meeting_id)
    if meeting is None:
        raise LookupError(f"meeting not found: {meeting_id}")

    if sub is None:
        return _read_meeting_record(store, meeting)
    if sub == "transcript":
        return _read_transcript(store, meeting)
    if sub == "summary":
        return _read_summary(meeting)
    if sub == "decisions":
        return _read_decisions(store, meeting)
    if sub == "actions":
        return _read_actions(store, meeting)
    raise KeyError(f"unknown resource subpath: {sub}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_meetings_index(store: Store) -> ResourceContent:
    meetings = store.list_meetings(limit=200)
    body = {
        "count": len(meetings),
        "meetings": [
            {
                "id": m.id,
                "title": m.title,
                "started_at": m.started_at.isoformat() if m.started_at else None,
                "duration_seconds": m.duration_seconds,
                "uri": f"meetmind://meeting/{m.id}",
                "subresources": {
                    "transcript": f"meetmind://meeting/{m.id}/transcript",
                    "summary": f"meetmind://meeting/{m.id}/summary",
                    "decisions": f"meetmind://meeting/{m.id}/decisions",
                    "actions": f"meetmind://meeting/{m.id}/actions",
                },
            }
            for m in meetings
        ],
    }
    return ResourceContent(
        uri="meetmind://meetings",
        mime_type="application/json",
        text=json.dumps(body, indent=2, default=str),
    )


def _read_people_index(store: Store) -> ResourceContent:
    rows = store.conn.execute(
        "SELECT id, display_name FROM speakers ORDER BY display_name COLLATE NOCASE"
    ).fetchall()
    body = {
        "count": len(rows),
        "people": [
            {
                "id": r["id"],
                "display_name": r["display_name"],
                "uri": f"meetmind://person/{r['id']}",
                "profile_uri": f"meetmind://person/{r['id']}/profile",
            }
            for r in rows
        ],
    }
    return ResourceContent(
        uri="meetmind://people",
        mime_type="application/json",
        text=json.dumps(body, indent=2, default=str),
    )


def _read_person_profile(store: Store, speaker_id: str) -> ResourceContent:
    speaker = store.get_speaker(speaker_id)
    if speaker is None:
        raise LookupError(f"speaker not found: {speaker_id}")
    # Aggregate talk-time, meetings spoken in, and a few example
    # quotes. Cheap to compute even for a year of meetings.
    rows = store.conn.execute(
        """
        SELECT meeting_id, text, start_ms, end_ms
          FROM transcript_segments
         WHERE speaker_id = ?
         ORDER BY start_ms DESC
         LIMIT 200
        """,
        (speaker_id,),
    ).fetchall()
    total_ms = sum((r["end_ms"] - r["start_ms"]) for r in rows)
    meetings = sorted({r["meeting_id"] for r in rows})
    sample_quotes = [r["text"] for r in rows[:5]]
    body = {
        "id": speaker.id,
        "display_name": speaker.display_name,
        "consent_ts": speaker.consent_ts.isoformat() if speaker.consent_ts else None,
        "voiceprint_present": speaker.voiceprint_centroid is not None,
        "talk_time_seconds": round(total_ms / 1000.0, 1),
        "segments_seen": len(rows),
        "meetings_seen": len(meetings),
        "meeting_ids": meetings,
        "sample_quotes": sample_quotes,
    }
    return ResourceContent(
        uri=f"meetmind://person/{speaker_id}/profile",
        mime_type="application/json",
        text=json.dumps(body, indent=2, default=str),
    )


def _read_meeting_record(store: Store, meeting: Meeting) -> ResourceContent:
    body = meeting.model_dump(mode="json", exclude={"transcript", "decisions", "summary"})
    body["segment_count"] = len(store.list_segments(meeting.id))
    body["action_count"] = len(store.list_action_items(meeting_id=meeting.id))
    body["decision_count"] = len(store.list_decisions(meeting.id))
    return ResourceContent(
        uri=f"meetmind://meeting/{meeting.id}",
        mime_type="application/json",
        text=json.dumps(body, indent=2, default=str),
    )


def _read_transcript(store: Store, meeting: Meeting) -> ResourceContent:
    segments = store.list_segments(meeting.id)
    return ResourceContent(
        uri=f"meetmind://meeting/{meeting.id}/transcript",
        mime_type="text/markdown",
        text=_transcript_to_markdown(meeting, segments),
    )


def _read_summary(meeting: Meeting) -> ResourceContent:
    summary = meeting.summary
    body = summary.tl_dr if summary is not None and summary.tl_dr else "_No summary generated yet._"
    return ResourceContent(
        uri=f"meetmind://meeting/{meeting.id}/summary",
        mime_type="text/markdown",
        text=f"# Summary — {meeting.title or meeting.id}\n\n{body}\n",
    )


def _read_decisions(store: Store, meeting: Meeting) -> ResourceContent:
    decisions = store.list_decisions(meeting.id)
    return ResourceContent(
        uri=f"meetmind://meeting/{meeting.id}/decisions",
        mime_type="text/markdown",
        text=_decisions_to_markdown(meeting, decisions),
    )


def _read_actions(store: Store, meeting: Meeting) -> ResourceContent:
    items = store.list_action_items(meeting_id=meeting.id)
    return ResourceContent(
        uri=f"meetmind://meeting/{meeting.id}/actions",
        mime_type="text/markdown",
        text=_actions_to_markdown(meeting, items),
    )


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------


def _meeting_blurb(m: Meeting) -> str:
    bits = []
    if m.started_at:
        bits.append(m.started_at.strftime("%Y-%m-%d %H:%M"))
    if m.duration_seconds:
        bits.append(f"{m.duration_seconds / 60:.0f} min")
    return " · ".join(bits) or "Meeting record"


def _transcript_to_markdown(meeting: Meeting, segments: list[TranscriptSegment]) -> str:
    lines = [f"# Transcript — {meeting.title or meeting.id}", ""]
    if not segments:
        lines.append("_No transcript segments stored._")
        return "\n".join(lines) + "\n"
    for s in segments:
        speaker = s.speaker_id or s.speaker or "speaker"
        ts = _ms_to_ts(s.start_ms)
        lines.append(f"**[{ts}] {speaker}:** {s.text}")
        lines.append("")
    return "\n".join(lines)


def _decisions_to_markdown(meeting: Meeting, decisions: list[Decision]) -> str:
    lines = [f"# Decisions — {meeting.title or meeting.id}", ""]
    if not decisions:
        lines.append("_No decisions recorded._")
        return "\n".join(lines) + "\n"
    for d in decisions:
        lines.append(f"- **{d.decision}**")
        if d.rationale:
            lines.append(f"  - Rationale: {d.rationale}")
        if d.dissenters:
            lines.append(f"  - Dissenters: {', '.join(d.dissenters)}")
        lines.append("")
    return "\n".join(lines)


def _actions_to_markdown(meeting: Meeting, items: list[ActionItem]) -> str:
    lines = [f"# Action items — {meeting.title or meeting.id}", ""]
    if not items:
        lines.append("_No action items extracted._")
        return "\n".join(lines) + "\n"
    for a in items:
        owner = a.owner or "unassigned"
        due = a.due or "no due date"
        status = a.status or "open"
        lines.append(f"- [{status}] **{a.description}** — {owner} ({due})")
        if a.evidence_quote:
            lines.append(f"  - Evidence: _{a.evidence_quote}_")
    return "\n".join(lines) + "\n"


def _ms_to_ts(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

"""Obsidian integration: writes a Markdown note per meeting into the vault.

Filesystem-only, with Dataview-compatible YAML frontmatter.

Module is a leaf per the import-linter contract: memory and models only,
via structural typing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from meetmind.models import ActionItem, Decision, Meeting, TranscriptSegment


class _StoreLike(Protocol):
    """Minimal store shape this module needs.

    Declared inline so ``integrations`` stays a leaf and never imports
    ``memory``; callers pass a real ``Store``.
    """

    def get_meeting(self, meeting_id: str) -> Meeting | None: ...
    def list_segments(self, meeting_id: str) -> list[TranscriptSegment]: ...
    def list_action_items(
        self, *, status: str | None = ..., meeting_id: str | None = ..., owner: str | None = ...
    ) -> list[ActionItem]: ...
    def list_decisions(self, meeting_id: str) -> list[Decision]: ...


@dataclass
class ObsidianExportResult:
    note_path: Path
    bytes_written: int


_FRONTMATTER_FIELD_RE = re.compile(r"[^a-z0-9_]+")


def _slug(s: str) -> str:
    """Conservative Obsidian-friendly slug for filenames."""
    s = (s or "meeting").strip().lower()
    s = _FRONTMATTER_FIELD_RE.sub("-", s)
    return s.strip("-") or "meeting"


def export_meeting(
    store: _StoreLike,
    meeting_id: str,
    *,
    vault: Path,
    folder: str = "MeetMind",
    overwrite: bool = False,
) -> ObsidianExportResult:
    """Export a single meeting as a Markdown note in the vault.

    Returns the path written and byte count. Raises ``LookupError`` if
    the meeting doesn't exist; ``FileExistsError`` if the target note
    already exists and ``overwrite=False``.
    """
    meeting = store.get_meeting(meeting_id)
    if meeting is None:
        raise LookupError(f"meeting not found: {meeting_id}")

    segments = store.list_segments(meeting_id)
    actions = store.list_action_items(meeting_id=meeting_id)
    decisions = store.list_decisions(meeting_id)

    target_dir = vault / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    date_str = meeting.started_at.date().isoformat() if meeting.started_at else "undated"
    title_slug = _slug(meeting.title or meeting.id)
    note_path = target_dir / f"{date_str}-{title_slug}.md"

    if note_path.exists() and not overwrite:
        raise FileExistsError(note_path)

    body = _render_markdown(meeting, segments, actions, decisions)
    note_path.write_text(body, encoding="utf-8")
    return ObsidianExportResult(note_path=note_path, bytes_written=len(body.encode("utf-8")))


def _render_markdown(
    meeting: Meeting,
    segments: list[TranscriptSegment],
    actions: list[ActionItem],
    decisions: list[Decision],
) -> str:
    started = meeting.started_at.isoformat() if meeting.started_at else ""
    ended = meeting.ended_at.isoformat() if meeting.ended_at else ""
    duration_min = round(meeting.duration_seconds / 60, 1) if meeting.duration_seconds else 0
    owners = sorted({a.owner for a in actions if a.owner})

    fm = [
        "---",
        f"meeting_id: {meeting.id}",
        f'title: "{meeting.title or meeting.id}"',
        f"date: {started[:10] if started else ''}",
        f"started_at: {started}",
        f"ended_at: {ended}",
        f"duration_minutes: {duration_min}",
        f"action_count: {len(actions)}",
        f"decision_count: {len(decisions)}",
        f"owners: [{', '.join(owners)}]",
        "tags:",
        "  - meetmind",
        "---",
    ]

    out = ["\n".join(fm), "", f"# {meeting.title or meeting.id}", ""]

    if decisions:
        out.append("## Decisions")
        for d in decisions:
            out.append(f"- **{d.decision}**")
            if d.rationale:
                out.append(f"  - Rationale: {d.rationale}")
        out.append("")

    if actions:
        out.append("## Action items")
        for a in actions:
            owner = a.owner or "unassigned"
            due = a.due or ""
            checkbox = "[x]" if a.status == "done" else "[ ]"
            line = f"- {checkbox} **{a.description}** — {owner}"
            if due:
                line += f" (due: {due})"
            out.append(line)
            if a.evidence_quote:
                out.append(f"  - Evidence: _{a.evidence_quote}_")
        out.append("")

    if segments:
        out.append("## Transcript")
        for s in segments:
            speaker = s.speaker_id or s.speaker or "speaker"
            ts = _ms_to_ts(s.start_ms)
            out.append(f"**[{ts}] {speaker}:** {s.text}")
            out.append("")

    return "\n".join(out) + "\n"


def _ms_to_ts(ms: int) -> str:
    sec = ms // 1000
    return f"{sec // 60:02d}:{sec % 60:02d}"

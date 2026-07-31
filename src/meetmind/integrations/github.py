"""GitHub Issues integration via the `gh` CLI.

Shells out to the user's authenticated `gh` install rather than holding its
own credentials, which avoids an OAuth flow or a PAT secret store entirely.
A missing or unauthenticated `gh` surfaces an error rather than a silent noop.

Module is a leaf: memory and models only.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol

from meetmind.models import ActionItem


class _StoreLike(Protocol):
    """Minimal store shape this module needs (see obsidian.py for rationale)."""

    def list_action_items(
        self, *, status: str | None = ..., meeting_id: str | None = ..., owner: str | None = ...
    ) -> list[ActionItem]: ...


@dataclass
class GhIssueRef:
    """Minimal record of a created issue."""

    number: int | None  # None when --dry-run
    url: str | None
    title: str
    repo: str


class GhCliMissingError(RuntimeError):
    """gh isn't on PATH."""


def _gh_path() -> str:
    path = shutil.which("gh")
    if not path:
        raise GhCliMissingError("`gh` CLI not found on PATH; install from https://cli.github.com/")
    return path


def export_action_items(
    store: _StoreLike,
    meeting_id: str,
    *,
    repo: str,
    label: str = "meetmind",
    dry_run: bool = False,
    gh_binary: str | None = None,
) -> list[GhIssueRef]:
    """Open one GitHub issue per open action item in the meeting.

    `repo` is "owner/name". Dedup against existing issues is left to the
    caller. Returns one ``GhIssueRef`` per item.
    """
    actions = store.list_action_items(meeting_id=meeting_id, status="open")
    if not actions:
        return []
    binary = gh_binary or _gh_path()
    out: list[GhIssueRef] = []
    for a in actions:
        title = a.description.strip().splitlines()[0][:80]
        body = _render_issue_body(a, meeting_id)
        if dry_run:
            out.append(GhIssueRef(number=None, url=None, title=title, repo=repo))
            continue
        result = subprocess.run(  # noqa: S603 — controlled, parameterized invocation
            [
                binary,
                "issue",
                "create",
                "--repo",
                repo,
                "--title",
                title,
                "--body",
                body,
                "--label",
                label,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        url = result.stdout.strip().splitlines()[-1] if result.stdout else None
        number = _extract_issue_number(url)
        out.append(GhIssueRef(number=number, url=url, title=title, repo=repo))
    return out


def _render_issue_body(item: ActionItem, meeting_id: str) -> str:
    lines = [
        item.description,
        "",
        "---",
        f"**Source meeting:** `{meeting_id}` (MeetMind)",
    ]
    if item.owner:
        lines.append(f"**Owner:** {item.owner}")
    if item.due:
        lines.append(f"**Due:** {item.due}")
    if item.evidence_quote:
        lines.append("")
        lines.append("**Evidence quote:**")
        lines.append(f"> {item.evidence_quote}")
    lines.append("")
    lines.append("_Created by `meetmind export-github`._")
    return "\n".join(lines)


def _extract_issue_number(url: str | None) -> int | None:
    if not url:
        return None
    try:
        return int(url.rstrip("/").rsplit("/", 1)[-1])
    except (ValueError, IndexError):
        return None

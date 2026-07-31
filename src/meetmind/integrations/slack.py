"""Slack integration via Incoming Webhooks.

No OAuth and no bot install: the user sets ``SLACK_WEBHOOK_URL`` and
``meetmind export slack <meeting_id>`` posts a Block Kit summary.

Egress: this is one of the few code paths that deliberately leaves the
device. It is unreachable unless a webhook URL is configured.

Module is a leaf per the import-linter contract: memory and models only,
via structural typing.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from meetmind.models import ActionItem, Decision, Meeting, TranscriptSegment

log = logging.getLogger(__name__)

_ENV = "SLACK_WEBHOOK_URL"
_TIMEOUT_S = 10.0


class _StoreLike(Protocol):
    """Minimal store shape this module needs (see obsidian.py for rationale)."""

    def get_meeting(self, meeting_id: str) -> Meeting | None: ...
    def list_segments(self, meeting_id: str) -> list[TranscriptSegment]: ...
    def list_action_items(
        self, *, status: str | None = ..., meeting_id: str | None = ..., owner: str | None = ...
    ) -> list[ActionItem]: ...
    def list_decisions(self, meeting_id: str) -> list[Decision]: ...
    def get_summary(self, meeting_id: str) -> dict | None: ...


@dataclass
class SlackPostResult:
    """Outcome of a webhook POST.

    Slack returns plain ``ok`` text on success; on failure the body
    explains why (invalid_payload, channel_is_archived, etc).
    """

    ok: bool
    status: int
    response: str
    payload_size: int


def export_meeting_to_slack(
    store: _StoreLike,
    meeting_id: str,
    *,
    webhook_url: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Post a meeting digest to a Slack channel via Incoming Webhook.

    Parameters
    ----------
    store
        Anything that satisfies the ``_StoreLike`` protocol.
    meeting_id
        ULID of the meeting to export.
    webhook_url
        Override env. When neither is set, returns ``{"ok": False,
        "error": "no webhook"}`` rather than raising — this is the
        common "user forgot to set it" path and the CLI should print
        a clear message.
    dry_run
        Skip the network call; render the payload only. Useful for tests
        and for letting users preview what they'd send.
    """
    url = webhook_url or os.environ.get(_ENV)
    if not url and not dry_run:
        return {
            "ok": False,
            "error": (
                f"No webhook configured. Set {_ENV} or pass --webhook-url. "
                "Create one at https://api.slack.com/messaging/webhooks."
            ),
        }
    meeting = store.get_meeting(meeting_id)
    if meeting is None:
        return {"ok": False, "error": f"meeting not found: {meeting_id}"}
    payload = _build_payload(meeting, store)
    body = json.dumps(payload).encode("utf-8")
    if dry_run:
        return {"ok": True, "dry_run": True, "payload": payload, "bytes": len(body)}
    result = _post(url, body)
    return {
        "ok": result.ok,
        "status": result.status,
        "response": result.response,
        "bytes": result.payload_size,
    }


def _build_payload(meeting: Meeting, store: _StoreLike) -> dict[str, Any]:
    """Block Kit payload — header, TL;DR, decisions, open actions."""
    summary = store.get_summary(meeting.id)
    decisions = store.list_decisions(meeting.id)
    open_actions = [a for a in store.list_action_items(meeting_id=meeting.id) if a.status == "open"]

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": meeting.title or "MeetMind digest"},
        }
    ]
    when = (meeting.started_at or meeting.created_at).isoformat() if meeting.created_at else "—"
    blocks.append(
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"_Recorded {when} · `{meeting.id}`_"}],
        }
    )

    if summary and summary.get("tl_dr"):
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary["tl_dr"]}})
    if decisions:
        bullets = "\n".join(f"• {d.decision}" for d in decisions[:10])
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Decisions*\n{bullets}"}}
        )
    if open_actions:
        bullets = "\n".join(
            f"• {a.description}" + (f" — @{a.owner}" if a.owner else "") for a in open_actions[:10]
        )
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Open actions*\n{bullets}"}}
        )

    # Slack still wants a plain-text fallback for notifications + a11y.
    fallback = f"MeetMind digest: {meeting.title or meeting.id}"
    return {"text": fallback, "blocks": blocks}


def _post(url: str, body: bytes) -> SlackPostResult:
    """POST the payload. Returns success/failure structurally — never raises.

    Stdlib urllib rather than httpx, to keep this leaf package dependency-free.
    """
    req = urllib.request.Request(  # noqa: S310 — URL is user-supplied webhook
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            text = resp.read().decode("utf-8", errors="replace")
            return SlackPostResult(
                ok=200 <= resp.status < 300,
                status=resp.status,
                response=text,
                payload_size=len(body),
            )
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return SlackPostResult(ok=False, status=e.code, response=text, payload_size=len(body))
    except urllib.error.URLError as e:
        log.warning("Slack webhook unreachable: %s", e)
        return SlackPostResult(ok=False, status=0, response=str(e), payload_size=len(body))

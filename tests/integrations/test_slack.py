"""Slack exporter tests.

No real network calls: every test uses ``dry_run=True`` or monkeypatches
``urllib.request.urlopen``.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, datetime
from typing import Any

import pytest

from meetmind.integrations.slack import _build_payload, export_meeting_to_slack
from meetmind.memory.store import Store
from meetmind.models import Meeting


@pytest.fixture
def store_with_meeting(tmp_path):
    s = Store.open(tmp_path / "slack.db", use_keychain=False)
    m = Meeting(
        id="01HXXXXXXXXXXXXXXXXXXXXXXX",
        title="Q2 planning",
        created_at=datetime(2026, 5, 12, 10, 0, tzinfo=UTC),
        started_at=datetime(2026, 5, 12, 10, 0, tzinfo=UTC),
    )
    s.upsert_meeting(m)
    s.upsert_summary(
        m.id, tl_dr="We agreed to ship v1.0 in June.", topics=["v1.0", "scope"], model="gemma4"
    )
    s.conn.execute(
        "INSERT INTO decisions (id, meeting_id, decision, rationale) "
        "VALUES ('d1', ?, 'Ship v1.0 in June', 'No new launch-blockers')",
        (m.id,),
    )
    s.conn.execute(
        "INSERT INTO action_items (id, meeting_id, description, status) "
        "VALUES ('a1', ?, 'Finalise homebrew tap', 'open')",
        (m.id,),
    )
    yield s, m.id
    s.close()


def test_dry_run_builds_payload_without_network(store_with_meeting) -> None:
    store, mid = store_with_meeting
    result = export_meeting_to_slack(store, mid, dry_run=True)
    assert result["ok"] is True
    payload = result["payload"]
    assert payload["text"].startswith("MeetMind digest")
    blocks = payload["blocks"]
    # header + context + tl_dr + decisions + actions = 5 blocks
    assert len(blocks) == 5
    assert blocks[0]["type"] == "header"
    assert "Q2 planning" in blocks[0]["text"]["text"]
    section_texts = [b["text"]["text"] for b in blocks if b["type"] == "section"]
    assert any("ship v1.0" in t.lower() for t in section_texts)
    assert any("Decisions" in t for t in section_texts)
    assert any("Finalise homebrew" in t for t in section_texts)


def test_missing_webhook_returns_structured_error(store_with_meeting, monkeypatch) -> None:
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    store, mid = store_with_meeting
    result = export_meeting_to_slack(store, mid)
    assert result["ok"] is False
    assert "SLACK_WEBHOOK_URL" in result["error"]
    assert "https://api.slack.com" in result["error"]


def test_unknown_meeting_returns_structured_error(store_with_meeting) -> None:
    store, _ = store_with_meeting
    result = export_meeting_to_slack(store, "01HZZZZZZZZZZZZZZZZZZZZZZZ", dry_run=True)
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_post_success_path(store_with_meeting, monkeypatch) -> None:
    """A 200 OK from Slack is surfaced as a successful result."""
    store, mid = store_with_meeting
    captured: dict[str, Any] = {}

    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"ok"

    def _fake_urlopen(req, timeout):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["method"] = req.get_method()
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    result = export_meeting_to_slack(store, mid, webhook_url="https://hooks.example/T00/B00/abc")
    assert result["ok"] is True
    assert result["status"] == 200
    assert captured["url"] == "https://hooks.example/T00/B00/abc"
    assert captured["method"] == "POST"
    payload = json.loads(captured["body"].decode("utf-8"))
    assert "blocks" in payload and "text" in payload


def test_post_http_error_is_surfaced(store_with_meeting, monkeypatch) -> None:
    store, mid = store_with_meeting

    def _raise_http_error(req, timeout):  # noqa: ARG001
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", hdrs={}, fp=io.BytesIO(b"invalid_payload")
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise_http_error)
    result = export_meeting_to_slack(store, mid, webhook_url="https://hooks.example/T00/B00/abc")
    assert result["ok"] is False
    assert result["status"] == 400
    assert "invalid_payload" in result["response"]


def test_build_payload_truncates_long_lists(tmp_path) -> None:
    """Sanity: building a payload from a meeting with 20 actions caps at 10."""
    s = Store.open(tmp_path / "x.db", use_keychain=False)
    m = Meeting(
        id="01HABCDEF01HABCDEF01HABCDE",
        title="Big meeting",
        created_at=datetime(2026, 5, 12, tzinfo=UTC),
    )
    s.upsert_meeting(m)
    for i in range(20):
        s.conn.execute(
            "INSERT INTO action_items (id, meeting_id, description, status) "
            "VALUES (?, ?, ?, 'open')",
            (f"a{i:02d}", m.id, f"Action {i}"),
        )
    payload = _build_payload(m, s)
    actions_block = next(
        b
        for b in payload["blocks"]
        if b["type"] == "section" and "Open actions" in b["text"]["text"]
    )
    # 10 bullets max
    assert actions_block["text"]["text"].count("•") == 10
    s.close()

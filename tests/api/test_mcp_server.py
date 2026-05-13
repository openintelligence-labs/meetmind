"""Tests for the MCP server (tool catalog + JSON-RPC dispatch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meetmind.api.mcp_server import JsonRpcServer, StoreContext, call_tool, get_tool, list_tools
from meetmind.memory.store import Store
from meetmind.memory.vector import HybridIndex, IndexedSegment, hash_embedder
from meetmind.models import ActionItem, ChannelKind, Decision, Meeting, TranscriptSegment


@pytest.fixture
def ctx(tmp_path: Path) -> StoreContext:
    store = Store.open(tmp_path / "store.db")
    index = HybridIndex.open(tmp_path / "vec", vector_dim=64, embedder=hash_embedder(64))

    m1 = Meeting(id="01M1", title="Snowflake migration kick-off")
    m2 = Meeting(id="01M2", title="Weekly standup")
    store.upsert_meeting(m1)
    store.upsert_meeting(m2)

    seg1 = TranscriptSegment(
        start_seconds=0.0,
        end_seconds=4.0,
        text="Sam mentioned the Snowflake migration timeline is Friday",
        channel=ChannelKind.LOOPBACK,
        speaker_id="remote-A",
    )
    seg2 = TranscriptSegment(
        start_seconds=4.0,
        end_seconds=8.0,
        text="Priya proposed adopting LanceDB for the vector store",
        channel=ChannelKind.LOOPBACK,
        speaker_id="remote-B",
    )
    sid1 = store.append_segment(m1.id, seg1)
    sid2 = store.append_segment(m1.id, seg2)
    store.append_segment(
        m2.id, TranscriptSegment(start_seconds=0.0, end_seconds=2.0, text="Routine standup notes")
    )

    index.add(
        [
            IndexedSegment(
                meeting_id=m1.id,
                segment_id=sid1,
                text=seg1.text,
                start_ms=seg1.start_ms,
                end_ms=seg1.end_ms,
                cluster_id="remote-A",
                channel="loopback",
                language="en",
            ),
            IndexedSegment(
                meeting_id=m1.id,
                segment_id=sid2,
                text=seg2.text,
                start_ms=seg2.start_ms,
                end_ms=seg2.end_ms,
                cluster_id="remote-B",
                channel="loopback",
                language="en",
            ),
        ]
    )

    store.upsert_action_item(
        m1.id,
        ActionItem(
            description="Send the migration deck",
            owner="remote-A",
            evidence_quote="I'll send the deck Friday",
            status="open",
        ),
    )
    store.upsert_decision(
        m1.id,
        Decision(decision="Adopt LanceDB", rationale="Better at 1M+ vectors"),
    )
    return StoreContext(store=store, index=index)


def test_catalog_has_15_tools():
    names = {t.name for t in list_tools()}
    expected = {
        "search_meetings",
        "get_meeting",
        "who_said",
        "list_action_items",
        "get_decisions",
        "find_unanswered_questions",
        "get_speaker_history",
        "summarize_period",
        "extract_quotes",
        "get_attendees",
        "attendee_stats",
        "link_to_moment",
        "compare_meetings",
        "get_followups",
        "export_to",
    }
    assert expected.issubset(names)
    assert get_tool("search_meetings") is not None
    assert get_tool("definitely-not-a-tool") is None


async def test_search_meetings_returns_relevant_hits(ctx: StoreContext):
    out = await call_tool(ctx, "search_meetings", {"query": "snowflake migration"})
    assert len(out["hits"]) >= 1
    top = out["hits"][0]
    assert "Snowflake" in top["text"]


async def test_get_meeting_includes_transcript(ctx: StoreContext):
    out = await call_tool(ctx, "get_meeting", {"meeting_id": "01M1"})
    assert out["meeting"]["title"] == "Snowflake migration kick-off"
    assert len(out["transcript"]) == 2


async def test_list_action_items_filters_by_meeting(ctx: StoreContext):
    out = await call_tool(ctx, "list_action_items", {"meeting_id": "01M1"})
    assert len(out["items"]) == 1
    assert out["items"][0]["description"] == "Send the migration deck"


async def test_get_decisions_lists_them(ctx: StoreContext):
    out = await call_tool(ctx, "get_decisions", {"meeting_id": "01M1"})
    assert len(out["decisions"]) == 1
    assert out["decisions"][0]["decision"] == "Adopt LanceDB"


async def test_get_attendees_aggregates_talk_time(ctx: StoreContext):
    out = await call_tool(ctx, "get_attendees", {"meeting_id": "01M1"})
    speakers = {a["speaker"] for a in out["attendees"]}
    assert "remote-A" in speakers
    assert "remote-B" in speakers


async def test_extract_quotes_filters_by_topic(ctx: StoreContext):
    out = await call_tool(ctx, "extract_quotes", {"meeting_id": "01M1", "topic": "lancedb"})
    assert len(out["quotes"]) == 1
    assert "LanceDB" in out["quotes"][0]["text"]


async def test_link_to_moment_returns_deep_link(ctx: StoreContext):
    out = await call_tool(ctx, "link_to_moment", {"meeting_id": "01M1", "timestamp_ms": 5000})
    assert "meetmind://meeting/01M1" in out["uri"]
    assert "t=5000" in out["deep_link"]


async def test_compare_meetings_returns_summary_per_id(ctx: StoreContext):
    out = await call_tool(ctx, "compare_meetings", {"meeting_ids": ["01M1", "01M2"]})
    assert len(out["comparison"]) == 2


async def test_unknown_tool_raises_keyerror(ctx: StoreContext):
    with pytest.raises(KeyError):
        await call_tool(ctx, "nonexistent_tool", {})


async def test_jsonrpc_tools_list(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
    obj = json.loads(out)
    assert obj["jsonrpc"] == "2.0"
    assert obj["id"] == 1
    names = [t["name"] for t in obj["result"]["tools"]]
    assert "search_meetings" in names


async def test_jsonrpc_tools_call(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "list_action_items",
                    "arguments": {"meeting_id": "01M1"},
                },
            }
        )
    )
    obj = json.loads(out)
    assert obj["id"] == 7
    payload = obj["result"]["content"][0]["data"]
    assert len(payload["items"]) == 1


async def test_jsonrpc_unknown_method_returns_error(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 9, "method": "nope"}))
    obj = json.loads(out)
    assert "error" in obj
    assert obj["error"]["code"] == -32601


async def test_jsonrpc_parse_error(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line("{not json")
    obj = json.loads(out)
    assert obj["error"]["code"] == -32700


# ─────────────────── de-stubbed tools (v0.21) ───────────────────


async def test_get_decisions_without_meeting_id_returns_cross_meeting(ctx: StoreContext):
    """Was a stub; should return up to `limit` recent decisions across meetings."""
    out = await call_tool(ctx, "get_decisions", {})
    assert "warning" not in out, out
    assert isinstance(out["decisions"], list)
    assert len(out["decisions"]) >= 1
    assert all("meeting_id" in d for d in out["decisions"])


async def test_find_unanswered_questions_substring_heuristic(ctx: StoreContext, tmp_path):
    """A question segment with no later same-speaker-different reply within 30s
    should surface as unanswered."""
    # Add a question with no follow-up.
    from meetmind.models import TranscriptSegment

    ctx.store.append_segment(
        "01M1",
        TranscriptSegment(
            start_seconds=20.0,
            end_seconds=22.0,
            text="Have we sized the snowflake bill against the q2 budget yet?",
            speaker_id="remote-A",
        ),
    )
    out = await call_tool(ctx, "find_unanswered_questions", {"meeting_id": "01M1"})
    assert "warning" not in out
    qs = out["questions"]
    assert any("snowflake bill" in q["text"].lower() for q in qs)


async def test_find_unanswered_questions_requires_meeting_id(ctx: StoreContext):
    out = await call_tool(ctx, "find_unanswered_questions", {})
    assert out["questions"] == []
    assert "meeting_id required" in out["warning"]


async def test_summarize_period_pulls_persisted_summaries(ctx: StoreContext):
    """Was a stub; should now aggregate per-meeting summaries."""
    ctx.store.upsert_summary("01M1", tl_dr="Migration deck ships Friday", topics=["snowflake"])
    out = await call_tool(ctx, "summarize_period", {})
    assert "warning" not in out
    assert out["count"] >= 1
    matching = [p for p in out["period"] if p["meeting_id"] == "01M1"]
    assert matching and matching[0]["tl_dr"] == "Migration deck ships Friday"
    assert "Adopt LanceDB" in matching[0]["decisions"]


async def test_export_to_unknown_target_lists_supported(ctx: StoreContext):
    out = await call_tool(ctx, "export_to", {"target": "salesforce", "meeting_id": "01M1"})
    assert out["ok"] is False
    assert "salesforce" in out["error"]
    assert set(out["supported"]) == {"obsidian", "github", "slack"}


async def test_export_to_slack_dry_run(ctx: StoreContext, monkeypatch):
    """Wire-up check: export_to dispatches to the slack exporter."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example/T/B/C")

    captured: dict = {}

    def _fake_urlopen(req, timeout):  # noqa: ARG001
        class _R:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"ok"

        captured["url"] = req.full_url
        return _R()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    out = await call_tool(ctx, "export_to", {"target": "slack", "meeting_id": "01M1"})
    assert out["ok"] is True
    assert out["target"] == "slack"
    assert captured["url"] == "https://hooks.example/T/B/C"


async def test_export_to_obsidian_writes_note(ctx: StoreContext, tmp_path):
    out = await call_tool(
        ctx,
        "export_to",
        {"target": "obsidian", "meeting_id": "01M1", "vault_path": str(tmp_path)},
    )
    assert out["ok"] is True
    assert out["target"] == "obsidian"
    assert (tmp_path / "MeetMind").exists()
    note = Path(out["path"])
    assert note.exists()
    assert "Snowflake" in note.read_text()

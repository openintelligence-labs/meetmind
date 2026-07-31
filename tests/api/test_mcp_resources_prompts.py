"""Tests for MCP `resources/*` and `prompts/*` primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meetmind.api.mcp_prompts import (
    PromptDescriptor,
    get_prompt,
    list_prompts,
    render_prompt,
)
from meetmind.api.mcp_resources import (
    ResourceContent,
    list_resources,
    read_resource,
)
from meetmind.api.mcp_server import JsonRpcServer, StoreContext
from meetmind.memory.store import Store
from meetmind.memory.vector import HybridIndex, hash_embedder
from meetmind.models import (
    ActionItem,
    ChannelKind,
    Decision,
    Meeting,
    Summary,
    TranscriptSegment,
)


@pytest.fixture
def populated_store(tmp_path: Path) -> Store:
    store = Store.open(tmp_path / "store.db")
    m1 = Meeting(
        id="01M1",
        title="Snowflake migration kick-off",
        summary=Summary(tl_dr="Decided to migrate to Snowflake by EOQ."),
    )
    m2 = Meeting(id="01M2", title="Weekly standup")
    store.upsert_meeting(m1)
    store.upsert_meeting(m2)

    store.append_segment(
        m1.id,
        TranscriptSegment(
            start_seconds=0.0,
            end_seconds=4.0,
            text="Sam mentioned the Snowflake migration timeline is Friday",
            channel=ChannelKind.LOOPBACK,
            speaker_id="remote-A",
        ),
    )
    store.append_segment(
        m1.id,
        TranscriptSegment(
            start_seconds=4.0,
            end_seconds=8.0,
            text="Priya proposed adopting LanceDB for the vector store",
            channel=ChannelKind.LOOPBACK,
            speaker_id="remote-B",
        ),
    )
    store.upsert_action_item(
        m1.id,
        ActionItem(
            description="Send the migration deck",
            owner="remote-A",
            due="2026-05-08",
            evidence_quote="I'll send the deck Friday",
            status="open",
        ),
    )
    store.upsert_decision(
        m1.id,
        Decision(decision="Adopt LanceDB", rationale="Better at 1M+ vectors"),
    )
    return store


@pytest.fixture
def ctx(populated_store: Store, tmp_path: Path) -> StoreContext:
    index = HybridIndex.open(tmp_path / "vec", vector_dim=64, embedder=hash_embedder(64))
    return StoreContext(store=populated_store, index=index)


def test_list_resources_starts_with_meetings_index(populated_store: Store):
    resources = list_resources(populated_store)
    assert resources[0].uri == "meetmind://meetings"
    assert resources[0].mime_type == "application/json"
    uris = {r.uri for r in resources}
    assert "meetmind://meeting/01M1" in uris
    assert "meetmind://meeting/01M2" in uris


def test_read_meetings_index_lists_subresource_uris(populated_store: Store):
    content = read_resource(populated_store, "meetmind://meetings")
    body = json.loads(content.text)
    assert body["count"] == 2
    first = body["meetings"][0]
    assert first["subresources"]["transcript"].endswith("/transcript")
    assert first["subresources"]["summary"].endswith("/summary")
    assert first["subresources"]["decisions"].endswith("/decisions")
    assert first["subresources"]["actions"].endswith("/actions")


def test_read_meeting_record_returns_counts(populated_store: Store):
    content = read_resource(populated_store, "meetmind://meeting/01M1")
    body = json.loads(content.text)
    assert body["title"] == "Snowflake migration kick-off"
    assert body["segment_count"] == 2
    assert body["action_count"] == 1
    assert body["decision_count"] == 1


def test_read_transcript_renders_markdown(populated_store: Store):
    content = read_resource(populated_store, "meetmind://meeting/01M1/transcript")
    assert content.mime_type == "text/markdown"
    assert "# Transcript" in content.text
    assert "Snowflake migration timeline" in content.text
    assert "remote-A" in content.text


def test_read_summary_handles_missing_summary(populated_store: Store):
    # `Meeting.summary` is not persisted in the schema; the handler falls back.
    content = read_resource(populated_store, "meetmind://meeting/01M1/summary")
    assert "_No summary generated yet._" in content.text
    content2 = read_resource(populated_store, "meetmind://meeting/01M2/summary")
    assert "_No summary generated yet._" in content2.text


def test_read_decisions_renders_markdown(populated_store: Store):
    content = read_resource(populated_store, "meetmind://meeting/01M1/decisions")
    assert "Adopt LanceDB" in content.text
    assert "Rationale" in content.text


def test_read_actions_renders_markdown(populated_store: Store):
    content = read_resource(populated_store, "meetmind://meeting/01M1/actions")
    assert "Send the migration deck" in content.text
    assert "remote-A" in content.text
    assert "[open]" in content.text
    assert "2026-05-08" in content.text


def test_read_unknown_uri_raises_keyerror(populated_store: Store):
    with pytest.raises(KeyError):
        read_resource(populated_store, "https://evil.example.com/")


def test_read_missing_meeting_raises_lookup(populated_store: Store):
    with pytest.raises(LookupError):
        read_resource(populated_store, "meetmind://meeting/01XNOPE")


def test_list_prompts_returns_curated_set():
    names = {p.name for p in list_prompts()}
    assert {
        "daily_digest",
        "what_changed",
        "prep_for_meeting",
        "follow_up_draft",
        "review_my_talk_time",
    }.issubset(names)


def test_get_prompt_unknown_returns_none():
    assert get_prompt("definitely_not_a_prompt") is None


def test_render_prompt_substitutes_args():
    text = render_prompt("what_changed", {"meeting_a_id": "01M1", "meeting_b_id": "01M2"})
    assert "meetmind://meeting/01M1" in text
    assert "meetmind://meeting/01M2" in text


def test_render_prompt_missing_required_arg_raises():
    with pytest.raises(KeyError):
        render_prompt("what_changed", {"meeting_a_id": "01M1"})


def test_render_prompt_optional_arg_defaults_to_empty():
    # `date` is optional; should not raise even when omitted.
    text = render_prompt("daily_digest", {})
    assert "MeetMind MCP server" in text


def test_prompt_descriptor_to_wire_shape():
    p = get_prompt("follow_up_draft")
    assert p is not None
    wire = p.to_wire()
    assert wire["name"] == "follow_up_draft"
    arg_names = {a["name"] for a in wire["arguments"]}
    assert "meeting_id" in arg_names
    required = {a["name"] for a in wire["arguments"] if a["required"]}
    assert required == {"meeting_id"}


async def test_jsonrpc_resources_list(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})
    )
    obj = json.loads(out)
    uris = [r["uri"] for r in obj["result"]["resources"]]
    assert "meetmind://meetings" in uris
    assert any(u.startswith("meetmind://meeting/01M1") for u in uris)


async def test_jsonrpc_resources_read(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {"uri": "meetmind://meeting/01M1/transcript"},
            }
        )
    )
    obj = json.loads(out)
    contents = obj["result"]["contents"]
    assert len(contents) == 1
    assert contents[0]["mimeType"] == "text/markdown"
    assert "Snowflake" in contents[0]["text"]


async def test_jsonrpc_resources_read_missing_uri(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {}})
    )
    obj = json.loads(out)
    assert obj["error"]["code"] == -32602


async def test_jsonrpc_resources_read_unknown_meeting(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": {"uri": "meetmind://meeting/01XNOPE"},
            }
        )
    )
    obj = json.loads(out)
    assert obj["error"]["code"] == -32001


async def test_jsonrpc_prompts_list(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps({"jsonrpc": "2.0", "id": 5, "method": "prompts/list"})
    )
    obj = json.loads(out)
    names = [p["name"] for p in obj["result"]["prompts"]]
    assert "daily_digest" in names


async def test_jsonrpc_prompts_get_renders_template(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "prompts/get",
                "params": {
                    "name": "what_changed",
                    "arguments": {"meeting_a_id": "01M1", "meeting_b_id": "01M2"},
                },
            }
        )
    )
    obj = json.loads(out)
    msg = obj["result"]["messages"][0]
    assert msg["role"] == "user"
    assert "01M1" in msg["content"]["text"]
    assert "01M2" in msg["content"]["text"]


async def test_jsonrpc_prompts_get_unknown(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "prompts/get",
                "params": {"name": "nope"},
            }
        )
    )
    obj = json.loads(out)
    assert obj["error"]["code"] == -32601


async def test_jsonrpc_prompts_get_missing_required(ctx: StoreContext):
    server = JsonRpcServer(ctx=ctx)
    out = await server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "prompts/get",
                "params": {"name": "what_changed", "arguments": {"meeting_a_id": "01M1"}},
            }
        )
    )
    obj = json.loads(out)
    assert obj["error"]["code"] == -32602


def test_resource_content_wire_shape():
    rc = ResourceContent(uri="x", mime_type="text/plain", text="hi")
    assert rc.to_wire() == {"uri": "x", "mimeType": "text/plain", "text": "hi"}


def test_prompt_descriptor_render_helper():
    # Smoke-test the dataclass directly (not just via render_prompt).
    p: PromptDescriptor | None = get_prompt("daily_digest")
    assert p is not None
    text = p.render({"date": "2026-05-06"})
    assert "2026-05-06" in text


def test_list_resources_includes_people_index(populated_store: Store):
    uris = {r.uri for r in list_resources(populated_store)}
    assert "meetmind://people" in uris


def test_read_people_index_returns_speakers(populated_store: Store):
    from meetmind.models import Speaker

    populated_store.upsert_speaker(Speaker(id="01ALICE", display_name="Alice"))
    populated_store.upsert_speaker(Speaker(id="01BOB", display_name="Bob"))

    content = read_resource(populated_store, "meetmind://people")
    body = json.loads(content.text)
    assert body["count"] == 2
    names = {p["display_name"] for p in body["people"]}
    assert names == {"Alice", "Bob"}


def test_read_person_profile_aggregates_talk_time(populated_store: Store):
    from meetmind.models import Speaker

    populated_store.upsert_speaker(Speaker(id="remote-A", display_name="Sam"))
    content = read_resource(populated_store, "meetmind://person/remote-A/profile")
    body = json.loads(content.text)
    assert body["display_name"] == "Sam"
    assert body["segments_seen"] >= 1
    assert body["talk_time_seconds"] > 0


def test_read_person_profile_unknown_raises(populated_store: Store):
    with pytest.raises(LookupError):
        read_resource(populated_store, "meetmind://person/nope/profile")


def test_new_prompts_present():
    names = {p.name for p in list_prompts()}
    assert "weekly_review" in names
    assert "quarterly_summary" in names


def test_quarterly_summary_requires_quarter():
    with pytest.raises(KeyError):
        render_prompt("quarterly_summary", {})
    text = render_prompt("quarterly_summary", {"quarter": "Q1 2026"})
    assert "Q1 2026" in text

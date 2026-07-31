"""MCP stdio server: JSON-RPC 2.0 over newline-delimited stdin/stdout.

Tools are backed by the SQLCipher store and the LanceDB hybrid index. Each is
an async function `(StoreContext, args) -> dict` registered in `_TOOL_TABLE`,
so tools can be exercised without running the JSON-RPC loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from meetmind.api.mcp_prompts import get_prompt, list_prompts, render_prompt
from meetmind.api.mcp_resources import list_resources, read_resource
from meetmind.memory.store import Store
from meetmind.memory.vector import HybridIndex

log = logging.getLogger(__name__)


@dataclass
class StoreContext:
    """Bundle of backends each tool may need."""

    store: Store
    index: HybridIndex | None = None


ToolFn = Callable[[StoreContext, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class ToolDescriptor:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolFn


async def _search_meetings(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query", "").strip()
    if not query:
        return {"hits": []}
    if ctx.index is None:
        return {"hits": [], "warning": "vector index not configured"}
    limit = int(args.get("limit", 10))
    meeting_id = args.get("meeting_id")
    hits = ctx.index.search(query, limit=limit, meeting_id=meeting_id)
    return {
        "hits": [
            {
                "meeting_id": h.segment.meeting_id,
                "segment_id": h.segment.segment_id,
                "text": h.segment.text,
                "start_ms": h.segment.start_ms,
                "end_ms": h.segment.end_ms,
                "cluster_id": h.segment.cluster_id,
                "score": h.score,
            }
            for h in hits
        ]
    }


async def _get_meeting(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    mid = args["meeting_id"]
    m = ctx.store.get_meeting(mid)
    if m is None:
        return {"meeting": None}
    include = set(args.get("include", ["transcript", "summary", "actions"]))
    out: dict[str, Any] = {"meeting": _meeting_to_dict(m)}
    if "transcript" in include:
        out["transcript"] = [_segment_to_dict(s) for s in ctx.store.list_segments(mid)]
    if "actions" in include:
        out["actions"] = [_action_to_dict(a) for a in ctx.store.list_action_items(meeting_id=mid)]
    if "summary" in include and m.summary is not None:
        out["summary"] = m.summary.model_dump(mode="json")
    return out


async def _who_said(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    """Approximate: searches the index for the quote, returns matching segments."""
    quote = args.get("quote_or_topic", "").strip()
    if not quote or ctx.index is None:
        return {"hits": []}
    hits = ctx.index.search(quote, limit=int(args.get("limit", 5)))
    return {
        "hits": [
            {
                "meeting_id": h.segment.meeting_id,
                "speaker": h.segment.cluster_id,
                "text": h.segment.text,
                "start_ms": h.segment.start_ms,
                "end_ms": h.segment.end_ms,
                "score": h.score,
            }
            for h in hits
        ]
    }


async def _list_action_items(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    items = ctx.store.list_action_items(
        status=args.get("status"),
        meeting_id=args.get("meeting_id"),
        owner=args.get("owner"),
    )
    return {"items": [_action_to_dict(a) for a in items]}


async def _get_decisions(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    mid = args.get("meeting_id")
    if mid:
        decisions = ctx.store.list_decisions(mid)
        return {"decisions": [d.model_dump(mode="json") for d in decisions]}
    # No meeting_id → return the most recent cross-meeting decisions.
    limit = int(args.get("limit", 50))
    pairs = ctx.store.list_all_decisions(limit=limit)
    return {
        "decisions": [
            {**d.model_dump(mode="json"), "meeting_id": meeting_id} for meeting_id, d in pairs
        ]
    }


async def _find_unanswered_questions(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    """Return question segments with no reply from another speaker in ``window_ms``.

    Purely heuristic (no LLM), so callers should treat the result as candidates.
    """
    mid = args.get("meeting_id")
    if not mid:
        return {"questions": [], "warning": "meeting_id required"}
    window_ms = int(args.get("window_ms", 30_000))
    segments = ctx.store.list_segments(mid)
    questions: list[dict[str, Any]] = []
    for i, s in enumerate(segments):
        text = (s.text or "").strip()
        if not text.endswith("?") or len(text) < 8:
            continue
        speaker_key = s.speaker_id or s.speaker
        cutoff = int(s.end_seconds * 1000) + window_ms
        answered = False
        for s2 in segments[i + 1 :]:
            if int(s2.start_seconds * 1000) > cutoff:
                break
            other_speaker = (s2.speaker_id or s2.speaker) != speaker_key
            if other_speaker and len((s2.text or "").strip()) > 10:
                answered = True
                break
        if not answered:
            questions.append(_segment_to_dict(s))
    return {"questions": questions, "meeting_id": mid, "window_ms": window_ms}


async def _get_speaker_history(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    person = args.get("person", "").strip()
    if not person or ctx.index is None:
        return {"hits": []}
    topic = args.get("topic", "").strip()
    query = f"{topic} {person}".strip()
    hits = ctx.index.search(query or person, limit=int(args.get("limit", 10)))
    person_lower = person.lower()
    relevant = [
        h for h in hits if h.segment.cluster_id and person_lower in h.segment.cluster_id.lower()
    ] or hits
    return {
        "hits": [
            {
                "meeting_id": h.segment.meeting_id,
                "text": h.segment.text,
                "start_ms": h.segment.start_ms,
                "end_ms": h.segment.end_ms,
                "score": h.score,
            }
            for h in relevant
        ]
    }


async def _summarize_period(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    """Aggregate already-persisted summaries and decisions across a date window.

    Never re-invokes the LLM; only stored per-meeting TL;DRs are returned.
    """
    from datetime import datetime as _dt  # noqa: PLC0415

    start = args.get("start")  # ISO date, inclusive
    end = args.get("end")  # ISO date, exclusive
    meetings = ctx.store.list_meetings(limit=int(args.get("limit", 200)))

    def _in_window(m_started: Any) -> bool:
        if m_started is None:
            return True
        try:
            ts = m_started if isinstance(m_started, _dt) else _dt.fromisoformat(str(m_started))
        except ValueError:
            return True
        if start and ts.isoformat() < start:
            return False
        return not (end and ts.isoformat() >= end)

    period: list[dict[str, Any]] = []
    for m in meetings:
        if not _in_window(m.started_at or m.created_at):
            continue
        summary = ctx.store.get_summary(m.id)
        decisions = ctx.store.list_decisions(m.id)
        actions = ctx.store.list_action_items(meeting_id=m.id)
        period.append(
            {
                "meeting_id": m.id,
                "title": m.title,
                "started_at": (m.started_at or m.created_at).isoformat()
                if (m.started_at or m.created_at)
                else None,
                "tl_dr": summary["tl_dr"] if summary else None,
                "topics": summary["topics"] if summary else [],
                "decisions": [d.decision for d in decisions],
                "open_actions": [a.description for a in actions if a.status == "open"],
            }
        )
    return {"period": period, "count": len(period), "start": start, "end": end}


async def _extract_quotes(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    mid = args["meeting_id"]
    speaker = args.get("speaker")
    topic = args.get("topic")
    segments = ctx.store.list_segments(mid)
    if speaker:
        segments = [s for s in segments if s.speaker_id == speaker or s.speaker == speaker]
    if topic:
        topic_lower = topic.lower()
        segments = [s for s in segments if topic_lower in s.text.lower()]
    return {"quotes": [_segment_to_dict(s) for s in segments]}


async def _get_attendees(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    mid = args["meeting_id"]
    segments = ctx.store.list_segments(mid)
    seen: dict[str, int] = {}
    for s in segments:
        key = s.speaker_id or s.speaker or "unknown"
        seen[key] = seen.get(key, 0) + (s.end_ms - s.start_ms)
    attendees = sorted(
        ({"speaker": k, "talk_time_ms": v} for k, v in seen.items()),
        key=lambda x: x["talk_time_ms"],
        reverse=True,
    )
    return {"attendees": attendees}


async def _attendee_stats(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    return await _get_attendees(ctx, args)


async def _link_to_moment(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    mid = args["meeting_id"]
    timestamp = int(args.get("timestamp_ms", 0))
    return {
        "uri": f"meetmind://meeting/{mid}/moment/{timestamp}",
        "deep_link": f"meetmind://meeting/{mid}#t={timestamp}",
    }


async def _compare_meetings(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    ids = args.get("meeting_ids") or []
    if len(ids) < 2:
        return {"comparison": [], "warning": "need ≥ 2 meeting_ids"}
    comparison = []
    for mid in ids:
        m = ctx.store.get_meeting(mid)
        if m is None:
            continue
        actions = ctx.store.list_action_items(meeting_id=mid)
        decisions = ctx.store.list_decisions(mid)
        comparison.append(
            {
                "meeting_id": mid,
                "title": m.title,
                "actions_count": len(actions),
                "decisions_count": len(decisions),
                "decisions": [d.decision for d in decisions],
            }
        )
    return {"comparison": comparison}


async def _get_followups(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    mid = args["meeting_id"]
    items = ctx.store.list_action_items(meeting_id=mid, status="follow_up_needed")
    return {"followups": [_action_to_dict(a) for a in items]}


async def _export_to(ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to an integration exporter: obsidian, github, or slack.

    Unsupported targets return a structured error rather than raising.
    """
    target = (args.get("target") or "").strip().lower()
    meeting_id = args.get("meeting_id")
    if not meeting_id:
        return {"ok": False, "error": "meeting_id required"}
    if target == "obsidian":
        from pathlib import Path as _P  # noqa: PLC0415

        from meetmind.integrations.obsidian import export_meeting  # noqa: PLC0415

        vault = args.get("vault_path")
        if not vault:
            return {"ok": False, "error": "vault_path required for obsidian"}
        res = export_meeting(ctx.store, meeting_id, vault=_P(vault), overwrite=True)
        return {"ok": True, "target": "obsidian", "path": str(res.note_path)}
    if target == "github":
        from meetmind.integrations.github import export_action_items  # noqa: PLC0415

        repo = args.get("repo")
        if not repo:
            return {"ok": False, "error": "repo (owner/name) required for github"}
        refs = export_action_items(
            ctx.store, meeting_id, repo=repo, dry_run=bool(args.get("dry_run", False))
        )
        return {
            "ok": True,
            "target": "github",
            "issues": [{"number": r.number, "url": r.url, "title": r.title} for r in refs],
        }
    if target == "slack":
        from meetmind.integrations.slack import export_meeting_to_slack  # noqa: PLC0415

        webhook = args.get("webhook_url")  # falls back to env in the exporter
        result = export_meeting_to_slack(ctx.store, meeting_id, webhook_url=webhook)
        return {"ok": True, "target": "slack", **result}
    return {
        "ok": False,
        "error": f"unknown target: {target!r}",
        "supported": ["obsidian", "github", "slack"],
        "note": "Notion/Linear are not yet implemented (require OAuth setup).",
    }


async def _start_recording(_ctx: StoreContext, args: dict[str, Any]) -> dict[str, Any]:
    """Start a recording via the in-process HTTP recording session.

    Under `meetmind serve` the MCP server shares a process with the HTTP API,
    so the session helper is called directly; standalone, this returns an error.
    """
    try:
        from meetmind.api.http import _start_recording_session  # noqa: PLC0415
    except ImportError:
        return {"ok": False, "error": "HTTP API not installed (pip install 'meetmind[api]')"}
    title = args.get("title")
    stream = args.get("stream", "both")
    try:
        result = await _start_recording_session(title=title, stream=stream)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, **result}


async def _stop_recording(_ctx: StoreContext, _args: dict[str, Any]) -> dict[str, Any]:
    try:
        from meetmind.api.http import _stop_recording_session  # noqa: PLC0415
    except ImportError:
        return {"ok": False, "error": "HTTP API not installed (pip install 'meetmind[api]')"}
    try:
        result = await _stop_recording_session()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, **result}


def _str_schema(props: dict[str, str], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": required or [],
    }


_TOOL_TABLE: list[ToolDescriptor] = [
    ToolDescriptor(
        name="search_meetings",
        description="Hybrid lexical + semantic search across all meeting transcripts.",
        input_schema=_str_schema({"query": "search query"}, required=["query"]),
        fn=_search_meetings,
    ),
    ToolDescriptor(
        name="get_meeting",
        description="Fetch a meeting by id with optional includes.",
        input_schema=_str_schema({"meeting_id": "meeting ULID"}, required=["meeting_id"]),
        fn=_get_meeting,
    ),
    ToolDescriptor(
        name="who_said",
        description="Find which speaker(s) said a quote or discussed a topic.",
        input_schema=_str_schema(
            {"quote_or_topic": "phrase to search"}, required=["quote_or_topic"]
        ),
        fn=_who_said,
    ),
    ToolDescriptor(
        name="list_action_items",
        description="List action items, optionally filtered by status, meeting, or owner.",
        input_schema=_str_schema({}),
        fn=_list_action_items,
    ),
    ToolDescriptor(
        name="get_decisions",
        description="List decisions made in a given meeting.",
        input_schema=_str_schema({"meeting_id": "meeting ULID"}, required=["meeting_id"]),
        fn=_get_decisions,
    ),
    ToolDescriptor(
        name="find_unanswered_questions",
        description="Surface questions raised but never resolved (requires live pipeline, v0.12).",
        input_schema=_str_schema({}),
        fn=_find_unanswered_questions,
    ),
    ToolDescriptor(
        name="get_speaker_history",
        description="What a person has said about a topic across meetings.",
        input_schema=_str_schema({"person": "speaker label"}, required=["person"]),
        fn=_get_speaker_history,
    ),
    ToolDescriptor(
        name="summarize_period",
        description="Generate a summary spanning a date range (requires LLM, v0.8).",
        input_schema=_str_schema({"start": "ISO date", "end": "ISO date"}),
        fn=_summarize_period,
    ),
    ToolDescriptor(
        name="extract_quotes",
        description="Verbatim quotes from a meeting, filtered by speaker or topic.",
        input_schema=_str_schema({"meeting_id": "meeting ULID"}, required=["meeting_id"]),
        fn=_extract_quotes,
    ),
    ToolDescriptor(
        name="get_attendees",
        description="Who participated in a meeting and how much they spoke.",
        input_schema=_str_schema({"meeting_id": "meeting ULID"}, required=["meeting_id"]),
        fn=_get_attendees,
    ),
    ToolDescriptor(
        name="attendee_stats",
        description="Talk-time per speaker for a meeting.",
        input_schema=_str_schema({"meeting_id": "meeting ULID"}, required=["meeting_id"]),
        fn=_attendee_stats,
    ),
    ToolDescriptor(
        name="link_to_moment",
        description="Build a deep link URI to a specific timestamp in a meeting.",
        input_schema={
            "type": "object",
            "properties": {
                "meeting_id": {"type": "string"},
                "timestamp_ms": {"type": "integer"},
            },
            "required": ["meeting_id", "timestamp_ms"],
        },
        fn=_link_to_moment,
    ),
    ToolDescriptor(
        name="compare_meetings",
        description="Side-by-side comparison across multiple meetings.",
        input_schema={
            "type": "object",
            "properties": {
                "meeting_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["meeting_ids"],
        },
        fn=_compare_meetings,
    ),
    ToolDescriptor(
        name="get_followups",
        description="Action items flagged as needing follow-up.",
        input_schema=_str_schema({"meeting_id": "meeting ULID"}, required=["meeting_id"]),
        fn=_get_followups,
    ),
    ToolDescriptor(
        name="export_to",
        description="Export to Notion/Linear/Slack/Obsidian/GitHub (v0.11.x).",
        input_schema=_str_schema({"target": "integration name"}),
        fn=_export_to,
    ),
    ToolDescriptor(
        name="start_recording",
        description="Begin a recording session (requires live pipeline).",
        input_schema=_str_schema({}),
        fn=_start_recording,
    ),
    ToolDescriptor(
        name="stop_recording",
        description="End the active recording session.",
        input_schema=_str_schema({}),
        fn=_stop_recording,
    ),
]


def list_tools() -> list[ToolDescriptor]:
    return list(_TOOL_TABLE)


def get_tool(name: str) -> ToolDescriptor | None:
    for t in _TOOL_TABLE:
        if t.name == name:
            return t
    return None


async def call_tool(ctx: StoreContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool = get_tool(name)
    if tool is None:
        raise KeyError(f"unknown tool: {name}")
    return await tool.fn(ctx, args)


@dataclass
class JsonRpcServer:
    ctx: StoreContext
    log_calls: bool = field(default=True)

    async def serve_stdio(
        self,
        reader: asyncio.StreamReader | None = None,
        writer: asyncio.StreamWriter | None = None,
    ) -> None:
        """Run the server on stdin/stdout (or any async stream pair)."""
        if reader is None or writer is None:
            loop = asyncio.get_running_loop()
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)
            transport, _ = await loop.connect_write_pipe(
                asyncio.streams.FlowControlMixin, sys.stdout
            )
            writer = asyncio.StreamWriter(transport, _, reader, loop)

        while True:
            line = await reader.readline()
            if not line:
                return
            response = await self.handle_line(line.decode("utf-8"))
            writer.write((response + "\n").encode("utf-8"))
            await writer.drain()

    async def handle_line(self, line: str) -> str:
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            return _rpc_error(None, -32700, f"parse error: {e}")
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0":
            rpc_id = req.get("id") if isinstance(req, dict) else None
            return _rpc_error(rpc_id, -32600, "invalid request")
        rpc_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}
        if method == "tools/list":
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in list_tools()
            ]
            return _rpc_result(rpc_id, {"tools": tools})
        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            try:
                result = await call_tool(self.ctx, name, arguments)
            except KeyError as e:
                return _rpc_error(rpc_id, -32601, str(e))
            except Exception as e:  # noqa: BLE001
                return _rpc_error(rpc_id, -32603, f"tool error: {e}")
            return _rpc_result(rpc_id, {"content": [{"type": "json", "data": result}]})
        if method == "resources/list":
            limit = int(params.get("limit", 50))
            resources = [r.to_wire() for r in list_resources(self.ctx.store, limit=limit)]
            return _rpc_result(rpc_id, {"resources": resources})
        if method == "resources/read":
            uri = params.get("uri", "")
            if not uri:
                return _rpc_error(rpc_id, -32602, "resources/read requires 'uri'")
            try:
                content = read_resource(self.ctx.store, uri)
            except KeyError as e:
                return _rpc_error(rpc_id, -32602, str(e))
            except LookupError as e:
                return _rpc_error(rpc_id, -32001, str(e))
            return _rpc_result(rpc_id, {"contents": [content.to_wire()]})
        if method == "prompts/list":
            return _rpc_result(rpc_id, {"prompts": [p.to_wire() for p in list_prompts()]})
        if method == "prompts/get":
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            prompt = get_prompt(name)
            if prompt is None:
                return _rpc_error(rpc_id, -32601, f"unknown prompt: {name}")
            try:
                text = render_prompt(name, arguments)
            except KeyError as e:
                return _rpc_error(rpc_id, -32602, str(e))
            return _rpc_result(
                rpc_id,
                {
                    "description": prompt.description,
                    "messages": [
                        {"role": "user", "content": {"type": "text", "text": text}},
                    ],
                },
            )
        return _rpc_error(rpc_id, -32601, f"unknown method: {method}")


def _rpc_result(rpc_id: Any, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": result}, default=str)


def _rpc_error(rpc_id: Any, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})


def _meeting_to_dict(m: Any) -> dict[str, Any]:
    return m.model_dump(mode="json", exclude={"transcript", "decisions", "summary"})


def _segment_to_dict(s: Any) -> dict[str, Any]:
    return {
        "start_ms": s.start_ms,
        "end_ms": s.end_ms,
        "speaker": s.speaker_id or s.speaker,
        "channel": s.channel.value if s.channel is not None else None,
        "text": s.text,
        "language": s.language,
    }


def _action_to_dict(a: Any) -> dict[str, Any]:
    return a.model_dump(mode="json")

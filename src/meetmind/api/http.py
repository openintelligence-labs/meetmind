"""Local HTTP API server (FastAPI on 127.0.0.1).

Endpoints:
  GET  /                     — overlay UI (static, served from tauri/ui/)
  GET  /v1/health            — unauthenticated liveness probe
  GET  /v1/info              — bound version + protocol metadata
  GET  /v1/auth/handshake    — localhost-only token handshake (no auth)
  GET  /v1/transcripts/live  — SSE stream of live transcript events
  GET  /v1/meetings          — list archived meetings (dashboard)
  GET  /v1/meeting/{id}      — single meeting (+ transcript, actions, decisions)
  GET  /v1/search            — hybrid lexical+dense search across the archive

Auth:
  All non-health endpoints require `Authorization: Bearer <token>`,
  where `<token>` is the ephemeral per-launch token written to
  `~/.meetmind/token` (mode 0600).

  The handshake endpoint is the exception: it returns the token to
  callers connecting *from the loopback interface itself*. That makes
  the same-origin overlay UI usable without copy-pasting a token —
  a remote attacker cannot reach 127.0.0.1 from another host, so the
  loopback check is a sufficient origin proof.

Bind:
  Always 127.0.0.1 only. The `serve()` helper enforces this — `host`
  is not a parameter.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import meetmind
from meetmind.api.auth import BearerAuth, generate_token, write_token
from meetmind.api.bus import EventBus, default_bus
from meetmind.api.events import Event
from meetmind.memory.store import Store

# Directory holding the static overlay UI. Resolved relative to the
# repo root; in an installed wheel we'd ship it under
# ``meetmind/_ui/`` and resolve via ``importlib.resources``, but for
# v0.x dev workflow this path is correct.
UI_DIR = Path(__file__).resolve().parents[3] / "tauri" / "ui"

log = logging.getLogger(__name__)


def _cors_regex(port: int | None) -> str:
    """Build the CORS ``allow_origin_regex`` for a given bound port.

    Three classes of origin are accepted:
      • ``null`` / ``file://`` — for the dev workflow of opening the
        bundled overlay straight from disk;
      • ``tauri://localhost`` — the Tauri WebView's hard-coded origin;
      • ``http(s)://(localhost|127.0.0.1):<port>`` — the running server's
        own port, or any localhost port if ``port`` is None.

    Narrowing to a single port shuts the door on a malicious browser
    tab running on another local server (e.g. a long-lived devtool on
    :3000) from speaking to the MeetMind API after a token leak.
    """
    if port is None:
        host_clause = r"(localhost|127\.0\.0\.1)(:\d+)?"
    else:
        host_clause = rf"(localhost|127\.0\.0\.1):{int(port)}"
    return rf"^(null|file://.*|https?://{host_clause}|tauri://localhost)$"


def create_app(
    token: str,
    bus: EventBus | None = None,
    *,
    port: int | None = None,
) -> FastAPI:
    """Construct the FastAPI app bound to a specific token + bus.

    ``port`` (when supplied) narrows the CORS allow-list to that exact
    port. Without it (legacy callers + tests) any localhost port is
    accepted. Production ``serve()`` always passes its bound port, so
    real deployments never expose the broad regex.
    """
    bus = bus or default_bus
    auth = BearerAuth(token)
    app = FastAPI(
        title="MeetMind local API",
        version=meetmind.__version__,
        docs_url=None,  # don't expose interactive docs to localhost browsers
        redoc_url=None,
    )
    # The bundled overlay (`tauri/ui/index.html`) is opened via `file://`
    # in dev — browsers send the `Origin: null` header for those, and
    # block the cross-origin fetch unless we explicitly allow it. Tauri's
    # WebView is `tauri://localhost`. Bearer-token auth on the endpoints
    # themselves remains the real authorization gate; CORS is defence in
    # depth against a same-machine malicious browser tab.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_cors_regex(port),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/info", dependencies=[Depends(auth)])
    async def info() -> dict[str, Any]:
        return {
            "version": meetmind.__version__,
            "protocol": "1.0.0",
            "endpoints": [
                "/v1/health",
                "/v1/info",
                "/v1/auth/handshake",
                "/v1/transcripts/live",
            ],
        }

    @app.get("/v1/auth/handshake")
    async def handshake(request: Request) -> dict[str, str]:
        """Return the bearer token to a loopback caller.

        Anyone who can reach 127.0.0.1 over TCP is already on the same
        host as the server (modulo a malicious local user, against
        which the file-mode-0600 token is the real defense). Returning
        the token here saves the user from copy-pasting it into the
        overlay UI on every launch.
        """
        client = request.client
        host = client.host if client is not None else ""
        if host not in {"127.0.0.1", "::1"}:
            raise HTTPException(status_code=403, detail="handshake is loopback-only")
        return {"token": token}

    @app.get("/v1/transcripts/live", dependencies=[Depends(auth)])
    async def transcripts_live() -> StreamingResponse:
        async def event_stream():
            async with bus.subscription() as q:
                # Send a comment line so curl-style clients see traffic
                # immediately and confirm the stream is alive.
                yield ":connected\n\n"
                while True:
                    event = await q.get()
                    yield _format_sse(event)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ─────────────── archive endpoints (powers dashboard.js) ───────────────
    # These open the SQLCipher store on demand, count segments, and return
    # dashboard-shaped JSON. The store path follows the same fallback
    # chain as the CLI: $MEETMIND_HOME/data/meetmind.db then
    # ~/.meetmind/data/meetmind.db. We never hold a long-lived handle from
    # the request thread — sqlite is cheap to reopen and threads make
    # connection pooling painful.

    @app.get("/v1/meetings", dependencies=[Depends(auth)])
    async def list_meetings_route(limit: int = 200) -> dict[str, Any]:
        return _list_meetings(limit=limit)

    @app.get("/v1/meeting/{meeting_id}", dependencies=[Depends(auth)])
    async def get_meeting_route(meeting_id: str) -> dict[str, Any]:
        result = _get_meeting(meeting_id)
        if result is None:
            raise HTTPException(status_code=404, detail="meeting not found")
        return result

    @app.get("/v1/search", dependencies=[Depends(auth)])
    async def search_route(q: str = "", limit: int = 20) -> dict[str, Any]:
        return _search(q, limit=limit)

    # ─────────────── recording control (UI start/stop) ───────────────

    @app.post("/v1/recording/start", dependencies=[Depends(auth)])
    async def start_recording_route(request: Request) -> dict[str, Any]:
        body: dict[str, Any] = {}
        import contextlib

        with contextlib.suppress(Exception):
            body = await request.json()
        title = body.get("title")
        stream = body.get("stream", "both")
        result = await _start_recording_session(title=title, stream=stream)
        return result

    @app.post("/v1/recording/stop", dependencies=[Depends(auth)])
    async def stop_recording_route() -> dict[str, Any]:
        return await _stop_recording_session()

    @app.get("/v1/recording/status", dependencies=[Depends(auth)])
    async def recording_status_route() -> dict[str, Any]:
        return _recording_status()

    @app.post("/v1/meeting/{meeting_id}/summarize", dependencies=[Depends(auth)])
    async def summarize_meeting_route(meeting_id: str) -> dict[str, Any]:
        return await _summarize_meeting_via_api(meeting_id)

    @app.get("/v1/meeting/{meeting_id}/audio/{stream}")
    async def meeting_audio_route(meeting_id: str, stream: str, request: Request) -> FileResponse:
        """Stream a persisted WAV file for one stream of a meeting.

        404 if the meeting wasn't recorded with ``--persist-audio`` (the
        WAV path on disk doesn't exist). Path is sanity-checked against
        the persisted ``Meeting.audio_path_*`` columns to prevent any
        path-traversal via the URL.

        Auth: HTML5 ``<audio>`` elements can't send ``Authorization``
        headers, so this route accepts the bearer token via either
        ``Authorization: Bearer <t>`` OR ``?t=<token>``. The query-param
        path is only accepted for this specific route — everything else
        requires the header.
        """
        from secrets import compare_digest  # noqa: PLC0415

        provided = request.query_params.get("t", "")
        header = request.headers.get("authorization", "")
        header_token = (
            header.split(None, 1)[1].strip() if header.lower().startswith("bearer ") else ""
        )
        if not (
            (provided and compare_digest(provided, token))
            or (header_token and compare_digest(header_token, token))
        ):
            raise HTTPException(status_code=401, detail="missing/invalid bearer token")
        if stream not in {"mic", "loopback"}:
            raise HTTPException(status_code=400, detail="stream must be mic|loopback")
        db = _default_db_path()
        with Store.open(db) as store:
            meeting = store.get_meeting(meeting_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="meeting not found")
        path = meeting.audio_path_mic if stream == "mic" else meeting.audio_path_loopback
        if path is None or not Path(path).exists():
            raise HTTPException(status_code=404, detail="no audio persisted for this stream")
        return FileResponse(str(path), media_type="audio/wav", filename=Path(path).name)

    @app.patch("/v1/meeting/{meeting_id}", dependencies=[Depends(auth)])
    async def patch_meeting_route(meeting_id: str, request: Request) -> dict[str, Any]:
        """Rename a meeting. Body: ``{"title": "new title"}``.

        Only ``title`` is mutable — started/ended/duration are derived
        from the recording session, and we don't want stale UI writes
        to overwrite them.
        """
        body = await request.json()
        new_title = (body.get("title") or "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="title is required")
        db = _default_db_path()
        with Store.open(db) as store:
            meeting = store.get_meeting(meeting_id)
            if meeting is None:
                raise HTTPException(status_code=404, detail="meeting not found")
            meeting.title = new_title[:200]
            store.upsert_meeting(meeting)
        return {"ok": True, "meeting_id": meeting_id, "title": new_title[:200]}

    @app.post("/v1/meeting/{meeting_id}/export/{target}", dependencies=[Depends(auth)])
    async def export_meeting_route(
        meeting_id: str, target: str, request: Request
    ) -> dict[str, Any]:
        """Run an integration exporter for a meeting.

        Body shape depends on target:
          • obsidian: ``{"vault_path": "/abs/path"}``
          • slack:    ``{"webhook_url": "https://..."}`` (or rely on env)
          • github:   ``{"repo": "owner/name", "dry_run": false}``

        Returns the exporter's structured result; never raises on
        integration-level errors (the body's ``ok`` flag indicates
        success/failure). Auth-required.
        """
        body: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        db = _default_db_path()
        target = target.lower()
        if target == "obsidian":
            vault = body.get("vault_path")
            if not vault:
                raise HTTPException(status_code=400, detail="vault_path required")
            from meetmind.integrations.obsidian import export_meeting

            try:
                with Store.open(db) as store:
                    result = export_meeting(store, meeting_id, vault=Path(vault), overwrite=True)
                return {"ok": True, "target": "obsidian", "path": str(result.note_path)}
            except LookupError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
        if target == "slack":
            from meetmind.integrations.slack import export_meeting_to_slack

            with Store.open(db) as store:
                result = export_meeting_to_slack(
                    store, meeting_id, webhook_url=body.get("webhook_url")
                )
            return {"target": "slack", **result}
        if target == "github":
            from meetmind.integrations.github import GhCliMissingError, export_action_items

            repo = body.get("repo")
            if not repo:
                raise HTTPException(status_code=400, detail="repo (owner/name) required")
            try:
                with Store.open(db) as store:
                    refs = export_action_items(
                        store, meeting_id, repo=repo, dry_run=bool(body.get("dry_run", False))
                    )
            except GhCliMissingError as e:
                return {"ok": False, "target": "github", "error": str(e)}
            return {
                "ok": True,
                "target": "github",
                "issues": [{"number": r.number, "url": r.url, "title": r.title} for r in refs],
            }
        raise HTTPException(status_code=400, detail=f"unknown target: {target!r}")

    @app.get("/v1/compliance/status", dependencies=[Depends(auth)])
    async def compliance_status_route() -> dict[str, Any]:
        """Surface compliance-posture facts the dashboard can render.

        Computed live so the UI never drifts from reality:
          • storage mode (encrypted vs unencrypted vs disabled),
          • retention TTLs in effect (env-tunable),
          • count of meetings + speakers currently subject to retention.
        """
        from meetmind.compliance.retention import RetentionPolicy  # noqa: PLC0415
        from meetmind.memory.keyring import get_or_create_dek  # noqa: PLC0415

        db = _default_db_path()
        n_meetings = 0
        n_speakers = 0
        n_consent_events = 0
        if db.exists():
            with Store.open(db) as store:
                n_meetings = len(store.list_meetings(limit=10_000))
                rows = store.conn.execute("SELECT COUNT(*) AS c FROM speakers").fetchone()
                n_speakers = int(rows["c"] if rows else 0)
                rows = store.conn.execute("SELECT COUNT(*) AS c FROM consent_events").fetchone()
                n_consent_events = int(rows["c"] if rows else 0)
        try:
            import pysqlcipher3  # noqa: F401, PLC0415

            driver = "pysqlcipher3"
        except ImportError:
            driver = "stdlib-sqlite3"
        dek_avail = get_or_create_dek() is not None
        encryption_mode = "encrypted" if driver == "pysqlcipher3" and dek_avail else "unencrypted"
        retention = RetentionPolicy.from_env()
        return {
            "encryption": {
                "mode": encryption_mode,
                "driver": driver,
                "dek_available": dek_avail,
            },
            "retention": {
                "meetings_days": retention.meetings_ttl_days,
                "voiceprints_days": retention.voiceprint_ttl_days,
            },
            "redaction_profiles": ["raw", "team_internal", "public_share"],
            "counts": {
                "meetings": n_meetings,
                "speakers": n_speakers,
                "consent_events": n_consent_events,
            },
            "egress": {
                "default_config": "loopback-only",
                "outbound_paths": [
                    "actants LLM (when MEETMIND_LLM_PROVIDER != ollama)",
                    "gh CLI (when export-github is invoked)",
                    "Slack webhook (when export-slack is invoked)",
                ],
            },
        }

    @app.post("/v1/meeting/{meeting_id}/diarize", dependencies=[Depends(auth)])
    async def diarize_meeting_route(meeting_id: str) -> dict[str, Any]:
        """Run post-hoc diarization on a recorded meeting from the UI.

        Requires the meeting was recorded with audio persistence;
        without it, falls back to the channel-label synthesizer (no
        clustering gain, but doesn't crash).
        """
        from meetmind.cli import _diarize_meeting  # noqa: PLC0415

        db = _default_db_path()
        if not db.exists():
            raise HTTPException(status_code=404, detail="no store")
        try:
            await _diarize_meeting(db, meeting_id, rms_threshold=0.005, match_enrolled=True)
        except Exception as e:  # noqa: BLE001 — surface to UI
            return {"ok": False, "error": str(e)}
        return {"ok": True, "meeting_id": meeting_id}

    @app.delete("/v1/meeting/{meeting_id}", dependencies=[Depends(auth)])
    async def delete_meeting_route(meeting_id: str) -> dict[str, Any]:
        """Hard-delete a meeting + cascade (segments, actions, decisions,
        summary). Also unlinks persisted WAV files for the meeting."""
        db = _default_db_path()
        with Store.open(db) as store:
            meeting = store.get_meeting(meeting_id)
            if meeting is None:
                raise HTTPException(status_code=404, detail="meeting not found")
            # Best-effort: remove persisted audio first. If unlink fails,
            # the row delete still proceeds — the orphaned WAV is
            # picked up later by the retention sweep.
            for p in (meeting.audio_path_mic, meeting.audio_path_loopback):
                if p is None:
                    continue
                with contextlib.suppress(FileNotFoundError, OSError):
                    Path(p).unlink()
            store.forget_meeting(meeting_id)
        return {"ok": True, "meeting_id": meeting_id, "deleted": True}

    # Static UI — same origin as the API kills CORS, lets the overlay
    # auto-handshake the bearer token on load. Mounted last so the
    # `/v1/*` routes always win over the static fallback.
    if UI_DIR.is_dir():

        @app.get("/")
        async def _index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

        app.mount(
            "/static",
            StaticFiles(directory=str(UI_DIR), html=False),
            name="static",
        )

    return app


def _format_sse(event: Event) -> str:
    """Serialize a single event as SSE 'event:'/'data:' frame."""
    payload = event.model_dump(mode="json")
    return f"event: {event.kind}\ndata: {json.dumps(payload)}\n\n"


# ─────────────────────── recording session (UI-driven) ───────────────────────
# Single shared session — the server hosts at most one recording at a time.
# Start spawns a `_run_record` task in this process so it publishes onto the
# same in-process bus the dashboard subscribes to. Stop fires a stop_event.

_recording_lock = asyncio.Lock()
_recording_state: dict[str, Any] = {
    "task": None,
    "stop_event": None,
    "started_at": None,
    "title": None,
    "stream": None,
    "meeting_id": None,  # filled by `_run_record` via the meeting it creates
}


def _recording_status() -> dict[str, Any]:
    task = _recording_state["task"]
    return {
        "recording": task is not None and not task.done(),
        "started_at": _recording_state["started_at"],
        "title": _recording_state["title"],
        "stream": _recording_state["stream"],
        "meeting_id": _recording_state["meeting_id"],
    }


async def _start_recording_session(
    *, title: str | None = None, stream: str = "both"
) -> dict[str, Any]:
    from datetime import UTC, datetime

    from meetmind.ipc import StreamId

    async with _recording_lock:
        existing = _recording_state["task"]
        if existing is not None and not existing.done():
            raise HTTPException(status_code=409, detail="recording already in progress")

        # Resolve sidecars + stream id. Lazy-imported from cli to keep
        # the api module decoupled.
        from meetmind.cli import _find_capture_sidecar, _find_mock_capture, _run_record
        from meetmind.stt.parakeet_v3 import find_stt_sidecar

        capture_bin = _find_capture_sidecar() or _find_mock_capture()
        stt_bin = find_stt_sidecar() or _find_mock_capture()
        if stream == "mic":
            sid: StreamId | None = StreamId.MIC
        elif stream == "loopback":
            sid = StreamId.LOOPBACK
        else:
            sid = None  # both

        stop_event = asyncio.Event()
        _recording_state["stop_event"] = stop_event
        _recording_state["started_at"] = datetime.now(UTC).isoformat()
        _recording_state["title"] = title
        _recording_state["stream"] = stream
        _recording_state["meeting_id"] = None  # populated by the runner

        async def _runner() -> str | None:
            db_path = _default_db_path()
            mid = await _run_record(
                capture_bin,
                stt_bin,
                duration=0.0,
                stream_id=sid,
                mock="_mock_sidecar" in capture_bin.name,
                emit_sse=True,
                sse_port=0,  # unused — publish_only mode shares this server's bus
                db_path=db_path,
                title=title,
                coach=False,
                publish_only=True,
                stop_event=stop_event,
            )
            return mid

        task = asyncio.create_task(_runner(), name="api-record")
        _recording_state["task"] = task

        # Best-effort: peek the meeting id once it's been created. We do
        # this by polling the store for new meetings within a short
        # window — `_run_record` writes the meeting row before any
        # transcripts arrive.
        async def _track_meeting_id() -> None:
            for _ in range(40):  # ~4s
                await asyncio.sleep(0.1)
                from meetmind.memory.store import Store

                with Store.open(_default_db_path()) as s:
                    most_recent = s.list_meetings(limit=1)
                    if most_recent and most_recent[0].title == title:
                        _recording_state["meeting_id"] = most_recent[0].id
                        return

        asyncio.create_task(_track_meeting_id(), name="api-record-track")

    return _recording_status()


async def _stop_recording_session() -> dict[str, Any]:
    from datetime import UTC, datetime

    async with _recording_lock:
        task = _recording_state["task"]
        stop_event = _recording_state["stop_event"]
        if task is None or task.done():
            return {"recording": False, "note": "no recording was active"}
        if stop_event is not None:
            stop_event.set()
        try:
            mid = await asyncio.wait_for(task, timeout=10.0)
        except TimeoutError:
            task.cancel()
            mid = _recording_state.get("meeting_id")

        result = {
            "recording": False,
            "stopped_at": datetime.now(UTC).isoformat(),
            "meeting_id": mid or _recording_state.get("meeting_id"),
            "title": _recording_state.get("title"),
        }
        _recording_state["task"] = None
        _recording_state["stop_event"] = None
        return result


async def _summarize_meeting_via_api(meeting_id: str) -> dict[str, Any]:
    """Run summarize in the API process, return the persisted summary.

    The CLI handler writes to stdout — that's fine for the terminal but
    we want JSON here. Reuse the analyze functions directly.
    """
    db = _default_db_path()
    if not db.exists():
        raise HTTPException(status_code=404, detail="no store yet")

    from meetmind.analyze.actions import (
        ExtractionPayload,
        build_user_prompt,
        extract_action_items,
    )
    from meetmind.analyze.decisions import extract_decisions
    from meetmind.analyze.llm import get_default_llm
    from meetmind.analyze.summarize import _DensePayload, _DraftPayload, summarize_meeting
    from meetmind.memory.store import Store

    with Store.open(db) as store:
        meeting = store.get_meeting(meeting_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="meeting not found")
        segments = store.list_segments(meeting_id)
        transcript_window = "\n".join(s.text for s in segments if s.text)

        # Run the LLM-heavy work off the event loop so the API stays
        # responsive. The actants LLM helpers already have their own
        # worker-thread bridge, but we still want async clarity here.
        loop = asyncio.get_running_loop()
        llm_holder: dict[str, Any] = {}

        def _do_summarize() -> dict[str, Any]:
            llm = get_default_llm()
            llm_holder["model"] = getattr(getattr(llm, "settings", None), "model", "auto")

            def _actions_llm(prompt: str) -> dict:
                from meetmind.analyze.llm import _run

                return _run(llm.extract(prompt, ExtractionPayload)).model_dump(mode="python")

            def _summary_llm(prompt: str) -> dict:
                from meetmind.analyze.llm import _run

                schema = _DensePayload if "previous_draft" in prompt else _DraftPayload
                return _run(llm.extract(prompt, schema)).model_dump(mode="python")

            actions_result = extract_action_items(
                build_user_prompt(transcript_window), _actions_llm
            )
            decisions_result = extract_decisions(transcript_window, _summary_llm)
            sresult = summarize_meeting(
                transcript_window,
                _summary_llm,
                densify_passes=1,
                key_decisions=[d.decision for d in decisions_result.accepted],
                action_items=actions_result.accepted,
            )
            return {
                "actions": actions_result,
                "decisions": decisions_result,
                "summary": sresult,
            }

        result = await loop.run_in_executor(None, _do_summarize)

        # Persist
        for a in result["actions"].accepted:
            store.upsert_action_item(meeting_id, a)
        for d in result["decisions"].accepted:
            store.upsert_decision(meeting_id, d)
        store.upsert_summary(
            meeting_id,
            tl_dr=result["summary"].summary.tl_dr,
            topics=list(result["summary"].headline_topics),
            model=llm_holder.get("model") or "auto",
        )

    return {
        "meeting_id": meeting_id,
        "tl_dr": result["summary"].summary.tl_dr,
        "topics": list(result["summary"].headline_topics),
        "actions_accepted": len(result["actions"].accepted),
        "actions_rejected": len(result["actions"].rejected),
        "decisions_accepted": len(result["decisions"].accepted),
        "decisions_rejected": len(result["decisions"].rejected),
        "model": llm_holder.get("model") or "auto",
    }


# ─────────────────────── archive helpers ───────────────────────


def _default_db_path() -> Path:
    """Same fallback chain as the CLI."""
    import os

    base = Path(os.environ.get("MEETMIND_HOME", str(Path.home() / ".meetmind")))
    return base / "data" / "meetmind.db"


def _default_lance_dir() -> Path:
    import os

    base = Path(os.environ.get("MEETMIND_HOME", str(Path.home() / ".meetmind")))
    return base / "data" / "lance"


def _list_meetings(*, limit: int = 200) -> dict[str, Any]:
    db = _default_db_path()
    if not db.exists():
        return {"meetings": [], "count": 0, "note": "no archive yet"}
    from meetmind.memory.store import Store  # local import — heavy deps

    out: list[dict[str, Any]] = []
    with Store.open(db) as store:
        meetings = store.list_meetings(limit=limit)
        for m in meetings:
            seg_count = store.conn.execute(
                "SELECT COUNT(*) FROM transcript_segments WHERE meeting_id = ?",
                (m.id,),
            ).fetchone()[0]
            out.append(
                {
                    "id": m.id,
                    "title": m.title,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "started_at": m.started_at.isoformat() if m.started_at else None,
                    "ended_at": m.ended_at.isoformat() if m.ended_at else None,
                    "duration_seconds": m.duration_seconds,
                    "segment_count": int(seg_count),
                }
            )
    return {"meetings": out, "count": len(out)}


def _get_meeting(meeting_id: str) -> dict[str, Any] | None:
    db = _default_db_path()
    if not db.exists():
        return None
    from meetmind.memory.store import Store

    with Store.open(db) as store:
        m = store.get_meeting(meeting_id)
        if m is None:
            return None
        segments = store.list_segments(meeting_id)
        actions = store.list_action_items(meeting_id=meeting_id)
        decisions = store.list_decisions(meeting_id)
        summary = store.get_summary(meeting_id)
    return {
        "meeting": {
            "id": m.id,
            "title": m.title,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "started_at": m.started_at.isoformat() if m.started_at else None,
            "ended_at": m.ended_at.isoformat() if m.ended_at else None,
            "duration_seconds": m.duration_seconds,
            "audio_path_mic": str(m.audio_path_mic) if m.audio_path_mic else None,
            "audio_path_loopback": (str(m.audio_path_loopback) if m.audio_path_loopback else None),
        },
        "summary": summary,
        "segments": [
            {
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "text": s.text,
                "speaker": s.speaker,
                "speaker_id": s.speaker_id,
                "channel": s.channel.value if s.channel is not None else None,
                "language": s.language,
            }
            for s in segments
        ],
        "actions": [a.model_dump(mode="json") for a in actions],
        "decisions": [d.model_dump(mode="json") for d in decisions],
    }


def _search(query: str, *, limit: int = 20) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"hits": [], "query": "", "count": 0}
    lance = _default_lance_dir()
    db = _default_db_path()
    if not lance.exists():
        return {"hits": [], "query": query, "count": 0, "note": "no index yet"}
    from meetmind.memory.store import Store
    from meetmind.memory.vector import HybridIndex, hash_embedder

    # Lightweight fallback embedder so search works even when nomic-embed
    # isn't installed locally. The real index was built with whatever
    # embedder `meetmind index` ran with — for cross-process search the
    # hash_embedder works as a passable approximation; with a real
    # embedder we'd plug it in here too.
    index = HybridIndex.open(lance, vector_dim=768, embedder=hash_embedder(768))
    hits = index.search(query, limit=limit)
    titles: dict[str, str] = {}
    if db.exists():
        with Store.open(db) as store:
            for mid in {h.segment.meeting_id for h in hits if h.segment.meeting_id}:
                m = store.get_meeting(mid)
                if m and m.title:
                    titles[mid] = m.title
    return {
        "query": query,
        "count": len(hits),
        "hits": [
            {
                "meeting_id": h.segment.meeting_id,
                "meeting_title": titles.get(h.segment.meeting_id),
                "segment_id": h.segment.segment_id,
                "text": h.segment.text,
                "start_ms": h.segment.start_ms,
                "end_ms": h.segment.end_ms,
                "cluster_id": h.segment.cluster_id,
                "score": float(h.score),
            }
            for h in hits
        ],
    }


async def serve(
    token: str | None = None,
    *,
    port: int = 7857,
    bus: EventBus | None = None,
) -> None:
    """Run the API server on 127.0.0.1:<port> until the task is cancelled.

    If `token` is None, generate one and write it to ~/.meetmind/token.
    Otherwise, the caller is responsible for token persistence.
    """
    import uvicorn  # local import — only needed if `[api]` extra is installed

    if token is None:
        token = generate_token()
        write_token(token)

    app = create_app(token, bus=bus, port=port)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",  # locked — never 0.0.0.0
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except asyncio.CancelledError:
        await server.shutdown()
        raise

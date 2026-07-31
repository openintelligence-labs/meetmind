"""Tests for the FastAPI surface (auth + SSE)."""

from __future__ import annotations

import asyncio
import json
import sys

import httpx
import pytest

import meetmind
from meetmind.api.auth import BearerAuthClient
from meetmind.api.bus import EventBus
from meetmind.api.events import FinalEvent, MetaEvent, PartialEvent
from meetmind.api.http import create_app


@pytest.fixture
def app_and_client() -> tuple:
    token = "tok-test-1234"
    bus = EventBus()
    app = create_app(token, bus=bus)
    transport = httpx.ASGITransport(app=app)
    return app, bus, token, transport


async def test_health_is_unauthenticated(app_and_client):
    _, _, _, transport = app_and_client
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


async def test_info_requires_auth(app_and_client):
    _, _, _, transport = app_and_client
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/v1/info")
        assert resp.status_code == 401


async def test_info_with_correct_token(app_and_client):
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.get("/v1/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == meetmind.__version__
        assert "/v1/transcripts/live" in body["endpoints"]
        assert "/v1/auth/handshake" in body["endpoints"]


async def test_handshake_returns_token_for_loopback():
    """Real-uvicorn round-trip: handshake should return the bearer token
    when the request is made from 127.0.0.1."""
    import socket

    import uvicorn

    from meetmind.api.http import create_app

    token = "tok-handshake-1234"
    app = create_app(token, bus=EventBus())

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.02)

    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=4.0) as c:
            resp = await c.get("/v1/auth/handshake")
            assert resp.status_code == 200
            assert resp.json() == {"token": token}
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=4.0)


async def _seed_meeting_with_audio(tmp_path):
    """Create a real meeting row with a real WAV file on disk."""
    import wave

    from meetmind.memory.store import Store
    from meetmind.models import Meeting

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav_path = audio_dir / "01HMEET_mic.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(b"\x00\x01" * 1600)  # 0.1s of nonzero PCM

    db_path = tmp_path / "data" / "meetmind.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    s = Store.open(db_path, use_keychain=False)
    m = Meeting(
        id="01HMEETXXXXXXXXXXXXXXXXXXX",
        title="Audio test meeting",
        audio_path_mic=wav_path,
    )
    s.upsert_meeting(m)
    s.close()
    return m.id, wav_path


async def test_audio_route_streams_persisted_wav(app_and_client, tmp_path, monkeypatch):
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, wav_path = await _seed_meeting_with_audio(tmp_path)
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.get(f"/v1/meeting/{mid}/audio/mic")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content == wav_path.read_bytes()


async def test_audio_route_404_when_not_persisted(app_and_client, tmp_path, monkeypatch):
    """Meeting exists but audio_path is None → 404."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    from meetmind.memory.store import Store
    from meetmind.models import Meeting

    db_path = tmp_path / "data" / "meetmind.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    s = Store.open(db_path, use_keychain=False)
    m = Meeting(id="01HNOAUDIOXXXXXXXXXXXXXXXX", title="No audio")
    s.upsert_meeting(m)
    s.close()

    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.get(f"/v1/meeting/{m.id}/audio/mic")
        assert resp.status_code == 404


async def test_audio_route_accepts_query_param_token(app_and_client, tmp_path, monkeypatch):
    """HTML5 <audio> can't send Authorization headers — `?t=` is allowed."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, _ = await _seed_meeting_with_audio(tmp_path)
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"/v1/meeting/{mid}/audio/mic", params={"t": token})
        assert resp.status_code == 200
        # Wrong token in query: 401, not 200.
        resp_bad = await c.get(f"/v1/meeting/{mid}/audio/mic", params={"t": "not-the-token"})
        assert resp_bad.status_code == 401


async def test_audio_route_rejects_invalid_stream(app_and_client, tmp_path, monkeypatch):
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, _ = await _seed_meeting_with_audio(tmp_path)
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.get(f"/v1/meeting/{mid}/audio/bogus")
        assert resp.status_code == 400


async def test_patch_meeting_renames_in_store(app_and_client, tmp_path, monkeypatch):
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, _ = await _seed_meeting_with_audio(tmp_path)
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.patch(f"/v1/meeting/{mid}", json={"title": "Renamed in UI"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "meeting_id": mid, "title": "Renamed in UI"}
    # Verify via the store.
    from meetmind.memory.store import Store

    with Store.open(tmp_path / "data" / "meetmind.db", use_keychain=False) as s:
        assert s.get_meeting(mid).title == "Renamed in UI"


async def test_patch_meeting_requires_title(app_and_client, tmp_path, monkeypatch):
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, _ = await _seed_meeting_with_audio(tmp_path)
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.patch(f"/v1/meeting/{mid}", json={"title": "   "})
        assert resp.status_code == 400


async def test_delete_meeting_unlinks_audio_and_cascades(app_and_client, tmp_path, monkeypatch):
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, wav_path = await _seed_meeting_with_audio(tmp_path)
    assert wav_path.exists()
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.delete(f"/v1/meeting/{mid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
    assert not wav_path.exists()
    from meetmind.memory.store import Store

    with Store.open(tmp_path / "data" / "meetmind.db", use_keychain=False) as s:
        assert s.get_meeting(mid) is None


async def test_export_obsidian_route_writes_note(app_and_client, tmp_path, monkeypatch):
    """The UI calls /v1/meeting/{id}/export/obsidian to fan out to the
    integration. Verify the wiring + that the note lands on disk."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, _ = await _seed_meeting_with_audio(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.post(
            f"/v1/meeting/{mid}/export/obsidian",
            json={"vault_path": str(vault)},
        )
        assert resp.status_code == 200, resp.text
        result = resp.json()
        assert result["ok"] is True
        assert (vault / "MeetMind").exists()
        from pathlib import Path

        assert Path(result["path"]).exists()


async def test_export_unknown_target_returns_400(app_and_client, tmp_path, monkeypatch):
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, _ = await _seed_meeting_with_audio(tmp_path)
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.post(f"/v1/meeting/{mid}/export/salesforce", json={})
        assert resp.status_code == 400


async def test_diarize_route_completes_for_a_meeting(app_and_client, tmp_path, monkeypatch):
    """Wire the /v1/meeting/{id}/diarize route end-to-end."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    mid, _ = await _seed_meeting_with_audio(tmp_path)
    # Seed at least one segment so the diarizer has something to attach to.
    from meetmind.memory.store import Store
    from meetmind.models import ChannelKind, TranscriptSegment

    with Store.open(tmp_path / "data" / "meetmind.db", use_keychain=False) as s:
        s.append_segment(
            mid,
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=0.1,
                text="Hello",
                channel=ChannelKind.MIC,
            ),
        )
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.post(f"/v1/meeting/{mid}/diarize")
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True


async def test_compliance_status_endpoint(app_and_client, tmp_path, monkeypatch):
    """Surface compliance posture: encryption mode + retention + counts."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    # Seed at least one meeting so counts.meetings > 0.
    await _seed_meeting_with_audio(tmp_path)
    _, _, token, transport = app_and_client
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", auth=BearerAuthClient(token)
    ) as c:
        resp = await c.get("/v1/compliance/status")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "encryption" in body
        assert body["encryption"]["mode"] in {"encrypted", "unencrypted"}
        assert "retention" in body
        assert body["retention"]["meetings_days"] > 0
        assert body["counts"]["meetings"] >= 1
        assert "raw" in body["redaction_profiles"]


async def test_static_ui_is_mounted_when_present(app_and_client):
    """When the tauri/ui directory is present, the index page is served at /.

    Skipped in an installed wheel, which does not package the UI files.
    """
    from meetmind.api.http import UI_DIR

    if not UI_DIR.is_dir():
        pytest.skip("tauri/ui not present in this checkout")

    _, _, _, transport = app_and_client
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/")
        assert resp.status_code == 200
        body = resp.text.lower()
        assert "meetmind" in body
        assert "<script" in body


@pytest.mark.timeout(10)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="live uvicorn startup is flaky on Windows CI runners; path covered on Linux/macOS",
)
async def test_transcripts_live_emits_via_real_uvicorn(app_and_client):
    """End-to-end SSE through a real uvicorn server on a 127.0.0.1 port.

    ASGITransport buffers SSE lines unhelpfully; running uvicorn for real
    is the only way to validate streaming line delivery.
    """
    _, bus, token, _ = app_and_client
    import socket

    import uvicorn

    from meetmind.api.http import create_app

    app = create_app(token, bus=bus)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    # Wait for the server to bind.
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.02)

    seen: list[dict] = []

    async def consume() -> None:
        async with (
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                auth=BearerAuthClient(token),
                timeout=8.0,
            ) as c,
            c.stream("GET", "/v1/transcripts/live") as resp,
        ):
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    seen.append(json.loads(line[6:]))
                    if len(seen) >= 3:
                        return

    consume_task = asyncio.create_task(consume())
    try:
        # Drive the producer.
        await asyncio.sleep(0.1)
        await bus.publish(MetaEvent(event="session_started"))
        await bus.publish(PartialEvent(text="hello", start_ms=0, end_ms=100))
        await bus.publish(FinalEvent(text="hello world", start_ms=0, end_ms=200))
        await asyncio.wait_for(consume_task, timeout=4.0)
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=4.0)

    kinds = [e["kind"] for e in seen]
    assert kinds == ["meta", "partial", "final"]
    assert seen[2]["text"] == "hello world"

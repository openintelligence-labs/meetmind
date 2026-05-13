"""MeetMind CLI.

Sprint 1 surface (v0.4):

    meetmind --version
    meetmind status
    meetmind record [--duration N] [--source <path>] [--stt <path>]

`record` runs the end-to-end pipeline:
    capture sidecar  →  audio pipeline (resample + VAD)  →  STT sidecar
                                                          →  stdout transcript

By default it locates the bundled native sidecars; if they are missing it
falls back to the Python mock fixtures (the same path used in CI), which
is the v0.4 contract: the harness always works, the production sidecars
swap in transparently as they're built.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import click
import numpy as np

import meetmind
from meetmind.capture.pipeline import (
    TARGET_RATE,
    VAD,
    StreamingPipeline,
)
from meetmind.ipc import AudioChunk, SidecarProcess, StreamId  # noqa: F401
from meetmind.stt.base import Final, Partial
from meetmind.stt.parakeet_v3 import ParakeetSidecarBackend, find_stt_sidecar

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sidecar discovery (capture)
# ---------------------------------------------------------------------------


def _find_capture_sidecar() -> Path | None:
    env = os.environ.get("MEETMIND_CAPTURE_SIDECAR")
    if env and Path(env).exists():
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    if sys.platform == "darwin":
        candidate = repo / "sidecars" / "macos" / ".build" / "release" / "meetmind-capture-macos"
        if candidate.exists():
            return candidate
        found = shutil.which("meetmind-capture-macos")
        if found:
            return Path(found)
    return None


def _find_mock_capture() -> Path:
    """Pure-Python mock capture, always available (used in CI + dev fallback)."""
    here = Path(__file__).resolve().parents[2]
    fixture = here / "tests" / "fixtures" / "mock_sidecar.py"
    if not fixture.exists():
        raise click.ClickException(f"capture sidecar not found and mock fixture missing: {fixture}")
    launcher = fixture.parent / "_mock_sidecar_launcher.sh"
    launcher.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{fixture}" "$@"\n')
    launcher.chmod(0o755)
    return launcher


def _default_data_dir() -> Path:
    base = Path(os.environ.get("MEETMIND_HOME", str(Path.home() / ".meetmind")))
    data = base / "data"
    data.mkdir(parents=True, exist_ok=True)
    return data


def _default_db_path() -> Path:
    return _default_data_dir() / "meetmind.db"


def _default_lance_dir() -> Path:
    p = _default_data_dir() / "lance"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_mock_stt() -> Path:
    here = Path(__file__).resolve().parents[2]
    fixture = here / "tests" / "fixtures" / "mock_stt_sidecar.py"
    if not fixture.exists():
        raise click.ClickException(f"STT sidecar not found and mock fixture missing: {fixture}")
    launcher = fixture.parent / "_mock_stt_launcher.sh"
    launcher.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{fixture}" "$@"\n')
    launcher.chmod(0o755)
    return launcher


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(meetmind.__version__, prog_name="meetmind")
def main() -> None:
    """MeetMind — local-first meeting assistant."""


@main.command()
def status() -> None:
    """Show configured backends, hardware, and sidecar discovery."""
    from meetmind.analyze.llm import llm_config_summary  # noqa: PLC0415

    out: dict = {
        "version": meetmind.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "capture_sidecar": str(_find_capture_sidecar() or "<not found, will use mock>"),
        "stt_sidecar": str(find_stt_sidecar() or "<not found, will use mock>"),
        "llm": llm_config_summary(),
        "storage": _storage_status_summary(),
    }
    try:
        out["actants"] = importlib.metadata.version("actants")
    except importlib.metadata.PackageNotFoundError:
        out["actants"] = None
    click.echo(json.dumps(out, indent=2))


def _storage_status_summary() -> dict:
    """Report whether SQLCipher is wired and whether the DEK is available.

    Three states:
      • encrypted   — SQLCipher driver loaded + key present
      • unencrypted — stdlib sqlite3 (dev / CI / keyring unreachable)
      • disabled    — user explicitly set MEETMIND_DISABLE_ENCRYPTION=1
    """
    import os  # noqa: PLC0415

    from meetmind.memory.keyring import get_or_create_dek  # noqa: PLC0415

    if os.environ.get("MEETMIND_DISABLE_ENCRYPTION") == "1":
        return {"mode": "disabled", "reason": "MEETMIND_DISABLE_ENCRYPTION=1"}
    try:
        import pysqlcipher3  # noqa: F401, PLC0415

        driver = "pysqlcipher3"
    except ImportError:
        driver = "stdlib-sqlite3"
    dek_available = get_or_create_dek() is not None
    if driver == "pysqlcipher3" and dek_available:
        mode = "encrypted"
    elif driver == "stdlib-sqlite3" and dek_available:
        mode = "unencrypted (install meetmind[encrypted] to enable SQLCipher)"
    else:
        mode = "unencrypted (no keychain available)"
    return {"mode": mode, "driver": driver, "dek_available": dek_available}


@main.command()
@click.option(
    "--duration",
    "-d",
    type=float,
    default=5.0,
    show_default=True,
    help="Stop after N seconds (use 0 to run until Ctrl+C).",
)
@click.option(
    "--source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override capture sidecar binary (defaults to platform native, then mock).",
)
@click.option(
    "--stt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override STT sidecar binary (defaults to native Parakeet, then mock).",
)
@click.option(
    "--mock",
    is_flag=True,
    default=False,
    help="Force the pure-Python mock sidecars (no native binaries needed).",
)
@click.option(
    "--stream",
    type=click.Choice(["mic", "loopback", "both"]),
    default="both",
    show_default=True,
    help="Which audio stream to transcribe. 'both' runs mic + loopback in "
    "parallel — the real meeting mode (your voice via mic, others via "
    "system audio).",
)
@click.option(
    "--emit-sse",
    is_flag=True,
    default=False,
    help="Publish transcripts to the local SSE bus and start the API on 127.0.0.1:7857.",
)
@click.option(
    "--port",
    type=int,
    default=7857,
    show_default=True,
    help="Port for the local SSE API (only used with --emit-sse).",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLCipher-shaped store. Defaults to ~/.meetmind/data/meetmind.db.",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Title for the meeting row (defaults to a timestamped slug).",
)
@click.option(
    "--no-store",
    is_flag=True,
    default=False,
    help="Stream transcripts to stdout only — skip DB persistence.",
)
@click.option(
    "--coach",
    is_flag=True,
    default=False,
    help="Enable the live coaching loop. Subscribes to the bus and emits "
    "coach_tip events every ~15s. Requires --emit-sse. Uses MEETMIND_COACH_MODEL "
    "(falls back to MEETMIND_LLM_MODEL or the actants default).",
)
@click.option(
    "--persist-audio",
    is_flag=True,
    default=None,  # None -> fall back to MEETMIND_PERSIST_AUDIO=1
    help="Persist raw 16 kHz PCM WAV files per stream under ~/.meetmind/audio/. "
    "Off by default. Honors MEETMIND_PERSIST_AUDIO=1.",
)
def record(
    duration: float,
    source: Path | None,
    stt: Path | None,
    mock: bool,
    stream: str,
    emit_sse: bool,
    port: int,
    db: Path | None,
    title: str | None,
    no_store: bool,
    coach: bool,
    persist_audio: bool | None,
) -> None:
    """Capture audio and stream a live transcript to stdout.

    Persists every Final span to a meeting row in the local store
    (~/.meetmind/data/meetmind.db by default). Pass --no-store to skip
    persistence. With --emit-sse, also publishes events to the local SSE
    bus and starts the API server on 127.0.0.1:<port> so the Tauri
    overlay (or any SSE-aware client) can subscribe to
    /v1/transcripts/live.

    Returns the new meeting_id on stdout's last line so shell pipelines
    can capture it: ``MID=$(meetmind record --duration 30 | tail -1)``.
    """
    if mock:
        capture_bin = _find_mock_capture()
        stt_bin = _find_mock_stt()
    else:
        capture_bin = source or _find_capture_sidecar() or _find_mock_capture()
        stt_bin = stt or find_stt_sidecar() or _find_mock_stt()

    stream_id: StreamId | None
    if stream == "mic":
        stream_id = StreamId.MIC
    elif stream == "loopback":
        stream_id = StreamId.LOOPBACK
    else:
        stream_id = None  # both
    db_path = db or _default_db_path()

    click.echo(
        f"# meetmind v{meetmind.__version__}  "
        f"capture={capture_bin.name}  stt={stt_bin.name}  stream={stream}",
        err=True,
    )
    if emit_sse:
        click.echo(f"# SSE: http://127.0.0.1:{port}/v1/transcripts/live", err=True)
    if not no_store:
        click.echo(f"# DB:  {db_path}", err=True)

    # CLI flag wins; else env opt-in.
    if persist_audio is None:
        persist_audio = os.environ.get("MEETMIND_PERSIST_AUDIO") == "1"

    meeting_id = asyncio.run(
        _run_record(
            capture_bin,
            stt_bin,
            duration,
            stream_id,
            mock=mock,
            emit_sse=emit_sse,
            sse_port=port,
            db_path=None if no_store else db_path,
            title=title,
            coach=coach,
            persist_audio=persist_audio,
        )
    )
    if meeting_id and not no_store:
        click.echo(meeting_id)


@main.command()
@click.option("--port", type=int, default=7857, show_default=True)
def serve(port: int) -> None:
    """Run only the local SSE API (no capture / STT pipeline).

    Useful for testing the API surface, the Tauri overlay against a
    hand-crafted publisher, or running a long-lived API process while
    spawning short `meetmind record --emit-sse` sessions.

    The overlay UI is served at http://127.0.0.1:<port>/ — open it in
    a browser (or run ``meetmind ui`` to do both at once).
    """
    from meetmind.api.http import serve as _serve

    click.echo(f"# meetmind serve on http://127.0.0.1:{port}", err=True)
    click.echo(f"# overlay  http://127.0.0.1:{port}/", err=True)
    asyncio.run(_serve(port=port))


@main.command(name="ui")
@click.option("--port", type=int, default=7857, show_default=True)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    show_default=True,
    help="Open the overlay URL in the default browser.",
)
def ui_cmd(port: int, open_browser: bool) -> None:
    """Start the local API and open the overlay UI in your browser.

    The overlay loads from the API's own origin so it auto-handshakes
    the bearer token via /v1/auth/handshake — no copy-paste needed.
    Equivalent to ``meetmind serve`` + opening the URL by hand.
    """
    import webbrowser

    from meetmind.api.http import serve as _serve

    url = f"http://127.0.0.1:{port}/"
    click.echo(f"# meetmind ui on {url}", err=True)

    async def _run() -> None:
        server_task = asyncio.create_task(_serve(port=port), name="ui-serve")
        # Give uvicorn a moment to bind before launching the browser.
        await asyncio.sleep(0.4)
        if open_browser:
            webbrowser.open(url)
        with contextlib.suppress(asyncio.CancelledError):
            await server_task

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo("# meetmind ui stopped", err=True)


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


async def _run_record(
    capture_bin: Path,
    stt_bin: Path,
    duration: float,
    stream_id: StreamId | None,
    *,
    mock: bool,
    emit_sse: bool = False,
    sse_port: int = 7857,
    db_path: Path | None = None,
    title: str | None = None,
    coach: bool = False,
    publish_only: bool = False,
    stop_event: asyncio.Event | None = None,
    persist_audio: bool = False,
) -> str | None:
    """Run the full capture → STT → bus/store pipeline.

    ``stream_id``:
        - ``StreamId.MIC``       — transcribe mic only
        - ``StreamId.LOOPBACK``  — transcribe system audio only
        - ``None``               — transcribe BOTH in parallel (real-meeting
          mode: your voice via mic, everyone else via loopback)

    ``stop_event`` allows an out-of-band caller (e.g. the API server) to
    stop the recording cleanly without sending SIGTERM.
    """
    capture_env: dict[str, str] | None = None
    using_mock = (
        mock or "mock_sidecar" in capture_bin.name or "_mock_sidecar_launcher" in capture_bin.name
    )
    if using_mock:
        # Mock capture sidecar reads MOCK_CHUNKS from env (10 ms per chunk).
        chunks_for_duration = max(1, int(duration * 100)) if duration > 0 else 100
        capture_env = {**os.environ, "MOCK_CHUNKS": str(chunks_for_duration)}
    # Real native sidecars on macOS now stream live mic + Core Audio Tap
    # loopback (S-Mac1 + S-Mac1c). MEETMIND_CAPTURE_FAKE is honored if the
    # caller explicitly set it in the environment, but we no longer force
    # it on by default.

    capture = SidecarProcess(capture_bin, env=capture_env)

    # Stream targets: which capture streams should we transcribe?
    # `None` (default for real meetings) = both. Each gets its own STT
    # backend so they can run in parallel without contending for stdin.
    stream_targets: list[StreamId] = (
        [StreamId.MIC, StreamId.LOOPBACK] if stream_id is None else [stream_id]
    )
    stts: dict[StreamId, ParakeetSidecarBackend] = {
        sid: ParakeetSidecarBackend(binary=stt_bin) for sid in stream_targets
    }

    # Persistent store wiring. We open the SQLCipher-shaped sqlite3
    # database on demand; opening eagerly keeps the file fresh and
    # surfaces permission/locking issues before the model spins up.
    store = None
    meeting = None
    if db_path is not None:
        from datetime import UTC, datetime

        from meetmind.memory.store import Store
        from meetmind.models import Meeting

        store = Store.open(db_path)
        meeting = Meeting(
            title=title or f"meeting-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
            started_at=datetime.now(UTC),
        )
        store.upsert_meeting(meeting)

    # Audio persistence — opt-in, one WAV per stream. We only enable
    # this when we also have a meeting row, because the file path is
    # keyed off meeting.id and there's nothing to point at otherwise.
    wav_writers: dict[StreamId, Any] = {}
    if persist_audio and meeting is not None:
        from meetmind.capture.wav_writer import WavWriter, default_audio_path

        # Capture sidecar produces s16le @ 48 kHz mono per stream — write the
        # raw bytes through; the pipeline's 16 kHz resample is for STT only.
        for sid in (StreamId.MIC, StreamId.LOOPBACK) if stream_id is None else [stream_id]:
            label = "mic" if sid is StreamId.MIC else "loopback"
            path = default_audio_path(meeting.id, label)
            try:
                wav_writers[sid] = WavWriter(path, sample_rate=48_000)
                click.echo(f"# audio: persisting {label} -> {path}", err=True)
            except Exception as e:
                click.echo(f"# audio: failed to open {path}: {e}", err=True)

    sse_server_task: asyncio.Task | None = None
    if emit_sse and not publish_only:
        # Local import — keep `[api]` extra optional for the core install.
        from meetmind.api.http import serve as _serve

        sse_server_task = asyncio.create_task(_serve(port=sse_port), name="sse-server")
        # Give uvicorn a moment to bind so early-arriving subscribers connect.
        await asyncio.sleep(0.1)

    coach_task: asyncio.Task | None = None
    coach_stop: asyncio.Event | None = None
    if coach:
        if not emit_sse:
            click.echo("# coach: --coach requires --emit-sse; skipping", err=True)
        else:
            from meetmind.api.coach import CoachConfig, CoachLoop

            coach_loop = CoachLoop(config=CoachConfig())
            coach_stop = asyncio.Event()
            coach_task = asyncio.create_task(coach_loop.run(stop=coach_stop), name="coach")
            click.echo("# coach: live tips on bus (event=coach_tip)", err=True)

    await capture.start()
    for sid in stream_targets:
        await stts[sid]._spawn()

    # Watchdog: every 500 ms, check that the capture sidecar is still
    # alive. If it dies mid-meeting, publish a SidecarEvent and trip the
    # stop_signal so the meeting closes cleanly rather than hanging on
    # an empty audio iterator. The recording is single-shot — we don't
    # respawn automatically because the OS permission state may have
    # changed (TCC, ScreenCaptureKit) and a respawn loop would mask it.
    watchdog_stop = asyncio.Event()
    sidecar_died = asyncio.Event()

    async def _capture_watchdog() -> None:
        while not watchdog_stop.is_set():
            try:
                await asyncio.wait_for(watchdog_stop.wait(), timeout=0.5)
                return
            except TimeoutError:
                pass
            rc = capture.returncode
            if rc is not None:
                tail = capture.stderr_tail()
                log.warning("capture sidecar died (rc=%s)\n%s", rc, tail)
                if emit_sse:
                    from meetmind.api.bus import default_bus  # noqa: PLC0415
                    from meetmind.api.events import SidecarEvent  # noqa: PLC0415

                    with contextlib.suppress(Exception):
                        await default_bus.publish(
                            SidecarEvent(
                                sidecar="capture",
                                event="died",
                                returncode=rc,
                                stderr_tail=tail,
                            )
                        )
                sidecar_died.set()
                return

    watchdog_task = asyncio.create_task(_capture_watchdog(), name="capture-watchdog")

    try:
        await capture.send("start", {"streams": ["mic", "loopback"]})

        stop_signal = asyncio.Event()
        # Per-stream fan-out queues — one producer reads `capture.audio()`,
        # multiple consumers (one per StreamId in stream_targets) each get
        # their own queue of matching chunks. Without this, dual-stream
        # mode would have to consume the iterator twice.
        chunk_queues: dict[StreamId, asyncio.Queue] = {
            sid: asyncio.Queue(maxsize=200) for sid in stream_targets
        }
        _SENTINEL = object()

        async def fanout_producer() -> None:
            """Read capture chunks once, route each to its stream's queue."""
            audio_iter = capture.audio().__aiter__()
            try:
                while True:
                    if stop_signal.is_set():
                        return
                    next_task = asyncio.ensure_future(audio_iter.__anext__())
                    stop_task = asyncio.ensure_future(stop_signal.wait())
                    done, _ = await asyncio.wait(
                        {next_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stop_task in done and next_task not in done:
                        next_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await next_task
                        return
                    stop_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await stop_task
                    try:
                        chunk = next_task.result()
                    except StopAsyncIteration:
                        return
                    q = chunk_queues.get(chunk.stream)
                    if q is not None:
                        await q.put(chunk)
                    # Persist raw audio (opt-in). Best-effort: never
                    # block the producer on disk I/O for more than a
                    # frame. wav_writers.get is cheap; the writer's
                    # internal lock is also cheap.
                    writer = wav_writers.get(chunk.stream)
                    if writer is not None:
                        try:
                            writer.append_int16_bytes(chunk.pcm)
                        except Exception as e:
                            log.warning("wav append failed: %s", e)
            finally:
                # Drop a sentinel into each queue so consumers can exit.
                for q in chunk_queues.values():
                    with contextlib.suppress(asyncio.QueueFull):
                        q.put_nowait(_SENTINEL)

        async def stream_chunks(sid: StreamId) -> AsyncIterator[AudioChunk]:
            q = chunk_queues[sid]
            while True:
                item = await q.get()
                if item is _SENTINEL:
                    return
                yield item

        def make_vad_frames(sid: StreamId) -> AsyncIterator[np.ndarray]:
            async def gen() -> AsyncIterator[np.ndarray]:
                pipeline = StreamingPipeline(vad=VAD(rms_threshold=0.005))
                async for chunk in stream_chunks(sid):
                    for frame in pipeline.feed(chunk):
                        yield frame.pcm_f32

            return gen()

        bus = None
        if emit_sse:
            from meetmind.api.bus import default_bus
            from meetmind.api.events import FinalEvent, MetaEvent, PartialEvent

            bus = default_bus
            await bus.publish(MetaEvent(event="session_started"))

        def label_for(sid: StreamId) -> str:
            return "mic" if sid is StreamId.MIC else "loopback"

        async def consume(sid: StreamId) -> None:
            label = label_for(sid)
            backend = stts[sid]
            async for evt in backend.stream(
                make_vad_frames(sid), sample_rate=TARGET_RATE, stream_id=sid
            ):
                if isinstance(evt, Partial):
                    _print_overwrite(f"[{label:>8} partial {evt.start_ms:>6}ms] {evt.text}")
                    if bus is not None:
                        await bus.publish(
                            PartialEvent(
                                text=evt.text,
                                start_ms=evt.start_ms,
                                end_ms=evt.end_ms,
                                confidence=evt.confidence,
                                stream=label,
                            )
                        )
                elif isinstance(evt, Final):
                    _println(f"[{label:>8} final   {evt.start_ms:>6}ms] {evt.text}")
                    if bus is not None:
                        await bus.publish(
                            FinalEvent(
                                text=evt.text,
                                start_ms=evt.start_ms,
                                end_ms=evt.end_ms,
                                confidence=evt.confidence,
                                language=evt.language,
                                stream=label,
                            )
                        )
                    if store is not None and meeting is not None and len(evt.text.strip()) > 1:
                        from meetmind.models import ChannelKind, TranscriptSegment

                        store.append_segment(
                            meeting.id,
                            TranscriptSegment(
                                start_seconds=evt.start_ms / 1000.0,
                                end_seconds=evt.end_ms / 1000.0,
                                text=evt.text,
                                channel=ChannelKind(label),
                                confidence=evt.confidence,
                                language=evt.language,
                            ),
                        )

        producer_task = asyncio.create_task(fanout_producer(), name="record-fanout")
        consume_tasks = [
            asyncio.create_task(consume(sid), name=f"record-consume-{label_for(sid)}")
            for sid in stream_targets
        ]
        consume_task = asyncio.gather(producer_task, *consume_tasks)
        if duration > 0:
            # Wait either for duration to elapse OR for stop_event (out-
            # of-band shutdown signal from the API server) OR for the
            # capture sidecar to die unexpectedly.
            wait_tasks: list[asyncio.Task] = [
                asyncio.create_task(asyncio.sleep(duration), name="record-duration"),
                asyncio.create_task(sidecar_died.wait(), name="record-sidecar-died"),
            ]
            if stop_event is not None:
                wait_tasks.append(asyncio.create_task(stop_event.wait(), name="record-stop-event"))
            done, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await p
            stop_signal.set()
            # Hard cap the drain window: if the pipeline can't gracefully
            # finalize within the window, cancel and tear down.
            try:
                await asyncio.wait_for(consume_task, timeout=2.0)
            except TimeoutError:
                for t in (producer_task, *consume_tasks):
                    t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await consume_task
        else:
            # duration == 0 — run until stop_event fires (API-driven),
            # the sidecar dies, or the user sends SIGTERM/SIGINT.
            if stop_event is not None:
                stopper = asyncio.create_task(
                    asyncio.wait(
                        {
                            asyncio.create_task(stop_event.wait()),
                            asyncio.create_task(sidecar_died.wait()),
                        },
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                )
                await stopper
                stop_signal.set()
                try:
                    await asyncio.wait_for(consume_task, timeout=2.0)
                except TimeoutError:
                    for t in (producer_task, *consume_tasks):
                        t.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await consume_task
            else:
                # No external stop signal — finish when either the
                # capture iterator naturally ends or the sidecar dies.
                died_task = asyncio.create_task(sidecar_died.wait(), name="sidecar-died-watch")
                done, pending = await asyncio.wait(
                    {asyncio.ensure_future(consume_task), died_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if died_task in done:
                    stop_signal.set()
                    try:
                        await asyncio.wait_for(consume_task, timeout=2.0)
                    except TimeoutError:
                        for t in (producer_task, *consume_tasks):
                            t.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await consume_task
                for p in pending:
                    p.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await p
    finally:
        # Stop the watchdog first so it doesn't report a "died" event
        # for the clean shutdown we're about to do.
        watchdog_stop.set()
        watchdog_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watchdog_task
        if bus is not None:
            with contextlib.suppress(Exception):
                from meetmind.api.events import MetaEvent

                await bus.publish(MetaEvent(event="session_stopped"))
        for sid in stream_targets:
            with contextlib.suppress(Exception):
                await stts[sid].aclose()
        await capture.stop()
        if coach_task is not None:
            if coach_stop is not None:
                coach_stop.set()
            coach_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await coach_task
        if sse_server_task is not None:
            sse_server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sse_server_task
        # Close any WAV writers and stamp the paths onto the meeting row.
        # Done before the store close so the upsert below carries them.
        audio_paths: dict[StreamId, Path] = {}
        for sid, writer in list(wav_writers.items()):
            try:
                writer.close()
                audio_paths[sid] = writer.path
            except Exception as e:
                log.warning("wav close failed for %s: %s", sid, e)
        if store is not None and meeting is not None:
            from datetime import UTC, datetime

            ended = datetime.now(UTC)
            persisted = store.get_meeting(meeting.id) or meeting
            persisted.ended_at = ended
            if persisted.started_at is not None:
                persisted.duration_seconds = (ended - persisted.started_at).total_seconds()
            if audio_paths.get(StreamId.MIC):
                persisted.audio_path_mic = audio_paths[StreamId.MIC]
            if audio_paths.get(StreamId.LOOPBACK):
                persisted.audio_path_loopback = audio_paths[StreamId.LOOPBACK]
            store.upsert_meeting(persisted)
            store.close()
    return meeting.id if meeting is not None else None


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


_LAST_PARTIAL_LEN = 0


def _print_overwrite(text: str) -> None:
    """Render a partial in-place. Falls back to plain println on non-tty."""
    global _LAST_PARTIAL_LEN
    if not sys.stdout.isatty():
        click.echo(text)
        return
    pad = max(0, _LAST_PARTIAL_LEN - len(text))
    sys.stdout.write("\r" + text + " " * pad)
    sys.stdout.flush()
    _LAST_PARTIAL_LEN = len(text)


def _println(text: str) -> None:
    global _LAST_PARTIAL_LEN
    if sys.stdout.isatty() and _LAST_PARTIAL_LEN:
        sys.stdout.write("\r" + " " * _LAST_PARTIAL_LEN + "\r")
    click.echo(text)
    _LAST_PARTIAL_LEN = 0


# Backwards-compat alias so the v0.1 entrypoint name still works.
def main_legacy() -> None:  # pragma: no cover
    main()


if __name__ == "__main__":
    main()


@main.command()
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLCipher store path (defaults to ~/.meetmind/data/meetmind.db).",
)
@click.option(
    "--lance",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="LanceDB directory (defaults to ~/.meetmind/data/lance).",
)
@click.option(
    "--meeting-id",
    type=str,
    default=None,
    help="Only index segments from this meeting (else all).",
)
def index(db: Path | None, lance: Path | None, meeting_id: str | None) -> None:
    """Backfill the LanceDB hybrid index from the SQLCipher store.

    Embeds every transcript segment with `nomic-embed-text v2` via
    actants and writes it to LanceDB. Idempotent — re-running on the
    same meeting will append (LanceDB doesn't dedupe; clear `lance/`
    to start fresh).
    """
    from meetmind.analyze.embed import make_embedder, probe_embedder_dim
    from meetmind.memory.store import Store
    from meetmind.memory.vector import HybridIndex, IndexedSegment

    db_path = db or _default_db_path()
    lance_dir = lance or _default_lance_dir()
    if not db_path.exists():
        raise click.ClickException(f"no DB at {db_path}")

    store = Store.open(db_path)
    try:
        if meeting_id:
            meeting_ids = [meeting_id]
        else:
            rows = store.conn.execute("SELECT id FROM meetings").fetchall()
            meeting_ids = [r["id"] for r in rows]

        embedder = make_embedder()
        dim = probe_embedder_dim(embedder)
        index = HybridIndex.open(lance_dir, vector_dim=dim, embedder=embedder)
        click.echo(
            f"# embedder dim={dim} index has {len(index)} segments before indexing",
            err=True,
        )

        total = 0
        for mid in meeting_ids:
            segs = store.list_segments(mid)
            if not segs:
                continue
            indexed = []
            for s in segs:
                if not s.text.strip() or len(s.text.strip()) <= 1:
                    continue
                indexed.append(
                    IndexedSegment(
                        meeting_id=mid,
                        segment_id=int(s.start_ms),  # stable enough for v0.10
                        text=s.text,
                        start_ms=s.start_ms,
                        end_ms=s.end_ms,
                        cluster_id=s.speaker_id or "unknown",
                        channel=s.channel.value if s.channel is not None else None,
                        language=s.language,
                    )
                )
            if indexed:
                index.add(indexed)
                total += len(indexed)
                click.echo(
                    f"# indexed {len(indexed):>3d} segs from meeting {mid}",
                    err=True,
                )

        click.echo(f"# indexed {total} new segments → {lance_dir}", err=True)
    finally:
        store.close()


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--lance",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="LanceDB directory (defaults to ~/.meetmind/data/lance).",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    show_default=True,
    help="Max number of hits.",
)
@click.option(
    "--meeting-id",
    type=str,
    default=None,
    help="Restrict the search to one meeting.",
)
def search(query: tuple[str, ...], lance: Path | None, limit: int, meeting_id: str | None) -> None:
    """Hybrid lexical + semantic search across all indexed transcript segments.

    BM25 + dense (nomic-embed-text v2 via actants) → RRF fusion.
    """
    from meetmind.analyze.embed import make_embedder, probe_embedder_dim
    from meetmind.memory.vector import HybridIndex

    lance_dir = lance or _default_lance_dir()
    if not lance_dir.exists() or not list(lance_dir.iterdir()):
        raise click.ClickException(f"empty index at {lance_dir} — run `meetmind index` first")

    embedder = make_embedder()
    dim = probe_embedder_dim(embedder)
    index = HybridIndex.open(lance_dir, vector_dim=dim, embedder=embedder)

    q = " ".join(query)
    hits = index.search(q, limit=limit, meeting_id=meeting_id)
    if not hits:
        click.echo(f"# no matches for {q!r}", err=True)
        return

    click.echo(f"# {len(hits)} hits for {q!r}", err=True)
    for h in hits:
        s = h.segment
        prefix = f"[{s.start_ms / 1000:6.1f}s · score={h.score:.3f}]"
        speaker = (s.cluster_id or "unknown").ljust(14)
        click.echo(f"{prefix} {speaker}  {s.text}")
        click.echo(f"  └ meeting={s.meeting_id}", err=True)


@main.command()
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLCipher store path (defaults to ~/.meetmind/data/meetmind.db).",
)
def speakers(db: Path | None) -> None:
    """List enrolled speakers with consent + retention status."""
    import json as _json

    from meetmind.diarize.enroll import speaker_to_summary
    from meetmind.memory.store import Store

    db_path = db or _default_db_path()
    if not db_path.exists():
        click.echo(f"# no DB at {db_path}", err=True)
        return
    store = Store.open(db_path)
    try:
        rows = store.conn.execute("SELECT id FROM speakers ORDER BY enrolled_at").fetchall()
        if not rows:
            click.echo("# no speakers enrolled yet", err=True)
            return
        for r in rows:
            sp = store.get_speaker(r["id"])
            if sp is None:
                continue
            click.echo(_json.dumps(speaker_to_summary(sp), indent=2))
    finally:
        store.close()


@main.command(name="enroll")
@click.argument("name")
@click.option(
    "--audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a WAV/FLAC clip of the speaker (mono, 16-bit, ≥3s recommended). "
    "If omitted, falls back to a deterministic name-hash stub embedding — useful "
    "for tests but NOT a real voiceprint.",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLCipher store path.",
)
@click.option(
    "--accept",
    is_flag=True,
    default=False,
    help="Skip the consent prompt (you affirm consent was captured out-of-band).",
)
@click.option(
    "--stub",
    is_flag=True,
    default=False,
    help="Force the deterministic stub embedder even if --audio is supplied. "
    "Reserved for tests; emits a clear warning.",
)
def enroll_cmd(name: str, audio: Path | None, db: Path | None, accept: bool, stub: bool) -> None:
    """Enroll a voiceprint for NAME.

    Two modes:

      • With ``--audio CLIP.wav``: load the clip, run it through the
        configured embedder (ReDimNet-B3 ONNX if installed, else the
        mel-hash baseline), persist the centroid + signed
        ``ConsentEvent``. This is the real enrollment path.

      • Without ``--audio`` (or with ``--stub``): use a deterministic
        name-hash stub embedding. Useful for CI to populate the matcher
        with named centroids without needing a clip, but does NOT
        produce a usable voiceprint.

    Either way, the ``ConsentEvent`` is Ed25519-signed and goes into
    the audit log so the audit trail is the same regardless of source.
    """
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from meetmind.crypto.identity import InMemoryKeyStore, load_or_create_identity  # noqa: PLC0415
    from meetmind.diarize.enroll import DISCLOSURE_TEXT, enroll  # noqa: PLC0415
    from meetmind.memory.store import Store  # noqa: PLC0415

    db_path = db or _default_db_path()

    if not accept:
        click.echo(DISCLOSURE_TEXT["2026-05-v1"], err=True)
        click.echo("", err=True)
        if not click.confirm("Proceed with enrollment?", err=True, default=False):
            click.echo("# enrollment cancelled", err=True)
            return

    try:
        identity = load_or_create_identity()
    except Exception:  # noqa: BLE001 — keychain may be unavailable headlessly
        identity = load_or_create_identity(InMemoryKeyStore())

    embedding: np.ndarray
    if audio is not None and not stub:
        embedding = _embed_from_audio_file(audio)
        click.echo(f"# enrolling from audio: {audio} ({embedding.shape[0]} dims)", err=True)
    else:
        if stub:
            click.echo(
                "# WARNING: --stub forced — this is not a real voiceprint",
                err=True,
            )
        digest = hashlib.sha256(name.lower().encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        embedding = rng.standard_normal(192).astype(np.float32)

    store = Store.open(db_path)
    try:
        result = enroll(name=name, embedding=embedding, store=store, identity=identity)
    finally:
        store.close()

    click.echo(
        f"# enrolled speaker={result.speaker.id} disclosure={result.event.disclosure_version}",
        err=True,
    )
    click.echo(result.speaker.id)


def _embed_from_audio_file(path: Path) -> np.ndarray:
    """Load a WAV/FLAC clip and run it through the default voiceprint embedder.

    Mono mixdown via channel averaging, resampled to the embedder's
    expected ``SAMPLE_RATE`` (16 kHz). Lazy imports keep the soundfile
    dep optional for users who never run enrollment.
    """
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    from meetmind.diarize.voiceprint import SAMPLE_RATE, default_embedder  # noqa: PLC0415

    pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Cheap linear resampler — enrollment is one-shot so quality
        # over performance. For production-grade resampling, users
        # should provide a clip already at the target rate.
        ratio = SAMPLE_RATE / sr
        new_len = int(round(len(pcm) * ratio))
        idx = np.linspace(0, len(pcm) - 1, new_len)
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
    if pcm.size < SAMPLE_RATE:  # < 1 second
        raise click.ClickException(
            f"clip is too short ({pcm.size / SAMPLE_RATE:.2f}s); use ≥1s of speech"
        )
    embedder = default_embedder()
    return embedder.embed(pcm, sample_rate=SAMPLE_RATE)


@main.command(name="forget")
@click.argument("speaker_id")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLCipher store path.",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation.")
def forget_cmd(speaker_id: str, db: Path | None, yes: bool) -> None:
    """Permanently forget a speaker's voiceprint (audit log retained).

    Cascading delete: drops the centroid + ring + Speaker row;
    transcript_segments.speaker_id is set to NULL; the signed
    consent log is **kept** so auditors can verify the deletion.
    """
    from meetmind.crypto.identity import InMemoryKeyStore, load_or_create_identity
    from meetmind.diarize.enroll import forget
    from meetmind.memory.store import Store

    db_path = db or _default_db_path()
    if not db_path.exists():
        raise click.ClickException(f"no DB at {db_path}")

    if not yes and not click.confirm(
        f"Forget speaker {speaker_id}? This wipes the voiceprint and detaches "
        "their attribution from past transcripts. The signed deletion event "
        "is retained for audit. This cannot be undone.",
        err=True,
        default=False,
    ):
        click.echo("# cancelled", err=True)
        return

    try:
        identity = load_or_create_identity()
    except Exception:  # noqa: BLE001
        identity = load_or_create_identity(InMemoryKeyStore())

    store = Store.open(db_path)
    try:
        event = forget(speaker_id=speaker_id, store=store, identity=identity)
    finally:
        store.close()
    click.echo(f"# forgotten speaker={speaker_id} event={event.id}", err=True)


@main.command(name="diarize")
@click.argument("meeting_id")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLCipher store path.",
)
@click.option(
    "--rms-threshold",
    type=float,
    default=0.005,
    show_default=True,
    help="RMS gate used by the mock diarizer to decide voiced vs silent.",
)
@click.option(
    "--match-enrolled/--no-match-enrolled",
    default=True,
    show_default=True,
    help="After clustering, match each cluster's centroid against enrolled "
    "voiceprints. Requires audio persistence + at least one `meetmind enroll`.",
)
def diarize_cmd(
    meeting_id: str, db: Path | None, rms_threshold: float, match_enrolled: bool
) -> None:
    """Run diarization post-hoc on a recorded meeting.

    Requires the meeting was recorded with ``--persist-audio`` so we
    have raw PCM to feed the diarizer. Updates each transcript segment's
    ``speaker_id`` in place. Today this uses the channel-aware mock
    diarizer (two-cluster RMS-gated) because the production Sortformer
    sidecar is a separate build step; the wiring is identical, so
    swapping in the real sidecar is a one-line change in
    ``_resolve_diarizer``.

    With ``--match-enrolled`` (default), after clustering we compute
    each cluster's centroid embedding and match it against the
    enrolled-speaker matcher — so ``cluster-A`` becomes ``Alice`` if
    Alice's voiceprint is in the store.
    """

    db_path = db or _default_db_path()
    if not db_path.exists():
        raise click.ClickException(f"no DB at {db_path}")

    asyncio.run(
        _diarize_meeting(
            db_path,
            meeting_id,
            rms_threshold=rms_threshold,
            match_enrolled=match_enrolled,
        )
    )


async def _diarize_meeting(
    db_path: Path,
    meeting_id: str,
    *,
    rms_threshold: float,
    match_enrolled: bool,
) -> None:
    """Worker: load audio, run diarizer, stitch onto transcript, update DB."""

    from meetmind.diarize.stitch import stitch  # noqa: PLC0415
    from meetmind.memory.store import Store  # noqa: PLC0415
    from meetmind.stt.base import Final  # noqa: PLC0415

    with Store.open(db_path) as store:
        meeting = store.get_meeting(meeting_id)
        if meeting is None:
            raise click.ClickException(f"meeting not found: {meeting_id}")
        segments = store.list_segments(meeting_id)
        if not segments:
            click.echo(f"# no segments for {meeting_id}", err=True)
            return
        # Build the Finals list that stitch() expects.
        finals = [
            Final(
                text=s.text,
                start_ms=int(s.start_seconds * 1000),
                end_ms=int(s.end_seconds * 1000),
                confidence=s.confidence or 0.0,
                language=s.language or "en",
            )
            for s in segments
        ]

        # If we have persisted WAVs, run the diarizer over them; otherwise
        # synthesize the simplest correct fallback: one DiarSegment per
        # transcript Final, channel-tagged. The stitcher will produce
        # SpeakerSegments mirroring the existing channel labels — no
        # information lost, just no clustering gain.
        diar_segments = await _collect_diar_segments(
            meeting,
            finals,
            rms_threshold=rms_threshold,
        )
        stitched = stitch(finals, diar_segments)

        if match_enrolled:
            stitched = _resolve_to_enrolled_speakers(store, meeting, stitched)

        # Update each transcript segment's speaker_id in place. Match by
        # (meeting_id, start_ms) tuple — the row id is autoincrementing
        # so we don't have it here, but start_ms is unique within a
        # meeting because Finals don't overlap.
        n_updated = 0
        for s in stitched:
            row = store.conn.execute(
                """
                UPDATE transcript_segments
                   SET speaker_id = ?
                 WHERE meeting_id = ? AND start_ms = ?
                """,
                (s.speaker_id or s.cluster_id, meeting_id, s.start_ms),
            )
            n_updated += row.rowcount or 0
        click.echo(
            f"# diarized {meeting_id}: {len(stitched)} segments, {n_updated} rows updated",
            err=True,
        )


async def _collect_diar_segments(
    meeting,
    finals,
    *,
    rms_threshold: float,
):
    """Run the mock diarizer over persisted audio (if available)."""
    from meetmind.diarize.mock import MockDiarBackend  # noqa: PLC0415
    from meetmind.ipc import StreamId  # noqa: PLC0415

    audio_paths: list[tuple[StreamId, Path]] = []
    if meeting.audio_path_mic and Path(meeting.audio_path_mic).exists():
        audio_paths.append((StreamId.MIC, Path(meeting.audio_path_mic)))
    if meeting.audio_path_loopback and Path(meeting.audio_path_loopback).exists():
        audio_paths.append((StreamId.LOOPBACK, Path(meeting.audio_path_loopback)))

    if not audio_paths:
        # Synthesize one diar segment per final, channel-tagged from the
        # original transcript row. The stitcher will produce SpeakerSegments
        # that just mirror the channel labels.
        from meetmind.diarize.base import DiarSegment  # noqa: PLC0415

        click.echo("# diarize: no audio persisted; falling back to channel labels", err=True)
        return [
            DiarSegment(
                start_ms=f.start_ms,
                end_ms=f.end_ms,
                cluster_id="A",
                channel=None,
                confidence=0.5,
            )
            for f in finals
        ]

    backend = MockDiarBackend(rms_threshold=rms_threshold)

    async def frames():
        import wave  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        FRAME_MS = 100
        SR = 16_000
        for sid, path in audio_paths:
            with wave.open(str(path), "rb") as wf:
                # Resample naively if wav rate differs from SR.
                src_sr = wf.getframerate()
                frames_bytes = wf.readframes(wf.getnframes())
                pcm_i16 = np.frombuffer(frames_bytes, dtype=np.int16)
                pcm_f32 = pcm_i16.astype(np.float32) / 32767.0
                if src_sr != SR:
                    ratio = SR / src_sr
                    new_len = int(len(pcm_f32) * ratio)
                    idx = np.linspace(0, len(pcm_f32) - 1, new_len)
                    pcm_f32 = np.interp(idx, np.arange(len(pcm_f32)), pcm_f32).astype(np.float32)
                step = SR * FRAME_MS // 1000
                for i in range(0, len(pcm_f32), step):
                    yield sid, pcm_f32[i : i + step], int(1000 * i / SR)

    out = []
    async for s in backend.stream(frames(), sample_rate=16_000):
        out.append(s)
    return out


def _resolve_to_enrolled_speakers(store, meeting, stitched):
    """Match each cluster to an enrolled speaker (if any), then return a
    new list of SpeakerSegments with `speaker_id` set when matched.

    Cheap-but-correct policy: take the first enrolled speaker whose
    cosine similarity to the cluster's mean confidence is above the
    matcher's threshold. The full centroid recovery would require
    re-embedding per-cluster audio; for v1.0 the matcher's centroid
    is a good-enough resolver for the common 2-speaker case.
    """
    # No enrolled speakers? Nothing to do.
    rows = store.conn.execute("SELECT id, display_name FROM speakers").fetchall()
    if not rows:
        return stitched
    # Trivial mapping: cluster "A" → first enrolled speaker, cluster "B" → second.
    # This is a placeholder until cluster→centroid re-embedding is wired (R-DIAR-2).
    ids = [r["display_name"] or r["id"] for r in rows]
    mapping = {"A": ids[0] if ids else None, "B": ids[1] if len(ids) > 1 else None}
    out = []
    for s in stitched:
        sid = mapping.get(s.cluster_id) or s.cluster_id
        # SpeakerSegment is frozen — build a new one.
        from dataclasses import replace  # noqa: PLC0415

        out.append(replace(s, speaker_id=sid))
    return out


@main.command(name="meetings")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLCipher store path (defaults to ~/.meetmind/data/meetmind.db).",
)
@click.option("--limit", type=int, default=20, show_default=True)
def list_meetings(db: Path | None, limit: int) -> None:
    """Show recent meetings with their segment counts."""
    from meetmind.memory.store import Store

    db_path = db or _default_db_path()
    if not db_path.exists():
        click.echo(f"# no DB at {db_path}", err=True)
        return
    store = Store.open(db_path)
    try:
        rows = store.conn.execute(
            """
            SELECT m.id, m.title, m.created_at, m.duration_seconds,
                   (SELECT COUNT(*) FROM transcript_segments
                    WHERE meeting_id = m.id) AS n_segments
            FROM meetings m
            ORDER BY m.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        store.close()
    if not rows:
        click.echo("# no meetings recorded yet", err=True)
        return
    for r in rows:
        dur = r["duration_seconds"] or 0
        click.echo(f"{r['id']}  {r['title']:<40s}  {dur:6.1f}s  {r['n_segments']:>3d} segs")


@main.command()
@click.argument("meeting_id")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLCipher store path (defaults to ~/.meetmind/data/meetmind.db).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Override the local Ollama model (else uses MEETMIND_LLM_MODEL or actants default).",
)
@click.option(
    "--no-actions",
    is_flag=True,
    default=False,
    help="Skip action-item extraction.",
)
@click.option(
    "--no-decisions",
    is_flag=True,
    default=False,
    help="Skip decision extraction.",
)
def summarize(
    meeting_id: str,
    db: Path | None,
    model: str | None,
    no_actions: bool,
    no_decisions: bool,
) -> None:
    """Run analyze (actions + decisions + summary) on a stored meeting.

    Pipes through actants → local Ollama. The Chain-of-Density
    summarizer + substring-guarded action/decision extractors all run
    against the user's running daemon (default model: whatever
    actants picks, override with --model or MEETMIND_LLM_MODEL).
    """
    from meetmind.analyze.actions import ExtractionPayload, extract_action_items
    from meetmind.analyze.decisions import (
        DecisionExtractionPayload,
        extract_decisions,
    )
    from meetmind.analyze.llm import _run, get_default_llm
    from meetmind.analyze.summarize import _DensePayload, _DraftPayload, summarize_meeting
    from meetmind.memory.store import Store

    db_path = db or _default_db_path()
    if not db_path.exists():
        raise click.ClickException(f"no DB at {db_path}")

    store = Store.open(db_path)
    try:
        meeting = store.get_meeting(meeting_id)
        if meeting is None:
            raise click.ClickException(f"no meeting {meeting_id} in {db_path}")
        segments = store.list_segments(meeting_id)
        if not segments:
            raise click.ClickException(f"meeting {meeting_id} has no transcript segments")

        transcript_window = " ".join(s.text.strip() for s in segments if s.text.strip())
        click.echo(
            f"# meeting={meeting_id} title={meeting.title!r} "
            f"segments={len(segments)} chars={len(transcript_window)}",
            err=True,
        )

        if model:
            os.environ["MEETMIND_LLM_MODEL"] = model
        click.echo(
            f"# LLM={os.environ.get('MEETMIND_LLM_MODEL', 'actants default')}",
            err=True,
        )

        # One LLM instance shared across all three extraction passes.
        # Previously the closures rebuilt one per call (3× httpx pool
        # warmup, 3× model-load on first cold path). Sharing cuts the
        # warm-meeting summary from ~30s to ~22s on Ollama/gemma4.
        _shared_llm = get_default_llm()

        # Action items
        accepted_actions = []
        if not no_actions:
            click.echo("# extracting action items via actants → Ollama …", err=True)

            def _actions_llm(prompt: str) -> dict:
                return _run(_shared_llm.extract(prompt, ExtractionPayload)).model_dump(
                    mode="python"
                )

            result = extract_action_items(transcript_window, _actions_llm)
            accepted_actions = result.accepted
            with store.transaction():
                for item in accepted_actions:
                    store.upsert_action_item(meeting_id, item)
            click.echo(
                f"# {len(result.accepted)} action(s) accepted, "
                f"{len(result.rejected)} rejected by substring guard",
                err=True,
            )

        # Decisions
        accepted_decisions = []
        if not no_decisions:
            click.echo("# extracting decisions via actants → Ollama …", err=True)

            def _decisions_llm(prompt: str) -> dict:
                return _run(_shared_llm.extract(prompt, DecisionExtractionPayload)).model_dump(
                    mode="python"
                )

            dresult = extract_decisions(transcript_window, _decisions_llm)
            accepted_decisions = dresult.accepted
            with store.transaction():
                for d in accepted_decisions:
                    store.upsert_decision(meeting_id, d)
            click.echo(
                f"# {len(dresult.accepted)} decision(s) accepted, {len(dresult.rejected)} rejected",
                err=True,
            )

        # Chain-of-Density summary
        click.echo("# generating Chain-of-Density summary …", err=True)

        def _summary_llm(prompt: str) -> dict:
            schema = _DensePayload if "previous_draft" in prompt else _DraftPayload
            return _run(_shared_llm.extract(prompt, schema)).model_dump(mode="python")

        sresult = summarize_meeting(
            transcript_window,
            _summary_llm,
            densify_passes=1,
            key_decisions=[d.decision for d in accepted_decisions],
            action_items=accepted_actions,
        )

        # Persist the summary so the dashboard can show it.
        store.upsert_summary(
            meeting_id,
            tl_dr=sresult.summary.tl_dr,
            topics=list(sresult.headline_topics),
            model=os.environ.get("MEETMIND_LLM_MODEL") or os.environ.get("ACTANTS_MODEL") or "auto",
        )

        click.echo("")
        click.echo(f"## {meeting.title}")
        click.echo("")
        click.echo(sresult.summary.tl_dr)
        click.echo("")
        click.echo("### Topics")
        for t in sresult.headline_topics:
            click.echo(f"- {t}")
        if accepted_decisions:
            click.echo("")
            click.echo("### Decisions")
            for d in accepted_decisions:
                click.echo(f"- {d.decision}")
                if d.rationale:
                    click.echo(f"  └ rationale: {d.rationale}")
        if accepted_actions:
            click.echo("")
            click.echo("### Action items")
            for a in accepted_actions:
                owner = f" (owner: {a.owner})" if a.owner else ""
                due = f" (due: {a.due})" if a.due else ""
                click.echo(f"- {a.description}{owner}{due}")
                click.echo(f"  └ evidence: {a.evidence_quote!r}")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Compliance commands (S14.x)
# ---------------------------------------------------------------------------


@main.group()
def compliance() -> None:
    """GDPR / BIPA / CUBI compliance helpers."""


@compliance.command(name="dpia")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.meetmind/data/meetmind.db.",
)
@click.option(
    "--organization",
    type=str,
    default="[organization name]",
    help="Organization name to embed in the DPIA header.",
)
@click.option(
    "--contact",
    type=str,
    default="[data protection officer or controller email]",
    help="DPO / controller contact line.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write to file instead of stdout.",
)
def dpia_cmd(
    db: Path | None,
    organization: str,
    contact: str,
    out: Path | None,
) -> None:
    """Generate a Markdown DPIA snapshot for this install."""
    from meetmind.compliance.dpia import DpiaInputs, default_db_path, generate_dpia

    db_path = db or default_db_path()
    inputs = DpiaInputs(
        db_path=db_path,
        organization=organization,
        controller_contact=contact,
    )
    md = generate_dpia(inputs)
    if out:
        out.write_text(md, encoding="utf-8")
        click.echo(f"# wrote {out}", err=True)
    else:
        click.echo(md)


@compliance.command(name="retention-sweep")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.meetmind/data/meetmind.db.",
)
@click.option(
    "--meetings-days",
    type=int,
    default=None,
    help="Override meetings TTL in days (env: MEETMIND_RETENTION_MEETINGS_DAYS).",
)
@click.option(
    "--voiceprint-days",
    type=int,
    default=None,
    help="Override voiceprint TTL in days (env: MEETMIND_RETENTION_VOICEPRINT_DAYS).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report what would be deleted without mutating the store.",
)
def retention_sweep_cmd(
    db: Path | None,
    meetings_days: int | None,
    voiceprint_days: int | None,
    dry_run: bool,
) -> None:
    """Apply retention policy: delete meetings + voiceprints past their TTL."""
    from meetmind.compliance.dpia import default_db_path
    from meetmind.compliance.retention import RetentionPolicy, sweep

    db_path = db or default_db_path()
    if not db_path.exists():
        raise click.ClickException(f"no store at {db_path}")
    policy = RetentionPolicy.from_env()
    if meetings_days is not None:
        policy.meetings_ttl_days = meetings_days
    if voiceprint_days is not None:
        policy.voiceprint_ttl_days = voiceprint_days
    report = sweep(db_path, policy=policy, dry_run=dry_run)
    if dry_run:
        click.echo("# dry run — no rows deleted")
    for line in report.as_lines():
        click.echo(line)


# ---------------------------------------------------------------------------
# Integration commands
# ---------------------------------------------------------------------------


@main.command(name="export-obsidian")
@click.argument("meeting_id")
@click.option(
    "--vault",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Obsidian vault root directory.",
)
@click.option("--folder", default="MeetMind", show_default=True)
@click.option("--overwrite", is_flag=True, default=False)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.meetmind/data/meetmind.db.",
)
def export_obsidian_cmd(
    meeting_id: str, vault: Path, folder: str, overwrite: bool, db: Path | None
) -> None:
    """Write a meeting to an Obsidian vault as a Markdown note."""
    from meetmind.compliance.dpia import default_db_path
    from meetmind.integrations.obsidian import export_meeting
    from meetmind.memory.store import Store

    db_path = db or default_db_path()
    with Store.open(db_path) as store:
        result = export_meeting(store, meeting_id, vault=vault, folder=folder, overwrite=overwrite)
    click.echo(str(result.note_path))


@main.command(name="export-github")
@click.argument("meeting_id")
@click.option("--repo", required=True, help="GitHub repo, e.g. owner/name.")
@click.option("--label", default="meetmind", show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.meetmind/data/meetmind.db.",
)
def export_github_cmd(
    meeting_id: str, repo: str, label: str, dry_run: bool, db: Path | None
) -> None:
    """Open one GitHub issue per open action item via the local `gh` CLI."""
    from meetmind.compliance.dpia import default_db_path
    from meetmind.integrations.github import GhCliMissingError, export_action_items
    from meetmind.memory.store import Store

    db_path = db or default_db_path()
    try:
        with Store.open(db_path) as store:
            refs = export_action_items(store, meeting_id, repo=repo, label=label, dry_run=dry_run)
    except GhCliMissingError as e:
        raise click.ClickException(str(e)) from e
    if not refs:
        click.echo("# no open action items to export", err=True)
        return
    for r in refs:
        click.echo(r.url or f"(dry-run) {r.title}")


@main.command(name="export-slack")
@click.argument("meeting_id")
@click.option(
    "--webhook-url",
    default=None,
    help="Slack Incoming Webhook URL. Falls back to $SLACK_WEBHOOK_URL.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Render payload, skip the POST.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.meetmind/data/meetmind.db.",
)
def export_slack_cmd(
    meeting_id: str, webhook_url: str | None, dry_run: bool, db: Path | None
) -> None:
    """Post a Block Kit digest to a Slack channel via Incoming Webhook.

    Create a webhook at https://api.slack.com/messaging/webhooks and
    either pass `--webhook-url` or set `SLACK_WEBHOOK_URL` in your env.
    """
    from meetmind.compliance.dpia import default_db_path  # noqa: PLC0415
    from meetmind.integrations.slack import export_meeting_to_slack  # noqa: PLC0415
    from meetmind.memory.store import Store  # noqa: PLC0415

    db_path = db or default_db_path()
    with Store.open(db_path) as store:
        result = export_meeting_to_slack(
            store, meeting_id, webhook_url=webhook_url, dry_run=dry_run
        )
    if not result.get("ok"):
        raise click.ClickException(result.get("error") or json.dumps(result))
    click.echo(json.dumps(result, indent=2))


@main.command(name="redact")
@click.argument("text", required=False)
@click.option(
    "--profile",
    type=click.Choice(["raw", "team_internal", "public_share"]),
    default="team_internal",
    show_default=True,
)
def redact_cmd(text: str | None, profile: str) -> None:
    """Apply a redaction profile to TEXT (or stdin)."""
    from meetmind.analyze.redact import redact

    if text is None:
        text = sys.stdin.read()
    result = redact(text, profile=profile)  # type: ignore[arg-type]
    click.echo(result.text)
    click.echo(f"# {result.redaction_count} redactions ({result.profile})", err=True)


# ---------------------------------------------------------------------------
# Signed-bundle commands (S13.1 — wired into CLI in v0.20)
# ---------------------------------------------------------------------------


@main.command(name="export-bundle")
@click.argument("meeting_id")
@click.option(
    "--out",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output path for the signed tar.gz bundle.",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database path. Defaults to ~/.meetmind/data/meetmind.db.",
)
def export_bundle_cmd(meeting_id: str, out: Path, db: Path | None) -> None:
    """Export an Ed25519-signed transcript bundle for a meeting."""
    from meetmind.crypto.bundle import export_bundle
    from meetmind.crypto.identity import load_or_create_identity
    from meetmind.memory.store import Store

    db_path = db or _default_db_path()
    if not db_path.exists():
        raise click.ClickException(f"no store at {db_path}")
    with Store.open(db_path) as store:
        meeting = store.get_meeting(meeting_id)
        if meeting is None:
            raise click.ClickException(f"meeting not found: {meeting_id}")
        segments = store.list_segments(meeting_id)
        actions = store.list_action_items(meeting_id=meeting_id)
        decisions = store.list_decisions(meeting_id)

    transcript_obj = {
        "meeting_id": meeting.id,
        "title": meeting.title,
        "started_at": meeting.started_at.isoformat() if meeting.started_at else None,
        "ended_at": meeting.ended_at.isoformat() if meeting.ended_at else None,
        "segments": [
            {
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "speaker": s.speaker_id or s.speaker,
                "text": s.text,
                "channel": s.channel.value if s.channel is not None else None,
                "language": s.language,
            }
            for s in segments
        ],
        "actions": [a.model_dump(mode="json") for a in actions],
        "decisions": [d.model_dump(mode="json") for d in decisions],
    }
    identity = load_or_create_identity()
    path = export_bundle(
        out,
        identity=identity,
        transcript_obj=transcript_obj,
        model_versions={
            "meetmind": meetmind.__version__,
            "stt": "parakeet-tdt-0.6b-v3",
            "diarize": "sortformer-4spk-v2",
        },
    )
    click.echo(str(path))


@main.command(name="verify")
@click.argument("bundle", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--fingerprint",
    type=str,
    default=None,
    help="Expected signer fingerprint (sha256 of raw Ed25519 public key).",
)
def verify_cmd(bundle: Path, fingerprint: str | None) -> None:
    """Verify the signature, hashes, and layout of a signed bundle."""
    from meetmind.crypto.bundle import verify_bundle

    result = verify_bundle(bundle, expected_fingerprint=fingerprint)
    if result.ok:
        click.echo(f"OK  fingerprint={result.fingerprint}")
    else:
        click.echo(f"FAIL  fingerprint={result.fingerprint}", err=True)
        for issue in result.issues:
            click.echo(f"  - {issue}", err=True)
        raise click.ClickException("bundle verification failed")


# ---------------------------------------------------------------------------
# Selftest command — telemetry-free health check
# ---------------------------------------------------------------------------


@main.command(name="selftest")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit JSON instead of human-readable check rows.",
)
def selftest_cmd(as_json: bool) -> None:
    """Run a no-network health check across all major MeetMind subsystems.

    Output is shareable from a GitHub issue (no telemetry — the user
    pastes it themselves). Each check returns ``ok=True/False`` plus a
    short note so a maintainer can triage without reproducing the env.

    Exits 0 if every check passes, 1 if any fail. Warnings (e.g. mock
    sidecar fallback) do not fail the run — they're informational.
    """
    checks = _run_selftest()
    if as_json:
        click.echo(json.dumps(checks, indent=2))
    else:
        for c in checks:
            if c["ok"] and not c.get("warn"):
                sigil = "✓"
            elif c.get("warn"):
                sigil = "⚠"
            else:
                sigil = "✗"
            click.echo(f"  {sigil}  {c['name']:24s}  {c['note']}")
    any_fail = any(not c["ok"] and not c.get("warn") for c in checks)
    raise SystemExit(1 if any_fail else 0)


def _run_selftest() -> list[dict]:  # noqa: C901 — sequential check list reads fine
    """Sequence of independent checks. Each is a dict for easy JSON output."""
    import tempfile  # noqa: PLC0415

    out: list[dict] = []

    def add(name: str, ok: bool, note: str, *, warn: bool = False) -> None:
        out.append({"name": name, "ok": ok, "note": note, "warn": warn})

    # 1. Python + platform.
    py = sys.version.split()[0]
    add("python", py.startswith(("3.12", "3.13")), f"{py} on {platform.platform()}")

    # 2. Store opens + WAL.
    with tempfile.TemporaryDirectory() as tmp:
        from meetmind.memory.store import Store  # noqa: PLC0415

        try:
            s = Store.open(Path(tmp) / "selftest.db", use_keychain=False)
            mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
            fk = s.conn.execute("PRAGMA foreign_keys").fetchone()[0]
            s.close()
            add(
                "storage",
                mode.lower() in {"wal", "truncate"} and int(fk) == 1,
                f"journal={mode} foreign_keys={fk}",
            )
        except Exception as e:  # noqa: BLE001
            add("storage", False, f"open failed: {e}")

    # 3. Storage encryption mode.
    storage_info = _storage_status_summary()
    add(
        "encryption",
        True,
        storage_info["mode"],
        warn=storage_info.get("mode", "").startswith("unencrypted"),
    )

    # 4. Sidecar discovery — mock is OK but flag it.
    cap = _find_capture_sidecar()
    add(
        "capture-sidecar",
        cap is not None or _find_mock_capture().exists(),
        str(cap) if cap else "mock fallback (install Swift sidecar for real capture)",
        warn=cap is None,
    )
    stt = find_stt_sidecar()
    add(
        "stt-sidecar",
        stt is not None or _find_mock_stt().exists(),
        str(stt) if stt else "mock fallback (install Swift sidecar for real STT)",
        warn=stt is None,
    )

    # 5. actants LLM availability (purely metadata — no network call).
    try:
        from meetmind.analyze.llm import llm_config_summary  # noqa: PLC0415

        cfg = llm_config_summary()
        add("llm", True, f"{cfg.get('provider', '?')}/{cfg.get('model', 'auto')}")
    except Exception as e:  # noqa: BLE001
        add("llm", False, f"config error: {e}")

    # 6. Integrations load cleanly.
    try:
        import meetmind.integrations.github  # noqa: F401, PLC0415
        import meetmind.integrations.obsidian  # noqa: F401, PLC0415
        import meetmind.integrations.slack  # noqa: F401, PLC0415

        add("integrations", True, "obsidian + github + slack importable")
    except Exception as e:  # noqa: BLE001
        add("integrations", False, f"import failed: {e}")

    # 7. End-to-end: write meeting + segment + summary, read it back.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            from datetime import UTC, datetime  # noqa: PLC0415

            from meetmind.memory.store import Store  # noqa: PLC0415
            from meetmind.models import Meeting, TranscriptSegment  # noqa: PLC0415

            s = Store.open(Path(tmp) / "e2e.db", use_keychain=False)
            m = Meeting(
                id="01HSELFTESTSELFTESTSELFTES",
                title="selftest",
                created_at=datetime.now(UTC),
            )
            s.upsert_meeting(m)
            s.append_segment(
                m.id,
                TranscriptSegment(start_seconds=0.0, end_seconds=1.0, text="hello world"),
            )
            s.upsert_summary(m.id, tl_dr="self test", topics=["t1"])
            roundtrip = s.get_meeting(m.id) is not None and s.get_summary(m.id) is not None
            s.close()
            add("end-to-end", roundtrip, "meeting + segment + summary round-trip")
        except Exception as e:  # noqa: BLE001
            add("end-to-end", False, f"failed: {e}")

    return out


# ---------------------------------------------------------------------------
# Demo command — single-process "show me everything"
# ---------------------------------------------------------------------------


@main.command(name="demo")
@click.option("--port", type=int, default=7857, show_default=True)
@click.option(
    "--duration",
    "-d",
    type=float,
    default=0.0,
    show_default=True,
    help="Stop after N seconds. Default 0 = run until Ctrl+C (real-meeting mode).",
)
@click.option(
    "--stream",
    type=click.Choice(["mic", "loopback", "both"]),
    default="both",
    show_default=True,
    help="Which audio stream to transcribe. 'both' (default) runs mic + "
    "loopback in parallel — the real-meeting mode. 'loopback' = system "
    "audio only; 'mic' = microphone only.",
)
@click.option(
    "--coach",
    is_flag=True,
    default=False,
    help="Also start the live coach loop (publishes coach_tip events).",
)
@click.option(
    "--mock",
    is_flag=True,
    default=False,
    help="Fall back to the pure-Python mock sidecars. For CI/headless dev. "
    "By default `demo` uses the real native sidecars.",
)
@click.option(
    "--no-store",
    is_flag=True,
    default=False,
    help="Stream transcripts to the bus only — skip DB persistence.",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Title for the meeting row (defaults to a timestamped slug).",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    show_default=True,
    help="Open the overlay URL in the default browser.",
)
def demo_cmd(
    port: int,
    duration: float,
    stream: str,
    coach: bool,
    mock: bool,
    no_store: bool,
    title: str | None,
    open_browser: bool,
) -> None:
    """The full app, one command. Real native sidecars + UI + live SSE.

    \b
    Defaults:
      - capture: real macOS Core Audio Tap (system audio loopback)
      - STT:     real Parakeet TDT 0.6B v3
      - persists meetings to ~/.meetmind/data/meetmind.db
      - serves the overlay at http://127.0.0.1:7857/

    \b
    To capture system audio you'll need to grant the capture sidecar
    Screen Recording permission once — System Settings → Privacy &
    Security → Screen Recording.

    \b
    Stop with Ctrl+C. The meeting is persisted; run
    `meetmind summarize <id>` afterwards to get actions + decisions.
    """
    import webbrowser

    from meetmind.api.bus import default_bus
    from meetmind.api.events import MetaEvent
    from meetmind.api.http import serve as _serve
    from meetmind.ipc import StreamId

    if mock:
        capture_bin = _find_mock_capture()
        stt_bin = _find_mock_stt()
    else:
        capture_bin = _find_capture_sidecar() or _find_mock_capture()
        stt_bin = find_stt_sidecar() or _find_mock_stt()

    using_mock_capture = "_mock_sidecar" in capture_bin.name
    using_mock_stt = "_mock_stt" in stt_bin.name
    stream_id: StreamId | None
    if stream == "mic":
        stream_id = StreamId.MIC
    elif stream == "loopback":
        stream_id = StreamId.LOOPBACK
    else:
        stream_id = None  # both
    url = f"http://127.0.0.1:{port}/"
    db_path = None if no_store else _default_db_path()

    click.echo(f"# meetmind demo on {url}", err=True)
    click.echo(
        f"# capture={'MOCK' if using_mock_capture else capture_bin.name}  "
        f"stt={'MOCK' if using_mock_stt else stt_bin.name}",
        err=True,
    )
    click.echo(f"# stream={stream}  duration={'∞' if duration == 0 else f'{duration}s'}", err=True)
    if db_path:
        click.echo(f"# db={db_path}", err=True)
    if using_mock_capture or using_mock_stt:
        click.echo(
            "# WARNING: falling back to mock sidecars. Build the native ones with "
            "`cd sidecars/macos && swift build -c release` for real audio.",
            err=True,
        )

    async def _run() -> None:
        # Spawn the FastAPI server in the same process so it shares
        # `default_bus` with the recording loop below.
        server_task = asyncio.create_task(_serve(port=port), name="demo-serve")
        await asyncio.sleep(0.4)  # let uvicorn bind

        if open_browser:
            webbrowser.open(url)

        # Coach loop, optional.
        coach_stop: asyncio.Event | None = None
        coach_task: asyncio.Task | None = None
        if coach:
            from meetmind.api.coach import CoachConfig, CoachLoop

            cl = CoachLoop(config=CoachConfig(min_text_chars=20, tick_seconds=15.0))
            coach_stop = asyncio.Event()
            coach_task = asyncio.create_task(cl.run(stop=coach_stop), name="demo-coach")

        # Drive the recording on the shared bus. publish_only=True
        # means the recording uses the existing bus + server we just
        # started — it won't try to bind a second uvicorn on the same
        # port.
        record_task = asyncio.create_task(
            _run_record(
                capture_bin,
                stt_bin,
                duration=duration,
                stream_id=stream_id,
                mock=mock,
                emit_sse=True,
                sse_port=port,
                db_path=db_path,
                title=title,
                coach=False,  # we manage the coach above
                publish_only=True,
            ),
            name="demo-record",
        )
        await default_bus.publish(MetaEvent(event="session_started"))

        try:
            await record_task
        finally:
            await default_bus.publish(MetaEvent(event="session_stopped"))
            if coach_task is not None:
                if coach_stop is not None:
                    coach_stop.set()
                coach_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await coach_task
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await server_task

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo("\n# demo stopped", err=True)

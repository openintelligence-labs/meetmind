"""Test the post-hoc `meetmind diarize <meeting_id>` command.

Builds a meeting with persisted WAV files (sine waves with silences
between to give the RMS-gated mock diarizer something to cluster),
then asserts that speaker_id columns get updated.
"""

from __future__ import annotations

import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from meetmind.cli import main
from meetmind.memory.store import Store
from meetmind.models import ChannelKind, Meeting, TranscriptSegment


def _write_clip(path: Path, voiced: bool, *, seconds: float = 1.0, sr: int = 16_000) -> None:
    """Write a WAV clip — voiced (sine) or silence."""
    n = int(sr * seconds)
    if voiced:
        t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float32)
        pcm = 0.4 * np.sin(2 * np.pi * 440 * t)
    else:
        pcm = np.zeros(n, dtype=np.float32)
    i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.tobytes())


def test_diarize_updates_speaker_ids_with_persisted_audio(tmp_path: Path, monkeypatch) -> None:
    """Run diarize over a meeting that has a persisted WAV and check that
    transcript_segments.speaker_id is updated to a cluster label."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    db_path = tmp_path / "data" / "meetmind.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav_path = audio_dir / "M_mic.wav"
    _write_clip(wav_path, voiced=True, seconds=2.0)

    s = Store.open(db_path, use_keychain=False)
    m = Meeting(
        id="01HDIARIZEMEETXXXXXXXXXXXXX",
        title="Diarize me",
        created_at=datetime(2026, 5, 12, tzinfo=UTC),
        started_at=datetime(2026, 5, 12, tzinfo=UTC),
        audio_path_mic=wav_path,
    )
    s.upsert_meeting(m)
    s.append_segment(
        m.id,
        TranscriptSegment(
            start_seconds=0.0,
            end_seconds=1.0,
            text="Hello there",
            channel=ChannelKind.MIC,
            speaker_id=None,
        ),
    )
    s.close()

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["diarize", m.id, "--db", str(db_path), "--no-match-enrolled"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    with Store.open(db_path, use_keychain=False) as s:
        row = s.conn.execute(
            "SELECT speaker_id FROM transcript_segments WHERE meeting_id = ?", (m.id,)
        ).fetchone()
    # Mock diarizer assigns "A" or "B" — we don't care which, only that
    # it's been written (was None before).
    assert row["speaker_id"] in {"A", "B", "unknown"}


def test_diarize_with_no_audio_falls_back_to_channel_labels(tmp_path: Path, monkeypatch) -> None:
    """A meeting with no persisted audio should still complete without crash,
    using the synthesized channel-based segments."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    db_path = tmp_path / "data" / "meetmind.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    s = Store.open(db_path, use_keychain=False)
    m = Meeting(
        id="01HNOAUDIODIARMEETXXXXXXXX",
        title="No audio",
        created_at=datetime(2026, 5, 12, tzinfo=UTC),
    )
    s.upsert_meeting(m)
    s.append_segment(
        m.id,
        TranscriptSegment(start_seconds=0.0, end_seconds=1.0, text="Word", channel=ChannelKind.MIC),
    )
    s.close()

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["diarize", m.id, "--db", str(db_path), "--no-match-enrolled"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "no audio persisted" in result.output or "falling back" in result.output


def test_diarize_unknown_meeting_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    db_path = tmp_path / "data" / "meetmind.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Store.open(db_path, use_keychain=False).close()

    runner = CliRunner()
    result = runner.invoke(
        main, ["diarize", "01HBOGUS", "--db", str(db_path)], catch_exceptions=False
    )
    assert result.exit_code != 0
    assert "not found" in result.output

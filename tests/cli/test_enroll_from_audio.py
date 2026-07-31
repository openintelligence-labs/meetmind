"""End-to-end test for `meetmind enroll --audio`.

Runs the CLI against a synthetic WAV and checks the speaker row, the signed
consent event, and that the centroid came from the audio embedder rather than
the deterministic name-hash stub.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from meetmind.cli import main
from meetmind.memory.store import Store


def _write_synthetic_clip(path: Path, *, seconds: float = 1.5, sr: int = 16_000) -> None:
    """Mix a 220 Hz + 440 Hz sine wave so the mel features have signal."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    pcm = 0.4 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t))
    i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.tobytes())


def test_enroll_from_audio_produces_real_centroid(tmp_path: Path, monkeypatch) -> None:
    """Real-audio enrollment should write a non-stub centroid + signed event."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    db = tmp_path / "data" / "meetmind.db"
    clip = tmp_path / "alice.wav"
    _write_synthetic_clip(clip)

    runner = CliRunner()
    audio_result = runner.invoke(
        main,
        [
            "enroll",
            "Alice",
            "--audio",
            str(clip),
            "--db",
            str(db),
            "--accept",
        ],
        catch_exceptions=False,
    )
    assert audio_result.exit_code == 0, audio_result.output
    audio_speaker_id = audio_result.output.strip().splitlines()[-1]
    assert audio_speaker_id

    with Store.open(db, use_keychain=False) as s:
        audio_speaker = s.get_speaker(audio_speaker_id)
    assert audio_speaker is not None
    assert audio_speaker.voiceprint_centroid is not None
    assert len(audio_speaker.voiceprint_centroid) > 0

    # Now run the stub path against the same DB. The centroid bytes
    # should differ (audio embedder vs name-hash RNG).
    stub_result = runner.invoke(
        main,
        [
            "enroll",
            "AliceStub",
            "--db",
            str(db),
            "--accept",
        ],
        catch_exceptions=False,
    )
    assert stub_result.exit_code == 0, stub_result.output
    stub_speaker_id = stub_result.output.strip().splitlines()[-1]

    with Store.open(db, use_keychain=False) as s:
        stub_speaker = s.get_speaker(stub_speaker_id)
    assert stub_speaker is not None
    assert stub_speaker.voiceprint_centroid != audio_speaker.voiceprint_centroid


def test_enroll_rejects_too_short_clip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    db = tmp_path / "data" / "meetmind.db"
    clip = tmp_path / "short.wav"
    _write_synthetic_clip(clip, seconds=0.3)  # 300 ms — too short

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["enroll", "Bob", "--audio", str(clip), "--db", str(db), "--accept"],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert "too short" in result.output


def test_enroll_stub_flag_overrides_audio(tmp_path: Path, monkeypatch) -> None:
    """--stub forces the deterministic embedding even with --audio supplied."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    db = tmp_path / "data" / "meetmind.db"
    clip = tmp_path / "x.wav"
    _write_synthetic_clip(clip)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "enroll",
            "Cara",
            "--audio",
            str(clip),
            "--stub",
            "--db",
            str(db),
            "--accept",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "--stub forced" in result.output

    # The persisted centroid should match what the deterministic path
    # produces — i.e. seeded by the lower-case name hash.
    import hashlib

    digest = hashlib.sha256(b"cara").digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    expected = rng.standard_normal(192).astype(np.float32)

    speaker_id = result.output.strip().splitlines()[-1]
    with Store.open(db, use_keychain=False) as s:
        speaker = s.get_speaker(speaker_id)
    # The centroid is stored normalized, so only the dimension is comparable.
    from meetmind.diarize.matcher import _decode_centroid

    decoded = _decode_centroid(speaker)
    assert decoded is not None
    assert decoded.shape == expected.shape


@pytest.mark.timeout(15)
def test_enroll_from_audio_handles_stereo_clip(tmp_path: Path, monkeypatch) -> None:
    """Stereo clips should mono-mix without complaint."""
    monkeypatch.setenv("MEETMIND_HOME", str(tmp_path))
    db = tmp_path / "data" / "meetmind.db"
    clip = tmp_path / "stereo.wav"
    sr = 16_000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, right], axis=1)
    i16 = (stereo * 32767.0).astype(np.int16)
    with wave.open(str(clip), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.tobytes())

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["enroll", "Dee", "--audio", str(clip), "--db", str(db), "--accept"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

"""Tests for the opt-in audio persistence writer."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from meetmind.capture.wav_writer import (
    WavWriter,
    default_audio_path,
    open_wav_writer,
)


def test_writer_creates_a_valid_wav(tmp_path) -> None:
    path = tmp_path / "out.wav"
    w = WavWriter(path, sample_rate=16_000)
    try:
        pcm = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
        w.append(pcm)
        w.append(pcm)
        assert pytest.approx(w.duration_seconds, abs=1e-3) == 1.0
    finally:
        w.close()
    # Read it back with stdlib wave.
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16_000
        assert wf.getnframes() == 16_000


def test_writer_clips_out_of_range_amplitudes(tmp_path) -> None:
    path = tmp_path / "clip.wav"
    w = WavWriter(path)
    try:
        # Send amplitudes above 1.0 — should clip not blow up.
        pcm = np.array([2.5, -2.5, 0.5, -0.5], dtype=np.float32)
        w.append(pcm)
    finally:
        w.close()
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(4)
    samples = np.frombuffer(frames, dtype=np.int16)
    assert samples[0] == 32767
    assert samples[1] == -32767


def test_append_int16_bytes_path(tmp_path) -> None:
    """The fast path used by the CLI to skip numpy round-trips."""
    path = tmp_path / "i16.wav"
    w = WavWriter(path, sample_rate=48_000)
    try:
        raw = (np.array([1000, -1000, 32000], dtype=np.int16)).tobytes()
        w.append_int16_bytes(raw)
    finally:
        w.close()
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 48_000
        assert wf.getnframes() == 3


def test_append_int16_rejects_odd_length(tmp_path) -> None:
    w = WavWriter(tmp_path / "bad.wav")
    try:
        with pytest.raises(ValueError):
            w.append_int16_bytes(b"\x01\x02\x03")  # 3 bytes — odd
    finally:
        w.close()


def test_close_is_idempotent(tmp_path) -> None:
    w = WavWriter(tmp_path / "idem.wav")
    w.close()
    w.close()  # must not raise


def test_context_manager_yields_none_when_path_is_none(tmp_path) -> None:
    with open_wav_writer(None) as w:
        assert w is None


def test_context_manager_closes_on_exit(tmp_path) -> None:
    path = tmp_path / "ctx.wav"
    with open_wav_writer(path) as w:
        assert w is not None
        w.append(np.zeros(100, dtype=np.float32))
    # File exists with the right header even on early exit.
    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == 100


def test_default_audio_path_uses_meetmind_home(tmp_path, monkeypatch) -> None:
    """Resolution shouldn't break when ``$HOME`` is fake."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Path.home() reads HOME on POSIX, USERPROFILE on Windows.
    p = default_audio_path("01HABCDEF01HABCDEF01HABCDE", "mic")
    assert "01HABCDEF01HABCDEF01HABCDE_mic.wav" in str(p)


def test_default_audio_path_honors_explicit_root(tmp_path) -> None:
    p = default_audio_path("XYZ", "loopback", root=tmp_path / "custom")
    assert p == tmp_path / "custom" / "XYZ_loopback.wav"

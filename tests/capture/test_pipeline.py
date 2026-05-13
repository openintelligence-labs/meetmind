"""Tests for the audio pre-processing pipeline."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from meetmind.capture.pipeline import (
    SOURCE_RATE,
    TARGET_RATE,
    VAD,
    VAD_FRAME_SAMPLES,
    chunks_to_frames,
    downsample_48k_to_16k,
)
from meetmind.ipc import AudioChunk, StreamId


def _silent_chunk(stream: StreamId, samples_48k: int, ts: int = 0) -> AudioChunk:
    pcm = struct.pack(f"<{samples_48k}h", *([0] * samples_48k))
    return AudioChunk(stream=stream, timestamp_us=ts, pcm=pcm)


def _tone_chunk(stream: StreamId, samples_48k: int, freq_hz: float, ts: int = 0) -> AudioChunk:
    t = np.arange(samples_48k) / SOURCE_RATE
    sig = (0.6 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    s16 = (sig * 32767).astype("<i2")
    return AudioChunk(stream=stream, timestamp_us=ts, pcm=s16.tobytes())


def test_downsample_ratio_is_exact():
    in_samples = 4800  # 100 ms @ 48 kHz
    pcm = np.zeros(in_samples, dtype=np.float32)
    out = downsample_48k_to_16k(pcm)
    assert len(out) == in_samples // 3
    assert out.dtype == np.float32


def test_downsample_preserves_low_freq_amplitude_within_5pct():
    # 1 kHz tone, well below the 7.5 kHz cutoff. Amplitude should survive.
    seconds = 0.2
    n = int(seconds * SOURCE_RATE)
    t = np.arange(n) / SOURCE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    out = downsample_48k_to_16k(tone)
    in_rms = float(np.sqrt(np.mean(tone**2)))
    out_rms = float(np.sqrt(np.mean(out**2)))
    assert out_rms == pytest.approx(in_rms, rel=0.05)


def test_vad_rms_fallback_is_not_voiced_for_silence():
    vad = VAD(rms_threshold=0.01)
    silence = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
    assert vad.is_voiced(silence) is False


def test_vad_rms_fallback_is_voiced_for_tone():
    vad = VAD(rms_threshold=0.01)
    t = np.arange(VAD_FRAME_SAMPLES) / TARGET_RATE
    tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert vad.is_voiced(tone) is True


def test_chunks_to_frames_yields_correct_frame_count():
    # 480 samples @ 48 kHz = 10 ms = 160 samples @ 16 kHz.
    # 4 chunks * 160 = 640 samples >= VAD_FRAME_SAMPLES (512) → exactly 1 frame.
    chunks = [_silent_chunk(StreamId.MIC, 480, ts=i * 10_000) for i in range(4)]
    frames = list(chunks_to_frames(chunks))
    assert len(frames) == 1
    assert frames[0].stream is StreamId.MIC
    assert len(frames[0].pcm_f32) == VAD_FRAME_SAMPLES
    assert frames[0].voiced is False  # silence


def test_chunks_to_frames_emits_voiced_for_tone():
    # 6 chunks * 480 = 2880 @48k → 960 @16k → ~1.87 VAD frames.
    chunks = [_tone_chunk(StreamId.MIC, 480, 800.0, ts=i * 10_000) for i in range(6)]
    frames = list(chunks_to_frames(chunks, vad=VAD(rms_threshold=0.001)))
    assert len(frames) >= 1
    assert all(f.voiced for f in frames)
    assert all(f.stream is StreamId.MIC for f in frames)


def test_chunks_to_frames_keeps_streams_independent():
    """Mic and loopback chunks share the same input cadence but are
    buffered independently — neither stream contaminates the other."""
    chunks = []
    for i in range(4):
        chunks.append(_silent_chunk(StreamId.MIC, 480, ts=i * 10_000))
        chunks.append(_silent_chunk(StreamId.LOOPBACK, 480, ts=i * 10_000))
    frames = list(chunks_to_frames(chunks))
    mic_frames = [f for f in frames if f.stream is StreamId.MIC]
    loop_frames = [f for f in frames if f.stream is StreamId.LOOPBACK]
    assert len(mic_frames) == 1
    assert len(loop_frames) == 1

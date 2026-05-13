"""Tests for legal-mode signed transcript bundles."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from meetmind.crypto.bundle import BUNDLE_FORMAT_VERSION, export_bundle, verify_bundle
from meetmind.crypto.identity import Identity


@pytest.fixture
def sample_transcript() -> dict:
    return {
        "id": "01ABCDEF",
        "title": "Test meeting",
        "started_at": "2026-05-05T12:00:00Z",
        "ended_at": "2026-05-05T12:30:00Z",
        "duration_seconds": 1800,
        "transcript": {
            "segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 5.0,
                    "text": "Hello there.",
                    "speaker": "self",
                },
                {
                    "start_seconds": 5.0,
                    "end_seconds": 10.0,
                    "text": "General Kenobi.",
                    "speaker": "remote-A",
                },
            ]
        },
        "summary": {"tl_dr": "Brief greetings exchanged.", "key_decisions": []},
    }


def test_round_trip_export_and_verify(tmp_path: Path, sample_transcript):
    identity = Identity.generate()
    bundle_path = export_bundle(
        tmp_path / "bundle.tar.gz",
        identity=identity,
        transcript_obj=sample_transcript,
        model_versions={"stt": "parakeet-v3", "diar": "sortformer-v2"},
    )
    result = verify_bundle(bundle_path)
    assert result.ok, result.issues
    assert result.fingerprint == identity.fingerprint


def test_bundle_with_audio_artifacts(tmp_path: Path, sample_transcript):
    identity = Identity.generate()
    audio_mic = b"\x00" * 4096
    audio_loop = b"\x01" * 4096
    bundle_path = export_bundle(
        tmp_path / "with_audio.tar.gz",
        identity=identity,
        transcript_obj=sample_transcript,
        model_versions={"stt": "parakeet-v3"},
        audio_mic=audio_mic,
        audio_loopback=audio_loop,
    )
    result = verify_bundle(bundle_path)
    assert result.ok, result.issues


def test_tampered_transcript_fails_verification(tmp_path: Path, sample_transcript):
    identity = Identity.generate()
    bundle_path = export_bundle(
        tmp_path / "tampered.tar.gz",
        identity=identity,
        transcript_obj=sample_transcript,
        model_versions={"stt": "parakeet-v3"},
    )
    tampered = tmp_path / "tampered2.tar.gz"
    with tarfile.open(bundle_path, "r:gz") as src:
        members = src.getmembers()
        with tarfile.open(tampered, "w:gz") as dst:
            for m in members:
                f = src.extractfile(m)
                data = f.read() if f is not None else b""
                if m.name == "transcript.json":
                    data = data.replace(b"Hello there.", b"Hello, intruder.")
                m.size = len(data)
                dst.addfile(m, io.BytesIO(data))

    result = verify_bundle(tampered)
    assert not result.ok
    assert any("transcript.json hash does not match" in i for i in result.issues)


def test_tampered_signature_fails_verification(tmp_path: Path, sample_transcript):
    identity = Identity.generate()
    bundle_path = export_bundle(
        tmp_path / "badsig.tar.gz",
        identity=identity,
        transcript_obj=sample_transcript,
        model_versions={"stt": "parakeet-v3"},
    )
    tampered = tmp_path / "badsig2.tar.gz"
    with tarfile.open(bundle_path, "r:gz") as src:
        members = src.getmembers()
        with tarfile.open(tampered, "w:gz") as dst:
            for m in members:
                f = src.extractfile(m)
                data = f.read() if f is not None else b""
                if m.name == "signature.bin":
                    data = bytes((b ^ 0xFF) for b in data)
                m.size = len(data)
                dst.addfile(m, io.BytesIO(data))

    result = verify_bundle(tampered)
    assert not result.ok
    assert any("signature does not verify" in i for i in result.issues)


def test_expected_fingerprint_pin_works(tmp_path: Path, sample_transcript):
    identity = Identity.generate()
    bundle_path = export_bundle(
        tmp_path / "fp.tar.gz",
        identity=identity,
        transcript_obj=sample_transcript,
        model_versions={"stt": "parakeet-v3"},
    )
    good = verify_bundle(bundle_path, expected_fingerprint=identity.fingerprint)
    assert good.ok
    bad = verify_bundle(bundle_path, expected_fingerprint="0" * 64)
    assert not bad.ok
    assert any("fingerprint mismatch" in i for i in bad.issues)


def test_missing_files_detected(tmp_path: Path):
    bad = tmp_path / "incomplete.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        info = tarfile.TarInfo(name="not-a-bundle.txt")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"hello"))
    result = verify_bundle(bad)
    assert not result.ok
    assert any("missing required files" in i for i in result.issues)


def test_format_version_pinned_in_manifest(tmp_path: Path, sample_transcript):
    identity = Identity.generate()
    bundle_path = export_bundle(
        tmp_path / "v.tar.gz",
        identity=identity,
        transcript_obj=sample_transcript,
        model_versions={"stt": "parakeet-v3"},
    )
    with tarfile.open(bundle_path, "r:gz") as t:
        f = t.extractfile("manifest.json")
        assert f is not None
        import json as _j

        manifest = _j.loads(f.read())
    assert manifest["bundle_format"] == BUNDLE_FORMAT_VERSION
    assert manifest["fingerprint"] == identity.fingerprint

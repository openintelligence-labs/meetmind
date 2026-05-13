"""Signed transcript bundle export + verification.

A bundle is a tar.gz containing:

    transcript.json         — canonicalized meeting + segments + summary
    audio_mic.opus.enc      — encrypted Opus, optional
    audio_loopback.opus.enc — encrypted Opus, optional
    model_versions.json     — {stt: ..., diar: ..., summary_llm: ..., bundle_format: 1}
    public_key.pem          — exporter's Ed25519 public key (for offline verify)
    fingerprint.txt         — sha256(public_key) hex
    manifest.json           — hashes of all the above
    signature.bin           — Ed25519 over canonical(manifest)

`meetmind verify` checks:
    1. Bundle layout is well-formed.
    2. Manifest hashes match the actual file contents.
    3. Signature verifies against the embedded public key.
    4. Fingerprint matches sha256(public_key.pem raw bytes).

Verifiers can additionally pin the expected fingerprint out-of-band
(e.g. for chain-of-custody) — `verify_bundle(..., expected_fingerprint=...)`.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from meetmind.crypto.canonicalize import canonical_json, sha256_hex
from meetmind.crypto.identity import Identity

log = logging.getLogger(__name__)

BUNDLE_FORMAT_VERSION = 1


@dataclass
class BundleVerification:
    ok: bool
    fingerprint: str
    issues: list[str]


def export_bundle(
    output_path: Path,
    *,
    identity: Identity,
    transcript_obj: dict[str, Any],
    model_versions: dict[str, str],
    audio_mic: bytes | None = None,
    audio_loopback: bytes | None = None,
) -> Path:
    """Pack a signed bundle to `output_path`. Returns the path."""
    transcript_bytes = canonical_json(transcript_obj)
    transcript_hash = sha256_hex(transcript_bytes)

    versions_obj = {**model_versions, "bundle_format": BUNDLE_FORMAT_VERSION}
    versions_bytes = canonical_json(versions_obj)
    versions_hash = sha256_hex(versions_bytes)

    public_pem = identity.public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    manifest: dict[str, Any] = {
        "transcript_hash": transcript_hash,
        "model_versions_hash": versions_hash,
        "fingerprint": identity.fingerprint,
        "bundle_format": BUNDLE_FORMAT_VERSION,
    }
    if audio_mic is not None:
        manifest["audio_mic_hash"] = sha256_hex(audio_mic)
    if audio_loopback is not None:
        manifest["audio_loopback_hash"] = sha256_hex(audio_loopback)

    manifest_bytes = canonical_json(manifest)
    signature = identity.sign(manifest_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        _add_file(tar, "transcript.json", transcript_bytes)
        _add_file(tar, "model_versions.json", versions_bytes)
        _add_file(tar, "manifest.json", manifest_bytes)
        _add_file(tar, "public_key.pem", public_pem)
        _add_file(tar, "fingerprint.txt", identity.fingerprint.encode("ascii"))
        _add_file(tar, "signature.bin", signature)
        if audio_mic is not None:
            _add_file(tar, "audio_mic.opus.enc", audio_mic)
        if audio_loopback is not None:
            _add_file(tar, "audio_loopback.opus.enc", audio_loopback)

    return output_path


def verify_bundle(
    bundle_path: Path,
    *,
    expected_fingerprint: str | None = None,
) -> BundleVerification:
    """Validate signature, hashes, and bundle layout."""
    issues: list[str] = []
    files: dict[str, bytes] = {}

    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                f = tar.extractfile(member)
                if f is not None:
                    files[name] = f.read()
    except (tarfile.TarError, OSError) as e:
        return BundleVerification(ok=False, fingerprint="", issues=[f"unreadable bundle: {e}"])

    required = {
        "transcript.json",
        "model_versions.json",
        "manifest.json",
        "public_key.pem",
        "fingerprint.txt",
        "signature.bin",
    }
    missing = required - set(files)
    if missing:
        return BundleVerification(
            ok=False, fingerprint="", issues=[f"missing required files: {sorted(missing)}"]
        )

    public_pem = files["public_key.pem"]
    try:
        public_key = serialization.load_pem_public_key(public_pem)
    except Exception as e:  # noqa: BLE001
        return BundleVerification(ok=False, fingerprint="", issues=[f"bad public key: {e}"])
    if not isinstance(public_key, Ed25519PublicKey):
        return BundleVerification(ok=False, fingerprint="", issues=["public key is not Ed25519"])

    raw_pubkey = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    import hashlib as _h

    fingerprint = _h.sha256(raw_pubkey).hexdigest()
    embedded_fingerprint = files["fingerprint.txt"].decode("ascii").strip()
    if fingerprint != embedded_fingerprint:
        issues.append("fingerprint.txt does not match public_key.pem")

    if expected_fingerprint and fingerprint != expected_fingerprint:
        issues.append(f"fingerprint mismatch: expected {expected_fingerprint}, got {fingerprint}")

    manifest_bytes = files["manifest.json"]
    signature = files["signature.bin"]
    try:
        public_key.verify(signature, manifest_bytes)
    except Exception:  # noqa: BLE001
        issues.append("signature does not verify against public key + manifest")

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        issues.append(f"manifest is not valid JSON: {e}")
        return BundleVerification(ok=False, fingerprint=fingerprint, issues=issues)

    if manifest.get("transcript_hash") != sha256_hex(files["transcript.json"]):
        issues.append("transcript.json hash does not match manifest")
    if manifest.get("model_versions_hash") != sha256_hex(files["model_versions.json"]):
        issues.append("model_versions.json hash does not match manifest")

    if "audio_mic.opus.enc" in files:
        if manifest.get("audio_mic_hash") != sha256_hex(files["audio_mic.opus.enc"]):
            issues.append("audio_mic.opus.enc hash does not match manifest")
    elif "audio_mic_hash" in manifest:
        issues.append("manifest references audio_mic.opus.enc but file is missing")

    if "audio_loopback.opus.enc" in files:
        if manifest.get("audio_loopback_hash") != sha256_hex(files["audio_loopback.opus.enc"]):
            issues.append("audio_loopback.opus.enc hash does not match manifest")
    elif "audio_loopback_hash" in manifest:
        issues.append("manifest references audio_loopback.opus.enc but file is missing")

    return BundleVerification(ok=not issues, fingerprint=fingerprint, issues=issues)


def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o600
    tar.addfile(info, io.BytesIO(data))

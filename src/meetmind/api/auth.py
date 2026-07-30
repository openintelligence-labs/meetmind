"""Local API auth: ephemeral per-launch bearer token.

The token is a 256-bit random URL-safe string written to
`~/.meetmind/token` with mode 0600 at server start. Every request to the
local API must carry it as `Authorization: Bearer <token>`. The token
is rotated on every server restart — there's no persistence beyond the
running process.

Why not a Unix domain socket? Browsers and Tauri WebViews can't connect
to UDS over `fetch`/`EventSource` cleanly. 127.0.0.1 + ephemeral bearer
matches the precedent set by Ollama, LM Studio, and Linear desktop.
"""

from __future__ import annotations

import getpass
import os
import secrets
import subprocess
import sys
from pathlib import Path

import httpx
from fastapi import HTTPException, Request


def token_path() -> Path:
    return Path.home() / ".meetmind" / "token"


def _restrict_windows_acl(target: Path) -> None:
    """Restrict `target` to the current user via an owner-only DACL.

    ``os.chmod(0o600)`` is a no-op for access control on Windows (files
    report 0666 regardless), so we shell out to ``icacls``:

      * ``/inheritance:r`` strips every inherited ACE (Everyone,
        BUILTIN\\Users, ...), and
      * ``/grant:r <user>:F`` replaces the grant list with a single
        full-control ACE for the current user.

    On failure we delete the token file and raise — a world-readable
    bearer token is strictly worse than no token at all.
    """
    user = getpass.getuser()
    try:
        subprocess.run(  # noqa: S603 — fixed argv, path is ours, no shell
            ["icacls", str(target), "/inheritance:r", "/grant:r", f"{user}:F"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"failed to restrict token file ACL on Windows ({target}): "
            f"{detail.strip()} — refusing to leave the token world-readable"
        ) from exc


def generate_token() -> str:
    """Mint a fresh URL-safe token. Caller is responsible for writing it."""
    return secrets.token_urlsafe(32)


def write_token(token: str, path: Path | None = None) -> Path:
    """Persist `token` to disk with mode 0600. Default path is `token_path()`.

    Two failure modes we close:
      • Created-as-0600 window: ``O_CREAT|...|0o600`` makes the file
        unreadable from the moment of creation.
      • Pre-existing-with-loose-perms: ``os.open`` honors the mode bits
        ONLY when creating. If the file already existed (e.g. user ran
        an older build), ``O_TRUNC`` truncates but leaves perms alone.
        We follow up with an explicit ``chmod(0o600)`` and assert it.

    On Windows the mode bits are meaningless for access control, so we
    additionally apply an owner-only DACL via ``icacls`` (see
    :func:`_restrict_windows_acl`).
    """
    target = path or token_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(target), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("ascii"))
    finally:
        os.close(fd)
    # Belt + suspenders: chmod even on Windows (no-op on the perms but
    # consistent with the contract). On POSIX this is the line that
    # actually closes the pre-existing-loose-perms gap.
    os.chmod(str(target), 0o600)
    if sys.platform == "win32":
        # chmod does nothing for Windows access control; apply an
        # owner-only DACL so the token matches the POSIX 0600 posture.
        _restrict_windows_acl(target)
    return target


def read_token(path: Path | None = None) -> str | None:
    """Read the token from disk; None if missing."""
    target = path or token_path()
    try:
        return target.read_text("ascii").strip()
    except FileNotFoundError:
        return None


class BearerAuth:
    """FastAPI-compatible bearer-token verifier.

    Constructed at app startup with the active token. Use as a dependency:

        auth = BearerAuth(token)
        @app.get("/protected", dependencies=[Depends(auth)])
        async def protected(): ...
    """

    def __init__(self, token: str) -> None:
        self._token = token

    async def __call__(self, request: Request) -> None:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        provided = header.split(None, 1)[1].strip()
        if not secrets.compare_digest(provided, self._token):
            raise HTTPException(status_code=403, detail="invalid bearer token")


class BearerAuthClient(httpx.Auth):
    """httpx Auth helper for tests and Tauri clients."""

    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request

"""Local API auth: ephemeral per-launch bearer token.

A 256-bit URL-safe token is written to `~/.meetmind/token` (mode 0600) at
server start and rotated on every restart. Requests carry it as
`Authorization: Bearer <token>`. A Unix domain socket would avoid the file
entirely, but browsers and Tauri WebViews cannot reach UDS from
`fetch`/`EventSource`.
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
    report 0666 regardless), so this shells out to ``icacls`` twice:

      * ``/reset`` first replaces the whole ACL with the inherited
        defaults — this clears any *explicit* ACE a previous writer may
        have added (e.g. an ``Everyone`` grant), which neither
        ``/inheritance:r`` nor ``/grant:r`` would remove on their own;
      * ``/inheritance:r`` then strips every inherited ACE (Everyone,
        BUILTIN\\Users, ...), and ``/grant:r <user>:F`` leaves a single
        full-control ACE for the current user.

    On failure the token file is deleted and the error re-raised: a
    world-readable bearer token is worse than no token at all.
    """
    user = getpass.getuser()
    try:
        for argv in (
            ["icacls", str(target), "/reset"],
            ["icacls", str(target), "/inheritance:r", "/grant:r", f"{user}:F"],
        ):
            subprocess.run(  # noqa: S603 — fixed argv, path is ours, no shell
                argv,
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
    """Persist `token` to disk with mode 0600, defaulting to `token_path()`.

    ``os.open`` applies the mode only when it creates the file, so a
    pre-existing token file keeps its old (possibly loose) permissions; the
    explicit ``chmod`` below closes that gap. On Windows mode bits do not
    control access, so an owner-only DACL is applied as well.
    """
    target = path or token_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(target), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("ascii"))
    finally:
        os.close(fd)
    # Closes the pre-existing-loose-perms gap on POSIX.
    os.chmod(str(target), 0o600)
    if sys.platform == "win32":
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
    """FastAPI dependency that verifies the bearer token."""

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

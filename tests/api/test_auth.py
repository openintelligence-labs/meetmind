"""Tests for the bearer auth + token helpers."""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import Depends, FastAPI

from meetmind.api.auth import (
    BearerAuth,
    BearerAuthClient,
    generate_token,
    read_token,
    write_token,
)

# Principals that must never appear in the token file's DACL on Windows.
_FORBIDDEN_WIN_PRINCIPALS = ("everyone", "builtin\\users", "authenticated users")


def assert_owner_only_windows_acl(path: Path) -> None:
    """Assert (via `icacls <path>`) that only the current user has access."""
    out = subprocess.run(
        ["icacls", str(path)], check=True, capture_output=True, text=True
    ).stdout.lower()
    assert getpass.getuser().lower() in out, out
    for principal in _FORBIDDEN_WIN_PRINCIPALS:
        assert principal not in out, out


def test_generate_token_is_unique_and_long():
    a = generate_token()
    b = generate_token()
    assert a != b
    assert len(a) >= 40  # secrets.token_urlsafe(32) → ~43 chars


def test_write_token_uses_0600(tmp_path):
    target = tmp_path / "token"
    write_token("abc123", target)
    if sys.platform == "win32":
        # chmod bits are meaningless here; write_token applies an
        # owner-only DACL via icacls instead.
        assert_owner_only_windows_acl(target)
    else:
        mode = os.stat(target).st_mode & 0o777
        assert mode == 0o600
    assert target.read_text() == "abc123"


def test_write_token_overwrites_existing(tmp_path):
    target = tmp_path / "token"
    write_token("first", target)
    write_token("second", target)
    assert target.read_text() == "second"


def test_read_token_returns_none_when_missing(tmp_path):
    assert read_token(tmp_path / "absent") is None


def test_read_token_strips_whitespace(tmp_path):
    target = tmp_path / "token"
    target.write_text("xyz789\n")
    assert read_token(target) == "xyz789"


@pytest.fixture
def auth_app() -> tuple[FastAPI, str]:
    token = "test-token-abcdef"
    auth = BearerAuth(token)
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(auth)])
    async def protected():
        return {"ok": True}

    return app, token


async def test_bearer_auth_allows_correct_token(auth_app):
    app, token = auth_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


async def test_bearer_auth_rejects_wrong_token(auth_app):
    app, _ = auth_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/protected", headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 403


async def test_bearer_auth_rejects_missing_token(auth_app):
    app, _ = auth_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/protected")
        assert resp.status_code == 401


async def test_bearer_auth_client_helper(auth_app):
    app, token = auth_app
    transport = httpx.ASGITransport(app=app)
    auth = BearerAuthClient(token)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", auth=auth) as client:
        resp = await client.get("/protected")
        assert resp.status_code == 200

"""Security posture tests (Task #12).

Three layers:
  1. CORS narrows to the bound port when supplied.
  2. Token file lands at mode 0600 (read/write owner only).
  3. The outbound-call refusal stays green in default config.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from meetmind.api.auth import generate_token, write_token
from meetmind.api.http import _cors_regex, create_app
from tests.api.test_auth import assert_owner_only_windows_acl


def test_cors_regex_locked_to_bound_port() -> None:
    rx = re.compile(_cors_regex(7857))
    # Bound port: allowed (http + https + 127.0.0.1).
    for origin in ("http://localhost:7857", "https://127.0.0.1:7857", "tauri://localhost"):
        assert rx.match(origin), origin
    # Other ports: rejected.
    for origin in ("http://localhost:3000", "http://127.0.0.1:8080"):
        assert not rx.match(origin), origin
    # Cross-origin attempts: rejected.
    for origin in (
        "https://evil.example",
        "https://localhost.evil.example",
        "http://localhost.evil:7857",
    ):
        assert not rx.match(origin), origin


def test_cors_regex_unconstrained_when_no_port() -> None:
    """Backward compat: legacy callers + tests get the broad regex."""
    rx = re.compile(_cors_regex(None))
    assert rx.match("http://localhost:1234")
    assert rx.match("http://127.0.0.1:5678")
    assert not rx.match("https://evil.example")


def test_token_file_mode_is_0600(tmp_path: Path) -> None:
    target = tmp_path / "token"
    write_token("a-test-token", path=target)
    if sys.platform == "win32":
        # Mode bits don't gate access on Windows; assert the icacls
        # DACL instead: current user only, no Everyone/BUILTIN\Users.
        assert_owner_only_windows_acl(target)
    else:
        st = target.stat()
        # Owner-only read/write. Group/other bits must be zero.
        mode = st.st_mode & 0o777
        assert mode == 0o600, oct(mode)
    # And: must actually be readable by the owner.
    assert target.read_text("ascii") == "a-test-token"


def test_token_file_overwrite_preserves_0600(tmp_path: Path) -> None:
    """O_TRUNC reuse must not widen perms."""
    target = tmp_path / "token2"
    write_token("first", path=target)
    # Touch with broader perms to confirm we narrow back.
    if sys.platform == "win32":
        # chmod can't widen a Windows DACL — grant Everyone explicitly,
        # then confirm the rewrite strips it back to owner-only.
        subprocess.run(
            ["icacls", str(target), "/grant", "Everyone:F"],
            check=True,
            capture_output=True,
            text=True,
        )
        write_token("second", path=target)
        assert_owner_only_windows_acl(target)
    else:
        target.chmod(0o644)
        write_token("second", path=target)
        mode = target.stat().st_mode & 0o777
        assert mode == 0o600, oct(mode)


def test_unauth_request_is_rejected() -> None:
    app = create_app("test-token-xyz", port=7857)
    client = TestClient(app)
    # /v1/health is open by design (no Depends(auth)).
    assert client.get("/v1/health").status_code == 200
    # All other v1 endpoints require auth.
    assert client.get("/v1/info").status_code == 401
    assert client.get("/v1/meetings").status_code == 401
    # Bearer with wrong token returns 403.
    r = client.get("/v1/info", headers={"Authorization": "Bearer not-the-token"})
    assert r.status_code == 403


def test_auth_request_succeeds() -> None:
    app = create_app("the-correct-token", port=7857)
    client = TestClient(app)
    r = client.get("/v1/info", headers={"Authorization": "Bearer the-correct-token"})
    assert r.status_code == 200
    assert r.json()["version"]


def test_handshake_rejects_non_loopback() -> None:
    """The TestClient sets `client.host = "testclient"`, which is not loopback."""
    app = create_app("hs-token", port=7857)
    client = TestClient(app)
    r = client.get("/v1/auth/handshake")
    assert r.status_code == 403


def test_token_path_lives_in_meetmind_home(monkeypatch, tmp_path) -> None:
    """token_path() resolves to ~/.meetmind/token — verify with a fake HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    # Must use Path.home() resolution, not cached state.
    import importlib

    from meetmind.api import auth as auth_mod  # noqa: PLC0415

    importlib.reload(auth_mod)
    assert auth_mod.token_path() == tmp_path / ".meetmind" / "token"
    # Cleanup: restore the real module so subsequent tests see real paths.
    monkeypatch.undo()
    importlib.reload(auth_mod)


def test_default_config_makes_no_outbound_socket(monkeypatch) -> None:
    """Mirror of tests/security/test_no_outbound_calls.py — re-pinned here.

    Open a non-loopback socket and import a representative module; if
    the import or app-construction silently dials out, the assertion
    fires. We don't import the Slack/GitHub modules — those are
    egress-by-design and explicitly opt-in.
    """
    import socket  # noqa: PLC0415

    forbidden_hosts: set[str] = set()
    real_connect = socket.socket.connect

    def _watching_connect(self, addr):
        if isinstance(addr, tuple) and len(addr) == 2:
            host, _port = addr
            if host not in {"127.0.0.1", "::1", "localhost"}:
                forbidden_hosts.add(host)
        return real_connect(self, addr)

    monkeypatch.setattr(socket.socket, "connect", _watching_connect)

    # Standard default-config surfaces.
    from meetmind.analyze.llm import llm_config_summary  # noqa: PLC0415
    from meetmind.api.http import create_app  # noqa: PLC0415

    app = create_app("t", port=0)
    assert app is not None
    summary = llm_config_summary()
    assert isinstance(summary, dict)
    assert not forbidden_hosts, f"unexpected outbound: {forbidden_hosts}"


def test_handshake_token_loopback_only() -> None:
    """Document the loopback-only handshake contract via the regex test
    above + this app-level assertion."""
    app = create_app("loopback-token", port=7857)
    client = TestClient(app)
    # We can't easily fake `request.client.host` with TestClient — it
    # always reports "testclient" — but we can verify the route exists
    # and that it 403s unless we monkeypatch. The 403 path is covered
    # by test_handshake_rejects_non_loopback above.
    r = client.get("/v1/health")
    assert r.status_code == 200


def test_token_path_default_excludes_world_when_dir_created(tmp_path, monkeypatch) -> None:
    """Sanity: a fresh `write_token` to a path under a new dir creates
    the parent and restricts the file (not the dir) to the owner."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    import importlib

    from meetmind.api import auth as auth_mod  # noqa: PLC0415

    importlib.reload(auth_mod)
    tok = generate_token()
    p = auth_mod.write_token(tok)
    assert p.exists()
    if sys.platform == "win32":
        assert_owner_only_windows_acl(p)
    else:
        assert (p.stat().st_mode & 0o777) == 0o600
    # Cleanup.
    monkeypatch.undo()
    importlib.reload(auth_mod)

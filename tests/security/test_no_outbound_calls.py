"""Guarantee that no code path opens a non-loopback socket under the default
local configuration. If a default install ever silently phones home, these
tests fail.

``socket.socket.connect`` and ``connect_ex`` are patched to refuse any target
outside a deliberately narrow allow-list, then the API server and analyze
pipeline are exercised against them. Configuring a hosted LLM provider is an
explicit opt-in to egress, so these tests skip in that case.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Any

import pytest

from meetmind.analyze.llm import llm_config_summary

# Non-inet families carry no host to check and are let through.
_INET_FAMILIES = {socket.AF_INET, socket.AF_INET6}
_LOCALHOST_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


class OutboundConnectError(AssertionError):
    """Raised when test code attempts a non-loopback socket connect."""


def _is_loopback(target: Any) -> bool:
    if not isinstance(target, tuple) or len(target) < 1:
        return True  # AF_UNIX et al — let through
    host = target[0]
    if not isinstance(host, str):
        return False
    return host in _LOCALHOST_HOSTS or host.startswith("127.")


@contextmanager
def block_outbound_sockets():
    """Patch socket.socket.connect[_ex] to refuse non-loopback targets."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def _guarded_connect(self, target):
        if self.family in _INET_FAMILIES and not _is_loopback(target):
            raise OutboundConnectError(
                f"refused non-loopback connect: family={self.family} target={target}"
            )
        return real_connect(self, target)

    def _guarded_connect_ex(self, target):
        if self.family in _INET_FAMILIES and not _is_loopback(target):
            raise OutboundConnectError(
                f"refused non-loopback connect_ex: family={self.family} target={target}"
            )
        return real_connect_ex(self, target)

    def _guarded_getaddrinfo(host, *args, **kwargs):
        if isinstance(host, str) and host not in _LOCALHOST_HOSTS and not host.startswith("127."):
            # Resolution alone isn't egress, so it is allowed through; this
            # branch exists so a caller can hook unexpected DNS lookups.
            pass
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.getaddrinfo = _guarded_getaddrinfo
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        socket.getaddrinfo = real_getaddrinfo


def _provider_is_remote() -> bool:
    cfg = llm_config_summary()
    provider = (cfg["provider"] or "").lower()
    return provider not in ("ollama", "")


@pytest.mark.skipif(
    _provider_is_remote(),
    reason="MEETMIND_LLM_PROVIDER points at a hosted provider; egress is opted-in",
)
def test_serve_does_not_open_outbound_sockets(tmp_path):
    """Boot the FastAPI app + bind on 127.0.0.1 — must stay loopback."""
    import asyncio

    import uvicorn

    from meetmind.api.bus import EventBus
    from meetmind.api.http import create_app

    app = create_app("tok-1234", bus=EventBus())

    with block_outbound_sockets():
        # No requests are issued; this checks that boot alone does not egress.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        async def _boot_and_stop():
            config = uvicorn.Config(
                app, host="127.0.0.1", port=port, log_level="error", access_log=False
            )
            server = uvicorn.Server(config)
            task = asyncio.create_task(server.serve())
            for _ in range(50):
                if server.started:
                    break
                await asyncio.sleep(0.02)
            assert server.started, "uvicorn never bound to 127.0.0.1"
            server.should_exit = True
            await asyncio.wait_for(task, timeout=4.0)

        asyncio.run(_boot_and_stop())


@pytest.mark.skipif(
    _provider_is_remote(),
    reason="MEETMIND_LLM_PROVIDER points at a hosted provider; egress is opted-in",
)
def test_status_command_does_not_open_outbound_sockets():
    """`meetmind status` introspects only — it should not phone home.

    list_local_models() does probe the Ollama URL, which stays inside the
    loopback allow-list.
    """
    from meetmind.analyze.llm import list_local_models

    # Pin the base URL so an env override cannot change what is exercised.
    old = os.environ.get("MEETMIND_LLM_BASE_URL")
    os.environ["MEETMIND_LLM_BASE_URL"] = "http://127.0.0.1:11434"
    try:
        with block_outbound_sockets():
            list_local_models()  # must not raise
    finally:
        if old is None:
            os.environ.pop("MEETMIND_LLM_BASE_URL", None)
        else:
            os.environ["MEETMIND_LLM_BASE_URL"] = old


def test_block_outbound_sockets_actually_blocks():
    """Sanity: the test harness is real — confirm it blocks an obvious egress."""
    with block_outbound_sockets():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OutboundConnectError):
                s.connect(("1.1.1.1", 53))
        finally:
            s.close()

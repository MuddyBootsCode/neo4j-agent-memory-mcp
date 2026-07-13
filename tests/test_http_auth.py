"""Tests for bearer-token authentication on the HTTP/SSE transport (R10).

The HTTP transport previously had no authentication at all — anyone who could
reach the port got full tool access (see docs/reviews/2026-07-02-agent-memory-
deep-dive.md §2.3). This adds a static bearer-token ASGI middleware applied to
the FastMCP HTTP app, plus a startup guard that refuses to bind a non-loopback
host when no token is configured.

Tests exercise the ASGI app in-process via Starlette's TestClient — no real
network bind needed.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from agent_memory_mcp.mcp.server import (
    BearerAuthMiddleware,
    assert_safe_http_bind,
)


def _make_app(token: str) -> Starlette:
    async def ok(request):
        return PlainTextResponse("ok")

    return Starlette(
        routes=[Route("/mcp", ok)],
        middleware=[Middleware(BearerAuthMiddleware, token=token)],
    )


class TestBearerAuthMiddleware:
    def test_missing_authorization_header_returns_401(self):
        app = _make_app("secret-token")
        client = TestClient(app)

        response = client.get("/mcp")

        assert response.status_code == 401

    def test_wrong_token_returns_401(self):
        app = _make_app("secret-token")
        client = TestClient(app)

        response = client.get("/mcp", headers={"Authorization": "Bearer wrong"})

        assert response.status_code == 401

    def test_malformed_header_returns_401(self):
        app = _make_app("secret-token")
        client = TestClient(app)

        response = client.get("/mcp", headers={"Authorization": "secret-token"})

        assert response.status_code == 401

    def test_correct_token_passes_through(self):
        app = _make_app("secret-token")
        client = TestClient(app)

        response = client.get(
            "/mcp", headers={"Authorization": "Bearer secret-token"}
        )

        assert response.status_code == 200
        assert response.text == "ok"


class TestAssertSafeHttpBind:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_allowed_without_token(self, host):
        # Should not raise — loopback is safe even with no auth configured.
        assert_safe_http_bind(host=host, token=None)

    @pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "example.com"])
    def test_non_loopback_without_token_raises(self, host):
        with pytest.raises(RuntimeError, match="(?i)token"):
            assert_safe_http_bind(host=host, token=None)

    @pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "example.com"])
    def test_non_loopback_with_token_allowed(self, host):
        # Should not raise once a token is configured.
        assert_safe_http_bind(host=host, token="secret-token")


class TestRunSseWrapper:
    """Regression tests for the Neo4jMemoryMCPServer.run_sse wrapper.

    The wrapper previously called ``FastMCP.run_async`` directly, bypassing
    both ``assert_safe_http_bind`` and ``BearerAuthMiddleware`` — so
    ``await server.run_sse(host="0.0.0.0")`` exposed the full tool surface
    with no authentication (GitHub issue #6). These tests mock out
    ``run_async`` (no real sockets) and assert the same guard + middleware
    behavior as the ``run_server`` path.
    """

    @pytest.fixture()
    def server(self, monkeypatch):
        """A wrapper server with bootstrap and run_async mocked out."""
        from unittest.mock import AsyncMock, MagicMock

        import agent_memory_mcp.mcp._bootstrap as bootstrap_mod
        from agent_memory_mcp.mcp.server import Neo4jMemoryMCPServer

        monkeypatch.setattr(
            bootstrap_mod, "bootstrap_upstream_patches", MagicMock()
        )
        monkeypatch.delenv("NAM_HTTP_TOKEN", raising=False)

        srv = Neo4jMemoryMCPServer(MagicMock())
        srv._mcp.run_async = AsyncMock()
        return srv

    @staticmethod
    def _middleware_of(server):
        """Extract the middleware kwarg passed to the mocked run_async."""
        return server._mcp.run_async.call_args.kwargs["middleware"]

    async def test_non_loopback_without_token_raises_before_binding(
        self, server
    ):
        with pytest.raises(RuntimeError, match="(?i)token"):
            await server.run_sse(host="0.0.0.0")

        server._mcp.run_async.assert_not_awaited()

    async def test_non_loopback_with_explicit_token_installs_auth(self, server):
        await server.run_sse(host="0.0.0.0", port=9090, http_token="secret")

        server._mcp.run_async.assert_awaited_once()
        kwargs = server._mcp.run_async.call_args.kwargs
        assert kwargs["transport"] == "sse"
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9090

        (mw,) = self._middleware_of(server)
        assert mw.cls is BearerAuthMiddleware
        assert mw.kwargs == {"token": "secret"}

    async def test_env_token_fallback(self, server, monkeypatch):
        monkeypatch.setenv("NAM_HTTP_TOKEN", "env-secret")

        await server.run_sse(host="0.0.0.0")

        (mw,) = self._middleware_of(server)
        assert mw.cls is BearerAuthMiddleware
        assert mw.kwargs == {"token": "env-secret"}

    async def test_explicit_token_wins_over_env(self, server, monkeypatch):
        monkeypatch.setenv("NAM_HTTP_TOKEN", "env-secret")

        await server.run_sse(host="0.0.0.0", http_token="explicit")

        (mw,) = self._middleware_of(server)
        assert mw.kwargs == {"token": "explicit"}

    async def test_loopback_default_without_token_runs_unauthenticated(
        self, server
    ):
        # Backward-compatible: default loopback bind needs no token.
        await server.run_sse()

        server._mcp.run_async.assert_awaited_once()
        kwargs = server._mcp.run_async.call_args.kwargs
        assert kwargs["transport"] == "sse"
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8080
        assert kwargs["middleware"] == []

    async def test_loopback_with_token_still_installs_auth(self, server):
        await server.run_sse(http_token="secret")

        (mw,) = self._middleware_of(server)
        assert mw.cls is BearerAuthMiddleware
        assert mw.kwargs == {"token": "secret"}

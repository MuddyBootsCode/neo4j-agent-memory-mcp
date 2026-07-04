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

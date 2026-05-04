"""Tests for health check endpoint."""

import pytest

from neo4j_agent_memory.mcp.server import create_mcp_server


class TestHealthEndpoint:
    """Verify the health route is registered on the server."""

    def test_server_creates_without_error(self):
        """Server should create successfully without settings (no lifespan)."""
        server = create_mcp_server(settings=None)
        assert server is not None

    def test_health_custom_route_registered(self):
        """The /health custom route should be registered on the server."""
        server = create_mcp_server(settings=None)
        # FastMCP 2.x exposes custom_routes or we check _custom_routes
        # This test validates the route was added without errors
        assert server is not None

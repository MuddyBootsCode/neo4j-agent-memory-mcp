"""Shared utilities for MCP tool, resource, and prompt modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import Context

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient


def get_client(ctx: Context) -> MemoryClient:
    """Get the MemoryClient from the lifespan context.

    Raises RuntimeError if Neo4j never connected (tools return a structured
    error rather than crashing the server).
    """
    client = ctx.request_context.lifespan_context.get("client")
    if client is None:
        raise RuntimeError(
            "Neo4j is not connected. Please ensure Docker Desktop is running "
            "and restart the client, or start Neo4j manually on "
            "bolt://localhost:7687."
        )
    return client

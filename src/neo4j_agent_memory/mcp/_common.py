"""Shared utilities for MCP tool, resource, and prompt modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import Context

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient
    from neo4j_agent_memory.mcp._registry import ClientRegistry


def get_client(ctx: Context) -> MemoryClient:
    """Get the general MemoryClient from lifespan context.

    Backward-compatible: works with both single-client and
    registry-based setups.
    """
    lifespan = ctx.request_context.lifespan_context

    # New registry-based access
    registry = lifespan.get("registry")
    if registry is not None:
        return registry.general

    # Legacy single-client access
    client = lifespan.get("client")
    if client is None:
        raise RuntimeError(
            "Neo4j is not connected. Please ensure Docker Desktop is running "
            "and restart Claude Desktop, or start Neo4j manually on "
            "bolt://localhost:7687."
        )
    return client


def get_registry(ctx: Context) -> ClientRegistry:
    """Get the ClientRegistry from lifespan context."""
    registry = ctx.request_context.lifespan_context.get("registry")
    if registry is None:
        raise RuntimeError(
            "ClientRegistry not available. Multi-database support "
            "requires NAM_VERTICALS to be configured."
        )
    return registry


def get_router(ctx: Context):
    """Get the QueryRouter from lifespan context."""
    from neo4j_agent_memory.routing.router import QueryRouter

    router = ctx.request_context.lifespan_context.get("router")
    if router is None:
        # Return a disabled router that always routes to general
        return QueryRouter(available_databases=["neo4j"])
    return router


def get_reranker(ctx: Context):
    """Get the ResultReranker from lifespan context."""
    from neo4j_agent_memory.routing.router import ResultReranker

    reranker = ctx.request_context.lifespan_context.get("reranker")
    if reranker is None:
        return ResultReranker(enabled=False)
    return reranker

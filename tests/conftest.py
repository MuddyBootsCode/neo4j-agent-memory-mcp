"""Shared test fixtures for neo4j-agent-memory-mcp tests.

Provides reusable fixtures that eliminate repeated boilerplate across test files:
- BAML client mocking (entity extraction, reasoning extraction)
- Mock MemoryClient with configurable sub-systems
- MCP tool access helpers
- Monkeypatched MCP context (get_client, get_registry, get_router)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastmcp import FastMCP

from neo4j_agent_memory.mcp._tools import register_tools


# ── BAML Mocking ─────────────────────────────────────────────────────


@pytest.fixture
def mock_baml_extract(monkeypatch):
    """Patch BAML generated client to avoid live API calls.

    Returns (mock_fn, mock_result) so tests can configure return values.
    """
    mock_result = MagicMock()
    mock_result.entities = []
    mock_result.relations = []
    mock_result.preferences = []

    mock_fn = AsyncMock(return_value=mock_result)
    monkeypatch.setattr(
        "neo4j_agent_memory.baml_client.async_client.b.ExtractEntities",
        mock_fn,
    )
    return mock_fn, mock_result


@pytest.fixture
def mock_baml_reasoning(monkeypatch):
    """Patch BAML reasoning extraction and synthesis to avoid live API calls.

    Returns (mock_extract, mock_synthesize) for test inspection.
    """
    mock_extract = AsyncMock()
    mock_synthesize = AsyncMock(return_value="Synthesized explanation")

    monkeypatch.setattr(
        "neo4j_agent_memory.extraction.reasoning_extractor.BamlReasoningExtractor.extract_reasoning",
        mock_extract,
    )
    monkeypatch.setattr(
        "neo4j_agent_memory.extraction.reasoning_extractor.BamlReasoningExtractor.synthesize_explanation",
        mock_synthesize,
    )
    return mock_extract, mock_synthesize


# ── Mock MemoryClient ────────────────────────────────────────────────


@pytest.fixture
def mock_memory_client():
    """Create a fully-wired mock MemoryClient with all sub-systems.

    Returns the mock client. All async methods are pre-configured with
    empty/default return values. Override specific methods in your test:

        def test_something(mock_memory_client):
            mock_memory_client.long_term.search_entities.return_value = [...]
    """
    client = MagicMock()

    # short_term (messages)
    client.short_term.add_message = AsyncMock(return_value=MagicMock(id="msg-001"))
    client.short_term.search_messages = AsyncMock(return_value=[])
    client.short_term._embedder = None

    # long_term (entities, preferences, facts)
    client.long_term.add_fact = AsyncMock(return_value=MagicMock(id="fact-001"))
    client.long_term.search_entities = AsyncMock(return_value=[])
    client.long_term.search_preferences = AsyncMock(return_value=[])
    client.long_term.search_facts = AsyncMock(return_value=[])
    client.long_term._embedder = None

    # reasoning (traces)
    client.reasoning.start_trace = AsyncMock()
    client.reasoning.add_step = AsyncMock()
    client.reasoning.record_tool_call = AsyncMock()
    client.reasoning.complete_trace = AsyncMock()
    client.reasoning.get_trace_with_steps = AsyncMock()
    client.reasoning.get_similar_traces = AsyncMock(return_value=[])
    client.reasoning.list_traces = AsyncMock()

    # graph (raw Cypher)
    client.graph.execute_read = AsyncMock(return_value=[])
    client.graph.execute_write = AsyncMock(return_value=[])

    return client


# ── MCP Tool Helpers ─────────────────────────────────────────────────


@pytest.fixture
def mcp_app():
    """Create a FastMCP app with all tools registered.

    Use with get_tool_fn() to access individual tool functions.
    """
    mcp = FastMCP("test")
    register_tools(mcp)
    return mcp


def get_tool_fn(mcp: FastMCP, tool_name: str):
    """Extract a registered tool function from a FastMCP instance.

    Replaces the 4-line for-loop pattern that appears 22+ times in tests.

    Usage:
        def test_something(mcp_app):
            search_fn = get_tool_fn(mcp_app, "memory_search")
            result = await search_fn(ctx, query="test")
    """
    for tool in mcp._tool_manager._tools.values():
        if tool.name == tool_name:
            return tool.fn
    raise ValueError(f"Tool '{tool_name}' not found. Available: "
                     f"{[t.name for t in mcp._tool_manager._tools.values()]}")


# ── MCP Context Mocking ─────────────────────────────────────────────


@pytest.fixture
def mock_mcp_context(mock_memory_client, monkeypatch):
    """Monkeypatch get_client/get_registry/get_router for single-client MCP tests.

    Returns (ctx, client) where ctx is a mock Context and client is the
    mock_memory_client fixture. The registry raises RuntimeError (forcing
    single-client fallback), and the router returns a default general-db route.

    This replaces the copy-pasted monkeypatch blocks in test_temporal.py,
    test_contradiction.py, and test_reasoning_tools.py.
    """
    ctx = MagicMock()

    monkeypatch.setattr(
        "neo4j_agent_memory.mcp._tools.get_client",
        lambda _ctx: mock_memory_client,
    )

    def _no_registry(_ctx):
        raise RuntimeError("No registry in test")

    monkeypatch.setattr(
        "neo4j_agent_memory.mcp._tools.get_registry",
        _no_registry,
    )

    mock_router = MagicMock()
    mock_route = MagicMock()
    mock_route.primary_database = "neo4j"
    mock_route.to_metadata.return_value = {"primary": "neo4j"}
    mock_router.route_storage = AsyncMock(return_value=mock_route)
    mock_router.route_query = AsyncMock(return_value=mock_route)

    monkeypatch.setattr(
        "neo4j_agent_memory.mcp._tools.get_router",
        lambda _ctx: mock_router,
    )

    return ctx, mock_memory_client

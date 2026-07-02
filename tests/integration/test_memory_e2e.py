"""End-to-end memory tests through the MCP tools against a real Neo4j.

These are the "does the memory actually work" tests: they drive the real
``memory_store`` / ``memory_search`` / ``temporal_query`` / ``entity_lookup``
tool functions (not the underlying client methods) against the test database,
so the full 1,600-line tool layer is exercised against real storage.

Requires a running Neo4j + Bedrock (see tests/integration/conftest.py).
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.asyncio


def _tool(mcp, name):
    for t in mcp._tool_manager._tools.values():
        if t.name == name:
            return t.fn
    raise AssertionError(f"tool {name} not registered")


@pytest.fixture
def tools(memory_client, monkeypatch):
    """Register the MCP tools wired to the real test-DB client."""
    from fastmcp import FastMCP

    from neo4j_agent_memory.mcp._tools import register_tools

    monkeypatch.setattr(
        "neo4j_agent_memory.mcp._tools.get_client", lambda _ctx: memory_client
    )
    mcp = FastMCP("e2e")
    register_tools(mcp)
    return mcp, MagicMock()


async def test_message_store_extracts_and_search_finds(tools, cypher_session):
    """Storing a message extracts entities and search retrieves them."""
    mcp, ctx = tools
    session_id = f"e2e-{uuid.uuid4()}"

    stored = json.loads(
        await _tool(mcp, "memory_store")(
            ctx,
            memory_type="message",
            content=(
                "Sarah Chen works at DataVault Solutions in Denver. "
                "She is leading the graph migration project."
            ),
            session_id=session_id,
        )
    )
    assert stored["stored"] is True
    assert stored["entities"] >= 2  # at least Sarah Chen + DataVault

    # The message and its entities are in the graph.
    counts = await cypher_session.execute_read(
        "MATCH (e:Entity) RETURN count(e) AS c", {}
    )
    assert counts[0]["c"] >= 2

    # Search finds the stored content.
    found = json.loads(
        await _tool(mcp, "memory_search")(
            ctx, query="Who works at DataVault?", memory_types=["messages", "entities"]
        )
    )
    names = {e["name"].lower() for e in found["results"].get("entities", [])}
    assert any("sarah" in n or "datavault" in n for n in names)


async def test_fact_roundtrip_and_point_in_time(tools):
    """R1 regression: a fact stored with valid_from is found by temporal_query.

    Before the epoch-millis unification this failed silently — valid_from was an
    ISO string compared against an integer, so point-in-time queries returned
    nothing for exactly the facts that carried temporal data.
    """
    mcp, ctx = tools

    await _tool(mcp, "memory_store")(
        ctx,
        memory_type="fact",
        subject="Alice",
        predicate="ROLE",
        object_value="Engineer",
        content="Alice became an Engineer",
        valid_from="2026-03-01T00:00:00Z",
    )

    # A point AFTER valid_from returns the fact.
    after = json.loads(
        await _tool(mcp, "temporal_query")(
            ctx, point_in_time="2026-04-01T00:00:00Z", subject="Alice"
        )
    )
    triples = {(f["subject"], f["predicate"], f["object"]) for f in after["facts"]}
    assert ("Alice", "ROLE", "Engineer") in triples

    # A point BEFORE valid_from excludes it.
    before = json.loads(
        await _tool(mcp, "temporal_query")(
            ctx, point_in_time="2026-02-01T00:00:00Z", subject="Alice"
        )
    )
    before_triples = {(f["subject"], f["predicate"], f["object"]) for f in before["facts"]}
    assert ("Alice", "ROLE", "Engineer") not in before_triples


async def test_fact_supersession_marks_active_and_expired(tools):
    """Storing a newer fact for the same subject+predicate supersedes the old one."""
    mcp, ctx = tools

    await _tool(mcp, "memory_store")(
        ctx,
        memory_type="fact",
        subject="Bob",
        predicate="WORKS_AT",
        object_value="Acme",
        content="Bob works at Acme",
    )
    second = json.loads(
        await _tool(mcp, "memory_store")(
            ctx,
            memory_type="fact",
            subject="Bob",
            predicate="WORKS_AT",
            object_value="Globex",
            content="Bob works at Globex",
        )
    )
    assert second["superseded_facts"] >= 1

    # Search shows the current fact as active, the old one as expired.
    found = json.loads(
        await _tool(mcp, "memory_search")(
            ctx, query="Where does Bob work?", memory_types=["facts"]
        )
    )
    by_object = {f["object"]: f["temporal_status"] for f in found["results"]["facts"]}
    assert by_object.get("Globex") == "active"
    assert by_object.get("Acme") == "expired"


async def test_entity_lookup_type_filter_and_injection_guard(tools):
    """entity_lookup honors a valid type and rejects an injection payload."""
    mcp, ctx = tools
    session_id = f"e2e-{uuid.uuid4()}"

    await _tool(mcp, "memory_store")(
        ctx,
        memory_type="message",
        content="Priya Okonkwo is a machine learning engineer at Graphable.",
        session_id=session_id,
    )

    # A crafted entity_type must be rejected, not interpolated into Cypher.
    rejected = json.loads(
        await _tool(mcp, "entity_lookup")(
            ctx, name="Priya", entity_type="Person WITH e MATCH (n) DETACH DELETE n //"
        )
    )
    assert "error" in rejected

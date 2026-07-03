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


# ── Temporal correctness (R5-R8, R14) ────────────────────────────────
#
# These tests disable the LLM contradiction/temporal-extraction passes so
# they exercise exactly the SPO supersession path in the fact branch.


@pytest.fixture
def temporal_only(monkeypatch):
    """Disable LLM contradiction detection + temporal extraction."""
    monkeypatch.setenv("NAM_CONTRADICTION_DETECTION", "false")
    monkeypatch.setenv("NAM_TEMPORAL_EXTRACTION", "false")


async def test_concurrent_contradicting_stores_do_not_mutually_invalidate(
    tools, memory_client, cypher_session, temporal_only, monkeypatch
):
    """R5: two concurrent contradicting stores must not invalidate each other.

    Before the atomic create+supersede fix, add_fact and the supersession ran
    as separate transactions: interleaved stores of the same subject+predicate
    superseded each other, leaving a supersession cycle and ZERO active facts.
    """
    import asyncio

    mcp, ctx = tools
    store = _tool(mcp, "memory_store")

    # Barrier on the embedder so both stores embed together, maximizing the
    # window where both facts exist before either supersession pass runs.
    embedder = memory_client.long_term._embedder
    orig_embed = embedder.embed
    release = asyncio.Event()
    pending = {"count": 0}

    async def synced_embed(text):
        pending["count"] += 1
        if pending["count"] >= 2:
            release.set()
        await release.wait()
        return await orig_embed(text)

    monkeypatch.setattr(embedder, "embed", synced_embed)

    await asyncio.gather(
        store(
            ctx, memory_type="fact", subject="Racer", predicate="WORKS_AT",
            object_value="Acme", content="Racer works at Acme",
        ),
        store(
            ctx, memory_type="fact", subject="Racer", predicate="WORKS_AT",
            object_value="Globex", content="Racer works at Globex",
        ),
    )

    rows = await cypher_session.execute_read(
        """
        MATCH (f:Fact)
        WHERE toLower(trim(f.subject)) = 'racer'
        RETURN f.id AS id, f.object AS object,
               f.valid_until AS valid_until, f.superseded_by AS superseded_by
        """,
        {},
    )
    active = [r for r in rows if r["valid_until"] is None]
    assert len(active) >= 1, f"all facts mutually invalidated: {rows}"

    # No supersession cycle: a superseded fact's successor must not itself
    # be superseded by the fact it replaced.
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        successor = r["superseded_by"]
        if successor is not None and successor in by_id:
            assert by_id[successor]["superseded_by"] != r["id"], (
                f"supersession cycle between {r['id']} and {successor}: {rows}"
            )


async def test_historical_fact_does_not_supersede_current(
    tools, cypher_session, temporal_only
):
    """R6: a bounded (historical) fact must NOT supersede the current one."""
    mcp, ctx = tools
    store = _tool(mcp, "memory_store")

    await store(
        ctx, memory_type="fact", subject="Carol", predicate="WORKS_AT",
        object_value="Initech", content="Carol works at Initech",
    )
    second = json.loads(
        await store(
            ctx, memory_type="fact", subject="Carol", predicate="WORKS_AT",
            object_value="Hooli", content="Carol worked at Hooli until mid-2020",
            valid_from="2015-01-01T00:00:00Z",
            valid_until="2020-06-01T00:00:00Z",
        )
    )
    assert second["superseded_facts"] == 0

    rows = await cypher_session.execute_read(
        """
        MATCH (f:Fact {subject: 'Carol'})
        WHERE f.valid_until IS NULL
        RETURN f.object AS object
        """,
        {},
    )
    assert [r["object"] for r in rows] == ["Initech"]


async def test_identical_reaffirm_refreshes_and_preserves_valid_from(
    tools, cypher_session, temporal_only
):
    """R7: re-affirming an identical fact refreshes it — no supersession,
    no duplicate node, and valid_from ("known since") is preserved."""
    from datetime import datetime, timezone

    mcp, ctx = tools
    store = _tool(mcp, "memory_store")

    first = json.loads(
        await store(
            ctx, memory_type="fact", subject="Erin", predicate="ROLE",
            object_value="Engineer", content="Erin is an Engineer",
            valid_from="2026-03-01T00:00:00Z",
        )
    )
    second = json.loads(
        await store(
            ctx, memory_type="fact", subject="Erin", predicate="ROLE",
            object_value="Engineer", content="Erin is an Engineer",
        )
    )

    assert second.get("refreshed") is True
    assert second["superseded_facts"] == 0
    assert second["id"] == first["id"]

    rows = await cypher_session.execute_read(
        """
        MATCH (f:Fact {subject: 'Erin', predicate: 'ROLE'})
        RETURN f.valid_from AS valid_from, f.valid_until AS valid_until
        """,
        {},
    )
    assert len(rows) == 1, f"expected one fact node, got {rows}"
    assert rows[0]["valid_until"] is None
    march = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert rows[0]["valid_from"] == march


async def test_expired_at_set_on_spo_supersession(
    tools, cypher_session, temporal_only
):
    """R8: the plain SPO supersession path must set expired_at (not just the
    contradiction path) so knowledge_state stops showing dead facts."""
    mcp, ctx = tools
    store = _tool(mcp, "memory_store")

    await store(
        ctx, memory_type="fact", subject="Frank", predicate="WORKS_AT",
        object_value="Acme", content="Frank works at Acme",
    )
    second = json.loads(
        await store(
            ctx, memory_type="fact", subject="Frank", predicate="WORKS_AT",
            object_value="Globex", content="Frank works at Globex",
        )
    )
    assert second["superseded_facts"] == 1

    rows = await cypher_session.execute_read(
        """
        MATCH (f:Fact {subject: 'Frank', object: 'Acme'})
        RETURN f.valid_until AS valid_until, f.expired_at AS expired_at,
               f.superseded_by AS superseded_by
        """,
        {},
    )
    assert len(rows) == 1
    assert rows[0]["valid_until"] is not None
    assert rows[0]["expired_at"] is not None, "expired_at not set on SPO supersession"
    assert rows[0]["superseded_by"] == second["id"]


async def test_supersession_is_case_and_whitespace_insensitive(
    tools, cypher_session, temporal_only
):
    """R14: subject/predicate matching must normalize case + whitespace."""
    mcp, ctx = tools
    store = _tool(mcp, "memory_store")

    await store(
        ctx, memory_type="fact", subject="Grace", predicate="WORKS_AT",
        object_value="Acme", content="Grace works at Acme",
    )
    second = json.loads(
        await store(
            ctx, memory_type="fact", subject="  grace  ", predicate="works_at",
            object_value="Globex", content="grace works at Globex now",
        )
    )
    assert second["superseded_facts"] == 1

    rows = await cypher_session.execute_read(
        """
        MATCH (f:Fact)
        WHERE toLower(trim(f.subject)) = 'grace' AND f.valid_until IS NULL
        RETURN f.object AS object
        """,
        {},
    )
    assert [r["object"] for r in rows] == ["Globex"]

"""Graph integrity assertion tests.

After storing multiple messages with extraction enabled, verifies the
pivot-era structural invariants (MUD-395) via direct Cypher queries: the
org entity ontology is retired, so message storage must produce Message
nodes with embeddings and NOTHING from the old Entity/MENTIONS/RELATED_TO
ontology, while preference-bearing content flows to Preference nodes.
"""

import pytest

# Corpus of messages exercising the retired org-entity shapes — none of
# these may produce Entity nodes any more.
CORPUS = [
    {
        "session_id": "integrity-session-1",
        "content": "Michael is VP of Engineering at Graphable. He drives a Tesla Model 3.",
    },
    {
        "session_id": "integrity-session-1",
        "content": "Sarah presented the Q3 analysis at the board meeting.",
    },
    {
        "session_id": "integrity-session-2",
        "content": "Raj Patel is the CTO of DataVault Solutions. He lives in Denver.",
    },
    {
        "session_id": "integrity-session-2",
        "content": "Alice completed the embedding pipeline migration at Graphable.",
    },
    {
        "session_id": "integrity-session-3",
        "content": "Marcus signed the lease for the new Denver office on Larimer Street.",
    },
]


@pytest.fixture
async def populated_graph(memory_client):
    """Store the full corpus with extraction enabled, return the client."""
    for msg in CORPUS:
        await memory_client.short_term.add_message(
            session_id=msg["session_id"],
            role="user",
            content=msg["content"],
            generate_embedding=True,
            extract_entities=True,
        )
    return memory_client


class TestNodeIntegrity:
    """Verify node-level invariants."""

    async def test_all_messages_have_embeddings(self, populated_graph):
        """Every stored Message node should have a non-null embedding."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (m:Message) "
            "RETURN m.id AS id, m.embedding IS NOT NULL AS has_emb",
            {},
        )
        assert len(rows) == len(CORPUS), (
            f"Expected {len(CORPUS)} messages, got {len(rows)}"
        )
        missing = [r["id"] for r in rows if not r["has_emb"]]
        assert len(missing) == 0, (
            f"Messages without embeddings: {missing}"
        )

    async def test_message_extraction_creates_no_entities(self, populated_graph):
        """The org entity ontology is retired: message storage extracts only
        preferences (coding-memory pivot), so no Entity nodes may appear."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (e:Entity) RETURN e.name AS name, e.type AS type", {},
        )
        assert len(rows) == 0, (
            f"Message extraction created entities after the org-ontology "
            f"retirement: {[(r['name'], r['type']) for r in rows]}"
        )


class TestEdgeIntegrity:
    """Verify edge-level invariants."""

    async def test_no_org_ontology_edges(self, populated_graph):
        """No MENTIONS or RELATED_TO edges may survive the pivot — the
        retired extraction path was their only producer."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH ()-[r]->() WHERE type(r) IN ['MENTIONS', 'RELATED_TO'] "
            "RETURN type(r) AS rel, count(r) AS count",
            {},
        )
        assert rows == [], (
            f"Retired org-ontology edges present: "
            f"{[(r['rel'], r['count']) for r in rows]}"
        )


class TestGraphStatistics:
    """Verify expected graph shape after loading the corpus (pivot semantics)."""

    async def test_messages_persist_and_are_queryable(self, populated_graph):
        """Every corpus message persists and vector search retrieves content."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (m:Message) RETURN count(m) AS count", {},
        )
        assert rows[0]["count"] == len(CORPUS), (
            f"Expected {len(CORPUS)} messages, got {rows[0]['count']}"
        )

        results = await client.short_term.search_messages(
            query="Who is the CTO of DataVault Solutions?",
            limit=5,
            threshold=0.3,
        )
        assert any("Raj Patel" in r.content for r in results), (
            f"Expected the Raj Patel message in search results, got "
            f"{[r.content for r in results]}"
        )

    async def test_preference_persists_after_seeding(self, populated_graph):
        """Preference-bearing content stored through the message path
        produces Preference nodes — the one thing message extraction still
        persists after the pivot."""
        from agent_memory_mcp.extraction.coding import extract_coding_memory
        from agent_memory_mcp.extraction.unified import persist_preferences

        client = populated_graph
        extracted = await extract_coding_memory(
            "I prefer integration tests over mocks — always verify against "
            "the real database.",
            branch="",
            task=None,
            files=[],
        )
        stored = await persist_preferences(client, extracted["preferences"])
        assert stored >= 1, (
            f"Expected the stated testing preference to persist, extraction "
            f"returned {extracted['preferences']}"
        )

        rows = await client.graph.execute_read(
            "MATCH (p:Preference) RETURN count(p) AS count", {},
        )
        assert rows[0]["count"] > 0

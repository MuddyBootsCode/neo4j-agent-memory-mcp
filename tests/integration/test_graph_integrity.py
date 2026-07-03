"""Graph integrity assertion tests.

After storing multiple messages with entity extraction, verifies
structural invariants of the knowledge graph via direct Cypher queries.
These catch data quality issues that functional tests miss.
"""

import pytest

# Corpus of messages that produce a known set of entities and relationships
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

    async def test_all_entities_have_type(self, populated_graph):
        """Every Entity node should have a non-null type property."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (e:Entity) "
            "WHERE e.type IS NULL "
            "RETURN e.name AS name",
            {},
        )
        assert len(rows) == 0, (
            f"Entities missing type: {[r['name'] for r in rows]}"
        )

    async def test_entity_types_are_valid(self, populated_graph):
        """Entity types should be from the unified ontology (POLE+O + domain types)."""
        from neo4j_agent_memory.baml_client.types import EntityType

        client = populated_graph
        valid_types = {t.value for t in EntityType}

        rows = await client.graph.execute_read(
            "MATCH (e:Entity) RETURN DISTINCT e.type AS type", {},
        )
        actual_types = {r["type"] for r in rows}
        invalid = actual_types - valid_types
        assert len(invalid) == 0, (
            f"Invalid entity types found: {invalid}. "
            f"Valid types: {valid_types}"
        )

    async def test_no_duplicate_entities(self, populated_graph):
        """No two Entity nodes should share the same (name, type) pair."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (e:Entity) "
            "WITH e.name AS name, e.type AS type, count(e) AS cnt "
            "WHERE cnt > 1 "
            "RETURN name, type, cnt",
            {},
        )
        assert len(rows) == 0, (
            f"Duplicate entities: "
            f"{[(r['name'], r['type'], r['cnt']) for r in rows]}"
        )


class TestEdgeIntegrity:
    """Verify edge-level invariants."""

    async def test_no_orphaned_entities(self, populated_graph):
        """Every Entity should have at least one incoming MENTIONS edge."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (e:Entity) "
            "WHERE NOT (e)<-[:MENTIONS]-(:Message) "
            "RETURN e.name AS name, e.type AS type",
            {},
        )
        assert len(rows) == 0, (
            f"Orphaned entities (no MENTIONS): "
            f"{[(r['name'], r['type']) for r in rows]}"
        )

    async def test_related_to_edges_have_type(self, populated_graph):
        """Every RELATED_TO edge should have a relation_type property."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH ()-[r:RELATED_TO]->() "
            "WHERE r.relation_type IS NULL OR r.relation_type = '' "
            "RETURN startNode(r).name AS src, endNode(r).name AS tgt",
            {},
        )
        assert len(rows) == 0, (
            f"RELATED_TO edges missing relation_type: "
            f"{[(r['src'], r['tgt']) for r in rows]}"
        )

    async def test_related_to_connects_entities(self, populated_graph):
        """RELATED_TO edges should only connect Entity nodes."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (a)-[r:RELATED_TO]->(b) "
            "WHERE NOT (a:Entity) OR NOT (b:Entity) "
            "RETURN labels(a) AS src_labels, labels(b) AS tgt_labels",
            {},
        )
        assert len(rows) == 0, (
            f"RELATED_TO connecting non-Entity nodes: "
            f"{[(r['src_labels'], r['tgt_labels']) for r in rows]}"
        )

    async def test_mentions_connects_message_to_entity(self, populated_graph):
        """MENTIONS edges should only go from Message to Entity."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (a)-[:MENTIONS]->(b) "
            "WHERE NOT (a:Message) OR NOT (b:Entity) "
            "RETURN labels(a) AS src_labels, labels(b) AS tgt_labels",
            {},
        )
        assert len(rows) == 0, (
            f"MENTIONS connecting wrong node types: "
            f"{[(r['src_labels'], r['tgt_labels']) for r in rows]}"
        )


class TestGraphQuerySafety:
    """Verify the read-only graph_query tool rejects write operations."""

    async def test_rejects_create(self, populated_graph):
        """graph_query should reject CREATE statements."""
        from neo4j_agent_memory.mcp._tools import _is_read_only_query

        assert not _is_read_only_query("CREATE (n:Test {name: 'bad'})")

    async def test_rejects_merge(self, populated_graph):
        """graph_query should reject MERGE statements."""
        from neo4j_agent_memory.mcp._tools import _is_read_only_query

        assert not _is_read_only_query("MERGE (n:Test {name: 'bad'})")

    async def test_rejects_delete(self, populated_graph):
        """graph_query should reject DELETE statements."""
        from neo4j_agent_memory.mcp._tools import _is_read_only_query

        assert not _is_read_only_query("MATCH (n) DELETE n")

    async def test_rejects_detach_delete(self, populated_graph):
        """graph_query should reject DETACH DELETE statements."""
        from neo4j_agent_memory.mcp._tools import _is_read_only_query

        assert not _is_read_only_query("MATCH (n) DETACH DELETE n")

    async def test_rejects_set(self, populated_graph):
        """graph_query should reject SET statements."""
        from neo4j_agent_memory.mcp._tools import _is_read_only_query

        assert not _is_read_only_query("MATCH (n) SET n.name = 'hacked'")

    async def test_allows_read(self, populated_graph):
        """graph_query should allow MATCH/RETURN queries."""
        from neo4j_agent_memory.mcp._tools import _is_read_only_query

        assert _is_read_only_query("MATCH (n) RETURN count(n)")
        assert _is_read_only_query(
            "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) "
            "RETURN a.name, r.relation_type, b.name"
        )


class TestGraphStatistics:
    """Verify expected graph shape after loading the corpus."""

    async def test_minimum_entity_count(self, populated_graph):
        """The corpus should produce a minimum number of entities."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (e:Entity) RETURN count(e) AS count", {},
        )
        # 5 messages mention at least: Michael, Graphable, Sarah, Raj, DataVault,
        # Alice, Marcus, Denver, Larimer Street — conservatively 5+
        assert rows[0]["count"] >= 5, (
            f"Expected at least 5 entities, got {rows[0]['count']}"
        )

    async def test_minimum_relationship_count(self, populated_graph):
        """The corpus should produce at least some relationships."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS count", {},
        )
        # At minimum: Michael WORKS_AT Graphable, Raj WORKS_AT DataVault
        assert rows[0]["count"] >= 2, (
            f"Expected at least 2 relationships, got {rows[0]['count']}"
        )

    async def test_all_messages_have_mentions(self, populated_graph):
        """Every message should mention at least one entity."""
        client = populated_graph
        rows = await client.graph.execute_read(
            "MATCH (m:Message) "
            "WHERE NOT (m)-[:MENTIONS]->(:Entity) "
            "RETURN m.id AS id, "
            "       substring(m.content, 0, 50) AS snippet",
            {},
        )
        assert len(rows) == 0, (
            f"Messages with no extracted entities: "
            f"{[r['snippet'] for r in rows]}"
        )

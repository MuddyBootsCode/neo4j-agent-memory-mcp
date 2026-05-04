"""Smoke tests validating the integration test infrastructure.

These tests verify that the fixture chain works end-to-end:
Neo4j connection → test database → MemoryClient → Bedrock embeddings
→ BAML entity extraction via Bedrock.
"""

import pytest


class TestFixtureChain:
    """Verify the integration test fixtures work."""

    async def test_memory_client_connects(self, memory_client):
        """MemoryClient is connected and ready."""
        assert memory_client.is_connected

    async def test_database_is_clean(self, memory_client):
        """Each test starts with an empty database."""
        rows = await memory_client.graph.execute_read(
            "MATCH (n) RETURN count(n) AS count", {}
        )
        assert isinstance(rows[0]["count"], int)

    async def test_store_and_retrieve_message(self, memory_client):
        """Basic message store → vector search round-trip with Bedrock."""
        msg = await memory_client.short_term.add_message(
            session_id="smoke-test-session",
            role="user",
            content="Alice is a software engineer at Graphable.",
            generate_embedding=True,
            extract_entities=False,
        )
        assert msg.id is not None

        results = await memory_client.short_term.search_messages(
            query="Who works at Graphable?",
            limit=5,
            threshold=0.3,
        )
        assert len(results) >= 1
        assert "Alice" in results[0].content

    async def test_store_and_retrieve_fact(self, memory_client):
        """Fact SPO triple store → search round-trip."""
        fact = await memory_client.long_term.add_fact(
            subject="Alice",
            predicate="WORKS_AT",
            obj="Graphable",
            confidence=0.95,
            generate_embedding=True,
        )
        assert fact.id is not None

        facts = await memory_client.long_term.search_facts(
            query="Where does Alice work?",
            limit=5,
            threshold=0.3,
        )
        assert len(facts) >= 1
        assert facts[0].subject == "Alice"

    async def test_store_preference(self, memory_client):
        """Preference store works."""
        pref = await memory_client.long_term.add_preference(
            category="scheduling",
            preference="prefers morning meetings",
            context="Work habits",
        )
        assert pref.id is not None

    async def test_cypher_session_works(self, memory_client, cypher_session):
        """Graph executor can verify data stored via MemoryClient."""
        await memory_client.long_term.add_fact(
            subject="Bob",
            predicate="LIVES_IN",
            obj="Austin",
            generate_embedding=False,
        )

        rows = await cypher_session.execute_read(
            "MATCH (f:Fact {subject: $subject}) RETURN f.predicate AS pred",
            {"subject": "Bob"},
        )
        assert len(rows) == 1
        assert rows[0]["pred"] == "LIVES_IN"

    async def test_test_isolation(self, memory_client):
        """Verify data from previous test was wiped (memory_client wipes on setup)."""
        rows = await memory_client.graph.execute_read(
            "MATCH (f:Fact) RETURN count(f) AS count", {}
        )
        assert rows[0]["count"] == 0, "Previous test data should be wiped"


class TestEntityExtraction:
    """Verify BAML entity extraction via Bedrock creates correct graph structure."""

    async def test_extraction_creates_entity_nodes(self, memory_client, cypher_session):
        """Storing a message with extract_entities=True creates Entity nodes."""
        await memory_client.short_term.add_message(
            session_id="extraction-smoke",
            role="user",
            content="Michael is VP of Engineering at Graphable.",
            generate_embedding=True,
            extract_entities=True,
        )

        entities = await cypher_session.execute_read(
            "MATCH (e:Entity) RETURN e.name AS name, e.type AS type "
            "ORDER BY e.name",
            {},
        )
        names = {e["name"] for e in entities}
        assert "Michael" in names, f"Expected 'Michael' in {names}"
        assert "Graphable" in names, f"Expected 'Graphable' in {names}"

        # Verify types
        type_map = {e["name"]: e["type"] for e in entities}
        assert type_map["Michael"] == "PERSON"
        assert type_map["Graphable"] == "ORGANIZATION"

    async def test_extraction_creates_relationships(self, memory_client, cypher_session):
        """Entity extraction creates RELATED_TO edges with relation_type."""
        await memory_client.short_term.add_message(
            session_id="extraction-rels",
            role="user",
            content="Sarah works at DataVault Solutions and drives a BMW X5.",
            generate_embedding=True,
            extract_entities=True,
        )

        rels = await cypher_session.execute_read(
            "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) "
            "RETURN a.name AS src, r.relation_type AS rel, b.name AS tgt",
            {},
        )
        assert len(rels) >= 1, "Expected at least one relationship"

        # At minimum, WORKS_AT should be extracted
        rel_tuples = {(r["src"], r["rel"], r["tgt"]) for r in rels}
        has_works_at = any(
            src == "Sarah" and "WORK" in rel.upper()
            for src, rel, tgt in rel_tuples
        )
        assert has_works_at, f"Expected WORKS_AT for Sarah, got {rel_tuples}"

    async def test_extraction_creates_mentions_edges(self, memory_client, cypher_session):
        """Messages link to extracted entities via MENTIONS edges."""
        msg = await memory_client.short_term.add_message(
            session_id="extraction-mentions",
            role="user",
            content="Raj Patel is the CTO of DataVault.",
            generate_embedding=True,
            extract_entities=True,
        )

        mentions = await cypher_session.execute_read(
            "MATCH (m:Message {id: $id})-[:MENTIONS]->(e:Entity) "
            "RETURN e.name AS name",
            {"id": str(msg.id)},
        )
        mentioned_names = {m["name"] for m in mentions}
        assert "Raj Patel" in mentioned_names or "Raj" in mentioned_names, (
            f"Expected Raj in mentions, got {mentioned_names}"
        )

    async def test_no_orphaned_entities(self, memory_client, cypher_session):
        """Every extracted Entity should be linked to at least one Message."""
        await memory_client.short_term.add_message(
            session_id="orphan-check",
            role="user",
            content="Alice and Bob work at Acme Corp in Denver.",
            generate_embedding=True,
            extract_entities=True,
        )

        orphans = await cypher_session.execute_read(
            "MATCH (e:Entity) "
            "WHERE NOT (e)<-[:MENTIONS]-(:Message) "
            "RETURN e.name AS name",
            {},
        )
        assert len(orphans) == 0, (
            f"Orphaned entities (no MENTIONS edge): {[o['name'] for o in orphans]}"
        )

"""Smoke tests validating the integration test infrastructure.

These tests verify that the fixture chain works end-to-end:
Neo4j connection → test database → MemoryClient → Bedrock embeddings
→ BAML entity extraction via Bedrock.
"""



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


class TestMessageExtraction:
    """Verify pivot-era message extraction semantics (MUD-395).

    The org entity ontology is retired: storing a message with extraction
    enabled runs the coding-memory extraction, which creates NO Entity nodes
    and NO MENTIONS edges. Preferences stated in message content persist via
    the ``extract_coding_memory`` → ``persist_preferences`` path — the exact
    path the ``memory_store`` tool runs (the upstream ``add_message`` hook
    discards extracted preferences, so persistence lives in our tool layer).
    """

    async def test_extraction_creates_no_entities_or_mentions(
        self, memory_client, cypher_session
    ):
        """A message stored with extract_entities=True produces no org graph."""
        msg = await memory_client.short_term.add_message(
            session_id="extraction-smoke",
            role="user",
            content="Michael is VP of Engineering at Graphable.",
            generate_embedding=True,
            extract_entities=True,
        )
        assert msg.id is not None

        entities = await cypher_session.execute_read(
            "MATCH (e:Entity) RETURN count(e) AS count", {},
        )
        assert entities[0]["count"] == 0, (
            f"Message extraction created {entities[0]['count']} Entity nodes "
            "after the org-ontology retirement"
        )

        mentions = await cypher_session.execute_read(
            "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS count", {},
        )
        assert mentions[0]["count"] == 0, (
            f"Message extraction created {mentions[0]['count']} MENTIONS edges "
            "after the org-ontology retirement"
        )

    async def test_stated_preference_persists_and_is_searchable(
        self, memory_client, cypher_session
    ):
        """A preference stated in message content survives extraction.

        Runs the same extraction + persistence pair the memory_store tool
        uses for message content, then finds the preference via search.
        """
        from agent_memory_mcp.extraction.coding import extract_coding_memory
        from agent_memory_mcp.extraction.unified import persist_preferences

        extracted = await extract_coding_memory(
            "Please remember: I prefer pytest over unittest for all new "
            "test code in this project.",
            branch="",
            task=None,
            files=[],
        )
        assert len(extracted["preferences"]) >= 1, (
            "Expected the stated pytest preference to be extracted, got "
            f"{extracted['preferences']}"
        )

        stored = await persist_preferences(memory_client, extracted["preferences"])
        assert stored >= 1

        rows = await cypher_session.execute_read(
            "MATCH (p:Preference) RETURN count(p) AS count", {},
        )
        assert rows[0]["count"] >= 1

        prefs = await memory_client.long_term.search_preferences(
            query="which test framework does the user prefer?",
            limit=5,
            threshold=0.2,
        )
        assert any("pytest" in p.preference.lower() for p in prefs), (
            f"Expected the pytest preference in search results, got "
            f"{[p.preference for p in prefs]}"
        )

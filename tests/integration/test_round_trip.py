"""Round-trip retrieval tests with paraphrased queries.

Stores known content via MemoryClient, then searches with queries that
range from near-verbatim to heavily paraphrased. Asserts that the memory
system retrieves the right content under natural language variation.

This is the highest-signal test for whether the system actually works
for agents that phrase things differently from how they were stored.
"""

import pytest


# ── Test Corpus ──────────────────────────────────────────────────────
# Each entry is stored once at the start of the test class.
# Queries reference these by corpus_id to assert expected matches.

MESSAGES = [
    {
        "id": "msg_michael",
        "session_id": "roundtrip-session-1",
        "content": "Michael is VP of Engineering at Graphable. He prefers morning meetings and drives a Tesla Model 3.",
    },
    {
        "id": "msg_sarah",
        "session_id": "roundtrip-session-1",
        "content": "Sarah presented the Q3 revenue analysis at Monday's board meeting. The findings show 23% growth in enterprise accounts.",
    },
    {
        "id": "msg_raj",
        "session_id": "roundtrip-session-2",
        "content": "Raj Patel is the CTO of DataVault Solutions. He's vegetarian and prefers Hyatt hotels when traveling.",
    },
    {
        "id": "msg_alice",
        "session_id": "roundtrip-session-2",
        "content": "Alice completed the migration pipeline task. The embedding approach uses graph-based RAG which outperforms vector-only retrieval.",
    },
    {
        "id": "msg_denver",
        "session_id": "roundtrip-session-3",
        "content": "The Denver office is moving to a new location on Larimer Street next quarter. The lease was signed by Marcus.",
    },
]

FACTS = [
    {
        "id": "fact_michael_role",
        "subject": "Michael",
        "predicate": "WORKS_AT",
        "obj": "Graphable",
    },
    {
        "id": "fact_raj_role",
        "subject": "Raj Patel",
        "predicate": "WORKS_AT",
        "obj": "DataVault Solutions",
    },
    {
        "id": "fact_office_location",
        "subject": "Denver office",
        "predicate": "LOCATED_AT",
        "obj": "Larimer Street",
    },
]

PREFERENCES = [
    {
        "id": "pref_michael_schedule",
        "category": "scheduling",
        "preference": "Michael prefers morning meetings",
        "context": "work habits",
    },
    {
        "id": "pref_raj_food",
        "category": "food",
        "preference": "Raj is vegetarian",
        "context": "dietary restrictions",
    },
    {
        "id": "pref_raj_hotel",
        "category": "travel",
        "preference": "Raj prefers Hyatt hotels",
        "context": "business travel",
    },
]


# ── Query Scenarios ──────────────────────────────────────────────────
# Each query has:
#   - difficulty: easy (near-verbatim), medium (synonym), hard (role/contextual)
#   - search_type: which search method to use
#   - expected_ids: corpus IDs that should appear in results
#   - threshold: similarity threshold for the search

MESSAGE_QUERIES = [
    # Easy: near-verbatim
    {
        "name": "verbatim_michael",
        "query": "Michael is VP of Engineering at Graphable",
        "expected_ids": ["msg_michael"],
        "difficulty": "easy",
        "threshold": 0.5,
    },
    {
        "name": "verbatim_sarah_meeting",
        "query": "Q3 revenue analysis board meeting",
        "expected_ids": ["msg_sarah"],
        "difficulty": "easy",
        "threshold": 0.4,
    },
    # Medium: synonym substitution
    {
        "name": "synonym_michael_role",
        "query": "Who is the engineering leader at Graphable?",
        "expected_ids": ["msg_michael"],
        "difficulty": "medium",
        "threshold": 0.3,
    },
    {
        "name": "synonym_migration_task",
        "query": "What's the status of the data migration work?",
        "expected_ids": ["msg_alice"],
        "difficulty": "medium",
        "threshold": 0.3,
    },
    {
        "name": "synonym_office_relocation",
        "query": "Where is the office relocating to?",
        "expected_ids": ["msg_denver"],
        "difficulty": "medium",
        "threshold": 0.3,
    },
    # Hard: role-based / contextual
    {
        "name": "contextual_raj_company",
        "query": "Who runs the technology side at that data company?",
        "expected_ids": ["msg_raj"],
        "difficulty": "hard",
        "threshold": 0.3,
    },
    {
        "name": "contextual_enterprise_growth",
        "query": "How are enterprise sales performing this quarter?",
        "expected_ids": ["msg_sarah"],
        "difficulty": "hard",
        "threshold": 0.3,
    },
    {
        "name": "contextual_rag_approach",
        "query": "What retrieval technique did we find works best?",
        "expected_ids": ["msg_alice"],
        "difficulty": "hard",
        "threshold": 0.3,
    },
    # Negative: should NOT match any stored content
    {
        "name": "negative_weather",
        "query": "What's the weather forecast for Austin this weekend?",
        "expected_ids": [],
        "difficulty": "negative",
        "threshold": 0.6,
    },
]

FACT_QUERIES = [
    {
        "name": "fact_michael_employment",
        "query": "Where does Michael work?",
        "expected_subjects": ["Michael"],
        "threshold": 0.3,
    },
    {
        "name": "fact_raj_employment",
        "query": "What company does Raj work at?",
        "expected_subjects": ["Raj Patel"],
        "threshold": 0.3,
    },
    {
        "name": "fact_office_address",
        "query": "Where is the Denver office?",
        "expected_subjects": ["Denver office"],
        "threshold": 0.3,
    },
]

PREFERENCE_QUERIES = [
    {
        "name": "pref_meeting_time",
        "query": "When does Michael prefer to have meetings?",
        "expected_keywords": ["morning"],
        "threshold": 0.3,
    },
    {
        "name": "pref_dietary",
        "query": "Does Raj have any food restrictions?",
        "expected_keywords": ["vegetarian"],
        "threshold": 0.3,
    },
    {
        "name": "pref_hotel_brand",
        "query": "What hotel chain does Raj like?",
        "expected_keywords": ["hyatt"],
        "threshold": 0.3,
    },
]


# ── Test Implementation ──────────────────────────────────────────────


@pytest.fixture
async def loaded_memory(memory_client):
    """Store the entire test corpus and return the client + ID mapping."""
    stored_ids = {}

    # Store messages
    for msg in MESSAGES:
        result = await memory_client.short_term.add_message(
            session_id=msg["session_id"],
            role="user",
            content=msg["content"],
            generate_embedding=True,
            extract_entities=False,  # extraction tested separately
        )
        stored_ids[msg["id"]] = str(result.id)

    # Store facts
    for fact in FACTS:
        result = await memory_client.long_term.add_fact(
            subject=fact["subject"],
            predicate=fact["predicate"],
            obj=fact["obj"],
            confidence=1.0,
            generate_embedding=True,
        )
        stored_ids[fact["id"]] = str(result.id)

    # Store preferences
    for pref in PREFERENCES:
        result = await memory_client.long_term.add_preference(
            category=pref["category"],
            preference=pref["preference"],
            context=pref.get("context"),
        )
        stored_ids[pref["id"]] = str(result.id)

    return memory_client, stored_ids


class TestMessageRetrieval:
    """Test message retrieval with paraphrased queries."""

    @pytest.mark.parametrize(
        "scenario",
        [s for s in MESSAGE_QUERIES if s["difficulty"] == "easy"],
        ids=lambda s: s["name"],
    )
    async def test_easy_queries(self, loaded_memory, scenario):
        """Near-verbatim queries should reliably retrieve the right message."""
        client, stored_ids = loaded_memory
        results = await client.short_term.search_messages(
            query=scenario["query"],
            limit=5,
            threshold=scenario["threshold"],
        )

        if scenario["expected_ids"]:
            assert len(results) >= 1, (
                f"Query '{scenario['query']}' returned no results"
            )
            result_contents = [r.content for r in results]
            # The expected message content should appear in the top results
            for expected_id in scenario["expected_ids"]:
                expected_content = next(
                    m["content"] for m in MESSAGES if m["id"] == expected_id
                )
                found = any(expected_content == c for c in result_contents)
                assert found, (
                    f"Expected content for {expected_id} not in results. "
                    f"Got: {[c[:60] for c in result_contents]}"
                )

    @pytest.mark.parametrize(
        "scenario",
        [s for s in MESSAGE_QUERIES if s["difficulty"] == "medium"],
        ids=lambda s: s["name"],
    )
    async def test_medium_queries(self, loaded_memory, scenario):
        """Synonym/rephrase queries should retrieve the right message."""
        client, stored_ids = loaded_memory
        results = await client.short_term.search_messages(
            query=scenario["query"],
            limit=5,
            threshold=scenario["threshold"],
        )

        assert len(results) >= 1, (
            f"Query '{scenario['query']}' returned no results"
        )
        result_contents = [r.content for r in results]
        for expected_id in scenario["expected_ids"]:
            expected_content = next(
                m["content"] for m in MESSAGES if m["id"] == expected_id
            )
            found = any(expected_content == c for c in result_contents)
            assert found, (
                f"Expected content for {expected_id} not in top-5 results. "
                f"Got: {[c[:60] for c in result_contents]}"
            )

    @pytest.mark.parametrize(
        "scenario",
        [s for s in MESSAGE_QUERIES if s["difficulty"] == "hard"],
        ids=lambda s: s["name"],
    )
    async def test_hard_queries(self, loaded_memory, scenario):
        """Contextual/role-based queries — the real test of semantic search."""
        client, stored_ids = loaded_memory
        results = await client.short_term.search_messages(
            query=scenario["query"],
            limit=5,
            threshold=scenario["threshold"],
        )

        assert len(results) >= 1, (
            f"Query '{scenario['query']}' returned no results"
        )
        # For hard queries, check that the expected message appears anywhere
        # in the top-5 (not necessarily rank 1)
        result_contents = [r.content for r in results]
        for expected_id in scenario["expected_ids"]:
            expected_content = next(
                m["content"] for m in MESSAGES if m["id"] == expected_id
            )
            found = any(expected_content == c for c in result_contents)
            assert found, (
                f"[{scenario['difficulty']}] Expected {expected_id} not in top-5. "
                f"Query: '{scenario['query']}'\n"
                f"Got: {[c[:60] for c in result_contents]}"
            )

    @pytest.mark.parametrize(
        "scenario",
        [s for s in MESSAGE_QUERIES if s["difficulty"] == "negative"],
        ids=lambda s: s["name"],
    )
    async def test_negative_queries(self, loaded_memory, scenario):
        """Queries about unstored topics should return no results above threshold."""
        client, stored_ids = loaded_memory
        results = await client.short_term.search_messages(
            query=scenario["query"],
            limit=5,
            threshold=scenario["threshold"],
        )
        assert len(results) == 0, (
            f"Negative query '{scenario['query']}' should return nothing "
            f"at threshold={scenario['threshold']}, "
            f"got {len(results)}: {[r.content[:40] for r in results]}"
        )


class TestFactRetrieval:
    """Test fact SPO triple retrieval with semantic queries."""

    @pytest.mark.parametrize(
        "scenario",
        FACT_QUERIES,
        ids=lambda s: s["name"],
    )
    async def test_fact_search(self, loaded_memory, scenario):
        """Semantic queries should retrieve matching facts."""
        client, stored_ids = loaded_memory
        results = await client.long_term.search_facts(
            query=scenario["query"],
            limit=5,
            threshold=scenario["threshold"],
        )

        assert len(results) >= 1, (
            f"Fact query '{scenario['query']}' returned no results"
        )
        result_subjects = [f.subject for f in results]
        for expected_subj in scenario["expected_subjects"]:
            assert expected_subj in result_subjects, (
                f"Expected subject '{expected_subj}' not in results. "
                f"Got: {result_subjects}"
            )


class TestPreferenceRetrieval:
    """Test preference retrieval with natural language queries."""

    @pytest.mark.parametrize(
        "scenario",
        PREFERENCE_QUERIES,
        ids=lambda s: s["name"],
    )
    async def test_preference_search(self, loaded_memory, scenario):
        """Natural language queries should find matching preferences."""
        client, stored_ids = loaded_memory
        results = await client.long_term.search_preferences(
            query=scenario["query"],
            limit=5,
            threshold=scenario["threshold"],
        )

        assert len(results) >= 1, (
            f"Preference query '{scenario['query']}' returned no results"
        )
        # Check that expected keywords appear in any returned preference
        all_prefs_text = " ".join(r.preference.lower() for r in results)
        for keyword in scenario["expected_keywords"]:
            assert keyword.lower() in all_prefs_text, (
                f"Expected keyword '{keyword}' not in any preference. "
                f"Got: {[r.preference for r in results]}"
            )


class TestCrossTypeRetrieval:
    """Test that related content is findable across memory types."""

    async def test_person_findable_via_messages_and_facts(self, loaded_memory):
        """A person stored in both a message and a fact should be findable via both."""
        client, _ = loaded_memory

        # Search messages for Michael
        messages = await client.short_term.search_messages(
            query="Michael at Graphable",
            limit=3,
            threshold=0.3,
        )
        assert len(messages) >= 1
        assert any("Michael" in m.content for m in messages)

        # Search facts for Michael
        facts = await client.long_term.search_facts(
            query="Michael employment",
            limit=3,
            threshold=0.3,
        )
        assert len(facts) >= 1
        assert any(f.subject == "Michael" for f in facts)

    async def test_preference_findable_by_person_context(self, loaded_memory):
        """Preferences should be findable by querying about the person."""
        client, _ = loaded_memory

        results = await client.long_term.search_preferences(
            query="Raj travel preferences",
            limit=5,
            threshold=0.3,
        )
        assert len(results) >= 1
        pref_texts = [r.preference.lower() for r in results]
        assert any("hyatt" in p for p in pref_texts), (
            f"Expected Hyatt preference, got: {pref_texts}"
        )


class TestRetrievalQuality:
    """Aggregate retrieval quality metrics across all query types."""

    async def test_overall_hit_rate(self, loaded_memory):
        """At least 70% of non-negative queries should return expected results."""
        client, stored_ids = loaded_memory
        hits = 0
        total = 0

        # Test message queries
        for scenario in MESSAGE_QUERIES:
            if scenario["difficulty"] == "negative":
                continue
            total += 1
            results = await client.short_term.search_messages(
                query=scenario["query"],
                limit=5,
                threshold=scenario["threshold"],
            )
            result_contents = {r.content for r in results}
            expected_contents = {
                next(m["content"] for m in MESSAGES if m["id"] == eid)
                for eid in scenario["expected_ids"]
            }
            if expected_contents & result_contents:
                hits += 1

        # Test fact queries
        for scenario in FACT_QUERIES:
            total += 1
            results = await client.long_term.search_facts(
                query=scenario["query"],
                limit=5,
                threshold=scenario["threshold"],
            )
            result_subjects = {f.subject for f in results}
            if set(scenario["expected_subjects"]) & result_subjects:
                hits += 1

        # Test preference queries
        for scenario in PREFERENCE_QUERIES:
            total += 1
            results = await client.long_term.search_preferences(
                query=scenario["query"],
                limit=5,
                threshold=scenario["threshold"],
            )
            all_text = " ".join(r.preference.lower() for r in results)
            if all(kw.lower() in all_text for kw in scenario["expected_keywords"]):
                hits += 1

        hit_rate = hits / total if total else 0
        assert hit_rate >= 0.70, (
            f"Overall hit rate {hit_rate:.0%} ({hits}/{total}) "
            f"below 70% threshold"
        )

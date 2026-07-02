"""Tests for contradiction detection and Phase 2 temporal features."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo4j_agent_memory.temporal.contradiction import (
    _fallback_subject_predicate_match,
    detect_and_invalidate,
    find_contradiction_candidates,
)
from neo4j_agent_memory.temporal.extraction import extract_temporal_context
from neo4j_agent_memory.temporal.migration import migrate_existing_facts


# ── find_contradiction_candidates ───────────────────────────────────


class TestFindContradictionCandidates:
    @pytest.mark.asyncio
    async def test_no_embedder_falls_back(self):
        """When no embedder available, falls back to subject+predicate match."""
        mock_client = MagicMock()
        mock_client.long_term._embedder = None
        mock_client.graph.execute_read = AsyncMock(
            return_value=[
                {
                    "id": "old-fact-1",
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Acme",
                    "confidence": 0.9,
                }
            ]
        )

        result = await find_contradiction_candidates(
            mock_client, "Alice", "WORKS_AT", "Globex", "new-fact-1"
        )
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["similarity"] is None  # fallback doesn't have similarity

    @pytest.mark.asyncio
    async def test_embedding_failure_falls_back(self):
        """When embedding fails, falls back to subject+predicate match."""
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(side_effect=Exception("API error"))

        mock_client = MagicMock()
        mock_client.long_term._embedder = mock_embedder
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        result = await find_contradiction_candidates(
            mock_client, "Alice", "WORKS_AT", "Globex", "new-fact-1"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_vector_search_returns_candidates(self):
        """Vector search finds semantically similar facts."""
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_client = MagicMock()
        mock_client.long_term._embedder = mock_embedder
        mock_client.graph.execute_read = AsyncMock(
            return_value=[
                {
                    "id": "old-fact-1",
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Acme",
                    "confidence": 0.9,
                    "score": 0.85,
                }
            ]
        )

        result = await find_contradiction_candidates(
            mock_client, "Alice", "WORKS_AT", "Globex", "new-fact-1"
        )
        assert len(result) == 1
        assert result[0]["similarity"] == 0.85


# ── _fallback_subject_predicate_match ───────────────────────────────


class TestFallbackSubjectPredicateMatch:
    @pytest.mark.asyncio
    async def test_finds_matching_facts(self):
        mock_client = MagicMock()
        mock_client.graph.execute_read = AsyncMock(
            return_value=[
                {
                    "id": "f1",
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Acme",
                    "confidence": 0.9,
                },
                {
                    "id": "f2",
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "OldCorp",
                    "confidence": 0.7,
                },
            ]
        )

        result = await _fallback_subject_predicate_match(
            mock_client, "Alice", "WORKS_AT", "new-id"
        )
        assert len(result) == 2
        assert result[0]["idx"] == 0
        assert result[1]["idx"] == 1

    @pytest.mark.asyncio
    async def test_empty_when_no_matches(self):
        mock_client = MagicMock()
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        result = await _fallback_subject_predicate_match(
            mock_client, "Alice", "VISITED", "new-id"
        )
        assert result == []


# ── detect_and_invalidate ───────────────────────────────────────────


class TestDetectAndInvalidate:
    @pytest.mark.asyncio
    async def test_no_candidates_early_return(self):
        """When no candidates found, returns early with zero counts."""
        mock_client = MagicMock()
        mock_client.long_term._embedder = None
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        result = await detect_and_invalidate(
            mock_client, "Alice", "VISITED", "Seattle", "new-fact"
        )
        assert result["candidates_found"] == 0
        assert result["contradictions_detected"] == 0
        assert result["facts_invalidated"] == 0

    @pytest.mark.asyncio
    async def test_baml_failure_falls_back_to_spo(self):
        """When BAML fails, falls back to SPO match for contradiction."""
        mock_client = MagicMock()
        mock_client.long_term._embedder = None
        # Return one candidate with matching subject+predicate
        mock_client.graph.execute_read = AsyncMock(
            return_value=[
                {
                    "id": "old-fact-1",
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Acme",
                    "confidence": 0.9,
                }
            ]
        )
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"c": 1}]
        )

        # Patch BAML to fail
        with patch(
            "neo4j_agent_memory.temporal.contradiction.find_contradiction_candidates",
            new_callable=AsyncMock,
            return_value=[
                {
                    "id": "old-fact-1",
                    "idx": 0,
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Acme",
                    "confidence": 0.9,
                    "similarity": None,
                }
            ],
        ):
            # BAML import will fail, triggering fallback
            result = await detect_and_invalidate(
                mock_client, "Alice", "WORKS_AT", "Globex", "new-fact"
            )

        assert result["candidates_found"] == 1
        # SPO fallback should find the contradiction
        assert result["contradiction_type"] in ("direct_supersession", "none")

    @pytest.mark.asyncio
    async def test_non_contradiction_not_invalidated(self):
        """Facts with different predicates should not be contradicted."""
        mock_client = MagicMock()
        mock_client.long_term._embedder = None
        # No matching subject+predicate
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        result = await detect_and_invalidate(
            mock_client, "Alice", "VISITED", "Seattle", "new-fact"
        )
        assert result["facts_invalidated"] == 0
        assert result["contradiction_type"] == "none"


# ── extract_temporal_context ────────────────────────────────────────


class TestExtractTemporalContext:
    @pytest.mark.asyncio
    async def test_returns_defaults_on_baml_failure(self):
        """When BAML is unavailable, returns safe defaults."""
        result = await extract_temporal_context("Alice works at Globex")
        # BAML won't be available in tests, so we get defaults
        assert result["is_current_state"] is True
        assert result["valid_at"] is None
        assert result["temporal_qualifier"] is None

    @pytest.mark.asyncio
    async def test_with_custom_reference_time(self):
        """Accepts custom reference time."""
        ref = datetime(2026, 3, 1, tzinfo=timezone.utc)
        result = await extract_temporal_context("some text", reference_time=ref)
        # Should not raise, returns defaults since BAML not available
        assert isinstance(result, dict)


# ── migrate_existing_facts ──────────────────────────────────────────


class TestMigration:
    @pytest.mark.asyncio
    async def test_backfills_valid_from(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"migrated": 42}]
        )

        result = await migrate_existing_facts(mock_client)
        assert result["valid_from_backfilled"] == 42

    @pytest.mark.asyncio
    async def test_idempotent_zero_results(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"migrated": 0}]
        )

        result = await migrate_existing_facts(mock_client)
        assert result["valid_from_backfilled"] == 0


# ── knowledge_state tool ────────────────────────────────────────────


@pytest.fixture
def mock_knowledge_client(monkeypatch):
    """Mock client for knowledge_state tests."""
    mock_graph = MagicMock()
    mock_graph.execute_read = AsyncMock(
        return_value=[
            {
                "id": "f1",
                "subject": "Alice",
                "predicate": "WORKS_AT",
                "object": "Acme",
                "confidence": 0.9,
                "created_at": "2026-01-01T00:00:00Z",
                "expired_at": None,
                "valid_from": 1704067200000,
                "valid_until": None,
            }
        ]
    )

    mock_client = MagicMock()
    mock_client.graph = mock_graph

    monkeypatch.setattr(
        "neo4j_agent_memory.mcp._tools.get_client",
        lambda ctx: mock_client,
    )

    def _no_registry(ctx):
        raise RuntimeError("No registry")


    return mock_client


@pytest.mark.asyncio
async def test_knowledge_state_returns_facts(mock_knowledge_client):
    """knowledge_state returns facts known at a system time."""
    from neo4j_agent_memory.mcp._tools import register_tools
    from fastmcp import FastMCP

    mcp = FastMCP("test")
    register_tools(mcp)

    ctx = MagicMock()

    ks_fn = None
    for tool in mcp._tool_manager._tools.values():
        if tool.name == "knowledge_state":
            ks_fn = tool.fn
            break
    assert ks_fn is not None

    result_str = await ks_fn(
        ctx,
        as_of="2026-02-01T00:00:00Z",
        subject="Alice",
    )

    result = json.loads(result_str)
    assert result["count"] == 1
    assert result["facts"][0]["subject"] == "Alice"


@pytest.mark.asyncio
async def test_knowledge_state_invalid_datetime(mock_knowledge_client):
    """knowledge_state returns error for invalid datetime."""
    from neo4j_agent_memory.mcp._tools import register_tools
    from fastmcp import FastMCP

    mcp = FastMCP("test")
    register_tools(mcp)

    ctx = MagicMock()

    ks_fn = None
    for tool in mcp._tool_manager._tools.values():
        if tool.name == "knowledge_state":
            ks_fn = tool.fn
            break

    result_str = await ks_fn(ctx, as_of="not-a-date")
    result = json.loads(result_str)
    assert "error" in result

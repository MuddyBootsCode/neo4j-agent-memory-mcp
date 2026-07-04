"""Tests for temporal lifecycle module and temporal MCP tools."""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_mcp.temporal.lifecycle import (
    get_fact_evolution,
    parse_iso_datetime,
    supersede_fact_by_id,
    supersede_matching_facts,
    temporal_fact_query,
)


# ── parse_iso_datetime ──────────────────────────────────────────────


class TestParseIsoDatetime:
    def test_valid_iso_with_tz(self):
        result = parse_iso_datetime("2026-03-09T12:00:00+00:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.tzinfo is not None

    def test_valid_iso_without_tz(self):
        result = parse_iso_datetime("2026-03-09T12:00:00")
        assert result is not None
        assert result.tzinfo == timezone.utc  # auto-set to UTC

    def test_valid_iso_z_suffix(self):
        result = parse_iso_datetime("2026-03-09T12:00:00Z")
        assert result is not None

    def test_none_input(self):
        assert parse_iso_datetime(None) is None

    def test_empty_string(self):
        assert parse_iso_datetime("") is None

    def test_garbage_input(self):
        assert parse_iso_datetime("not-a-date") is None

    def test_date_only(self):
        result = parse_iso_datetime("2026-03-09")
        assert result is not None
        assert result.year == 2026


# ── supersede_matching_facts ────────────────────────────────────────


class TestSupersedeMatchingFacts:
    @pytest.mark.asyncio
    async def test_supersedes_matching_facts(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"superseded_count": 2}]
        )

        count = await supersede_matching_facts(
            mock_client, "Alice", "WORKS_AT", "new-fact-123"
        )
        assert count == 2

        # Verify the Cypher was called with normalized match params (R14)
        call_args = mock_client.graph.execute_write.call_args
        params = call_args[0][1]
        assert params["subject_norm"] == "alice"
        assert params["predicate_norm"] == "works_at"
        assert params["new_fact_id"] == "new-fact-123"

    @pytest.mark.asyncio
    async def test_no_matches(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"superseded_count": 0}]
        )

        count = await supersede_matching_facts(
            mock_client, "Alice", "WORKS_AT", "new-fact-123"
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_result(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(return_value=[])

        count = await supersede_matching_facts(
            mock_client, "Alice", "WORKS_AT", "new-fact-123"
        )
        assert count == 0


# ── supersede_fact_by_id ────────────────────────────────────────────


class TestSupersedeFactById:
    @pytest.mark.asyncio
    async def test_supersedes_single_fact(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"superseded_count": 1}]
        )

        count = await supersede_fact_by_id(mock_client, "old-fact-1", "new-fact-2")
        assert count == 1

    @pytest.mark.asyncio
    async def test_fact_not_found(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"superseded_count": 0}]
        )

        count = await supersede_fact_by_id(mock_client, "nonexistent", "new-fact-2")
        assert count == 0


# ── get_fact_evolution ──────────────────────────────────────────────


class TestGetFactEvolution:
    @pytest.mark.asyncio
    async def test_returns_evolution_chain(self):
        mock_client = MagicMock()
        mock_client.graph.execute_read = AsyncMock(
            return_value=[
                {
                    "id": "fact-1",
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Acme",
                    "confidence": 0.95,
                    "created_at": "2026-01-01T00:00:00Z",
                    "valid_from": "2026-01-01T00:00:00Z",
                    "valid_until": 1709827200000,  # epoch millis
                    "superseded_by": "fact-2",
                },
                {
                    "id": "fact-2",
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Globex",
                    "confidence": 1.0,
                    "created_at": "2026-03-07T00:00:00Z",
                    "valid_from": "2026-03-07T00:00:00Z",
                    "valid_until": None,
                    "superseded_by": None,
                },
            ]
        )

        result = await get_fact_evolution(mock_client, "Alice", predicate="WORKS_AT")
        assert len(result) == 2
        assert result[0]["is_current"] is False
        assert result[0]["superseded_by"] == "fact-2"
        assert result[1]["is_current"] is True
        assert result[1]["superseded_by"] is None

    @pytest.mark.asyncio
    async def test_with_predicate_filter(self):
        mock_client = MagicMock()
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        await get_fact_evolution(mock_client, "Alice", predicate="WORKS_AT")

        call_args = mock_client.graph.execute_read.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        assert "f.predicate = $predicate" in query
        assert params["predicate"] == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_without_predicate_filter(self):
        mock_client = MagicMock()
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        await get_fact_evolution(mock_client, "Alice")

        call_args = mock_client.graph.execute_read.call_args
        query = call_args[0][0]
        assert "f.predicate = $predicate" not in query


# ── temporal_fact_query ─────────────────────────────────────────────


class TestTemporalFactQuery:
    @pytest.mark.asyncio
    async def test_basic_point_in_time_query(self):
        mock_client = MagicMock()
        mock_client.graph.execute_read = AsyncMock(
            return_value=[
                {
                    "id": "fact-1",
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Acme",
                    "confidence": 0.95,
                    "created_at": "2026-01-01T00:00:00Z",
                    "valid_from": 1704067200000,
                    "valid_until": None,
                }
            ]
        )

        pit = datetime(2026, 2, 1, tzinfo=timezone.utc)
        result = await temporal_fact_query(mock_client, pit)
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"

    @pytest.mark.asyncio
    async def test_with_subject_and_predicate_filter(self):
        mock_client = MagicMock()
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        pit = datetime(2026, 2, 1, tzinfo=timezone.utc)
        await temporal_fact_query(
            mock_client, pit, subject="Alice", predicate="WORKS_AT"
        )

        call_args = mock_client.graph.execute_read.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        assert "f.subject = $subject" in query
        assert "f.predicate = $predicate" in query
        assert params["subject"] == "Alice"
        assert params["predicate"] == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_pit_epoch_conversion(self):
        mock_client = MagicMock()
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        pit = datetime(2026, 3, 1, tzinfo=timezone.utc)
        await temporal_fact_query(mock_client, pit)

        call_args = mock_client.graph.execute_read.call_args
        params = call_args[0][1]
        expected_epoch = int(pit.timestamp() * 1000)
        assert params["pit"] == expected_epoch


# ── memory_store temporal integration ───────────────────────────────


@pytest.fixture
def mock_fact_client(monkeypatch):
    """Mock client for memory_store fact tests (atomic store path)."""
    fact_id = str(uuid.uuid4())

    mock_long_term = MagicMock()
    mock_long_term._embedder = MagicMock()
    mock_long_term._embedder.embed = AsyncMock(return_value=[0.1] * 8)

    march_epoch = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
    mock_graph = MagicMock()
    mock_graph.execute_write = AsyncMock(
        return_value=[
            {
                "fact_id": fact_id,
                "refreshed": False,
                "superseded_count": 1,
                "valid_from": march_epoch,
            }
        ]
    )

    mock_client = MagicMock()
    mock_client.long_term = mock_long_term
    mock_client.graph = mock_graph

    monkeypatch.setattr(
        "agent_memory_mcp.mcp._tools.get_client",
        lambda ctx: mock_client,
    )

    return mock_client, fact_id


@pytest.mark.asyncio
async def test_memory_store_fact_with_temporal_params(mock_fact_client, monkeypatch):
    """memory_store runs one atomic create+supersede write with epoch params."""
    from agent_memory_mcp.mcp._tools import register_tools
    from fastmcp import FastMCP

    monkeypatch.setenv("NAM_CONTRADICTION_DETECTION", "false")

    mcp = FastMCP("test")
    register_tools(mcp)

    mock_client, fact_id = mock_fact_client
    ctx = MagicMock()

    # Get the memory_store function
    store_fn = None
    for tool in mcp._tool_manager._tools.values():
        if tool.name == "memory_store":
            store_fn = tool.fn
            break
    assert store_fn is not None

    result_str = await store_fn(
        ctx,
        memory_type="fact",
        content="Alice works at Globex",
        subject="Alice",
        predicate="WORKS_AT",
        object_value="Globex",
        valid_from="2026-03-01T00:00:00Z",
        confidence=0.9,
    )

    result = json.loads(result_str)
    assert result["stored"] is True
    assert result["type"] == "fact"
    assert result["id"] == fact_id
    assert result["confidence"] == 0.9
    # R1: valid_from is normalized to epoch millis (not an ISO string).
    assert isinstance(result["valid_from"], int) and result["valid_from"] > 0
    assert result["superseded_facts"] == 1  # SPO supersession ran
    assert result["refreshed"] is False

    # R5: exactly ONE write — create + supersession are a single atomic
    # Cypher statement, with epoch-millis temporal params.
    assert mock_client.graph.execute_write.await_count == 1
    params = mock_client.graph.execute_write.call_args[0][1]
    assert params["confidence"] == 0.9
    march_epoch = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert params["valid_from"] == march_epoch
    assert params["valid_until"] is None
    assert params["can_supersede"] is True
    assert params["subject_norm"] == "alice"


# ── memory_search temporal filtering ────────────────────────────────


@pytest.fixture
def mock_search_client(monkeypatch):
    """Mock client for memory_search tests with temporal facts."""
    mock_active_fact = MagicMock()
    mock_active_fact.id = uuid.uuid4()
    mock_active_fact.subject = "Alice"
    mock_active_fact.predicate = "WORKS_AT"
    mock_active_fact.object = "Globex"
    mock_active_fact.confidence = 1.0
    mock_active_fact.valid_from = None
    mock_active_fact.valid_until = None
    mock_active_fact.metadata = {"similarity": 0.95}
    mock_active_fact.superseded_by = None

    mock_expired_fact = MagicMock()
    mock_expired_fact.id = uuid.uuid4()
    mock_expired_fact.subject = "Alice"
    mock_expired_fact.predicate = "WORKS_AT"
    mock_expired_fact.object = "Acme"
    mock_expired_fact.confidence = 0.9
    mock_expired_fact.valid_from = None
    mock_expired_fact.valid_until = 1709827200000  # epoch millis in past
    mock_expired_fact.metadata = {"similarity": 0.90}
    mock_expired_fact.superseded_by = None

    mock_long_term = MagicMock()
    mock_long_term.search_facts = AsyncMock(
        return_value=[mock_active_fact, mock_expired_fact]
    )
    mock_long_term.search_entities = AsyncMock(return_value=[])
    mock_long_term.search_preferences = AsyncMock(return_value=[])

    mock_short_term = MagicMock()
    mock_short_term.search_messages = AsyncMock(return_value=[])

    mock_reasoning = MagicMock()
    mock_reasoning.get_similar_traces = AsyncMock(return_value=[])

    # Temporal fields are read straight from the graph (the upstream Fact
    # model can't round-trip epoch ints), so serve them via execute_read.
    mock_graph = MagicMock()
    mock_graph.execute_read = AsyncMock(
        return_value=[
            {
                "id": str(mock_active_fact.id),
                "valid_from": None,
                "valid_until": None,
                "superseded_by": None,
            },
            {
                "id": str(mock_expired_fact.id),
                "valid_from": None,
                "valid_until": 1709827200000,  # epoch millis in past
                "superseded_by": None,
            },
        ]
    )

    mock_client = MagicMock()
    mock_client.long_term = mock_long_term
    mock_client.short_term = mock_short_term
    mock_client.reasoning = mock_reasoning
    mock_client.graph = mock_graph

    monkeypatch.setattr(
        "agent_memory_mcp.mcp._tools.get_client",
        lambda ctx: mock_client,
    )

    return mock_client


@pytest.mark.asyncio
async def test_memory_search_annotates_temporal_status(mock_search_client):
    """memory_search annotates facts with temporal_status field."""
    from agent_memory_mcp.mcp._tools import register_tools
    from fastmcp import FastMCP

    mcp = FastMCP("test")
    register_tools(mcp)

    ctx = MagicMock()

    search_fn = None
    for tool in mcp._tool_manager._tools.values():
        if tool.name == "memory_search":
            search_fn = tool.fn
            break
    assert search_fn is not None

    result_str = await search_fn(
        ctx,
        query="Alice works",
        memory_types=["facts"],
    )

    result = json.loads(result_str)
    facts = result["results"]["facts"]
    assert len(facts) == 2

    # Active fact should be first (sorted)
    assert facts[0]["temporal_status"] == "active"
    assert facts[0]["object"] == "Globex"

    # Expired fact second
    assert facts[1]["temporal_status"] == "expired"
    assert facts[1]["object"] == "Acme"


@pytest.mark.asyncio
async def test_memory_search_exclude_expired(mock_search_client):
    """memory_search with include_expired=False filters out expired facts."""
    from agent_memory_mcp.mcp._tools import register_tools
    from fastmcp import FastMCP

    mcp = FastMCP("test")
    register_tools(mcp)

    ctx = MagicMock()

    search_fn = None
    for tool in mcp._tool_manager._tools.values():
        if tool.name == "memory_search":
            search_fn = tool.fn
            break

    result_str = await search_fn(
        ctx,
        query="Alice works",
        memory_types=["facts"],
        include_expired=False,
    )

    result = json.loads(result_str)
    facts = result["results"]["facts"]
    assert len(facts) == 1
    assert facts[0]["temporal_status"] == "active"


# ── database_init temporal indexes ──────────────────────────────────


class TestTemporalIndexes:
    @pytest.mark.asyncio
    async def test_ensure_indexes(self):
        from agent_memory_mcp.mcp._database_init import ensure_indexes

        mock_graph = MagicMock()
        mock_graph.execute_write = AsyncMock()

        await ensure_indexes(mock_graph)

        # One write per temporal index statement (Fact valid_from/until/etc.)
        assert mock_graph.execute_write.await_count >= 6
        stmts = " ".join(
            call.args[0] for call in mock_graph.execute_write.await_args_list
        )
        assert "f.valid_from" in stmts
        assert "f.subject, f.predicate" in stmts


# ── R16: portable Z-suffix parsing ──────────────────────────────────


class TestParseIsoZSuffixPortable:
    def test_uppercase_z_suffix(self):
        result = parse_iso_datetime("2026-03-09T12:00:00Z")
        assert result == datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)

    def test_lowercase_z_suffix(self):
        # datetime.fromisoformat rejects a lowercase 'z' (and rejects 'Z'
        # entirely on Python 3.10) — parse must normalize it portably.
        result = parse_iso_datetime("2026-03-09T12:00:00z")
        assert result == datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)

    def test_z_suffix_with_millis(self):
        result = parse_iso_datetime("2026-03-09T12:00:00.500Z")
        assert result == datetime(
            2026, 3, 9, 12, 0, 0, 500000, tzinfo=timezone.utc
        )


# ── R5/R6/R7: store_fact_atomic parameter gating ────────────────────


def _atomic_client(row=None):
    mock_client = MagicMock()
    mock_client.long_term._embedder = MagicMock()
    mock_client.long_term._embedder.embed = AsyncMock(return_value=[0.1] * 8)
    mock_client.graph.execute_write = AsyncMock(
        return_value=[
            row
            or {
                "fact_id": "new-id",
                "refreshed": False,
                "superseded_count": 0,
                "valid_from": None,
            }
        ]
    )
    return mock_client


class TestStoreFactAtomic:
    @pytest.mark.asyncio
    async def test_single_write_and_normalized_params(self):
        """R5/R14: exactly one write; normalized match params passed."""
        from agent_memory_mcp.temporal.lifecycle import store_fact_atomic

        client = _atomic_client()
        outcome = await store_fact_atomic(
            client,
            subject="  Alice ",
            predicate="WORKS_AT",
            obj="Globex",
            confidence=0.9,
        )

        assert client.graph.execute_write.await_count == 1
        params = client.graph.execute_write.call_args[0][1]
        assert params["subject_norm"] == "alice"
        assert params["predicate_norm"] == "works_at"
        assert params["object_norm"] == "globex"
        assert params["can_supersede"] is True
        assert outcome["fact_id"] == "new-id"
        assert outcome["refreshed"] is False

    @pytest.mark.asyncio
    async def test_bounded_valid_until_disables_supersession(self):
        """R6: a fact with a bounded valid_until must not supersede."""
        from agent_memory_mcp.temporal.lifecycle import store_fact_atomic

        client = _atomic_client()
        await store_fact_atomic(
            client,
            subject="Alice",
            predicate="WORKS_AT",
            obj="Acme",
            valid_until_epoch=1_600_000_000_000,
        )
        params = client.graph.execute_write.call_args[0][1]
        assert params["can_supersede"] is False
        assert params["allow_refresh"] is False

    @pytest.mark.asyncio
    async def test_non_current_state_disables_supersession(self):
        """R6: is_current_state=False (historical phrasing) must not supersede."""
        from agent_memory_mcp.temporal.lifecycle import store_fact_atomic

        client = _atomic_client()
        await store_fact_atomic(
            client,
            subject="Alice",
            predicate="WORKS_AT",
            obj="Acme",
            is_current_state=False,
        )
        params = client.graph.execute_write.call_args[0][1]
        assert params["can_supersede"] is False

    @pytest.mark.asyncio
    async def test_embedding_failure_stores_without_embedding(self):
        from agent_memory_mcp.temporal.lifecycle import store_fact_atomic

        client = _atomic_client()
        client.long_term._embedder.embed = AsyncMock(
            side_effect=RuntimeError("bedrock down")
        )
        outcome = await store_fact_atomic(
            client, subject="Alice", predicate="WORKS_AT", obj="Acme"
        )
        params = client.graph.execute_write.call_args[0][1]
        assert params["embedding"] is None
        assert outcome["fact_id"] == "new-id"

    @pytest.mark.asyncio
    async def test_explicit_supersedes_id_passed_through(self):
        from agent_memory_mcp.temporal.lifecycle import store_fact_atomic

        client = _atomic_client()
        await store_fact_atomic(
            client,
            subject="Alice",
            predicate="WORKS_AT",
            obj="Acme",
            supersedes_id="old-42",
        )
        params = client.graph.execute_write.call_args[0][1]
        assert params["supersedes_id"] == "old-42"
        # Explicit supersession must not silently turn into a refresh.
        assert params["allow_refresh"] is False


# ── R8/R14: legacy supersede helpers set expired_at + normalize ─────


class TestSupersedeSetsExpiredAtAndNormalizes:
    @pytest.mark.asyncio
    async def test_supersede_matching_facts_sets_expired_at(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"superseded_count": 1}]
        )
        await supersede_matching_facts(mock_client, "Alice", "WORKS_AT", "new-1")
        query = mock_client.graph.execute_write.call_args[0][0]
        assert "expired_at" in query, "R8: SPO supersession must set expired_at"

    @pytest.mark.asyncio
    async def test_supersede_matching_facts_normalizes_match(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"superseded_count": 0}]
        )
        await supersede_matching_facts(mock_client, "  Alice ", "Works_At", "new-1")
        params = mock_client.graph.execute_write.call_args[0][1]
        assert params["subject_norm"] == "alice"
        assert params["predicate_norm"] == "works_at"

    @pytest.mark.asyncio
    async def test_supersede_fact_by_id_sets_expired_at(self):
        mock_client = MagicMock()
        mock_client.graph.execute_write = AsyncMock(
            return_value=[{"superseded_count": 1}]
        )
        await supersede_fact_by_id(mock_client, "old-1", "new-1")
        query = mock_client.graph.execute_write.call_args[0][0]
        assert "expired_at" in query, "R8: explicit supersession must set expired_at"

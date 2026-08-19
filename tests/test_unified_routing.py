"""Conversational storage routes through the coding extraction (MUD-395).

``memory_store`` (message path) and :class:`UnifiedBamlExtractor` both call
``extract_coding_memory`` with best-effort context — branch="", task=None,
files=[]. With no anchors available, the anchored types (decisions, gotchas,
dead ends) all drop by design; only preferences flow. The hook capture path,
not conversational storage, is the source of anchored coding memory.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from conftest import get_tool_fn


def _stub_coding_baml(monkeypatch, *, decisions=(), gotchas=(), dead_ends=(),
                      preferences=()):
    """Stub the ExtractCodingMemory BAML call; returns the mock."""
    result = SimpleNamespace(
        decisions=list(decisions),
        gotchas=list(gotchas),
        dead_ends=list(dead_ends),
        preferences=list(preferences),
    )
    mock_fn = AsyncMock(return_value=result)
    monkeypatch.setattr(
        "agent_memory_mcp.baml_client.async_client.b.ExtractCodingMemory",
        mock_fn,
    )
    return mock_fn


def _decision():
    return SimpleNamespace(
        text="chose asyncpg over psycopg2",
        reason="async driver matches the event loop",
        anchor_files=["src/db.py"],
        concerns_task=True,
        confidence=0.9,
    )


def _gotcha():
    return SimpleNamespace(
        text="never run migrations against the shared test DB",
        anchor_files=["src/db.py"],
        concerns_task=True,
        confidence=0.8,
    )


def _dead_end():
    return SimpleNamespace(
        attempt="patched the pool size in app startup",
        why_failed="pool is created before config loads",
        anchor_files=["src/app.py"],
        concerns_task=True,
        confidence=0.6,
    )


def _preference():
    return SimpleNamespace(
        category="testing",
        preference="write the failing test before the fix",
        confidence=0.9,
    )


class TestMemoryStoreRouting:
    """memory_store's message path runs the coding extraction."""

    async def test_preferences_flow_and_anchored_types_drop(
        self, mcp_app, mock_mcp_context, monkeypatch
    ):
        ctx, client = mock_mcp_context
        client.long_term.add_preference = AsyncMock(
            return_value=SimpleNamespace(id="pref-001")
        )
        mock_fn = _stub_coding_baml(
            monkeypatch,
            decisions=[_decision()],
            gotchas=[_gotcha()],
            dead_ends=[_dead_end()],
            preferences=[_preference()],
        )

        store_fn = get_tool_fn(mcp_app, "memory_store")
        result = json.loads(
            await store_fn(
                ctx,
                memory_type="message",
                content="a conversational message",
                session_id="s-1",
            )
        )

        # Extraction ran over the message with best-effort (empty) context.
        mock_fn.assert_awaited_once()
        context = mock_fn.call_args.kwargs["context"]
        assert context.branch == ""
        assert context.task is None
        assert context.files == []

        # Response shape is stable; anchored types dropped without context.
        assert result["stored"] is True
        assert result["entities"] == 0
        assert result["relations"] == 0
        assert result["preferences"] == 1

        # The preference went through the existing add_preference pipeline.
        client.long_term.add_preference.assert_awaited_once()
        kwargs = client.long_term.add_preference.call_args.kwargs
        assert kwargs["category"] == "testing"
        assert kwargs["preference"] == "write the failing test before the fix"

        # No entity/relation persistence was attempted.
        client.graph.execute_write.assert_not_called()

    async def test_extraction_failure_still_stores_message(
        self, mcp_app, mock_mcp_context, monkeypatch
    ):
        ctx, client = mock_mcp_context
        monkeypatch.setattr(
            "agent_memory_mcp.baml_client.async_client.b.ExtractCodingMemory",
            AsyncMock(side_effect=RuntimeError("bedrock down")),
        )

        store_fn = get_tool_fn(mcp_app, "memory_store")
        result = json.loads(
            await store_fn(
                ctx,
                memory_type="message",
                content="a conversational message",
                session_id="s-1",
            )
        )

        assert result["stored"] is True
        assert "entities" not in result  # extraction skipped, counts omitted


class TestUnifiedBamlExtractorRouting:
    """The upstream-protocol adapter maps coding output to ExtractionResult."""

    @pytest.fixture
    def stubbed(self, monkeypatch):
        return _stub_coding_baml(
            monkeypatch,
            decisions=[_decision()],
            gotchas=[_gotcha()],
            dead_ends=[_dead_end()],
            preferences=[_preference()],
        )

    async def test_returns_preferences_only(self, stubbed):
        from agent_memory_mcp.extraction.unified import UnifiedBamlExtractor

        result = await UnifiedBamlExtractor().extract("some message text")

        assert result.entities == []
        assert result.relations == []
        assert len(result.preferences) == 1
        pref = result.preferences[0]
        assert pref.category == "testing"
        assert pref.preference == "write the failing test before the fix"
        assert result.source_text == "some message text"

        context = stubbed.call_args.kwargs["context"]
        assert context.branch == ""
        assert context.task is None
        assert context.files == []

    async def test_extract_preferences_false_yields_empty_result(self, stubbed):
        from agent_memory_mcp.extraction.unified import UnifiedBamlExtractor

        result = await UnifiedBamlExtractor().extract(
            "some message text", extract_preferences=False
        )

        assert result.entities == []
        assert result.relations == []
        assert result.preferences == []

    async def test_baml_failure_raises_extraction_error(self, monkeypatch):
        from neo4j_agent_memory.core.exceptions import ExtractionError

        from agent_memory_mcp.extraction.unified import UnifiedBamlExtractor

        monkeypatch.setattr(
            "agent_memory_mcp.baml_client.async_client.b.ExtractCodingMemory",
            AsyncMock(side_effect=RuntimeError("bedrock down")),
        )

        with pytest.raises(ExtractionError, match="Coding BAML extraction failed"):
            await UnifiedBamlExtractor().extract("text")

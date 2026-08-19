"""Tests for reasoning MCP tools: add_reasoning_trace, explain_reasoning, extract_reasoning."""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_trace(
    *,
    task="Fix the bug",
    outcome="Fixed it",
    success=True,
    session_id="test-session",
    steps=None,
):
    """Create a mock ReasoningTrace object."""
    trace = MagicMock()
    trace.id = uuid.uuid4()
    trace.task = task
    trace.outcome = outcome
    trace.success = success
    trace.session_id = session_id
    trace.started_at = datetime(2026, 2, 25, 10, 0, 0, tzinfo=timezone.utc)
    trace.completed_at = datetime(2026, 2, 25, 10, 5, 0, tzinfo=timezone.utc)
    trace.steps = steps or []
    return trace


def _make_step(*, thought="Check logs", action="grep", observation="Found error", step_number=1):
    """Create a mock ReasoningStep object."""
    step = MagicMock()
    step.id = uuid.uuid4()
    step.thought = thought
    step.action = action
    step.observation = observation
    step.step_number = step_number
    step.tool_calls = []
    return step


def _make_tool_call(*, tool_name="grep", arguments=None, status_value="success"):
    """Create a mock ToolCall object."""
    tc = MagicMock()
    tc.tool_name = tool_name
    tc.arguments = arguments or {}
    tc.status = MagicMock()
    tc.status.value = status_value
    return tc


@pytest.fixture
def mock_reasoning_client(monkeypatch):
    """Mock the MemoryClient's reasoning attribute for tool-level tests.

    Returns (mock_client, mock_reasoning) for test inspection.
    """
    mock_reasoning = MagicMock()
    mock_reasoning.start_trace = AsyncMock()
    mock_reasoning.add_step = AsyncMock()
    mock_reasoning.record_tool_call = AsyncMock()
    mock_reasoning.complete_trace = AsyncMock()
    mock_reasoning.get_trace_with_steps = AsyncMock()
    mock_reasoning.get_similar_traces = AsyncMock()
    mock_reasoning.list_traces = AsyncMock()

    mock_client = MagicMock()
    mock_client.reasoning = mock_reasoning

    monkeypatch.setattr(
        "agent_memory_mcp.mcp._tools.get_client",
        lambda ctx: mock_client,
    )

    # Ensure multi-DB helpers fall back to single-client mode


    return mock_client, mock_reasoning


@pytest.fixture
def mock_ctx():
    """Create a mock FastMCP Context."""
    return MagicMock()


class TestAddReasoningTraceNewFormat:
    """Test add_reasoning_trace with new steps format (thought/observation/alternatives)."""

    async def test_steps_with_thought_and_observation(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        step = _make_step()
        mock_reasoning.start_trace.return_value = trace
        mock_reasoning.add_step.return_value = step

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        # Access the registered tool function
        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "add_reasoning_trace":
                tool_fn = tool.fn
                break
        assert tool_fn is not None

        result_str = await tool_fn(
            mock_ctx,
            session_id="test-session",
            task="Debug connection error",
            steps=[
                {
                    "tool_name": "read_file",
                    "thought": "The config might have a typo",
                    "observation": "Found incorrect database URL",
                    "alternatives_considered": "Could be a network issue",
                    "arguments": {"path": "config.yaml"},
                    "result": "db_url: postgre://...",
                    "success": True,
                }
            ],
            outcome="Fixed config typo",
            success=True,
        )

        result = json.loads(result_str)
        assert result["success"] is True
        assert result["step_count"] == 1

        # Verify thought was passed (not auto-generated)
        call_args = mock_reasoning.add_step.call_args
        assert "The config might have a typo" in call_args.kwargs.get("thought", call_args[1].get("thought", ""))

    async def test_steps_with_alternatives_appended_to_thought(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        step = _make_step()
        mock_reasoning.start_trace.return_value = trace
        mock_reasoning.add_step.return_value = step

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "add_reasoning_trace":
                tool_fn = tool.fn
                break

        await tool_fn(
            mock_ctx,
            session_id="s1",
            task="t1",
            steps=[{
                "tool_name": "search",
                "thought": "Try searching",
                "alternatives_considered": "Could also grep logs",
            }],
        )

        call_args = mock_reasoning.add_step.call_args
        thought = call_args.kwargs.get("thought", call_args[1].get("thought", ""))
        assert "Try searching" in thought
        assert "Alternatives considered: Could also grep logs" in thought

    async def test_success_false_maps_to_failure_status(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        step = _make_step()
        mock_reasoning.start_trace.return_value = trace
        mock_reasoning.add_step.return_value = step

        from agent_memory_mcp.mcp._tools import register_tools
        from neo4j_agent_memory.memory.reasoning import ToolCallStatus
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "add_reasoning_trace":
                tool_fn = tool.fn
                break

        await tool_fn(
            mock_ctx,
            session_id="s1",
            task="t1",
            steps=[{"tool_name": "api_call", "success": False}],
        )

        call_args = mock_reasoning.record_tool_call.call_args
        assert call_args.kwargs.get("status") == ToolCallStatus.FAILURE


class TestAddReasoningTraceBackwardCompat:
    """Test add_reasoning_trace with old tool_calls format still works."""

    async def test_old_tool_calls_format(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        step = _make_step()
        mock_reasoning.start_trace.return_value = trace
        mock_reasoning.add_step.return_value = step

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "add_reasoning_trace":
                tool_fn = tool.fn
                break

        result_str = await tool_fn(
            mock_ctx,
            session_id="test-session",
            task="Old format test",
            tool_calls=[
                {"tool_name": "memory_search", "arguments": {"query": "test"}, "result": "found"},
            ],
            outcome="Done",
        )

        result = json.loads(result_str)
        assert result["success"] is True
        assert result["step_count"] == 1

        # When no thought provided, should auto-generate
        call_args = mock_reasoning.add_step.call_args
        thought = call_args.kwargs.get("thought", call_args[1].get("thought", ""))
        assert "Calling memory_search" in thought

    async def test_observation_from_result_when_no_observation(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        step = _make_step()
        mock_reasoning.start_trace.return_value = trace
        mock_reasoning.add_step.return_value = step

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "add_reasoning_trace":
                tool_fn = tool.fn
                break

        await tool_fn(
            mock_ctx,
            session_id="s1",
            task="t1",
            tool_calls=[{"tool_name": "test", "result": "some result"}],
        )

        call_args = mock_reasoning.add_step.call_args
        observation = call_args.kwargs.get("observation", call_args[1].get("observation", ""))
        assert observation == "some result"

    async def test_non_string_observation_converted(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        step = _make_step()
        mock_reasoning.start_trace.return_value = trace
        mock_reasoning.add_step.return_value = step

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "add_reasoning_trace":
                tool_fn = tool.fn
                break

        await tool_fn(
            mock_ctx,
            session_id="s1",
            task="t1",
            tool_calls=[{"tool_name": "test", "result": {"key": "value"}}],
        )

        call_args = mock_reasoning.add_step.call_args
        observation = call_args.kwargs.get("observation", call_args[1].get("observation", ""))
        assert isinstance(observation, str)


class TestExplainReasoning:
    """Test explain_reasoning tool."""

    async def test_by_trace_id(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        step = _make_step()
        tc = _make_tool_call()
        step.tool_calls = [tc]
        trace = _make_trace(steps=[step])
        mock_reasoning.get_trace_with_steps.return_value = trace

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "explain_reasoning":
                tool_fn = tool.fn
                break
        assert tool_fn is not None

        result_str = await tool_fn(mock_ctx, trace_id=str(trace.id))
        result = json.loads(result_str)

        assert result["trace_id"] == str(trace.id)
        assert result["task"] == "Fix the bug"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["thought"] == "Check logs"
        assert result["steps"][0]["action"] == "grep"

    async def test_by_query(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        mock_reasoning.get_similar_traces.return_value = [trace]
        mock_reasoning.get_trace_with_steps.return_value = trace

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "explain_reasoning":
                tool_fn = tool.fn
                break

        result_str = await tool_fn(mock_ctx, query="how to fix bugs")
        result = json.loads(result_str)

        assert result["task"] == "Fix the bug"
        mock_reasoning.get_similar_traces.assert_called_once_with(
            task="how to fix bugs", limit=1, success_only=False,
        )

    async def test_by_session_id(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        mock_reasoning.list_traces.return_value = [trace]
        mock_reasoning.get_trace_with_steps.return_value = trace

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "explain_reasoning":
                tool_fn = tool.fn
                break

        result_str = await tool_fn(mock_ctx, session_id="test-session")
        result = json.loads(result_str)

        assert result["task"] == "Fix the bug"
        mock_reasoning.list_traces.assert_called_once_with(
            session_id="test-session", limit=1,
        )

    async def test_no_params_returns_error(self, mock_reasoning_client, mock_ctx):
        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "explain_reasoning":
                tool_fn = tool.fn
                break

        result_str = await tool_fn(mock_ctx)
        result = json.loads(result_str)

        assert "error" in result
        assert "Provide trace_id, query, or session_id" in result["error"]

    async def test_query_no_match(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        mock_reasoning.get_similar_traces.return_value = []

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "explain_reasoning":
                tool_fn = tool.fn
                break

        result_str = await tool_fn(mock_ctx, query="nonexistent task")
        result = json.loads(result_str)

        assert result["found"] is False

    async def test_trace_not_found(self, mock_reasoning_client, mock_ctx):
        mock_client, mock_reasoning = mock_reasoning_client
        mock_reasoning.get_trace_with_steps.return_value = None

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "explain_reasoning":
                tool_fn = tool.fn
                break

        result_str = await tool_fn(mock_ctx, trace_id="nonexistent")
        result = json.loads(result_str)

        assert "error" in result

    async def test_synthesize(self, mock_reasoning_client, mock_ctx, monkeypatch):
        mock_client, mock_reasoning = mock_reasoning_client
        step = _make_step()
        trace = _make_trace(steps=[step])
        mock_reasoning.get_trace_with_steps.return_value = trace

        mock_synthesize = AsyncMock(return_value="I checked the logs and found the error.")
        monkeypatch.setattr(
            "agent_memory_mcp.extraction.reasoning_extractor.BamlReasoningExtractor.synthesize_explanation",
            mock_synthesize,
        )

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "explain_reasoning":
                tool_fn = tool.fn
                break

        result_str = await tool_fn(mock_ctx, trace_id=str(trace.id), synthesize=True)
        result = json.loads(result_str)

        assert "synthesized_explanation" in result
        assert result["synthesized_explanation"] == "I checked the logs and found the error."


class TestExtractReasoning:
    """Test extract_reasoning tool."""

    async def test_extract_and_store(self, mock_reasoning_client, mock_ctx, monkeypatch):
        mock_client, mock_reasoning = mock_reasoning_client
        trace = _make_trace()
        mock_reasoning.start_trace.return_value = trace
        mock_reasoning.add_step.return_value = _make_step()

        mock_extract = AsyncMock(return_value={
            "task": "Debug API error",
            "steps": [
                {
                    "thought": "Check the logs",
                    "action": "Read log file",
                    "observation": "Found 500 error",
                    "alternatives_considered": None,
                    "confidence": 0.85,
                }
            ],
            "final_conclusion": "Fixed the API handler",
            "success": True,
        })
        monkeypatch.setattr(
            "agent_memory_mcp.extraction.reasoning_extractor.BamlReasoningExtractor.extract_reasoning",
            mock_extract,
        )

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "extract_reasoning":
                tool_fn = tool.fn
                break
        assert tool_fn is not None

        result_str = await tool_fn(
            mock_ctx,
            text="User: The API is broken\nAssistant: Let me check the logs...",
            session_id="test-session",
            store=True,
        )

        result = json.loads(result_str)
        assert result["extracted"] is True
        assert result["task"] == "Debug API error"
        assert result["step_count"] == 1
        assert result["stored"] is True
        assert "trace_id" in result

        # Verify storage calls
        mock_reasoning.start_trace.assert_called_once()
        mock_reasoning.add_step.assert_called_once()
        mock_reasoning.complete_trace.assert_called_once()

    async def test_extract_without_store(self, mock_reasoning_client, mock_ctx, monkeypatch):
        mock_client, mock_reasoning = mock_reasoning_client

        mock_extract = AsyncMock(return_value={
            "task": "Some task",
            "steps": [
                {
                    "thought": "t",
                    "action": "a",
                    "observation": "o",
                    "alternatives_considered": None,
                    "confidence": 0.8,
                }
            ],
            "final_conclusion": "Done",
            "success": True,
        })
        monkeypatch.setattr(
            "agent_memory_mcp.extraction.reasoning_extractor.BamlReasoningExtractor.extract_reasoning",
            mock_extract,
        )

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "extract_reasoning":
                tool_fn = tool.fn
                break

        result_str = await tool_fn(
            mock_ctx,
            text="Some text",
            session_id="test-session",
            store=False,
        )

        result = json.loads(result_str)
        assert result["extracted"] is True
        assert "stored" not in result
        assert "trace_id" not in result

        # Verify no storage calls
        mock_reasoning.start_trace.assert_not_called()


class TestMemorySearchDefaultTypes:
    """Test that memory_search default types now include traces."""

    async def test_default_includes_traces(self, monkeypatch, mock_ctx):
        """memory_search with no memory_types should search traces too."""
        mock_messages = AsyncMock(return_value=[])
        mock_entities = AsyncMock(return_value=[])
        mock_preferences = AsyncMock(return_value=[])
        mock_traces = AsyncMock(return_value=[])
        mock_facts = AsyncMock(return_value=[])

        mock_client = MagicMock()
        mock_client.short_term.search_messages = mock_messages
        mock_client.long_term.search_entities = mock_entities
        mock_client.long_term.search_preferences = mock_preferences
        mock_client.reasoning.get_similar_traces = mock_traces
        mock_client.long_term.search_facts = mock_facts
        mock_client.graph.execute_read = AsyncMock(return_value=[])

        monkeypatch.setattr(
            "agent_memory_mcp.mcp._tools.get_client",
            lambda ctx: mock_client,
        )

        # Ensure multi-DB helpers fall back to single-client mode


        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "memory_search":
                tool_fn = tool.fn
                break
        assert tool_fn is not None

        result_str = await tool_fn(mock_ctx, query="test query")
        result = json.loads(result_str)

        # Traces should be searched by default
        mock_traces.assert_called_once()
        assert "traces" in result["results"]


class TestMemorySearchThresholdForwarding:
    """memory_search must forward its threshold to every branch that accepts one.

    Regression test for MUD-396: the preferences branch called
    search_preferences() without threshold, so the upstream 0.7 default
    always applied regardless of what the caller requested.
    """

    def _make_client(self):
        mock_messages = AsyncMock(return_value=[])
        mock_entities = AsyncMock(return_value=[])
        mock_preferences = AsyncMock(return_value=[])
        mock_traces = AsyncMock(return_value=[])
        mock_facts = AsyncMock(return_value=[])

        mock_client = MagicMock()
        mock_client.short_term.search_messages = mock_messages
        mock_client.long_term.search_entities = mock_entities
        mock_client.long_term.search_preferences = mock_preferences
        mock_client.reasoning.get_similar_traces = mock_traces
        mock_client.long_term.search_facts = mock_facts
        mock_client.graph.execute_read = AsyncMock(return_value=[])
        return mock_client

    async def _run_search(self, monkeypatch, mock_ctx, mock_client, **kwargs):
        monkeypatch.setattr(
            "agent_memory_mcp.mcp._tools.get_client",
            lambda ctx: mock_client,
        )

        from agent_memory_mcp.mcp._tools import register_tools
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "memory_search":
                tool_fn = tool.fn
                break
        assert tool_fn is not None

        result_str = await tool_fn(mock_ctx, query="test query", **kwargs)
        return json.loads(result_str)

    async def test_custom_threshold_forwarded_to_search_preferences(self, monkeypatch, mock_ctx):
        """A caller-supplied threshold must reach search_preferences."""
        mock_client = self._make_client()

        await self._run_search(monkeypatch, mock_ctx, mock_client, threshold=0.5)

        mock_client.long_term.search_preferences.assert_called_once()
        _, kwargs = mock_client.long_term.search_preferences.call_args
        assert kwargs["threshold"] == 0.5

    async def test_default_threshold_forwarded_to_search_preferences(self, monkeypatch, mock_ctx):
        """With no threshold argument, the tool's own 0.7 default must still be forwarded."""
        mock_client = self._make_client()

        await self._run_search(monkeypatch, mock_ctx, mock_client)

        mock_client.long_term.search_preferences.assert_called_once()
        _, kwargs = mock_client.long_term.search_preferences.call_args
        assert kwargs["threshold"] == 0.7

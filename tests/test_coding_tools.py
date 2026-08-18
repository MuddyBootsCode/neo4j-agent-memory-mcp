"""Tests for coding-memory MCP tools: record_coding_activity, coding_recall."""

import json
from unittest.mock import MagicMock

import pytest


class FakeGraph:
    """Captures (query, params) pairs and serves canned read results in order."""

    def __init__(self, read_results=None):
        self.writes = []
        self.reads = []
        self._read_results = list(read_results or [])

    async def execute_write(self, query, params):
        self.writes.append((query, params))
        return []

    async def execute_read(self, query, params):
        self.reads.append((query, params))
        if self._read_results:
            return self._read_results.pop(0)
        return []


@pytest.fixture
def mock_ctx():
    return MagicMock()


def _register(monkeypatch, graph):
    """Register the coding tools against a fake client wrapping ``graph``."""
    mock_client = MagicMock()
    mock_client.graph = graph
    monkeypatch.setattr(
        "agent_memory_mcp.mcp._coding_tools.get_client",
        lambda ctx: mock_client,
    )

    from fastmcp import FastMCP

    from agent_memory_mcp.mcp._coding_tools import register_coding_tools

    mcp = FastMCP("test")
    register_coding_tools(mcp)
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


class TestRecordCodingActivity:
    async def test_happy_path_writes_in_order_with_shared_ts(
        self, monkeypatch, mock_ctx
    ):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)

        result_str = await tools["record_coding_activity"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            task_key="MUD-395",
            edited_files=["a.py", "b.py"],
            commits=[{"sha": "abc123", "message": "fix", "files": ["a.py"], "ts": "ignored"}],
        )

        assert len(graph.writes) == 3
        # Session upsert MUST run first — later builders MATCH the session.
        assert "CodingSession" in graph.writes[0][0]
        assert "CodeAgent" in graph.writes[0][0]
        assert "EDITING" in graph.writes[1][0]
        assert "Change {sha" in graph.writes[2][0]

        # One ISO timestamp shared across every write in the call.
        ts_values = {params["ts"] for _, params in graph.writes}
        assert len(ts_values) == 1

        result = json.loads(result_str)
        assert result == {
            "session": "sess-1",
            "edited_files": 2,
            "commits": 1,
            "skipped_commits": 0,
        }

    async def test_no_files_no_commits_only_session_write(self, monkeypatch, mock_ctx):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)

        result_str = await tools["record_coding_activity"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            edited_files=[],
        )

        assert len(graph.writes) == 1
        assert "CodingSession" in graph.writes[0][0]

        result = json.loads(result_str)
        assert result == {
            "session": "sess-1",
            "edited_files": 0,
            "commits": 0,
            "skipped_commits": 0,
        }

    async def test_commit_without_sha_skipped_and_counted(self, monkeypatch, mock_ctx):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)

        result_str = await tools["record_coding_activity"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            commits=[
                {"message": "no sha here"},
                {"sha": "def456", "message": "ok", "files": []},
            ],
        )

        # session + one valid commit
        assert len(graph.writes) == 2
        assert graph.writes[1][1]["sha"] == "def456"

        result = json.loads(result_str)
        assert result["commits"] == 1
        assert result["skipped_commits"] == 1

    async def test_input_caps_100_files_50_commits(self, monkeypatch, mock_ctx):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)

        result_str = await tools["record_coding_activity"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            edited_files=[f"file{i}.py" for i in range(101)],
            commits=[{"sha": f"sha{i}", "message": "m", "files": []} for i in range(51)],
        )

        # session + editing + 50 commits
        assert len(graph.writes) == 52
        assert len(graph.writes[1][1]["paths"]) == 100

        result = json.loads(result_str)
        assert result["edited_files"] == 100
        assert result["commits"] == 50
        assert result["skipped_commits"] == 0


class TestCodingRecall:
    async def test_memories_with_files(self, monkeypatch, mock_ctx):
        mem_rows = [
            {
                "labels": ["Decision"],
                "props": {"text": "use uv everywhere"},
                "files": ["a.py"],
                "task": "MUD-395",
                "at": "2026-08-18T10:00:00+00:00",
            },
            {
                "labels": ["DeadEnd"],
                "props": {"attempt": "tried asyncio.gather", "why_failed": "ordering"},
                "files": ["b.py"],
                "task": None,
                "at": "2026-08-17T10:00:00+00:00",
            },
        ]
        graph = FakeGraph(read_results=[mem_rows, []])
        tools = _register(monkeypatch, graph)

        result_str = await tools["coding_recall"](
            mock_ctx,
            prompt="what did we decide",
            agent_id="agent-1",
            repo="my-repo",
            files=["a.py"],
        )

        assert len(graph.reads) == 2
        mem_query, mem_params = graph.reads[0]
        assert mem_params["repo"] == "my-repo"
        assert mem_params["files"] == ["a.py"]
        assert mem_params["task_key"] is None

        result = json.loads(result_str)
        assert result["fallback"] is False
        assert result["memories"] == [
            {
                "kind": "Decision",
                "text": "use uv everywhere",
                "files": ["a.py"],
                "task": "MUD-395",
                "at": "2026-08-18T10:00:00+00:00",
            },
            {
                "kind": "DeadEnd",
                "text": "tried asyncio.gather — failed: ordering",
                "files": ["b.py"],
                "task": None,
                "at": "2026-08-17T10:00:00+00:00",
            },
        ]

    async def test_no_files_no_task_key_is_fallback_no_queries(
        self, monkeypatch, mock_ctx
    ):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)

        result_str = await tools["coding_recall"](
            mock_ctx,
            prompt="anything",
            agent_id="agent-1",
            repo="my-repo",
        )

        assert graph.reads == []
        result = json.loads(result_str)
        assert result == {"memories": [], "fallback": True, "overlaps": []}

    async def test_overlaps_shape_window_and_self_exclusion(
        self, monkeypatch, mock_ctx
    ):
        overlap_rows = [
            {
                "agent": "agent-2",
                "files": ["a.py"],
                "task": "MUD-395",
                "last_seen": "2026-08-18T09:00:00+00:00",
            }
        ]
        graph = FakeGraph(read_results=[[], overlap_rows])
        tools = _register(monkeypatch, graph)

        # task_key-only recall: overlap query must still run.
        result_str = await tools["coding_recall"](
            mock_ctx,
            prompt="p",
            agent_id="agent-1",
            repo="my-repo",
            task_key="MUD-395",
            overlap_window_hours=6.0,
        )

        assert len(graph.reads) == 2
        overlap_query, overlap_params = graph.reads[1]
        assert "a2.id <> $agent_id" in overlap_query
        assert overlap_params["window"] == 6.0
        assert overlap_params["agent_id"] == "agent-1"

        result = json.loads(result_str)
        assert result["fallback"] is False
        assert result["overlaps"] == [
            {
                "agent": "agent-2",
                "files": ["a.py"],
                "task": "MUD-395",
                "last_seen": "2026-08-18T09:00:00+00:00",
            }
        ]

    async def test_no_user_value_ever_lands_in_query_text(self, monkeypatch, mock_ctx):
        sentinel = "__INJECT__') DETACH DELETE n //"
        graph = FakeGraph(read_results=[[], []])
        tools = _register(monkeypatch, graph)

        await tools["record_coding_activity"](
            mock_ctx,
            agent_id=sentinel,
            session_id=sentinel,
            repo=sentinel,
            branch=sentinel,
            task_key=sentinel,
            edited_files=[sentinel],
            commits=[{"sha": sentinel, "message": sentinel, "files": [sentinel]}],
        )
        await tools["coding_recall"](
            mock_ctx,
            prompt=sentinel,
            agent_id=sentinel,
            repo=sentinel,
            files=[sentinel],
            task_key=sentinel,
        )

        for query, _params in graph.writes + graph.reads:
            assert sentinel not in query


class TestServerRegistration:
    def test_create_mcp_server_registers_coding_tools(self):
        from agent_memory_mcp.mcp.server import create_mcp_server

        mcp = create_mcp_server()
        names = {t.name for t in mcp._tool_manager._tools.values()}
        assert "record_coding_activity" in names
        assert "coding_recall" in names

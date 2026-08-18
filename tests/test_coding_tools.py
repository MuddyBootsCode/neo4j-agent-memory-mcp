"""Tests for coding-memory MCP tools: record_coding_activity, coding_recall."""

import json
from unittest.mock import MagicMock

import pytest


class FakeGraph:
    """Captures (query, params) pairs and serves canned read results in order.

    ``fail_on_write`` makes the Nth (1-indexed) execute_write raise before
    the call is recorded, so ``writes`` holds only the successful writes.
    """

    def __init__(self, read_results=None, fail_on_write=None):
        self.writes = []
        self.reads = []
        self._read_results = list(read_results or [])
        self._fail_on_write = fail_on_write
        self._write_calls = 0

    async def execute_write(self, query, params):
        self._write_calls += 1
        if self._fail_on_write == self._write_calls:
            raise RuntimeError("neo4j write failed")
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

    async def test_write_failure_reports_partial_progress(
        self, monkeypatch, mock_ctx
    ):
        # Writes: 1=session, 2=editing, 3=first sha'd commit (fails).
        graph = FakeGraph(fail_on_write=3)
        tools = _register(monkeypatch, graph)

        result_str = await tools["record_coding_activity"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            edited_files=["a.py", "b.py"],
            commits=[
                {"message": "no sha"},
                {"sha": "abc123", "message": "m", "files": []},
                {"sha": "def456", "message": "m", "files": []},
            ],
        )

        # The writes before the failure landed and are recorded.
        assert len(graph.writes) == 2
        assert "CodingSession" in graph.writes[0][0]
        assert "EDITING" in graph.writes[1][0]

        result = json.loads(result_str)
        assert result == {
            "error": "neo4j write failed",
            "edited_files": 2,
            "commits": 0,
            "skipped_commits": 1,
        }


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
        # Anchored from the data, not an all-agents scan.
        assert "(f:CodeFile {repo: $repo})<-[e:EDITING]-" in overlap_query
        assert "<-[w:WORKING_ON]-" in overlap_query
        assert "LIMIT 20" in overlap_query
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
        assert "capture_session_memory" in names


def _extraction(**overrides):
    """A canned extract_coding_memory result, tweakable per test."""
    result = {
        "decisions": [
            {
                "text": "use uv everywhere",
                "reason": "lockfile reproducibility",
                "anchor_files": ["a.py"],
                "concerns_task": True,
                "confidence": 0.9,
            },
            {
                "text": "keep hooks thin",
                "reason": "fail-open budget",
                "anchor_files": ["b.py"],
                "concerns_task": False,
                "confidence": 0.8,
            },
        ],
        "gotchas": [
            {
                "text": "pytest asyncio_mode is auto here",
                "anchor_files": ["a.py", "b.py"],
                "concerns_task": False,
                "confidence": 0.7,
            }
        ],
        "dead_ends": [
            {
                "attempt": "tried asyncio.gather",
                "why_failed": "write ordering matters",
                "anchor_files": [],
                "concerns_task": True,
                "confidence": 0.7,
            }
        ],
        "preferences": [
            {
                "category": "tooling",
                "preference": "prefers uv over pip",
                "confidence": 0.8,
            }
        ],
        "anchor_rate": 0.8,
        "dropped_unanchored": 1,
    }
    result.update(overrides)
    return result


def _stub_extract(monkeypatch, result=None, exc=None):
    """Replace extract_coding_memory in the tool module; returns call log."""
    calls = []

    async def fake(transcript, *, branch, task, files):
        calls.append(
            {"transcript": transcript, "branch": branch, "task": task, "files": files}
        )
        if exc is not None:
            raise exc
        return result if result is not None else _extraction()

    monkeypatch.setattr(
        "agent_memory_mcp.mcp._coding_tools.extract_coding_memory", fake
    )
    return calls


class TestCaptureSessionMemory:
    async def test_happy_path_writes_props_edges_and_counts(
        self, monkeypatch, mock_ctx
    ):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)
        calls = _stub_extract(monkeypatch)

        result_str = await tools["capture_session_memory"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="feature/x",
            transcript="user: do the thing\nassistant: done",
            task_key="MUD-395",
            files=["a.py", "b.py"],
        )

        assert calls == [
            {
                "transcript": "user: do the thing\nassistant: done",
                "branch": "feature/x",
                "task": "MUD-395",
                "files": ["a.py", "b.py"],
            }
        ]

        # Session upsert first, then one write per kept item.
        assert len(graph.writes) == 6
        assert "CodingSession" in graph.writes[0][0]
        assert "CodeAgent" in graph.writes[0][0]

        # One ISO timestamp shared across every write in the call.
        ts_values = {params["ts"] for _, params in graph.writes}
        assert len(ts_values) == 1

        d1_query, d1_params = graph.writes[1]
        assert "CREATE (m:Decision)" in d1_query
        # Prop names must match coding_recall's _MEMORIES_QUERY contract.
        assert set(d1_params["props"]) == {"text", "reason", "confidence"}
        assert d1_params["props"]["text"] == "use uv everywhere"
        assert d1_params["anchor_paths"] == ["a.py"]
        # concerns_task=True carries the task edge.
        assert d1_params["task_key"] == "MUD-395"
        assert "CONCERNS" in d1_query

        d2_query, d2_params = graph.writes[2]
        assert "CREATE (m:Decision)" in d2_query
        # concerns_task=False: no task edge even though task_key was given.
        assert "task_key" not in d2_params
        assert "CONCERNS" not in d2_query
        assert d2_params["anchor_paths"] == ["b.py"]

        g_query, g_params = graph.writes[3]
        assert "CREATE (m:Gotcha)" in g_query
        assert set(g_params["props"]) == {"text", "confidence"}
        assert g_params["anchor_paths"] == ["a.py", "b.py"]

        de_query, de_params = graph.writes[4]
        assert "CREATE (m:DeadEnd)" in de_query
        assert set(de_params["props"]) == {"attempt", "why_failed", "confidence"}
        assert de_params["task_key"] == "MUD-395"

        p_query, p_params = graph.writes[5]
        assert "CREATE (m:CodingPreference)" in p_query
        assert set(p_params["props"]) == {"category", "preference", "confidence"}
        # Preferences are session-anchored: no files, no task edge.
        assert p_params["anchor_paths"] == []
        assert "task_key" not in p_params

        result = json.loads(result_str)
        assert result == {
            "stored": 5,
            "by_kind": {
                "Decision": 2,
                "Gotcha": 1,
                "DeadEnd": 1,
                "CodingPreference": 1,
            },
            "dropped_unanchored": 1,
            "anchor_rate": 0.8,
        }

    async def test_empty_transcript_skips_extraction_and_writes(
        self, monkeypatch, mock_ctx
    ):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)
        calls = _stub_extract(monkeypatch)

        result_str = await tools["capture_session_memory"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            transcript="   \n  ",
        )

        assert calls == []
        assert graph.writes == []
        assert json.loads(result_str) == {
            "stored": 0,
            "dropped_unanchored": 0,
            "anchor_rate": None,
        }

    async def test_transcript_tail_and_files_are_capped(self, monkeypatch, mock_ctx):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)
        calls = _stub_extract(monkeypatch)

        transcript = ("x" * 90_000) + "TAIL-MARKER"
        await tools["capture_session_memory"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            transcript=transcript,
            files=[f"f{i}.py" for i in range(120)],
        )

        sent = calls[0]["transcript"]
        assert len(sent) == 80_000
        assert sent.endswith("TAIL-MARKER")
        assert len(calls[0]["files"]) == 100

    async def test_extraction_failure_returns_error_without_writes(
        self, monkeypatch, mock_ctx
    ):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)
        _stub_extract(monkeypatch, exc=RuntimeError("baml exploded"))

        result_str = await tools["capture_session_memory"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            transcript="user: hi",
        )

        assert graph.writes == []
        result = json.loads(result_str)
        assert result["error"] == "baml exploded"
        assert result["stored"] == 0

    async def test_nothing_extracted_skips_session_upsert(
        self, monkeypatch, mock_ctx
    ):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)
        _stub_extract(
            monkeypatch,
            result=_extraction(
                decisions=[],
                gotchas=[],
                dead_ends=[],
                preferences=[],
                anchor_rate=None,
                dropped_unanchored=2,
            ),
        )

        result_str = await tools["capture_session_memory"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            transcript="user: hi",
        )

        assert graph.writes == []
        result = json.loads(result_str)
        assert result == {
            "stored": 0,
            "by_kind": {
                "Decision": 0,
                "Gotcha": 0,
                "DeadEnd": 0,
                "CodingPreference": 0,
            },
            "dropped_unanchored": 2,
            "anchor_rate": None,
        }

    async def test_write_failure_reports_partial_counts(
        self, monkeypatch, mock_ctx
    ):
        # Writes: 1=session, 2=first decision, 3=second decision (fails).
        graph = FakeGraph(fail_on_write=3)
        tools = _register(monkeypatch, graph)
        _stub_extract(monkeypatch)

        result_str = await tools["capture_session_memory"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            transcript="user: hi",
            task_key="MUD-395",
            files=["a.py", "b.py"],
        )

        assert len(graph.writes) == 2
        result = json.loads(result_str)
        assert result["error"] == "neo4j write failed"
        assert result["stored"] == 1
        assert result["by_kind"]["Decision"] == 1
        assert result["by_kind"]["Gotcha"] == 0
        assert result["dropped_unanchored"] == 1

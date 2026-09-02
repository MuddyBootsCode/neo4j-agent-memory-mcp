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


class FakeEmbedder:
    """Embedder returning a fixed vector, or raising when asked to."""

    def __init__(self, dimensions=4, fail=False):
        self.dimensions = dimensions
        self.fail = fail
        self.texts = []

    async def embed(self, text):
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("embedder down")
        return [0.1] * self.dimensions


def _register(monkeypatch, graph, embedder=None):
    """Register the coding tools against a fake client wrapping ``graph``.

    ``embedder`` is attached where the production code looks for it
    (``client.long_term._embedder``); the default MagicMock has no usable
    one, which is what keeps the anchor-only paths under test.
    """
    mock_client = MagicMock()
    mock_client.graph = graph
    if embedder is not None:
        mock_client.long_term._embedder = embedder
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

        # session, editing, commit, then the RESOLVED_BY sweep (MUD-405).
        assert len(graph.writes) == 4
        # Session upsert MUST run first — later builders MATCH the session.
        assert "CodingSession" in graph.writes[0][0]
        assert "CodeAgent" in graph.writes[0][0]
        assert "EDITING" in graph.writes[1][0]
        assert "Change {sha" in graph.writes[2][0]
        assert "RESOLVED_BY" in graph.writes[3][0]

        # One ISO timestamp shared across every write in the call.
        ts_values = {params["ts"] for _, params in graph.writes[:3]}
        assert len(ts_values) == 1

        result = json.loads(result_str)
        assert result == {
            "session": "sess-1",
            "edited_files": 2,
            "commits": 1,
            "skipped_commits": 0,
            "resolved": 0,
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
            "resolved": 0,
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

        # session + one valid commit + the resolve sweep
        assert len(graph.writes) == 3
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

        # session + editing + 50 commits + resolve sweep
        assert len(graph.writes) == 53
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
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "GATE_ENABLED", False)
        graph = FakeGraph(read_results=[mem_rows, []])
        tools = _register(monkeypatch, graph)

        result_str = await tools["coding_recall"](
            mock_ctx,
            prompt="what did we decide",
            agent_id="agent-1",
            repo="my-repo",
            files=["a.py"],
        )

        # No embedder, so the BM25 leg alone serves (MUD-406), then overlaps.
        assert len(graph.reads) == 2
        mem_query, mem_params = graph.reads[0]
        assert "db.index.fulltext.queryNodes" in mem_query
        assert mem_params["repo"] == "my-repo"
        assert mem_params["files"] == ["a.py"]
        assert mem_params["task_key"] is None

        result = json.loads(result_str)
        assert result["fallback"] is False
        assert result["strategy"] == "fulltext"
        for m in result["memories"]:
            for key in ("score", "anchored", "ranks"):
                m.pop(key, None)
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

        # No embedder in the fake client, so only the BM25 leg runs; it
        # finds nothing, and with no files and no task there is no anchor
        # leg either.
        assert len(graph.reads) == 1
        assert "db.index.fulltext.queryNodes" in graph.reads[0][0]
        result = json.loads(result_str)
        assert set(result.pop("timing_ms")) == {"embed", "vector"}
        assert result == {
            "memories": [],
            "fallback": True,
            "strategy": None,
            "overlaps": [],
        }

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

    async def fake(transcript, *, branch, task, files, trace_meta=None):
        calls.append(
            {"transcript": transcript, "branch": branch, "task": task, "files": files}
        )
        if exc is not None:
            raise exc
        return result if result is not None else _extraction()

    monkeypatch.setattr(
        "agent_memory_mcp.mcp._coding_tools.extract_coding_memory", fake
    )
    _stub_curator(monkeypatch)
    return calls


def _stub_curator(monkeypatch, keep=None):
    """Curator keeps everything (or the given predicate) and no neighbour
    lookup hits the graph, so the capture tests exercise the write path."""
    async def fake_curate(candidates, transcript, existing, trace_meta=None):
        kept = [c for c in candidates if keep is None or keep(c)]
        return {"kept": kept, "counts": {"write": len(kept), "already_known": 0, "supersedes": 0,
                                         "not_durable": len(candidates) - len(kept), "unsupported": 0},
                "known": []}

    async def no_neighbors(client, repo, embedding):
        return []

    monkeypatch.setattr("agent_memory_mcp.mcp._coding_tools.curate_coding_memory", fake_curate)
    monkeypatch.setattr("agent_memory_mcp.mcp._coding_tools._neighbors", no_neighbors)


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

        # Session upsert first, then one write per kept lesson. Preferences
        # go through the upstream store (MUD-405), not a CodingPreference node.
        assert len(graph.writes) == 5
        assert "CodingSession" in graph.writes[0][0]
        assert "CodeAgent" in graph.writes[0][0]

        # One ISO timestamp shared across every write in the call.
        ts_values = {params["ts"] for _, params in graph.writes}
        assert len(ts_values) == 1

        d1_query, d1_params = graph.writes[1]
        assert "CREATE (m:Decision:CodingMemory)" in d1_query
        # Prop names must match coding_recall's _MEMORIES_QUERY contract.
        assert set(d1_params["props"]) == {"text", "reason", "confidence"}
        assert d1_params["props"]["text"] == "use uv everywhere"
        assert d1_params["anchor_paths"] == ["a.py"]
        # concerns_task=True carries the task edge.
        assert d1_params["task_key"] == "MUD-395"
        assert "CONCERNS" in d1_query

        d2_query, d2_params = graph.writes[2]
        assert "CREATE (m:Decision:CodingMemory)" in d2_query
        # concerns_task=False: no task edge even though task_key was given.
        assert "task_key" not in d2_params
        assert "CONCERNS" not in d2_query
        assert d2_params["anchor_paths"] == ["b.py"]

        g_query, g_params = graph.writes[3]
        assert "CREATE (m:Gotcha:CodingMemory)" in g_query
        assert set(g_params["props"]) == {"text", "confidence"}
        assert g_params["anchor_paths"] == ["a.py", "b.py"]

        de_query, de_params = graph.writes[4]
        assert "CREATE (m:DeadEnd:CodingMemory)" in de_query
        assert set(de_params["props"]) == {"attempt", "why_failed", "confidence"}
        assert de_params["task_key"] == "MUD-395"

        assert not any("CodingPreference" in q for q, _ in graph.writes)

        result = json.loads(result_str)
        assert result == {
            "stored": 4,
            "by_kind": {
                "Decision": 2,
                "Gotcha": 1,
                "DeadEnd": 1,
                "CodingPreference": 0,
            },
            "dropped_unanchored": 1,
            "anchor_rate": 0.8,
            "embedded": 0,
            "windows": 1,
            "curated": {"write": 5, "already_known": 0, "supersedes": 0, "not_durable": 0, "unsupported": 0},
            "reasserted": 0,
            "superseded": 0,
            "preferences": 0,
            "rated": {"served": 0, "helpful": 0, "harmful": 0, "unused": 0},
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
            "by_kind": {
                "Decision": 0,
                "Gotcha": 0,
                "DeadEnd": 0,
                "CodingPreference": 0,
            },
            "dropped_unanchored": 0,
            "anchor_rate": None,
            "embedded": 0,
            "windows": 0,
            "curated": {"write": 0, "already_known": 0, "supersedes": 0, "not_durable": 0, "unsupported": 0},
            "reasserted": 0,
            "superseded": 0,
            "preferences": 0,
            "rated": {"served": 0, "helpful": 0, "harmful": 0, "unused": 0},
        }

    async def test_transcript_tail_and_files_are_capped(self, monkeypatch, mock_ctx):
        graph = FakeGraph()
        tools = _register(monkeypatch, graph)
        calls = _stub_extract(monkeypatch, result=_extraction(
            decisions=[], gotchas=[], dead_ends=[], preferences=[], anchor_rate=None, dropped_unanchored=0,
        ))

        # 450k chars of one-line filler: the server keeps the 400k tail and
        # cuts it into windows, each an extraction call; the newest window
        # carries the tail marker. Files are capped at 100.
        transcript = "\n".join("x" * 99 for _ in range(4_500)) + "\nTAIL-MARKER"
        result_str = await tools["capture_session_memory"](
            mock_ctx,
            agent_id="agent-1",
            session_id="sess-1",
            repo="my-repo",
            branch="main",
            transcript=transcript,
            files=[f"f{i}.py" for i in range(150)],
        )

        assert calls, "extractor was never called"
        assert sum(len(c["transcript"]) for c in calls) <= 400_000
        assert calls[-1]["transcript"].endswith("TAIL-MARKER")
        assert all(len(c["files"]) == 100 for c in calls)
        result = json.loads(result_str)
        assert result["windows"] == len(calls)
        assert 1 <= result["windows"] <= 4

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
            "embedded": 0,
            "windows": 1,
            "curated": {"write": 0, "already_known": 0, "supersedes": 0, "not_durable": 0, "unsupported": 0},
            "reasserted": 0,
            "superseded": 0,
            "preferences": 0,
            "rated": {"served": 0, "helpful": 0, "harmful": 0, "unused": 0},
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


class TestHybridRecall:
    """MUD-401: rank by prompt similarity, anchor as a boost not a gate."""

    def test_embedding_text_joins_dead_end_fields(self):
        from agent_memory_mcp.mcp._coding_tools import memory_embedding_text

        assert memory_embedding_text("Gotcha", {"text": "pin the version"}) == (
            "pin the version"
        )
        assert memory_embedding_text(
            "DeadEnd", {"attempt": "used a stash", "why_failed": "it was a no-op"}
        ) == "used a stash — failed: it was a no-op"

    async def test_prompt_alone_is_enough_no_anchors_needed(
        self, monkeypatch, mock_ctx
    ):
        """v1 returned nothing without files or a task; the vector leg does not need them."""
        from agent_memory_mcp.mcp._coding_tools import CODING_MEMORY_INDEX

        rows = [{
            "labels": ["Gotcha", "CodingMemory"], "props": {"text": "pin it"},
            "files": [], "task": None, "at": "2026-08-21T00:00:00Z",
            "score": 0.61, "anchored": False,
        }]
        graph = FakeGraph(read_results=[rows])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="why is the version floating?",
            agent_id="agent-1", repo="my-repo",
        ))

        assert result["strategy"] == "vector+gate"
        assert result["fallback"] is False
        assert result["memories"][0]["text"] == "pin it"
        # Fused score (RRF), not raw cosine; rank 0 on leg 0.
        assert result["memories"][0]["score"] == round(1 / 61, 4)
        assert result["memories"][0]["ranks"] == {"0": 0}
        assert result["memories"][0]["anchored"] is False
        # Two reads: the vector leg and the BM25 leg. No anchors, so no overlaps query.
        assert len(graph.reads) == 2
        assert graph.reads[0][1]["index"] == CODING_MEMORY_INDEX

    async def test_legs_pass_threshold_limit_and_anchors(
        self, monkeypatch, mock_ctx
    ):
        from agent_memory_mcp.mcp._coding_tools import (
            CODING_MEMORY_TEXT_INDEX,
            HYBRID_THRESHOLD,
            LEG_LIMIT,
        )

        embedder = FakeEmbedder()
        graph = FakeGraph(read_results=[[], [], []])
        tools = _register(monkeypatch, graph, embedder=embedder)

        await tools["coding_recall"](
            mock_ctx, prompt="what broke the deploy?", agent_id="agent-1",
            repo="my-repo", files=["infra/api.ts"], task_key="MUD-401",
        )

        assert embedder.texts == ["what broke the deploy?"]
        vec = graph.reads[0][1]
        assert vec["threshold"] == HYBRID_THRESHOLD
        assert vec["limit"] == LEG_LIMIT
        assert vec["files"] == ["infra/api.ts"]
        assert vec["task_key"] == "MUD-401"
        text = graph.reads[1][1]
        assert text["index"] == CODING_MEMORY_TEXT_INDEX
        assert text["query"] == "broke deploy"
        assert text["files"] == ["infra/api.ts"]
        # Anchors present, so overlaps still runs after both legs.
        assert len(graph.reads) == 3

    async def test_missing_index_falls_back_to_anchor_first(
        self, monkeypatch, mock_ctx
    ):
        """Nothing captured since the upgrade means no index; anchor-first still works."""
        class ExplodingGraph(FakeGraph):
            async def execute_read(self, query, params):
                if "db.index.vector.queryNodes" in query:
                    raise RuntimeError("no such index")
                return await super().execute_read(query, params)

        graph = ExplodingGraph(read_results=[[], [], []])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="anything", agent_id="agent-1", repo="my-repo",
            files=["a.py"],
        ))

        assert result["strategy"] == "anchor"
        assert result["fallback"] is False

    async def test_empty_hybrid_result_does_not_fall_back(
        self, monkeypatch, mock_ctx
    ):
        """An anchored candidate the prompt does not match is the 88% we stopped injecting."""
        graph = FakeGraph(read_results=[[], [], []])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="unrelated question", agent_id="agent-1",
            repo="my-repo", files=["a.py"],
        ))

        assert result["strategy"] == "vector+gate"
        assert result["memories"] == []
        # Only the two legs and overlaps — the anchor-first query never ran.
        assert not any("ORDER BY m.created_at DESC" in q for q, _ in graph.reads)

    async def test_broken_embedder_uses_anchor_path(self, monkeypatch, mock_ctx):
        graph = FakeGraph(read_results=[[], []])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder(fail=True))

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="anything", agent_id="agent-1", repo="my-repo",
            files=["a.py"],
        ))

        assert result["strategy"] == "anchor"


def _fused_row(text, eid, kind="Gotcha", **props):
    return {
        "labels": [kind, "CodingMemory"],
        "props": {"text": text, **props},
        "eid": eid,
        "files": [],
        "task": None,
        "at": "2026-08-21T00:00:00Z",
    }


class TestRecallDedup:
    """MUD-407: the same lesson captured in several sessions is several
    nodes; recall must not spend slots serving the identical text twice."""

    def test_duplicates_collapse_to_highest_ranked_instance(self):
        from agent_memory_mcp.mcp._coding_tools import dedupe_fused

        rows = [
            _fused_row("run npx vitest run", "eid-1"),
            _fused_row("check the lockfile", "eid-2"),
            _fused_row("run npx vitest run", "eid-3"),
            _fused_row("run npx vitest run", "eid-4"),
        ]
        kept = dedupe_fused(rows)
        assert [r["eid"] for r in kept] == ["eid-1", "eid-2"]

    def test_distinct_texts_survive(self):
        from agent_memory_mcp.mcp._coding_tools import dedupe_fused

        rows = [
            _fused_row("pin the version", "eid-1"),
            _fused_row("unpin the version", "eid-2"),
            _fused_row("delete the lockfile", "eid-3"),
        ]
        assert dedupe_fused(rows) == rows

    def test_whitespace_and_case_variants_collapse(self):
        from agent_memory_mcp.mcp._coding_tools import dedupe_fused

        rows = [
            _fused_row("Run  the full\tsuite", "eid-1"),
            _fused_row("run the full suite ", "eid-2"),
        ]
        kept = dedupe_fused(rows)
        assert [r["eid"] for r in kept] == ["eid-1"]

    def test_dead_end_duplicates_compare_on_joined_fields(self):
        from agent_memory_mcp.mcp._coding_tools import dedupe_fused

        dead_end = {
            "labels": ["DeadEnd", "CodingMemory"],
            "props": {"attempt": "used a stash", "why_failed": "no-op"},
            "eid": "eid-1",
        }
        twin = dict(dead_end, eid="eid-2")
        kept = dedupe_fused([dead_end, twin])
        assert [r["eid"] for r in kept] == ["eid-1"]

    def test_rows_without_text_are_not_collapsed_together(self):
        from agent_memory_mcp.mcp._coding_tools import dedupe_fused

        rows = [_fused_row("", "eid-1"), _fused_row("", "eid-2")]
        assert dedupe_fused(rows) == rows

    def test_empty_and_single_item_pass_through(self):
        from agent_memory_mcp.mcp._coding_tools import dedupe_fused

        assert dedupe_fused([]) == []
        single = [_fused_row("pin it", "eid-1")]
        assert dedupe_fused(single) == single

    async def test_backfill_keeps_limit_distinct_results(self, monkeypatch):
        """Dedup runs before the top-k cut, so freed slots backfill with the
        next-ranked distinct lessons instead of shrinking the recall."""
        from unittest.mock import MagicMock

        from agent_memory_mcp.mcp._coding_tools import retrieve_candidates

        vector_rows = [
            dict(_fused_row("run npx vitest run", f"eid-{i}"), score=0.9 - i / 100)
            for i in range(3)
        ] + [
            dict(_fused_row("check the lockfile", "eid-10"), score=0.5),
            dict(_fused_row("pin the version", "eid-11"), score=0.4),
            dict(_fused_row("delete node_modules", "eid-12"), score=0.3),
        ]
        graph = FakeGraph(read_results=[vector_rows, []])
        client = MagicMock()
        client.graph = graph
        client.long_term._embedder = FakeEmbedder()

        fused, strategy = await retrieve_candidates(
            client, prompt="why does the test suite fail?", repo="my-repo",
            files=[], task_key=None, limit=3,
        )

        assert strategy == "vector"
        assert [r["eid"] for r in fused] == ["eid-0", "eid-10", "eid-11"]


class TestCaptureEmbeds:
    async def test_lessons_are_embedded_preferences_are_not(
        self, monkeypatch, mock_ctx
    ):
        embedder = FakeEmbedder()
        graph = FakeGraph()
        tools = _register(monkeypatch, graph, embedder=embedder)
        _stub_extract(monkeypatch)

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="agent-1", session_id="sess-1", repo="my-repo",
            branch="main", transcript="user: hi", files=["a.py"],
        ))

        # 4 of the 5 stubbed items are recall kinds; the preference is not.
        assert result["embedded"] == 4
        assert len(embedder.texts) == 4
        embedded_writes = [
            params for _q, params in graph.writes if "embedding" in params
        ]
        assert len(embedded_writes) == 4
        assert all(p["embedding"] == [0.1] * 4 for p in embedded_writes)

    async def test_index_is_ensured_only_when_something_was_embedded(
        self, monkeypatch, mock_ctx
    ):
        from agent_memory_mcp.mcp._coding_tools import CODING_MEMORY_INDEX

        graph = FakeGraph()
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch)

        await tools["capture_session_memory"](
            mock_ctx, agent_id="agent-1", session_id="sess-1", repo="my-repo",
            branch="main", transcript="user: hi", files=["a.py"],
        )
        ddl = [q for q, _ in graph.writes if CODING_MEMORY_INDEX in q]
        assert len(ddl) == 1
        assert "`vector.dimensions`: 4" in ddl[0]

        # No embedder: nothing to index, so no DDL and no behaviour change.
        graph2 = FakeGraph()
        tools2 = _register(monkeypatch, graph2)
        _stub_extract(monkeypatch)
        result = json.loads(await tools2["capture_session_memory"](
            mock_ctx, agent_id="agent-1", session_id="sess-1", repo="my-repo",
            branch="main", transcript="user: hi", files=["a.py"],
        ))
        assert result["embedded"] == 0
        assert not any(CODING_MEMORY_INDEX in q for q, _ in graph2.writes)


class TestRecallGate:
    """MUD-401 follow-up: cosine ranking cannot tell relevant from irrelevant."""

    @staticmethod
    def _screen(monkeypatch, verdicts=None, raises=False):
        """Patch the BAML screen; verdicts is {id: keep}."""
        import sys
        import types
        from unittest.mock import MagicMock

        async def fake(query, candidates, baml_options=None):
            if raises:
                raise RuntimeError("judge down")
            out = MagicMock()
            out.verdicts = [
                types.SimpleNamespace(id=i, keep=k)
                for i, k in (verdicts or {}).items()
            ]
            return out

        mod = types.ModuleType("agent_memory_mcp.baml_client.async_client")
        mod.b = MagicMock()
        mod.b.ScreenRecalledMemories = fake
        monkeypatch.setitem(
            sys.modules, "agent_memory_mcp.baml_client.async_client", mod
        )

    @staticmethod
    def _rows(n):
        return [{
            "eid": f"e{i}",
            "labels": ["Gotcha", "CodingMemory"], "props": {"text": f"lesson {i}"},
            "files": [], "task": None, "at": "2026-08-21T00:00:00Z",
            "score": 0.6, "anchored": False,
        } for i in range(n)]

    async def test_gate_drops_what_the_judge_rejects(self, monkeypatch, mock_ctx):
        self._screen(monkeypatch, {0: True, 1: False, 2: True})
        graph = FakeGraph(read_results=[self._rows(3)])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="q", agent_id="a", repo="r",
        ))

        assert result["strategy"] == "vector+gate"
        assert [m["text"] for m in result["memories"]] == ["lesson 0", "lesson 2"]

    async def test_gate_failure_returns_ungated(self, monkeypatch, mock_ctx):
        """A dead judge must degrade to MUD-401 behaviour, never to empty recall."""
        self._screen(monkeypatch, raises=True)
        graph = FakeGraph(read_results=[self._rows(3)])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="q", agent_id="a", repo="r",
        ))

        assert len(result["memories"]) == 3

    async def test_gate_timeout_returns_ungated(self, monkeypatch, mock_ctx):
        """A hung judge must not hold the hook; the cap is the hook's budget, not BAML's."""
        import asyncio
        import sys
        import types
        from unittest.mock import MagicMock

        import agent_memory_mcp.mcp._coding_tools as ct

        async def hang(query, candidates, baml_options=None):
            await asyncio.sleep(5)

        mod = types.ModuleType("agent_memory_mcp.baml_client.async_client")
        mod.b = MagicMock()
        mod.b.ScreenRecalledMemories = hang
        monkeypatch.setitem(sys.modules, "agent_memory_mcp.baml_client.async_client", mod)
        monkeypatch.setattr(ct, "GATE_TIMEOUT_S", 0.05)
        graph = FakeGraph(read_results=[self._rows(3)])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="q", agent_id="a", repo="r",
        ))

        assert len(result["memories"]) == 3
        assert result["timing_ms"]["gate"] < 1000

    async def test_partial_verdicts_are_discarded_whole(self, monkeypatch, mock_ctx):
        """A truncated judge must not look like a decisive one."""
        self._screen(monkeypatch, {0: True})  # 1 verdict for 3 candidates
        graph = FakeGraph(read_results=[self._rows(3)])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="q", agent_id="a", repo="r",
        ))

        assert len(result["memories"]) == 3

    async def test_gate_can_be_disabled(self, monkeypatch, mock_ctx):
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "GATE_ENABLED", False)
        graph = FakeGraph(read_results=[self._rows(3)])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="q", agent_id="a", repo="r",
        ))

        assert result["strategy"] == "vector"
        # Legs always fetch LEG_LIMIT; fusion cuts to the caller's limit.
        assert graph.reads[0][1]["limit"] == ct.LEG_LIMIT
        assert len(result["memories"]) == 3

    async def test_gate_retrieves_at_gate_depth(self, monkeypatch, mock_ctx):
        import agent_memory_mcp.mcp._coding_tools as ct

        self._screen(monkeypatch, {i: True for i in range(10)})
        graph = FakeGraph(read_results=[self._rows(10)])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](mock_ctx, prompt="q", agent_id="a", repo="r"))

        assert graph.reads[0][1]["limit"] == ct.LEG_LIMIT
        assert len(result["memories"]) == max(ct.GATE_DEPTH, ct._RECALL_LIMIT)

    async def test_result_is_capped_at_recall_limit(self, monkeypatch, mock_ctx):
        """Gate depth may exceed the return limit; the caller's budget is fixed."""
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "GATE_DEPTH", 14)
        self._screen(monkeypatch, {i: True for i in range(14)})
        graph = FakeGraph(read_results=[self._rows(14)])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="q", agent_id="a", repo="r",
        ))

        assert len(result["memories"]) == ct._RECALL_LIMIT

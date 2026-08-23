"""Lessons accumulate instead of duplicating (MUD-405).

Builders are pure; the pipeline and tool paths run against the fake graph
from test_coding_tools with the curator stubbed.
"""

import json

import pytest

from agent_memory_mcp.capture.cypher import (
    anchored_memory_write,
    expire_write,
    reassert_write,
    resolve_write,
    served_write,
)
from test_coding_tools import FakeEmbedder, FakeGraph, _extraction, _register, _stub_extract


@pytest.fixture
def mock_ctx():
    from unittest.mock import MagicMock

    return MagicMock()


class TestBuilders:
    def test_new_lesson_carries_lifecycle_defaults_and_returns_its_id(self):
        q, p = anchored_memory_write("Gotcha", {"text": "x", "confidence": 0.8}, "s", "r", [], None, "2026-01-01T00:00:00Z")
        assert "m.valid_from = datetime($ts)" in q
        assert "m.evidence_count = 1" in q
        assert "m.served_count = 0, m.helpful = 0, m.harmful = 0" in q
        assert q.strip().endswith("RETURN DISTINCT elementId(m) AS eid")

    def test_reassert_increments_evidence_and_links_session(self):
        q, p = reassert_write("4:abc:1", "sess", "2026-01-01T00:00:00Z")
        assert "coalesce(m.evidence_count, 1) + 1" in q
        assert "REASSERTED_IN" in q
        assert p == {"eid": "4:abc:1", "session_id": "sess", "ts": "2026-01-01T00:00:00Z"}

    def test_expire_sets_fields_and_supersedes_edge_only_with_successor(self):
        q, p = expire_write("old", "new", "t")
        assert "m.expired_at = datetime($ts)" in q
        assert "SUPERSEDES" in q
        assert p["superseded_by"] == "new"
        q2, p2 = expire_write("old", None, "t")
        assert "SUPERSEDES" not in q2
        assert p2["superseded_by"] is None
        assert "DELETE" not in q and "DELETE" not in q2

    def test_served_write_is_none_for_nothing_served(self):
        assert served_write([], "s", "t") is None
        q, p = served_write(["a", "b"], "s", "t")
        assert "SERVED_TO" in q and "served_count" in q
        assert p["eids"] == ["a", "b"]

    def test_resolve_links_gotchas_and_dead_ends_only(self):
        q, p = resolve_write("s")
        assert "(m:Gotcha OR m:DeadEnd)" in q
        assert "RESOLVED_BY" in q and "c.at >= sv.at" in q
        assert p == {"session_id": "s"}

    @pytest.mark.parametrize("builder", [reassert_write, expire_write, served_write, resolve_write])
    def test_no_argument_interpolation(self, builder):
        args = {
            reassert_write: ("EVIL", "EVIL2", "t"),
            expire_write: ("EVIL", "EVIL2", "t"),
            served_write: (["EVIL"], "EVIL2", "t"),
            resolve_write: ("EVIL",),
        }[builder]
        q, _ = builder(*args)
        assert "EVIL" not in q


def _stub_curator_with(monkeypatch, *, known=(), supersedes=None):
    """Curator: every candidate WRITE, except those named in ``known``
    (index -> existing index) which become ALREADY_KNOWN, and the candidate
    at ``supersedes[0]`` which SUPERSEDES existing ``supersedes[1]``."""
    async def fake_curate(candidates, transcript, existing):
        kept, known_pairs = [], []
        for i, c in enumerate(candidates):
            if i in dict(known):
                known_pairs.append((c, dict(known)[i]))
            elif supersedes and i == supersedes[0]:
                c["supersedes"] = supersedes[1]
                kept.append(c)
            else:
                kept.append(c)
        return {"kept": kept, "counts": {"write": len(kept), "already_known": len(known_pairs),
                                         "supersedes": 1 if supersedes else 0, "not_durable": 0, "unsupported": 0},
                "known": known_pairs}

    async def neighbors(client, repo, embedding):
        return [("4:old:1", "[Gotcha] old lesson"), ("4:old:2", "[Gotcha] older lesson")]

    monkeypatch.setattr("agent_memory_mcp.mcp._coding_tools.curate_coding_memory", fake_curate)
    monkeypatch.setattr("agent_memory_mcp.mcp._coding_tools._neighbors", neighbors)


class ReturningGraph(FakeGraph):
    """Returns an elementId from lesson writes, like Neo4j does."""

    async def execute_write(self, query, params):
        out = await super().execute_write(query, params)
        if "RETURN DISTINCT elementId(m) AS eid" in query:
            return [{"eid": f"4:new:{len(self.writes)}"}]
        return out


class TestCapturePipelineLifecycle:
    async def test_already_known_reasserts_the_existing_node_instead_of_writing(self, monkeypatch, mock_ctx):
        graph = ReturningGraph()
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch, result=_extraction(preferences=[]))
        _stub_curator_with(monkeypatch, known=[(1, 0)])  # 2nd decision == existing 0

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="r", branch="b", transcript="user: hi", files=["a.py", "b.py"],
        ))

        assert result["reasserted"] == 1
        assert result["stored"] == 3
        reassert = [p for q, p in graph.writes if "REASSERTED_IN" in q]
        assert reassert == [{"eid": "4:old:1", "session_id": "s", "ts": reassert[0]["ts"]}]
        assert not any("keep hooks thin" in json.dumps(p) for q, p in graph.writes if "CREATE (m:" in q)

    async def test_supersedes_writes_the_new_lesson_and_expires_the_old(self, monkeypatch, mock_ctx):
        graph = ReturningGraph()
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch, result=_extraction(preferences=[]))
        _stub_curator_with(monkeypatch, supersedes=(2, 1))  # the gotcha replaces existing 1

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="r", branch="b", transcript="user: hi", files=["a.py", "b.py"],
        ))

        assert result["superseded"] == 1
        assert result["stored"] == 4
        expire = [(q, p) for q, p in graph.writes if "expired_at" in q]
        assert len(expire) == 1
        assert expire[0][1]["eid"] == "4:old:2"
        assert expire[0][1]["superseded_by"].startswith("4:new:")
        assert "SUPERSEDES" in expire[0][0]

    async def test_preferences_go_to_the_upstream_store(self, monkeypatch, mock_ctx):
        from unittest.mock import AsyncMock

        graph = FakeGraph()
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch)
        import agent_memory_mcp.mcp._coding_tools as ct

        ct.get_client(mock_ctx).long_term.add_preference = AsyncMock()

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="r", branch="b", transcript="user: hi", files=["a.py"],
        ))

        assert result["preferences"] == 1
        assert result["by_kind"]["CodingPreference"] == 0
        ct.get_client(mock_ctx).long_term.add_preference.assert_awaited_once()


class TestRecallLifecycle:
    def _rows(self):
        return [{
            "eid": "4:l:1", "labels": ["Gotcha", "CodingMemory"],
            "props": {"text": "pin it", "evidence_count": 3, "served_count": 1},
            "files": [], "task": None, "at": "t", "score": 0.7, "anchored": False,
        }]

    async def test_legs_exclude_expired_lessons(self):
        from agent_memory_mcp.mcp._coding_tools import _FULLTEXT_LEG_QUERY, _NEIGHBORS_QUERY, _VECTOR_LEG_QUERY

        for q in (_VECTOR_LEG_QUERY, _FULLTEXT_LEG_QUERY, _NEIGHBORS_QUERY):
            assert "m.expired_at IS NULL" in q

    async def test_served_lessons_are_recorded_and_counters_rendered(self, monkeypatch, mock_ctx):
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "GATE_ENABLED", False)
        graph = FakeGraph(read_results=[self._rows(), []])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](
            mock_ctx, prompt="q", agent_id="agent-1", repo="r", session_id="sess-9",
        ))

        m = result["memories"][0]
        assert m["evidence_count"] == 3 and m["served_count"] == 1
        assert "eid" not in m
        served = [(q, p) for q, p in graph.writes if "SERVED_TO" in q]
        assert len(served) == 1
        assert served[0][1]["eids"] == ["4:l:1"]
        assert served[0][1]["session_id"] == "sess-9"

    async def test_served_write_failure_does_not_break_recall(self, monkeypatch, mock_ctx):
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "GATE_ENABLED", False)
        graph = FakeGraph(read_results=[self._rows(), []], fail_on_write=1)
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        result = json.loads(await tools["coding_recall"](mock_ctx, prompt="q", agent_id="a", repo="r"))
        assert result["memories"][0]["text"] == "pin it"
        assert result["fallback"] is False

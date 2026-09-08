"""Files as a fused retrieval leg instead of a boost (MUD-461, P5 E6).

Sharing an edited file with a lesson explains 21% of labelled relevance at
about 3x lift over chance, but only 11% precision. As a score boost it
measured at zero effect (MUD-403). This gives it its own leg, ranked by
cosine among the anchored lessons, fused by RRF — and a hard cap on how
many of the returned slots lessons found *only* that way may take, since
an anchor on its own is weak evidence.
"""

from unittest.mock import MagicMock

from test_coding_tools import FakeEmbedder, FakeGraph


def _row(eid: str, text: str, score: float = 0.5, files=None):
    return {
        "eid": eid, "labels": ["Gotcha", "CodingMemory"], "props": {"text": text},
        "files": files or [], "task": None, "at": "2026-09-01T00:00:00Z",
        "score": score, "anchored": bool(files),
    }


def _client(graph, embedder=None):
    client = MagicMock()
    client.graph = graph
    client.long_term._embedder = embedder or FakeEmbedder()
    return client


class TestAnchorSlots:
    def test_lessons_found_only_by_the_anchor_leg_are_capped(self):
        """Two slots means two, however many anchored lessons the query
        drags in — the rest of the list stays the ranker's."""
        from agent_memory_mcp.mcp._coding_tools import cap_anchor_slots

        rows = [
            {"eid": "a", "ranks": {2: 0}},
            {"eid": "b", "ranks": {0: 1, 2: 1}},
            {"eid": "c", "ranks": {2: 2}},
            {"eid": "d", "ranks": {2: 3}},
            {"eid": "e", "ranks": {0: 4}},
        ]

        kept = cap_anchor_slots(rows, limit=10, anchor_leg=2, slots=2)

        assert [r["eid"] for r in kept] == ["a", "b", "c", "e"]

    def test_the_cap_still_truncates_to_the_limit(self):
        from agent_memory_mcp.mcp._coding_tools import cap_anchor_slots

        rows = [{"eid": str(i), "ranks": {0: i}} for i in range(5)]

        assert len(cap_anchor_slots(rows, limit=3, anchor_leg=2, slots=2)) == 3


class TestAnchorLegWiring:
    async def test_off_by_default_the_legs_are_vector_and_fulltext(self, monkeypatch):
        from agent_memory_mcp.mcp import _coding_tools as ct

        monkeypatch.setattr(ct, "ANCHOR_LEG_ENABLED", False)
        graph = FakeGraph(read_results=[[_row("v", "vector hit")], [_row("t", "text hit")]])

        rows, strategy = await ct.retrieve_candidates(
            _client(graph), prompt="why does the limiter return 200?", repo="r",
            files=["src/rate_limit.py"], task_key=None, limit=10,
        )

        assert strategy == "fused"
        assert len(graph.reads) == 2
        assert not any("ABOUT" in q and "vector.similarity" in q for q, _ in graph.reads)

    async def test_enabled_it_reads_anchored_lessons_ranked_by_cosine(self, monkeypatch):
        from agent_memory_mcp.mcp import _coding_tools as ct

        monkeypatch.setattr(ct, "ANCHOR_LEG_ENABLED", True)
        graph = FakeGraph(read_results=[
            [_row("v", "vector hit")], [_row("t", "text hit")],
            [_row("a", "anchored hit", files=["src/rate_limit.py"])],
        ])

        rows, strategy = await ct.retrieve_candidates(
            _client(graph), prompt="why does the limiter return 200?", repo="r",
            files=["src/rate_limit.py"], task_key=None, limit=10,
        )

        anchor_query, params = graph.reads[2]
        assert "vector.similarity.cosine" in anchor_query
        assert "ABOUT" in anchor_query
        assert params["files"] == ["src/rate_limit.py"]
        assert {r["eid"] for r in rows} == {"v", "t", "a"}

    async def test_a_query_with_no_files_never_runs_the_leg(self, monkeypatch):
        """The anchor leg's whole input is the edited files; without them
        it would match every lesson in the repo."""
        from agent_memory_mcp.mcp import _coding_tools as ct

        monkeypatch.setattr(ct, "ANCHOR_LEG_ENABLED", True)
        graph = FakeGraph(read_results=[[_row("v", "vector hit")], [_row("t", "text hit")]])

        await ct.retrieve_candidates(
            _client(graph), prompt="what changed?", repo="r", files=[],
            task_key=None, limit=10,
        )

        assert len(graph.reads) == 2

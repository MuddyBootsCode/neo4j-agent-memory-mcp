"""Hypothetical lessons as extra fused legs (MUD-459, P5 E4).

The stored text is answer-shaped and a prompt is a question, so HyDE
embeds a guessed answer instead. Each guess is its own vector leg fused by
RRF — never concatenated into the prompt, because the situation card
(MUD-458) measured what more text in the query string costs.
"""

from unittest.mock import MagicMock

from test_coding_tools import FakeEmbedder, FakeGraph


def _row(eid: str, text: str, score: float = 0.5):
    return {
        "eid": eid, "labels": ["Gotcha", "CodingMemory"], "props": {"text": text},
        "files": [], "task": None, "at": "2026-09-01T00:00:00Z",
        "score": score, "anchored": False,
    }


def _client(graph):
    client = MagicMock()
    client.graph = graph
    client.long_term._embedder = FakeEmbedder()
    return client


class TestExtraEmbeddingLegs:
    async def test_each_extra_embedding_becomes_its_own_vector_leg(self, monkeypatch):
        from agent_memory_mcp.mcp import _coding_tools as ct

        monkeypatch.setattr(ct, "ANCHOR_LEG_ENABLED", False)
        graph = FakeGraph(read_results=[
            [_row("v", "vector hit")], [_row("t", "text hit")],
            [_row("h1", "first guess")], [_row("h2", "second guess")],
        ])

        rows, strategy = await ct.retrieve_candidates(
            _client(graph), prompt="why does the deploy fail?", repo="r", files=[],
            task_key=None, limit=10, extra_embeddings=[[0.2] * 4, [0.3] * 4],
        )

        assert len(graph.reads) == 4
        assert {r["eid"] for r in rows} == {"v", "t", "h1", "h2"}
        assert strategy == "fused"

    async def test_no_extra_embeddings_leaves_the_legs_alone(self, monkeypatch):
        from agent_memory_mcp.mcp import _coding_tools as ct

        monkeypatch.setattr(ct, "ANCHOR_LEG_ENABLED", False)
        graph = FakeGraph(read_results=[[_row("v", "vector hit")], [_row("t", "text hit")]])

        await ct.retrieve_candidates(
            _client(graph), prompt="why does the deploy fail?", repo="r", files=[],
            task_key=None, limit=10, extra_embeddings=[],
        )

        assert len(graph.reads) == 2

    async def test_a_guess_alone_cannot_outrank_agreement(self, monkeypatch):
        """RRF's point: a lesson two legs found beats one only a guess did,
        even when the guess ranked it first."""
        from agent_memory_mcp.mcp import _coding_tools as ct

        monkeypatch.setattr(ct, "ANCHOR_LEG_ENABLED", False)
        graph = FakeGraph(read_results=[
            [_row("x", "second on the vector leg"), _row("agreed", "agreed")],
            [_row("agreed", "agreed")],
            [_row("guessed", "only the guess found it"), _row("x", "x")],
        ])

        rows, _ = await ct.retrieve_candidates(
            _client(graph), prompt="p", repo="r", files=[], task_key=None,
            limit=10, extra_embeddings=[[0.2] * 4],
        )

        assert rows[0]["eid"] == "agreed"

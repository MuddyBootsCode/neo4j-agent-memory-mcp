"""The outcome loop: served lessons are rated, counted, and ranked (MUD-407).

P2 recorded what recall served. This is what came of it: a session-end pass
rates each served lesson from the transcript, the counters move by an EMA,
the ranker reads them, and a lesson enough sessions have asserted is shown
as a guardrail rather than a tip.
"""

import json

import pytest

from agent_memory_mcp.capture.cypher import (
    OUTCOME_ALPHA,
    OUTCOME_SEED,
    outcome_write,
    served_lessons_read,
)
from agent_memory_mcp.extraction.coding import rate_served_lessons
from test_coding_tools import FakeEmbedder, FakeGraph, _extraction, _register, _stub_extract


@pytest.fixture
def mock_ctx():
    from unittest.mock import MagicMock

    return MagicMock()


class TestBuilders:
    def test_served_read_skips_already_rated_servings(self):
        q, p = served_lessons_read("sess")
        assert "SERVED_TO" in q and "sv.rated_at IS NULL" in q
        assert p == {"session_id": "sess", "limit": 40}

    def test_outcome_write_is_none_for_nothing_rated(self):
        assert outcome_write([], "s", "t") is None

    def test_outcome_write_moves_counts_weight_and_marks_the_edge(self):
        rating = {"eid": "4:a:1", "helpful": True, "reason": "used the flag"}
        q, p = outcome_write([rating], "s", "t")
        assert "coalesce(m.helpful, 0) + CASE WHEN rating.helpful THEN 1 ELSE 0 END" in q
        assert "coalesce(m.harmful, 0) + CASE WHEN rating.helpful THEN 0 ELSE 1 END" in q
        assert "m.outcome_weight" in q and "$alpha" in q
        # Idempotent per serving: a re-run finds the edge rated and skips it.
        assert "sv.rated_at = datetime($ts)" in q
        # MUD-427: the judge's reason lands on the rated edge.
        assert "sv.reason = rating.reason" in q
        assert p["alpha"] == OUTCOME_ALPHA and p["seed"] == OUTCOME_SEED
        assert p["ratings"] == [rating]

    @pytest.mark.parametrize("builder", [served_lessons_read, outcome_write])
    def test_no_argument_interpolation(self, builder):
        args = {
            served_lessons_read: ("EVIL",),
            outcome_write: ([{"eid": "EVIL", "helpful": True}], "EVIL2", "t"),
        }[builder]
        q, _ = builder(*args)
        assert "EVIL" not in q

    def test_ema_formula_matches_the_ticket(self):
        """w + alpha(r - w), seeded neutral: helpful climbs, harmful falls,
        and neither saturates on one rating."""
        def step(w, helpful):
            return w + OUTCOME_ALPHA * ((1.0 if helpful else 0.0) - w)

        w = OUTCOME_SEED
        assert step(w, True) == pytest.approx(0.65)
        assert step(w, False) == pytest.approx(0.35)
        for _ in range(20):
            w = step(w, True)
        assert 0.99 < w < 1.0


class TestRater:
    async def _rate(self, monkeypatch, verdicts, lessons=("a", "b", "c")):
        import agent_memory_mcp.baml_client.async_client as ac

        class FakeVerdict:
            def __init__(self, id, outcome, reason=None):
                self.id, self.outcome, self.reason = id, outcome, reason

        class FakeRatings:
            def __init__(self, verdicts):
                self.verdicts = verdicts

        async def fake_rate(lessons, transcript, baml_options=None):
            return FakeRatings([FakeVerdict(*v) for v in verdicts])

        monkeypatch.setattr(ac.b, "RateServedLessons", fake_rate, raising=False)
        return await rate_served_lessons(list(lessons), "transcript text")

    async def test_maps_outcomes_to_helpful_harmful_and_unused(self, monkeypatch):
        out = await self._rate(monkeypatch, [(0, "HELPFUL"), (1, "HARMFUL"), (2, "UNUSED")])
        assert out == [(True, None), (False, None), (None, None)]

    async def test_reasons_ride_along_with_the_verdicts(self, monkeypatch):
        """MUD-427: the judge's citation comes back with each rating."""
        out = await self._rate(monkeypatch, [
            (0, "HELPFUL", "used the flag"),
            (1, "HARMFUL", "sent the session down the asyncpg path"),
            (2, "UNUSED", "transcript is about the deploy"),
        ])
        assert out == [
            (True, "used the flag"),
            (False, "sent the session down the asyncpg path"),
            (None, None),  # unused stays unrated; its reason moves nothing
        ]

    async def test_out_of_range_and_missing_ids_stay_unrated(self, monkeypatch):
        out = await self._rate(monkeypatch, [(0, "HELPFUL"), (7, "HARMFUL")])
        assert out == [(True, None), (None, None), (None, None)]

    async def test_rater_failure_leaves_every_lesson_unrated(self, monkeypatch):
        import agent_memory_mcp.baml_client.async_client as ac

        async def boom(lessons, transcript, baml_options=None):
            raise RuntimeError("model down")

        monkeypatch.setattr(ac.b, "RateServedLessons", boom, raising=False)
        assert await rate_served_lessons(["a", "b"], "t") == [(None, None), (None, None)]

    async def test_kill_switch_and_empty_input_skip_the_call(self, monkeypatch):
        monkeypatch.setenv("NAM_OUTCOME_RATER", "off")
        assert await rate_served_lessons(["a"], "t") == [(None, None)]
        monkeypatch.delenv("NAM_OUTCOME_RATER")
        assert await rate_served_lessons([], "t") == []
        assert await rate_served_lessons(["a"], "   ") == [(None, None)]


def _stub_rater(monkeypatch, verdicts):
    async def fake_rate(lessons, transcript, trace_meta=None):
        return [(v, None) if not isinstance(v, tuple) else v for v in verdicts]

    monkeypatch.setattr("agent_memory_mcp.mcp._coding_tools.rate_served_lessons", fake_rate)


def _served_rows():
    return [
        {"eid": "4:l:1", "labels": ["Gotcha", "CodingMemory"], "props": {"text": "pin it", "confidence": 0.8}},
        {"eid": "4:l:2", "labels": ["Decision", "CodingMemory"],
         "props": {"text": "use asyncpg", "reason": "blocks the loop", "confidence": 0.9}},
    ]


class TestSessionEndPass:
    async def test_rates_served_lessons_and_writes_the_counters(self, monkeypatch, mock_ctx):
        graph = FakeGraph(read_results=[_served_rows()])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch, result=_extraction(decisions=[], gotchas=[], dead_ends=[], preferences=[]))
        _stub_rater(monkeypatch, [(True, "pinned it"), (False, "reverted the choice")])

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="r", branch="b",
            transcript="user: hi", files=["a.py"],
        ))

        assert result["rated"] == {"served": 2, "helpful": 1, "harmful": 1, "unused": 0}
        writes = [(q, p) for q, p in graph.writes if "outcome_weight" in q]
        assert len(writes) == 1
        assert writes[0][1]["ratings"] == [
            {"eid": "4:l:1", "helpful": True, "reason": "pinned it"},
            {"eid": "4:l:2", "helpful": False, "reason": "reverted the choice"},
        ]

    async def test_unused_lessons_are_not_written_as_harmful(self, monkeypatch, mock_ctx):
        graph = FakeGraph(read_results=[_served_rows()])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch, result=_extraction(decisions=[], gotchas=[], dead_ends=[], preferences=[]))
        _stub_rater(monkeypatch, [None, None])

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="r", branch="b",
            transcript="user: hi", files=["a.py"],
        ))

        assert result["rated"] == {"served": 2, "helpful": 0, "harmful": 0, "unused": 2}
        assert not [q for q, _ in graph.writes if "outcome_weight" in q]

    async def test_nothing_served_costs_no_model_call(self, monkeypatch, mock_ctx):
        called = []

        async def fake_rate(lessons, transcript, trace_meta=None):
            called.append(lessons)
            return [(None, None)] * len(lessons)

        monkeypatch.setattr("agent_memory_mcp.mcp._coding_tools.rate_served_lessons", fake_rate)
        graph = FakeGraph(read_results=[[]])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch, result=_extraction(decisions=[], gotchas=[], dead_ends=[], preferences=[]))

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="r", branch="b",
            transcript="user: hi", files=["a.py"],
        ))

        assert result["rated"]["served"] == 0
        assert called == []

    async def test_rating_failure_costs_the_ratings_only(self, monkeypatch, mock_ctx):
        """Capture has already committed by the time the pass runs, so a
        rater failure must not turn a successful capture into an error."""
        graph = FakeGraph(read_results=[_served_rows()])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch, result=_extraction(decisions=[], gotchas=[], dead_ends=[], preferences=[]))

        async def boom(lessons, transcript, trace_meta=None):
            raise RuntimeError("rater down")

        monkeypatch.setattr("agent_memory_mcp.mcp._coding_tools.rate_served_lessons", boom)

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="r", branch="b",
            transcript="user: hi", files=["a.py"],
        ))

        assert "error" not in result
        assert result["rated"] == {"served": 2, "helpful": 0, "harmful": 0, "unused": 2}
        assert not [q for q, _ in graph.writes if "outcome_weight" in q]

    async def test_write_failure_leaves_the_counters_untouched(self, monkeypatch, mock_ctx):
        graph = FakeGraph(read_results=[_served_rows()], fail_on_write=1)
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())
        _stub_extract(monkeypatch, result=_extraction(decisions=[], gotchas=[], dead_ends=[], preferences=[]))
        _stub_rater(monkeypatch, [True, True])

        result = json.loads(await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="r", branch="b",
            transcript="user: hi", files=["a.py"],
        ))

        assert "error" not in result
        assert result["rated"] == {"served": 2, "helpful": 0, "harmful": 0, "unused": 2}


class TestOutcomePrior:
    def test_no_history_ranks_exactly_as_before(self):
        from agent_memory_mcp.mcp._coding_tools import outcome_prior

        assert outcome_prior({}) == 1.0
        assert outcome_prior(None) == 1.0

    def test_proven_lessons_gain_and_harmful_ones_lose(self):
        from agent_memory_mcp.mcp._coding_tools import outcome_prior

        proven = outcome_prior({"outcome_weight": 0.95, "evidence_count": 6})
        neutral = outcome_prior({"outcome_weight": 0.5, "evidence_count": 1})
        harmful = outcome_prior({"outcome_weight": 0.05, "evidence_count": 1})
        assert proven > neutral > harmful
        assert harmful > 0

    def test_influence_is_non_zero_by_default(self):
        from agent_memory_mcp.mcp import _coding_tools as ct

        assert ct.OUTCOME_WEIGHT > 0 and ct.EVIDENCE_WEIGHT > 0

    def test_malformed_counters_do_not_break_the_ranker(self):
        from agent_memory_mcp.mcp._coding_tools import outcome_prior

        assert outcome_prior({"outcome_weight": "junk"}) == 1.0
        assert outcome_prior({"evidence_count": None, "outcome_weight": None}) == 1.0

    def test_fusion_applies_the_prior_and_can_reorder_a_tie(self):
        from agent_memory_mcp.mcp._coding_tools import rrf_fuse

        leg = [
            {"eid": "cold", "props": {}},
            {"eid": "proven", "props": {"outcome_weight": 0.95, "evidence_count": 5}},
        ]
        fused = rrf_fuse([leg, list(reversed(leg))])
        assert [r["eid"] for r in fused] == ["proven", "cold"]
        assert fused[0]["prior"] > 1.0 and fused[1]["prior"] == 1.0


class TestGuardrailSurfacing:
    def test_repeatedly_asserted_lessons_are_marked(self):
        from agent_memory_mcp.hook.recall_hook import format_coding_memories

        out = format_coding_memories([
            {"kind": "Gotcha", "text": "set NAM_TEST_DB", "evidence_count": 4},
            {"kind": "Gotcha", "text": "one-off", "evidence_count": 2},
            {"kind": "Decision", "text": "no count"},
        ])
        lines = out.splitlines()
        assert lines[0] == "[gotcha · guardrail] set NAM_TEST_DB"
        assert lines[1] == "[gotcha] one-off"
        assert lines[2] == "[decision] no count"

    def test_malformed_evidence_is_not_a_guardrail(self):
        from agent_memory_mcp.hook.recall_hook import format_coding_memories

        out = format_coding_memories([{"kind": "Gotcha", "text": "x", "evidence_count": "many"}])
        assert out == "[gotcha] x"


class TestServedSessionMerge:
    """The hook calls coding_recall before record_coding_activity, so the
    session node may not exist when recall records what it served."""

    def test_served_write_merges_the_session_with_its_repo(self):
        from agent_memory_mcp.capture.cypher import served_write

        q, p = served_write(["4:a:1"], "s", "t", "my-repo")
        assert "MERGE (s:CodingSession {id: $session_id})" in q
        assert "MATCH (s:CodingSession" not in q
        assert "ON CREATE SET s.repo = $repo" in q
        assert p["repo"] == "my-repo"

    async def test_first_prompt_of_a_session_still_records_what_it_served(
        self, monkeypatch, mock_ctx
    ):
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "GATE_ENABLED", False)
        rows = [{
            "eid": "4:l:1", "labels": ["Gotcha", "CodingMemory"],
            "props": {"text": "pin it"}, "files": [], "task": None,
            "at": "t", "score": 0.7, "anchored": False,
        }]
        graph = FakeGraph(read_results=[rows, []])
        tools = _register(monkeypatch, graph, embedder=FakeEmbedder())

        await tools["coding_recall"](
            mock_ctx, prompt="q", agent_id="a", repo="my-repo", session_id="brand-new",
        )

        served = [(q, p) for q, p in graph.writes if "SERVED_TO" in q]
        assert len(served) == 1
        assert served[0][1]["session_id"] == "brand-new"
        assert served[0][1]["repo"] == "my-repo"

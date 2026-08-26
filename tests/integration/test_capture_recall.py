"""Integration test: capture_session_memory → coding_recall round trip.

The missing loop in the coding-memory pivot: a session-end capture (real
BAML/Bedrock extraction over a transcript) followed by a recall anchored to
the same files must surface the captured decision and gotcha — and must NOT
surface the captured preference, because anchor-first recall serves only
Decision/Gotcha/DeadEnd.

Requires a running Neo4j + Bedrock (see tests/integration/conftest.py).
Costs one live Bedrock extraction call.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

REPO = "capture-recall-int-repo"
DECISION_FILE = "src/agent_memory_mcp/mcp/_coding_tools.py"
GOTCHA_FILE = "tests/integration/conftest.py"
FILES = [DECISION_FILE, GOTCHA_FILE]

TRANSCRIPT = f"""\
User: Let's wire the overlap query for the recall hook.

Agent: I compared two shapes for the overlap query in {DECISION_FILE}.
I chose to anchor the traversal from the CodeFile nodes instead of scanning
every agent-session pair, because the anchored traversal stays fast as the
number of agents grows.

Agent: One gotcha cost me an hour here: the integration tests fail unless
NAM_TEST_DB is set to a dedicated database name — {GOTCHA_FILE} drops and
recreates whatever database it is pointed at, so pointing it at a shared
database wipes it.

User: Good to know. By the way, I prefer integration tests over mocked unit
tests for anything that touches the database.
"""


def _tool(mcp, name):
    for t in mcp._tool_manager._tools.values():
        if t.name == name:
            return t.fn
    raise AssertionError(f"tool {name} not registered")


@pytest.fixture
def coding_tools(memory_client, monkeypatch):
    """Register the coding-memory MCP tools wired to the real test-DB client."""
    from fastmcp import FastMCP

    from agent_memory_mcp.mcp._coding_tools import register_coding_tools

    monkeypatch.setattr(
        "agent_memory_mcp.mcp._coding_tools.get_client", lambda _ctx: memory_client
    )
    mcp = FastMCP("capture-recall-int")
    register_coding_tools(mcp)
    return mcp, MagicMock()


async def test_capture_then_recall_round_trip(coding_tools):
    """One live extraction: capture a decision, gotcha, and preference, then
    recall by the anchor files and get the decision and gotcha back."""
    mcp, ctx = coding_tools

    captured = json.loads(
        await _tool(mcp, "capture_session_memory")(
            ctx,
            agent_id="agent-capture",
            session_id="sess-capture-1",
            repo=REPO,
            branch="feature/recall-hook",
            transcript=TRANSCRIPT,
            task_key="MUD-395",
            files=FILES,
        )
    )
    assert "error" not in captured, captured
    assert captured["stored"] >= 2, captured
    assert captured["by_kind"]["Decision"] >= 1, captured
    assert captured["by_kind"]["Gotcha"] >= 1, captured
    assert captured["by_kind"]["CodingPreference"] >= 1, captured
    assert "anchor_rate" in captured
    # Decisions/gotchas were extracted, so the rate is a real number.
    assert captured["anchor_rate"] is not None
    assert 0.0 < captured["anchor_rate"] <= 1.0

    recalled = json.loads(
        await _tool(mcp, "coding_recall")(
            ctx,
            prompt="anything I should know before touching these files?",
            agent_id="agent-recall",
            repo=REPO,
            files=FILES,
            task_key="MUD-395",
        )
    )
    assert "error" not in recalled, recalled
    assert recalled["fallback"] is False
    memories = recalled["memories"]
    assert memories, "capture stored memories but recall returned none"

    # Recall serves only the anchored kinds — never CodingPreference.
    assert all(m["kind"] in ("Decision", "Gotcha", "DeadEnd") for m in memories), (
        f"unexpected kinds in recall: {[m['kind'] for m in memories]}"
    )
    all_text = " ".join(m["text"].lower() for m in memories)
    assert "mocked unit tests" not in all_text, (
        "the captured preference leaked into anchor-first recall"
    )

    # Every recalled anchor must be a file this session actually passed.
    for m in memories:
        assert set(m["files"]) <= set(FILES), m

    decisions = [m for m in memories if m["kind"] == "Decision"]
    assert decisions, f"decision missing from recall: {memories}"
    decision_text = " ".join(d["text"].lower() for d in decisions)
    assert "codefile" in decision_text or "anchor" in decision_text, decisions

    gotchas = [m for m in memories if m["kind"] == "Gotcha"]
    assert gotchas, f"gotcha missing from recall: {memories}"
    gotcha_text = " ".join(g["text"].lower() for g in gotchas)
    assert "nam_test_db" in gotcha_text, gotchas

    # At least one recalled memory is anchored to a file (not only to the
    # task) — the transcript names both files explicitly.
    assert any(m["files"] for m in memories), memories


async def test_served_lessons_are_rated_and_counters_move(coding_tools, memory_client, monkeypatch):
    """The outcome loop against a real database (MUD-407): recall serves a
    lesson, the session-end pass rates it, and the counters move by the EMA.

    The rater is stubbed — what is under test is the Cypher, not the model:
    the CASE arms, the EMA arithmetic, and the rated_at stamp that makes a
    second capture of the same session a no-op.
    """
    from agent_memory_mcp.capture.cypher import OUTCOME_ALPHA, OUTCOME_SEED

    mcp, ctx = coding_tools
    session = "sess-outcome-1"

    await _tool(mcp, "capture_session_memory")(
        ctx, agent_id="agent-outcome", session_id="sess-outcome-seed", repo=REPO,
        branch="main", transcript=TRANSCRIPT, task_key="MUD-407", files=FILES,
    )
    recalled = json.loads(
        await _tool(mcp, "coding_recall")(
            ctx, prompt="what breaks when I point the tests at a shared database?",
            agent_id="agent-outcome", repo=REPO, files=FILES, session_id=session,
        )
    )
    assert recalled["memories"], recalled

    async def all_helpful(lessons, transcript):
        return [True] * len(lessons)

    monkeypatch.setattr("agent_memory_mcp.mcp._coding_tools.rate_served_lessons", all_helpful)

    captured = json.loads(
        await _tool(mcp, "capture_session_memory")(
            ctx, agent_id="agent-outcome", session_id=session, repo=REPO,
            branch="main", transcript=TRANSCRIPT, task_key="MUD-407", files=FILES,
        )
    )
    served = captured["rated"]["served"]
    assert served == len(recalled["memories"]), captured
    assert captured["rated"]["helpful"] == served
    assert captured["rated"]["harmful"] == 0

    rows = await memory_client.graph.execute_read(
        """
        MATCH (m)-[sv:SERVED_TO]->(:CodingSession {id: $session})
        RETURN m.helpful AS helpful, m.harmful AS harmful,
               m.outcome_weight AS weight, sv.rated_at IS NOT NULL AS rated
        """,
        {"session": session},
    )
    assert len(rows) == served, rows
    expected = OUTCOME_SEED + OUTCOME_ALPHA * (1.0 - OUTCOME_SEED)
    for row in rows:
        assert row["helpful"] == 1 and row["harmful"] == 0, row
        assert row["weight"] == pytest.approx(expected), row
        assert row["rated"] is True, row

    # A re-run finds every serving already rated and must not count it twice.
    again = json.loads(
        await _tool(mcp, "capture_session_memory")(
            ctx, agent_id="agent-outcome", session_id=session, repo=REPO,
            branch="main", transcript=TRANSCRIPT, task_key="MUD-407", files=FILES,
        )
    )
    assert again["rated"] == {"served": 0, "helpful": 0, "harmful": 0, "unused": 0}, again
    rows = await memory_client.graph.execute_read(
        "MATCH (m)-[:SERVED_TO]->(:CodingSession {id: $session}) RETURN m.helpful AS helpful",
        {"session": session},
    )
    assert {row["helpful"] for row in rows} == {1}, rows

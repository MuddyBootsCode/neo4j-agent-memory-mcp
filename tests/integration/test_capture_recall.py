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

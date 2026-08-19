"""Integration tests for cross-agent overlap detection (coding_recall).

Drives the real ``record_coding_activity`` / ``coding_recall`` tool
functions against the test database: one agent's recorded activity must
surface as an overlap warning for another agent in the same repo, with
correct file intersection, self-exclusion, task-only overlap via
WORKING_ON, and expiry past the recency window.

Requires a running Neo4j (see tests/integration/conftest.py).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agent_memory_mcp.capture.cypher import editing_upsert, session_upsert

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

REPO = "overlap-int-repo"


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
    mcp = FastMCP("overlap-int")
    register_coding_tools(mcp)
    return mcp, MagicMock()


async def _record(tools, agent, session, files=None, task=None, branch="main"):
    mcp, ctx = tools
    result = json.loads(
        await _tool(mcp, "record_coding_activity")(
            ctx,
            agent_id=agent,
            session_id=session,
            repo=REPO,
            branch=branch,
            task_key=task,
            edited_files=list(files or []),
            commits=[],
        )
    )
    assert "error" not in result, result
    return result


async def _recall(tools, agent, files=None, task=None, window=24.0):
    mcp, ctx = tools
    result = json.loads(
        await _tool(mcp, "coding_recall")(
            ctx,
            prompt="does anything overlap?",
            agent_id=agent,
            repo=REPO,
            files=list(files or []),
            task_key=task,
            overlap_window_hours=window,
        )
    )
    assert "error" not in result, result
    return result


async def test_other_agent_sees_overlap_with_intersecting_files_only(coding_tools):
    """Agent B sees agent A's overlap listing only the shared files."""
    await _record(
        coding_tools,
        "agent-a",
        "sess-a1",
        files=["src/shared.py", "src/only_a.py"],
    )

    result = await _recall(
        coding_tools, "agent-b", files=["src/shared.py", "src/only_b.py"]
    )

    assert result["fallback"] is False
    assert len(result["overlaps"]) == 1
    row = result["overlaps"][0]
    assert row["agent"] == "agent-a"
    assert row["files"] == ["src/shared.py"]
    assert row["task"] is None
    assert row["last_seen"]


async def test_agent_does_not_see_its_own_sessions(coding_tools):
    """Self-exclusion: an agent recalling its own files gets no overlap."""
    files = ["src/self.py"]
    await _record(coding_tools, "agent-a", "sess-a1", files=files)

    result = await _recall(coding_tools, "agent-a", files=files)

    assert result["fallback"] is False
    assert result["overlaps"] == []


async def test_task_only_overlap_via_working_on(coding_tools):
    """Two agents on the same task overlap even with no shared files."""
    await _record(coding_tools, "agent-a", "sess-a1", task="MUD-395")

    result = await _recall(coding_tools, "agent-b", task="MUD-395")

    assert result["fallback"] is False
    assert len(result["overlaps"]) == 1
    row = result["overlaps"][0]
    assert row["agent"] == "agent-a"
    assert row["files"] == []
    assert row["task"] == "MUD-395"


async def test_stale_editing_edge_outside_window_is_excluded(
    coding_tools, memory_client
):
    """An EDITING edge older than the window does not warn.

    The stale edge is seeded through the pure builders with an old ISO
    timestamp — record_coding_activity always stamps now, so it cannot
    write history.
    """
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()

    query, params = session_upsert(
        "agent-old", "sess-old", REPO, "main", None, stale_ts
    )
    await memory_client.graph.execute_write(query, params)
    built = editing_upsert("sess-old", REPO, ["src/stale.py"], stale_ts)
    assert built is not None
    query, params = built
    await memory_client.graph.execute_write(query, params)

    # Inside a wide window the seeded edge is visible — proves the seed
    # is well-formed and the exclusion below is the window, not a typo.
    wide = await _recall(coding_tools, "agent-b", files=["src/stale.py"], window=72.0)
    assert [row["agent"] for row in wide["overlaps"]] == ["agent-old"]

    narrow = await _recall(coding_tools, "agent-b", files=["src/stale.py"], window=24.0)
    assert narrow["overlaps"] == []

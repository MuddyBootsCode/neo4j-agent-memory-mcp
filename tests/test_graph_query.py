"""Tests for graph_query read-only enforcement.

Covers the two-layer guard on the graph_query tool:
1. _is_read_only_query — a cheap first-pass keyword reject (defense-in-depth).
2. _execute_read_only — the authoritative guard: runs the query in a genuine
   READ-access-mode transaction so the Neo4j server itself rejects writes,
   including write-mode APOC procedures the keyword denylist cannot catch.

The keyword pre-check has been shipping untested; the transaction wiring is new.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_memory_mcp.mcp._tools import _execute_read_only, _is_read_only_query


def get_tool_fn(mcp, tool_name):
    """Extract a registered tool function from a FastMCP instance."""
    for tool in mcp._tool_manager._tools.values():
        if tool.name == tool_name:
            return tool.fn
    raise ValueError(f"Tool '{tool_name}' not found")


# ── _is_read_only_query: first-pass keyword reject ───────────────────


class TestIsReadOnlyQuery:
    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (n) RETURN n",
            "MATCH (a:Entity)-[r:RELATED_TO]->(b) RETURN a, r, b",
            "RETURN 1 AS n",
            # Read procedures are intentionally allowed
            "CALL db.index.vector.queryNodes('idx', 5, $vec) YIELD node RETURN node",
            "CALL apoc.meta.data() YIELD label RETURN label",
        ],
    )
    def test_allows_reads(self, query):
        assert _is_read_only_query(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "CREATE (n:Person) RETURN n",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n) SET n.x = 1",
            "MERGE (n:Person {name: 'A'})",
            "MATCH (n) REMOVE n.x",
            "DROP INDEX idx",
            "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
            "MATCH (n) FOREACH (x IN [1] | SET n.y = x)",
            "CALL { CREATE (n) } RETURN 1",
            "MATCH (n) CALL { WITH n CREATE (:Copy) } IN TRANSACTIONS",
            # lowercase still caught (query is uppercased before matching)
            "create (n:Person) return n",
        ],
    )
    def test_blocks_writes(self, query):
        assert _is_read_only_query(query) is False

    def test_documents_apoc_write_gap(self):
        """The keyword denylist does NOT catch write-mode APOC procedures.

        Passing the write Cypher as a bound parameter keeps every write keyword
        out of the query text, so the pre-check waves apoc.cypher.doIt through.
        This is the exact gap the READ-access-mode transaction closes, so
        _execute_read_only must be the real guard (verified against a live
        server in the integration test).
        """
        payload = "CALL apoc.cypher.doIt($q, {}) YIELD value RETURN value"
        assert _is_read_only_query(payload) is True


# ── _execute_read_only: authoritative read-transaction guard ─────────


def _mock_client_with_read_session(rows):
    """Build a mock memory client whose graph runs work via a read transaction.

    Returns (client, tx, session, driver) so tests can assert on the wiring.
    """
    tx = MagicMock()
    result = MagicMock()
    result.data = AsyncMock(return_value=rows)
    tx.run = AsyncMock(return_value=result)

    session = MagicMock()

    async def _fake_execute_read(work):
        # Exercise the actual _work closure passed by _execute_read_only.
        return await work(tx)

    session.execute_read = AsyncMock(side_effect=_fake_execute_read)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session = MagicMock(return_value=session_cm)

    client = MagicMock()
    client.graph._ensure_connected = MagicMock(return_value=driver)
    client.graph._config.database = "meetings"

    return client, tx, session, driver


@pytest.mark.asyncio
class TestExecuteReadOnly:
    async def test_runs_in_read_transaction(self):
        client, tx, session, driver = _mock_client_with_read_session([{"n": 1}])

        rows = await _execute_read_only(client, "RETURN 1 AS n", {"p": 2})

        assert rows == [{"n": 1}]
        # Uses a managed READ transaction, not execute_write or a bare session.run.
        session.execute_read.assert_awaited_once()
        assert not hasattr(session, "execute_write") or not session.execute_write.called
        # The query + params reach the transaction as bound parameters.
        tx.run.assert_awaited_once_with("RETURN 1 AS n", {"p": 2})

    async def test_targets_the_clients_database(self):
        client, _tx, _session, driver = _mock_client_with_read_session([])

        await _execute_read_only(client, "RETURN 1", None)

        driver.session.assert_called_once_with(database="meetings")

    async def test_none_parameters_default_to_empty_dict(self):
        client, tx, _session, _driver = _mock_client_with_read_session([])

        await _execute_read_only(client, "RETURN 1", None)

        tx.run.assert_awaited_once_with("RETURN 1", {})


# ── graph_query tool: pre-check + delegation ─────────────────────────


@pytest.mark.asyncio
class TestGraphQueryTool:
    async def test_write_query_rejected_by_precheck(self, mcp_app, mock_mcp_context, monkeypatch):
        import json

        ctx, _client = mock_mcp_context

        # If the pre-check fails to block, this would be called — assert it isn't.
        sentinel = AsyncMock(return_value=[])
        monkeypatch.setattr(
            "agent_memory_mcp.mcp._tools._execute_read_only", sentinel
        )

        graph_query = get_tool_fn(mcp_app, "graph_query")
        result = json.loads(await graph_query(ctx, query="CREATE (n:Person) RETURN n"))

        assert "error" in result
        sentinel.assert_not_awaited()

    async def test_read_query_delegates_to_read_only_executor(
        self, mcp_app, mock_mcp_context, monkeypatch
    ):
        import json

        ctx, _client = mock_mcp_context

        executor = AsyncMock(return_value=[{"name": "Alice"}])
        monkeypatch.setattr(
            "agent_memory_mcp.mcp._tools._execute_read_only", executor
        )

        graph_query = get_tool_fn(mcp_app, "graph_query")
        result = json.loads(
            await graph_query(ctx, query="MATCH (n:Entity) RETURN n.name AS name")
        )

        assert result["success"] is True
        assert result["row_count"] == 1
        assert result["rows"] == [{"name": "Alice"}]
        executor.assert_awaited_once()

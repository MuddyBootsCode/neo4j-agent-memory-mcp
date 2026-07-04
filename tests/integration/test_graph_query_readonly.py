"""Integration tests: graph_query read-only enforcement against a live Neo4j.

These verify the claim the unit tests cannot: that running a query in a
READ-access-mode transaction (_execute_read_only) makes the *server* reject
writes — including write-mode APOC procedures that the keyword pre-check
(_is_read_only_query) deliberately lets through.

Requires a running Neo4j (see tests/integration/conftest.py).
"""

import pytest

from agent_memory_mcp.mcp._tools import _execute_read_only, _is_read_only_query

pytestmark = pytest.mark.asyncio


async def _apoc_available(client) -> bool:
    try:
        await _execute_read_only(
            client, "RETURN apoc.version() AS v", {}
        )
        return True
    except Exception:
        return False


async def _count_label(cypher_session, label: str) -> int:
    rows = await cypher_session.execute_read(
        f"MATCH (n:{label}) RETURN count(n) AS c", {}
    )
    return rows[0]["c"]


async def test_read_only_allows_reads(memory_client):
    rows = await _execute_read_only(memory_client, "RETURN 1 AS n", {})
    assert rows == [{"n": 1}]


async def test_read_only_rejects_plain_write(memory_client, cypher_session):
    """A CREATE submitted through the read executor is rejected by the server."""
    with pytest.raises(Exception):
        await _execute_read_only(
            memory_client, "CREATE (n:PwnedPlain) RETURN n", {}
        )
    # The write did not land.
    assert await _count_label(cypher_session, "PwnedPlain") == 0


async def test_read_only_rejects_apoc_write(memory_client, cypher_session):
    """The case the regex misses: apoc.cypher.doIt is a WRITE-mode procedure.

    The keyword pre-check passes it, but the READ-access-mode transaction makes
    the server reject it — so nothing is written.
    """
    if not await _apoc_available(memory_client):
        pytest.skip("APOC not installed on this Neo4j")

    # Passing the write Cypher as a bound parameter keeps every write keyword
    # out of the query text, so the cheap pre-check does NOT catch it.
    payload = "CALL apoc.cypher.doIt($q, {}) YIELD value RETURN value"
    params = {"q": "CREATE (n:PwnedApoc) RETURN n"}
    assert _is_read_only_query(payload) is True

    with pytest.raises(Exception):
        await _execute_read_only(memory_client, payload, params)

    assert await _count_label(cypher_session, "PwnedApoc") == 0

"""Tests for the optional dedicated read-only Neo4j account (RBAC).

Defense-in-depth on top of the READ-access-mode transaction in
_execute_read_only (see tests/test_graph_query.py): if a dedicated read-only
Neo4j account is configured via NAM_NEO4J_READONLY_USERNAME /
NAM_NEO4J_READONLY_PASSWORD, the graph_query path authenticates as that
least-privilege account instead of the primary read-write credentials.

Provisioning the account itself is a docs/deploy deliverable (see README) —
these tests only cover the config + driver-selection plumbing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neo4j_agent_memory.mcp import _tools


@pytest.fixture(autouse=True)
def _reset_readonly_driver_cache(monkeypatch):
    """Ensure each test starts with a clean module-level driver cache."""
    monkeypatch.setattr(_tools, "_READONLY_DRIVER", None)
    monkeypatch.setattr(_tools, "_READONLY_DRIVER_KEY", None)


def _mock_graph(uri="bolt://localhost:7687", database="neo4j"):
    graph = MagicMock()
    graph._config.uri = uri
    graph._config.database = database
    graph._ensure_connected = MagicMock(return_value="primary-driver")
    return graph


class TestReadOnlyAccountConfig:
    def test_no_env_vars_returns_none(self, monkeypatch):
        monkeypatch.delenv("NAM_NEO4J_READONLY_USERNAME", raising=False)
        monkeypatch.delenv("NAM_NEO4J_READONLY_PASSWORD", raising=False)

        assert _tools._read_only_account_config() is None

    def test_only_username_set_returns_none(self, monkeypatch):
        monkeypatch.setenv("NAM_NEO4J_READONLY_USERNAME", "ro_user")
        monkeypatch.delenv("NAM_NEO4J_READONLY_PASSWORD", raising=False)

        assert _tools._read_only_account_config() is None

    def test_both_set_returns_credentials(self, monkeypatch):
        monkeypatch.setenv("NAM_NEO4J_READONLY_USERNAME", "ro_user")
        monkeypatch.setenv("NAM_NEO4J_READONLY_PASSWORD", "ro_pass")

        assert _tools._read_only_account_config() == ("ro_user", "ro_pass")


class TestGetReadOnlyDriver:
    def test_falls_back_to_primary_driver_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("NAM_NEO4J_READONLY_USERNAME", raising=False)
        monkeypatch.delenv("NAM_NEO4J_READONLY_PASSWORD", raising=False)
        graph = _mock_graph()

        driver = _tools._get_read_only_driver(graph)

        assert driver == "primary-driver"
        graph._ensure_connected.assert_called_once()

    def test_uses_dedicated_account_when_configured(self, monkeypatch):
        monkeypatch.setenv("NAM_NEO4J_READONLY_USERNAME", "ro_user")
        monkeypatch.setenv("NAM_NEO4J_READONLY_PASSWORD", "ro_pass")
        graph = _mock_graph(uri="bolt://neo4j-host:7687")

        built = MagicMock(name="ro-driver")
        build_fn = MagicMock(return_value=built)
        monkeypatch.setattr(_tools, "_build_readonly_driver", build_fn)

        driver = _tools._get_read_only_driver(graph)

        assert driver is built
        build_fn.assert_called_once_with("bolt://neo4j-host:7687", "ro_user", "ro_pass")
        # Must NOT touch the primary (read-write) connection.
        graph._ensure_connected.assert_not_called()

    def test_caches_driver_across_calls(self, monkeypatch):
        monkeypatch.setenv("NAM_NEO4J_READONLY_USERNAME", "ro_user")
        monkeypatch.setenv("NAM_NEO4J_READONLY_PASSWORD", "ro_pass")
        graph = _mock_graph()

        build_fn = MagicMock(side_effect=lambda *a: MagicMock())
        monkeypatch.setattr(_tools, "_build_readonly_driver", build_fn)

        first = _tools._get_read_only_driver(graph)
        second = _tools._get_read_only_driver(graph)

        assert first is second
        build_fn.assert_called_once()


@pytest.mark.asyncio
class TestExecuteReadOnlyUsesReadOnlyDriverSelector:
    async def test_delegates_driver_selection(self, monkeypatch):
        """_execute_read_only must obtain its driver via _get_read_only_driver,
        not client.graph._ensure_connected() directly, so the dedicated
        read-only account (when configured) is actually used.
        """
        from unittest.mock import AsyncMock

        tx = MagicMock()
        result = MagicMock()
        result.data = AsyncMock(return_value=[])
        tx.run = AsyncMock(return_value=result)

        session = MagicMock()

        async def _fake_execute_read(work):
            return await work(tx)

        session.execute_read = AsyncMock(side_effect=_fake_execute_read)

        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        selected_driver = MagicMock()
        selected_driver.session = MagicMock(return_value=session_cm)

        selector = MagicMock(return_value=selected_driver)
        monkeypatch.setattr(_tools, "_get_read_only_driver", selector)

        client = MagicMock()
        client.graph._config.database = "neo4j"

        await _tools._execute_read_only(client, "RETURN 1", {})

        selector.assert_called_once_with(client.graph)
        selected_driver.session.assert_called_once_with(database="neo4j")

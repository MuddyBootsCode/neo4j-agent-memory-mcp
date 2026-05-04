"""Registry managing MemoryClient instances for multiple databases."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient

logger = logging.getLogger(__name__)


class ClientRegistry:
    """Manages MemoryClient instances for multiple Neo4j databases.

    Holds a dict of {database_name: MemoryClient} and provides
    lookup, iteration, and parallel query execution.
    """

    def __init__(self) -> None:
        self._clients: dict[str, MemoryClient] = {}
        self._context_managers: dict[str, Any] = {}

    @property
    def databases(self) -> list[str]:
        """List of registered database names."""
        return list(self._clients.keys())

    def get(self, database: str) -> MemoryClient:
        """Get client for a specific database."""
        if database not in self._clients:
            raise KeyError(
                f"No client registered for database '{database}'. "
                f"Available: {self.databases}"
            )
        return self._clients[database]

    @property
    def general(self) -> MemoryClient:
        """Get the general (default) database client."""
        for name in ("neo4j", "general"):
            if name in self._clients:
                return self._clients[name]
        raise RuntimeError("No general database client registered")

    def register(
        self, database: str, client: MemoryClient, context_manager: Any = None
    ) -> None:
        """Register a client for a database."""
        self._clients[database] = client
        if context_manager is not None:
            self._context_managers[database] = context_manager

    async def close_all(self) -> None:
        """Close all registered clients."""
        for name, cm in self._context_managers.items():
            try:
                await cm.__aexit__(None, None, None)
                logger.info("Closed client for database '%s'", name)
            except Exception as e:
                logger.warning("Error closing client for '%s': %s", name, e)
        self._clients.clear()
        self._context_managers.clear()

    async def query_multiple(
        self,
        databases: list[str],
        query_fn,
    ) -> dict[str, Any]:
        """Execute a query function against multiple databases in parallel.

        Args:
            databases: List of database names to query.
            query_fn: Async callable(client, db_name) -> result.

        Returns:
            Dict of {database_name: result}.
        """

        async def _run(db_name: str):
            client = self.get(db_name)
            try:
                return db_name, await query_fn(client, db_name)
            except Exception as e:
                logger.warning("Query failed on '%s': %s", db_name, e)
                return db_name, {"error": str(e)}

        results = await asyncio.gather(
            *[_run(db) for db in databases],
            return_exceptions=False,
        )
        return dict(results)

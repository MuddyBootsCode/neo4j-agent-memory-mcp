"""MemoryClient construction against the sweepcorpus database.

Mirrors tests/integration/conftest.py's local-mode fixture (LOCAL_EMBEDDING_CONFIG,
_apply_patches / bootstrap_upstream_patches) but targets the standalone
`sweepcorpus` database instead of the pytest test database, and is a plain
async function rather than a fixture so it can be reused across the sweep's
scripts.
"""

from __future__ import annotations

import os

from lib import SWEEP_DB  # noqa: E402

LOCAL_EMBEDDING_CONFIG = {
    "provider": "sentence_transformers",
    "model": "all-MiniLM-L6-v2",
    "dimensions": 384,
}

_patched = False


def apply_patches() -> None:
    global _patched
    if _patched:
        return
    from agent_memory_mcp.mcp._bootstrap import bootstrap_upstream_patches

    bootstrap_upstream_patches()
    _patched = True


def build_settings(database: str = SWEEP_DB):
    from neo4j_agent_memory.config.settings import (
        EmbeddingConfig,
        MemorySettings,
        Neo4jConfig,
    )

    return MemorySettings(
        neo4j=Neo4jConfig(
            password=os.environ.get("NEO4J_PASSWORD", "graphmemory"),
            database=database,
        ),
        embedding=EmbeddingConfig(**LOCAL_EMBEDDING_CONFIG),
    )


def open_client(database: str = SWEEP_DB):
    """Async context manager yielding a connected MemoryClient.

    Usage:
        async with open_client() as client:
            ...
    """
    apply_patches()
    from neo4j_agent_memory import MemoryClient

    return MemoryClient(settings=build_settings(database))

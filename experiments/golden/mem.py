"""MemoryClient against the golden scratch database (mirrors recall_sweep/mem.py)."""

from __future__ import annotations

import os

from lib import GOLDEN_DB

LOCAL_EMBEDDING_CONFIG = {
    "provider": os.environ.get("NAM_EMBEDDING_PROVIDER", "sentence_transformers"),
    "model": os.environ.get("NAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "dimensions": int(os.environ.get("NAM_EMBEDDING_DIMENSIONS", "384")),
}

_patched = False


def apply_patches() -> None:
    global _patched
    if _patched:
        return
    from agent_memory_mcp.mcp._bootstrap import bootstrap_upstream_patches

    bootstrap_upstream_patches()
    _patched = True


def build_settings(database: str = GOLDEN_DB):
    from neo4j_agent_memory.config.settings import EmbeddingConfig, MemorySettings, Neo4jConfig

    return MemorySettings(
        neo4j=Neo4jConfig(password=os.environ.get("NEO4J_PASSWORD", "graphmemory"), database=database),
        embedding=EmbeddingConfig(**LOCAL_EMBEDDING_CONFIG),
    )


def open_client(database: str = GOLDEN_DB):
    apply_patches()
    from neo4j_agent_memory import MemoryClient

    return MemoryClient(settings=build_settings(database))

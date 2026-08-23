"""Index initialization for the single memory graph."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Indexes on Fact temporal properties for efficient point-in-time and
# supersession queries. valid_from / valid_until are stored as epoch millis.
_INDEX_STATEMENTS = [
    "CREATE INDEX fact_valid_from IF NOT EXISTS FOR (f:Fact) ON (f.valid_from)",
    "CREATE INDEX fact_valid_until IF NOT EXISTS FOR (f:Fact) ON (f.valid_until)",
    "CREATE INDEX fact_subject_predicate IF NOT EXISTS FOR (f:Fact) ON (f.subject, f.predicate)",
    "CREATE INDEX fact_superseded_by IF NOT EXISTS FOR (f:Fact) ON (f.superseded_by)",
    "CREATE INDEX fact_expired_at IF NOT EXISTS FOR (f:Fact) ON (f.expired_at)",
    "CREATE INDEX fact_created_at IF NOT EXISTS FOR (f:Fact) ON (f.created_at)",
    # Coding plane (MUD-405): the deterministic nodes are MERGEd by these keys
    # from concurrent hooks; without constraints a race creates twins that
    # split ABOUT/EDITING edges across copies.
    "CREATE CONSTRAINT code_agent_id IF NOT EXISTS FOR (a:CodeAgent) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT coding_session_id IF NOT EXISTS FOR (s:CodingSession) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT work_task_key IF NOT EXISTS FOR (t:WorkTask) REQUIRE t.key IS UNIQUE",
    "CREATE CONSTRAINT change_sha IF NOT EXISTS FOR (c:Change) REQUIRE c.sha IS UNIQUE",
    "CREATE CONSTRAINT code_file_repo_path IF NOT EXISTS FOR (f:CodeFile) REQUIRE (f.repo, f.path) IS UNIQUE",
    "CREATE INDEX coding_memory_expired_at IF NOT EXISTS FOR (m:CodingMemory) ON (m.expired_at)",
]


async def ensure_indexes(graph: Any) -> None:
    """Create temporal/graph indexes on the configured database.

    Args:
        graph: The upstream Neo4j client (``client.graph``) — writes run against
            the client's configured database.
    """
    for stmt in _INDEX_STATEMENTS:
        try:
            await graph.execute_write(stmt, {})
        except Exception as e:  # noqa: BLE001 - index may already exist
            logger.warning("Index creation failed: %s (may already exist)", e)
    logger.info("Temporal/graph indexes ensured")

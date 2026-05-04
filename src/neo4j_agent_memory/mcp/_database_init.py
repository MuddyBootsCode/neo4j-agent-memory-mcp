"""Database initialization for vertical databases."""

from __future__ import annotations

import logging
import os

from neo4j_agent_memory.verticals import get_default_vertical_names

logger = logging.getLogger(__name__)


def get_configured_verticals() -> list[str]:
    """Get list of vertical databases from env or defaults."""
    env_val = os.environ.get("NAM_VERTICALS", "")
    if env_val.strip():
        return [v.strip() for v in env_val.split(",") if v.strip()]
    return get_default_vertical_names()


async def ensure_databases_exist(driver) -> list[str]:
    """Create vertical databases if they don't exist.

    Must be called with a driver connected to the Neo4j instance.
    Database creation commands run against the 'system' database.

    Returns:
        List of database names that were created or already existed.
    """
    verticals = get_configured_verticals()
    created = []

    async with driver.session(database="system") as session:
        for db_name in verticals:
            try:
                await session.run(
                    f"CREATE DATABASE {db_name} IF NOT EXISTS"
                )
                created.append(db_name)
                logger.info("Database '%s' ready", db_name)
            except Exception as e:
                logger.error(
                    "Failed to create database '%s': %s", db_name, e
                )

    # Create temporal indexes on all databases
    await ensure_temporal_indexes(driver, created + ["neo4j"])

    return created


async def ensure_temporal_indexes(driver, databases: list[str] | str | None = None) -> None:
    """Create indexes on temporal properties for efficient temporal queries.

    Creates indexes on Fact nodes for valid_from, valid_until, subject+predicate.
    Runs against each vertical database plus the general 'neo4j' database.
    """
    if databases is None:
        databases = get_configured_verticals() + ["neo4j"]
    elif isinstance(databases, str):
        databases = [databases]

    index_statements = [
        "CREATE INDEX fact_valid_from IF NOT EXISTS FOR (f:Fact) ON (f.valid_from)",
        "CREATE INDEX fact_valid_until IF NOT EXISTS FOR (f:Fact) ON (f.valid_until)",
        "CREATE INDEX fact_subject_predicate IF NOT EXISTS FOR (f:Fact) ON (f.subject, f.predicate)",
        "CREATE INDEX fact_superseded_by IF NOT EXISTS FOR (f:Fact) ON (f.superseded_by)",
        "CREATE INDEX fact_expired_at IF NOT EXISTS FOR (f:Fact) ON (f.expired_at)",
        "CREATE INDEX fact_created_at IF NOT EXISTS FOR (f:Fact) ON (f.created_at)",
    ]

    for db_name in databases:
        try:
            async with driver.session(database=db_name) as session:
                for stmt in index_statements:
                    try:
                        await session.run(stmt)
                    except Exception as e:
                        logger.warning(
                            "Index creation in '%s' failed: %s (may already exist)", db_name, e
                        )
            logger.info("Temporal indexes ensured for database '%s'", db_name)
        except Exception as e:
            logger.warning("Could not create indexes for database '%s': %s", db_name, e)

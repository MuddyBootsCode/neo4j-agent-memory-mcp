"""One-time migration to backfill temporal properties on existing facts."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def migrate_existing_facts(client: Any) -> dict[str, int]:
    """Backfill temporal properties on existing Fact nodes.

    Sets:
    - valid_from = created_at (for facts that have no valid_from)
    - expired_at remains NULL (all existing facts treated as current)

    This is safe to run multiple times (idempotent).

    Returns:
        Dict with migration counts.
    """
    # Backfill valid_from from created_at where missing
    result = await client.graph.execute_write(
        """
        MATCH (f:Fact)
        WHERE f.valid_from IS NULL AND f.created_at IS NOT NULL
        SET f.valid_from = f.created_at.epochMillis
        RETURN count(f) AS migrated
        """,
    )
    valid_from_count = result[0]["migrated"] if result else 0

    logger.info("Migration: set valid_from on %d facts", valid_from_count)

    return {
        "valid_from_backfilled": valid_from_count,
    }

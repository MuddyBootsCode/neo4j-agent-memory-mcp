"""Fact supersession and temporal lifecycle operations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO 8601 string to datetime, or return None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.warning("Could not parse datetime: %s", value)
        return None


async def supersede_matching_facts(
    client: Any,
    subject: str,
    predicate: str,
    new_fact_id: str,
) -> int:
    """Invalidate active facts with the same subject+predicate.

    Sets valid_until=now() and superseded_by on all matching facts
    that don't already have a valid_until set and aren't the new fact.

    Returns:
        Number of facts superseded.
    """
    result = await client.graph.execute_write(
        """
        MATCH (f:Fact)
        WHERE f.subject = $subject
          AND f.predicate = $predicate
          AND f.id <> $new_fact_id
          AND f.valid_until IS NULL
        SET f.valid_until = datetime().epochMillis,
            f.superseded_by = $new_fact_id
        RETURN count(f) AS superseded_count
        """,
        {
            "subject": subject,
            "predicate": predicate,
            "new_fact_id": new_fact_id,
        },
    )
    count = result[0]["superseded_count"] if result else 0
    if count > 0:
        logger.info(
            "Superseded %d fact(s) for %s/%s with %s",
            count, subject, predicate, new_fact_id,
        )
    return count


async def supersede_fact_by_id(
    client: Any,
    old_fact_id: str,
    new_fact_id: str,
) -> int:
    """Explicitly supersede a single fact by ID.

    Returns:
        1 if the fact was superseded, 0 if not found.
    """
    result = await client.graph.execute_write(
        """
        MATCH (f:Fact {id: $old_fact_id})
        WHERE f.valid_until IS NULL
        SET f.valid_until = datetime().epochMillis,
            f.superseded_by = $new_fact_id
        RETURN count(f) AS superseded_count
        """,
        {"old_fact_id": old_fact_id, "new_fact_id": new_fact_id},
    )
    return result[0]["superseded_count"] if result else 0


async def get_fact_evolution(
    client: Any,
    subject: str,
    predicate: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve the full version history of facts about a subject.

    Returns facts ordered by created_at, showing the evolution
    of knowledge including supersession chains.
    """
    params: dict[str, Any] = {"subject": subject, "limit": limit}

    predicate_clause = ""
    if predicate:
        predicate_clause = "AND f.predicate = $predicate"
        params["predicate"] = predicate

    result = await client.graph.execute_read(
        f"""
        MATCH (f:Fact)
        WHERE f.subject = $subject {predicate_clause}
        RETURN f.id AS id,
               f.subject AS subject,
               f.predicate AS predicate,
               f.object AS object,
               f.confidence AS confidence,
               f.created_at AS created_at,
               f.valid_from AS valid_from,
               f.valid_until AS valid_until,
               f.superseded_by AS superseded_by
        ORDER BY f.created_at ASC
        LIMIT $limit
        """,
        params,
    )

    return [
        {
            "id": row["id"],
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "confidence": row["confidence"],
            "created_at": str(row["created_at"]) if row["created_at"] else None,
            "valid_from": str(row["valid_from"]) if row["valid_from"] else None,
            "valid_until": str(row["valid_until"]) if row["valid_until"] else None,
            "superseded_by": row["superseded_by"],
            "is_current": row["valid_until"] is None,
        }
        for row in result
    ]


async def temporal_fact_query(
    client: Any,
    point_in_time: datetime | str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Query facts valid at a specific point in time.

    Uses event-time filtering: valid_from <= point_in_time AND
    (valid_until IS NULL OR valid_until > point_in_time).

    Args:
        point_in_time: datetime object or ISO 8601 string. Defaults to now.
    """
    if point_in_time is None:
        point_in_time = datetime.now(timezone.utc)
    elif isinstance(point_in_time, str):
        parsed = parse_iso_datetime(point_in_time)
        if parsed is None:
            raise ValueError(f"Invalid datetime string: {point_in_time}")
        point_in_time = parsed
    pit_epoch = int(point_in_time.timestamp() * 1000)
    params: dict[str, Any] = {"pit": pit_epoch, "limit": limit}

    where_clauses = [
        "(f.valid_from IS NULL OR f.valid_from <= $pit)",
        "(f.valid_until IS NULL OR f.valid_until > $pit)",
    ]
    if subject:
        where_clauses.append("f.subject = $subject")
        params["subject"] = subject
    if predicate:
        where_clauses.append("f.predicate = $predicate")
        params["predicate"] = predicate

    where = " AND ".join(where_clauses)

    result = await client.graph.execute_read(
        f"""
        MATCH (f:Fact)
        WHERE {where}
        RETURN f.id AS id,
               f.subject AS subject,
               f.predicate AS predicate,
               f.object AS object,
               f.confidence AS confidence,
               f.created_at AS created_at,
               f.valid_from AS valid_from,
               f.valid_until AS valid_until
        ORDER BY f.confidence DESC, f.created_at DESC
        LIMIT $limit
        """,
        params,
    )

    return [
        {
            "id": row["id"],
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "confidence": row["confidence"],
            "created_at": str(row["created_at"]) if row["created_at"] else None,
            "valid_from": str(row["valid_from"]) if row["valid_from"] else None,
            "valid_until": str(row["valid_until"]) if row["valid_until"] else None,
        }
        for row in result
    ]

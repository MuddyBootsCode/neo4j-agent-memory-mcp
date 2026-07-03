"""Fact supersession and temporal lifecycle operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def normalize_term(value: str) -> str:
    """Normalize a fact subject/predicate/object for matching.

    Casefold + trim, so ``" Alice "``/``"works_at"`` match
    ``"alice"``/``"WORKS_AT"``. The Cypher side of every match uses
    ``toLower(trim(...))`` on the stored property, which agrees with
    this for ASCII input (casefold vs toLower only diverge for a few
    non-ASCII code points like the German eszett).
    """
    return value.strip().casefold()


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO 8601 string to datetime, or return None.

    Handles the ``Z``/``z`` UTC suffix portably: Python 3.10's
    ``datetime.fromisoformat`` rejects ``Z`` entirely, and 3.11+ still
    rejects a lowercase ``z``, so it is normalized to ``+00:00`` first.
    """
    if not value:
        return None
    try:
        if value.endswith(("Z", "z")):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.warning("Could not parse datetime: %s", value)
        return None


# Atomic create/refresh + supersession for facts.
#
# One Cypher statement (one auto-commit transaction) does all of:
#   - R7: if an identical active fact exists (same normalized S/P/O),
#     refresh it (bump confidence, set last_confirmed) instead of creating
#     a duplicate — preserving its valid_from ("known since").
#   - Otherwise CREATE the new fact node.
#   - R5/R6/R8: supersede matching active facts (normalized subject +
#     predicate, different object) in the same transaction, setting
#     valid_until, expired_at, and superseded_by — but only when the new
#     assertion is current/open-ended ($can_supersede), or when an explicit
#     $supersedes_id was given.
#
# Being a single transaction fixes (a) the concurrent-store race in which
# two contradicting stores mutually invalidated each other (create and
# supersede used to be separate transactions), and (b) crash windows that
# left a created fact without its supersession (or vice versa).
STORE_FACT_ATOMIC = """
OPTIONAL MATCH (dup:Fact)
WHERE $allow_refresh
  AND dup.valid_until IS NULL
  AND toLower(trim(dup.subject)) = $subject_norm
  AND toLower(trim(dup.predicate)) = $predicate_norm
  AND toLower(trim(dup.object)) = $object_norm
WITH dup ORDER BY dup.created_at ASC LIMIT 1
FOREACH (d IN CASE WHEN dup IS NULL THEN [] ELSE [dup] END |
    SET d.confidence = CASE
            WHEN $confidence > coalesce(d.confidence, 0.0) THEN $confidence
            ELSE d.confidence
        END,
        d.last_confirmed = $now
)
FOREACH (_ IN CASE WHEN dup IS NULL THEN [1] ELSE [] END |
    CREATE (:Fact {
        id: $fact_id,
        subject: $subject,
        predicate: $predicate,
        object: $object,
        confidence: $confidence,
        embedding: $embedding,
        valid_from: $valid_from,
        valid_until: $valid_until,
        created_at: datetime(),
        metadata: $metadata
    })
)
WITH dup
OPTIONAL MATCH (old:Fact)
WHERE old.valid_until IS NULL
  AND old.id <> $fact_id
  AND (
      ($supersedes_id IS NOT NULL AND old.id = $supersedes_id)
      OR (
          $supersedes_id IS NULL
          AND $can_supersede
          AND toLower(trim(old.subject)) = $subject_norm
          AND toLower(trim(old.predicate)) = $predicate_norm
          AND toLower(trim(old.object)) <> $object_norm
      )
  )
SET old.valid_until = $now,
    old.expired_at = $now,
    old.superseded_by = CASE WHEN dup IS NULL THEN $fact_id ELSE dup.id END
WITH dup, count(old) AS superseded_count
RETURN CASE WHEN dup IS NULL THEN $fact_id ELSE dup.id END AS fact_id,
       dup IS NOT NULL AS refreshed,
       superseded_count,
       CASE WHEN dup IS NULL THEN $valid_from ELSE dup.valid_from END AS valid_from
"""


async def store_fact_atomic(
    client: Any,
    *,
    subject: str,
    predicate: str,
    obj: str,
    confidence: float = 1.0,
    valid_from_epoch: int | None = None,
    valid_until_epoch: int | None = None,
    is_current_state: bool = True,
    supersedes_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create (or refresh) a fact and supersede matching facts atomically.

    Runs create + supersession as ONE Cypher statement in a single
    transaction (see STORE_FACT_ATOMIC). The embedding is computed first
    (read-only, outside the transaction) so the write itself stays atomic.

    Supersession is gated on the new fact being a current, open-ended
    assertion (``valid_until_epoch is None`` and ``is_current_state``);
    historical facts are stored without invalidating the current one.
    An explicit ``supersedes_id`` bypasses that gate (user intent).

    Returns:
        Dict with keys: fact_id, refreshed, superseded_count, valid_from.
        ``refreshed`` is True when an identical active fact already existed
        and was refreshed instead of duplicated — ``fact_id``/``valid_from``
        then refer to the existing fact.
    """
    embedding = None
    embedder = getattr(client.long_term, "_embedder", None)
    if embedder is not None:
        try:
            embedding = await embedder.embed(f"{subject} {predicate} {obj}")
        except Exception as e:
            logger.warning("Fact embedding failed: %s — storing without embedding", e)

    can_supersede = valid_until_epoch is None and bool(is_current_state)
    allow_refresh = can_supersede and supersedes_id is None
    fact_id = str(uuid4())
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows = await client.graph.execute_write(
        STORE_FACT_ATOMIC,
        {
            "fact_id": fact_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "subject_norm": normalize_term(subject),
            "predicate_norm": normalize_term(predicate),
            "object_norm": normalize_term(obj),
            "confidence": confidence,
            "embedding": embedding,
            "valid_from": valid_from_epoch,
            "valid_until": valid_until_epoch,
            "metadata": json.dumps(metadata) if metadata else None,
            "now": now_ms,
            "can_supersede": can_supersede,
            "allow_refresh": allow_refresh,
            "supersedes_id": supersedes_id,
        },
    )
    row = rows[0]
    if row["superseded_count"] > 0:
        logger.info(
            "Superseded %d fact(s) for %s/%s with %s",
            row["superseded_count"], subject, predicate, row["fact_id"],
        )
    if row["refreshed"]:
        logger.info(
            "Refreshed existing fact %s (%s/%s/%s re-affirmed)",
            row["fact_id"], subject, predicate, obj,
        )
    return {
        "fact_id": row["fact_id"],
        "refreshed": bool(row["refreshed"]),
        "superseded_count": row["superseded_count"],
        "valid_from": row["valid_from"],
    }


async def supersede_matching_facts(
    client: Any,
    subject: str,
    predicate: str,
    new_fact_id: str,
) -> int:
    """Invalidate active facts with the same subject+predicate.

    Matching is normalized (casefold + trim). Sets valid_until, expired_at,
    and superseded_by on all matching facts that don't already have a
    valid_until set and aren't the new fact.

    Note: prefer ``store_fact_atomic`` when creating a fact — it runs the
    create and this supersession in one transaction.

    Returns:
        Number of facts superseded.
    """
    result = await client.graph.execute_write(
        """
        MATCH (f:Fact)
        WHERE toLower(trim(f.subject)) = $subject_norm
          AND toLower(trim(f.predicate)) = $predicate_norm
          AND f.id <> $new_fact_id
          AND f.valid_until IS NULL
        SET f.valid_until = datetime().epochMillis,
            f.expired_at = datetime().epochMillis,
            f.superseded_by = $new_fact_id
        RETURN count(f) AS superseded_count
        """,
        {
            "subject_norm": normalize_term(subject),
            "predicate_norm": normalize_term(predicate),
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
            f.expired_at = datetime().epochMillis,
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

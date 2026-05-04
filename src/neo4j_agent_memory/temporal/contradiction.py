"""LLM-powered contradiction detection for temporal fact management."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Similarity threshold for contradiction candidate search
CONTRADICTION_CANDIDATE_THRESHOLD = 0.7
CONTRADICTION_CANDIDATE_LIMIT = 10


async def find_contradiction_candidates(
    client: Any,
    subject: str,
    predicate: str,
    obj: str,
    new_fact_id: str,
) -> list[dict[str, Any]]:
    """Find existing facts that might be contradicted by a new fact.

    Uses vector similarity on the fact embedding to find semantically
    similar facts, then filters to active (non-superseded) ones.
    """
    # Build the same embedding text the upstream uses
    embedding_text = f"{subject} {predicate} {obj}"

    embedder = getattr(client.long_term, "_embedder", None)
    if embedder is None:
        logger.warning("No embedder — falling back to subject+predicate match")
        return await _fallback_subject_predicate_match(
            client, subject, predicate, new_fact_id
        )

    try:
        embedding = await embedder.embed(embedding_text)
    except Exception as e:
        logger.warning("Embedding failed: %s — falling back", e)
        return await _fallback_subject_predicate_match(
            client, subject, predicate, new_fact_id
        )

    if not embedding:
        return await _fallback_subject_predicate_match(
            client, subject, predicate, new_fact_id
        )

    # Vector search for similar facts
    rows = await client.graph.execute_read(
        """
        CALL db.index.vector.queryNodes(
            'fact_embedding_idx', $limit, $embedding
        )
        YIELD node, score
        WHERE score >= $threshold
          AND node.id <> $new_fact_id
          AND node.valid_until IS NULL
        RETURN node.id AS id,
               node.subject AS subject,
               node.predicate AS predicate,
               node.object AS object,
               node.confidence AS confidence,
               score
        ORDER BY score DESC
        """,
        {
            "embedding": embedding,
            "limit": CONTRADICTION_CANDIDATE_LIMIT,
            "threshold": CONTRADICTION_CANDIDATE_THRESHOLD,
            "new_fact_id": new_fact_id,
        },
    )

    return [
        {
            "id": row["id"],
            "idx": i,
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "confidence": row["confidence"],
            "similarity": row["score"],
        }
        for i, row in enumerate(rows)
    ]


async def _fallback_subject_predicate_match(
    client: Any,
    subject: str,
    predicate: str,
    new_fact_id: str,
) -> list[dict[str, Any]]:
    """Fallback: find candidates by exact subject+predicate match."""
    rows = await client.graph.execute_read(
        """
        MATCH (f:Fact)
        WHERE f.subject = $subject
          AND f.predicate = $predicate
          AND f.id <> $new_fact_id
          AND f.valid_until IS NULL
        RETURN f.id AS id,
               f.subject AS subject,
               f.predicate AS predicate,
               f.object AS object,
               f.confidence AS confidence
        ORDER BY f.confidence DESC, f.created_at DESC
        LIMIT 10
        """,
        {"subject": subject, "predicate": predicate, "new_fact_id": new_fact_id},
    )
    return [
        {
            "id": row["id"],
            "idx": i,
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "confidence": row["confidence"],
            "similarity": None,
        }
        for i, row in enumerate(rows)
    ]


async def detect_and_invalidate(
    client: Any,
    subject: str,
    predicate: str,
    obj: str,
    new_fact_id: str,
) -> dict[str, Any]:
    """Full contradiction detection + invalidation pipeline.

    1. Find semantically similar active facts
    2. If candidates found, call BAML DetectContradictions
    3. Invalidate contradicted facts (set expired_at, superseded_by)

    Returns:
        Dict with keys: candidates_found, contradictions_detected,
        facts_invalidated, contradiction_type, reasoning
    """
    # Step 1: Find candidates
    candidates = await find_contradiction_candidates(
        client, subject, predicate, obj, new_fact_id
    )

    if not candidates:
        return {
            "candidates_found": 0,
            "contradictions_detected": 0,
            "facts_invalidated": 0,
            "contradiction_type": "none",
            "reasoning": "No similar active facts found",
        }

    # Step 2: Call BAML for contradiction detection
    try:
        from baml_client import b as baml

        baml_candidates = [
            {
                "idx": c["idx"],
                "subject": c["subject"],
                "predicate": c["predicate"],
                "object": c["object"],
                "confidence": c["confidence"],
            }
            for c in candidates
        ]

        result = await baml.DetectContradictions(
            new_fact_subject=subject,
            new_fact_predicate=predicate,
            new_fact_object=obj,
            candidates=baml_candidates,
        )

        contradicted_indices = result.contradicted_indices or []
        contradiction_type = result.contradiction_type or "none"
        reasoning = result.reasoning or ""

    except Exception as e:
        logger.warning(
            "BAML contradiction detection failed: %s — falling back to SPO match", e
        )
        # Fallback: simple subject+predicate match supersession
        contradicted_indices = [
            c["idx"] for c in candidates
            if c["subject"] == subject and c["predicate"] == predicate
        ]
        contradiction_type = "direct_supersession" if contradicted_indices else "none"
        reasoning = f"BAML unavailable, fell back to SPO match (error: {e})"

    if not contradicted_indices:
        return {
            "candidates_found": len(candidates),
            "contradictions_detected": 0,
            "facts_invalidated": 0,
            "contradiction_type": contradiction_type,
            "reasoning": reasoning,
        }

    # Step 3: Invalidate contradicted facts
    invalidated = 0
    for idx in contradicted_indices:
        if idx < len(candidates):
            old_fact_id = candidates[idx]["id"]
            try:
                result = await client.graph.execute_write(
                    """
                    MATCH (f:Fact {id: $old_fact_id})
                    WHERE f.valid_until IS NULL
                    SET f.valid_until = datetime().epochMillis,
                        f.expired_at = datetime().epochMillis,
                        f.superseded_by = $new_fact_id
                    RETURN count(f) AS c
                    """,
                    {"old_fact_id": old_fact_id, "new_fact_id": new_fact_id},
                )
                if result and result[0]["c"] > 0:
                    invalidated += 1
                    logger.info(
                        "Invalidated fact %s (contradicted by %s): %s",
                        old_fact_id, new_fact_id, reasoning,
                    )
            except Exception as e:
                logger.error("Failed to invalidate fact %s: %s", old_fact_id, e)

    return {
        "candidates_found": len(candidates),
        "contradictions_detected": len(contradicted_indices),
        "facts_invalidated": invalidated,
        "contradiction_type": contradiction_type,
        "reasoning": reasoning,
    }

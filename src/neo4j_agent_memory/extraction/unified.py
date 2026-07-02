"""Unified single-pass memory extraction and persistence.

One BAML call (``ExtractMemory``) pulls POLE+O + domain entities, relations, and
preferences from a message; :func:`persist_memory` writes them to the single
graph. Entity ``type`` becomes a Neo4j label via the upstream dedup pipeline;
relations are stored on ``RELATED_TO`` edges keyed by ``relation_type`` so
distinct relationships between the same pair are preserved.

This replaces the former POLE+O-only ``BamlEntityExtractor`` (wired through the
upstream extractor factory) plus the separate per-vertical extractor — a single
path, one LLM call, global dedup.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Domain properties captured on entities when the LLM provides them.
_DOMAIN_PROPS = ("status", "priority", "date")


def _clamp(value: Any, default: float = 0.8) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


async def extract_memory(text: str) -> dict[str, list[dict[str, Any]]]:
    """Run the unified BAML extraction over a block of text.

    Returns a dict with ``entities``, ``relations`` and ``preferences`` lists.
    Relations whose endpoints are not among the extracted entities are dropped
    (prevents the LLM from inventing edges to entities it never surfaced).
    Raises nothing for empty input; callers should treat a BAML failure as a
    non-fatal skip so a message is still stored.
    """
    if not text or not text.strip():
        return {"entities": [], "relations": [], "preferences": []}

    from neo4j_agent_memory.baml_client.async_client import b

    result = await b.ExtractMemory(text=text)

    entities: list[dict[str, Any]] = []
    for e in result.entities:
        entity: dict[str, Any] = {
            "name": e.name,
            "type": e.type.value if hasattr(e.type, "value") else str(e.type),
            "subtype": getattr(e, "subtype", None),
            "confidence": _clamp(e.confidence),
        }
        for prop in _DOMAIN_PROPS:
            val = getattr(e, prop, None)
            if val:
                entity[prop] = val
        entities.append(entity)

    entity_names = {e["name"].lower().strip() for e in entities}
    relations = [
        {
            "source": r.source,
            "target": r.target,
            "relation_type": str(r.relation_type).upper(),
            "confidence": _clamp(r.confidence, 0.7),
        }
        for r in result.relations
        if r.source.lower().strip() in entity_names
        and r.target.lower().strip() in entity_names
    ]

    preferences = [
        {
            "category": p.category,
            "preference": p.preference,
            "context": getattr(p, "context", None),
            "confidence": _clamp(p.confidence, 0.7),
        }
        for p in result.preferences
    ]

    logger.debug(
        "ExtractMemory: %d entities, %d relations, %d preferences",
        len(entities),
        len(relations),
        len(preferences),
    )
    return {"entities": entities, "relations": relations, "preferences": preferences}


async def persist_memory(
    client: Any,
    message_id: str,
    extraction: dict[str, list[dict[str, Any]]],
    *,
    link_messages: bool = True,
) -> dict[str, int]:
    """Persist extracted entities, relations, and preferences to the graph.

    - Entities go through the upstream ``add_entity`` dedup pipeline; the domain
      ``type`` becomes a node label and status/priority/date are stored as
      attributes.
    - Each entity is linked to its source message via ``MENTIONS``.
    - Relations are MERGEd on ``RELATED_TO`` keyed by ``relation_type`` so two
      different relationships between the same pair remain distinct.
    - Preferences go through the upstream ``add_preference`` pipeline.

    Only relations whose *both* endpoints were persisted in this batch are
    written, so there is no arbitrary name-based fallback to the wrong entity.
    """
    entities_stored = 0
    relations_stored = 0
    preferences_stored = 0
    merged = 0
    name_to_id: dict[str, str] = {}

    for entity in extraction.get("entities", []):
        name = entity["name"]
        confidence = _clamp(entity.get("confidence"), 0.8)
        attributes = {k: entity[k] for k in _DOMAIN_PROPS if entity.get(k)}
        try:
            stored, dedup = await client.long_term.add_entity(
                name=name,
                entity_type=entity["type"],
                subtype=entity.get("subtype"),
                attributes=attributes or None,
                resolve=True,
                generate_embedding=True,
                deduplicate=True,
                geocode=False,
                enrich=False,
                metadata={"confidence": confidence},
            )
            entity_id = str(stored.id)
            name_to_id[name.lower().strip()] = entity_id
            entities_stored += 1
            if getattr(dedup, "action", None) == "merged":
                merged += 1

            if link_messages:
                await client.graph.execute_write(
                    """
                    MATCH (m:Message {id: $message_id})
                    MATCH (e:Entity {id: $entity_id})
                    MERGE (m)-[r:MENTIONS]->(e)
                    ON CREATE SET r.confidence = $confidence
                    """,
                    {
                        "message_id": message_id,
                        "entity_id": entity_id,
                        "confidence": confidence,
                    },
                )
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort per entity
            logger.warning("Failed to persist entity '%s': %s", name, exc)

    for relation in extraction.get("relations", []):
        source_id = name_to_id.get(relation["source"].lower().strip())
        target_id = name_to_id.get(relation["target"].lower().strip())
        if not source_id or not target_id:
            continue
        rel_type = relation["relation_type"]
        confidence = _clamp(relation.get("confidence"), 0.7)
        try:
            await client.graph.execute_write(
                """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATED_TO {relation_type: $relation_type}]->(target)
                ON CREATE SET r.confidence = $confidence, r.created_at = datetime()
                ON MATCH SET r.confidence =
                    CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                    r.updated_at = datetime()
                """,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": rel_type,
                    "confidence": confidence,
                },
            )
            relations_stored += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to persist relation %s-[%s]->%s: %s",
                relation["source"], rel_type, relation["target"], exc,
            )

    for pref in extraction.get("preferences", []):
        try:
            await client.long_term.add_preference(
                category=pref["category"],
                preference=pref["preference"],
                context=pref.get("context"),
                confidence=_clamp(pref.get("confidence"), 0.7),
            )
            preferences_stored += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist preference '%s': %s", pref.get("preference"), exc)

    logger.info(
        "Persisted memory: %d entities (%d merged), %d relations, %d preferences",
        entities_stored, merged, relations_stored, preferences_stored,
    )
    return {
        "entities": entities_stored,
        "relations": relations_stored,
        "preferences": preferences_stored,
        "merged": merged,
    }

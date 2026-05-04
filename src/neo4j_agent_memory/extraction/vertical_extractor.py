"""Vertical-aware entity extraction dispatcher and persistence."""

from __future__ import annotations

import logging
from typing import Any

from neo4j_agent_memory.verticals import get_vertical_extractors

logger = logging.getLogger(__name__)

# Maps database name to BAML extraction function name
VERTICAL_EXTRACTORS = get_vertical_extractors()


async def extract_for_vertical(
    text: str,
    database: str,
) -> dict[str, Any] | None:
    """Extract entities using the vertical-specific ontology.

    Args:
        text: Text to extract from.
        database: Target database name.

    Returns:
        Extraction result dict or None if no vertical extractor exists.
    """
    if database not in VERTICAL_EXTRACTORS:
        return None  # Use default POLE+O extraction

    try:
        from neo4j_agent_memory.baml_client.async_client import b

        func_name = VERTICAL_EXTRACTORS[database]
        extract_fn = getattr(b, func_name)
        result = await extract_fn(text=text)

        entities = []
        for e in result.entities:
            entity_dict: dict[str, Any] = {
                "name": e.name,
                "type": e.type.value if hasattr(e.type, "value") else str(e.type),
                "confidence": e.confidence,
            }
            # Capture domain-specific fields when present
            if hasattr(e, "date") and e.date:
                entity_dict["date"] = e.date
            if hasattr(e, "status") and e.status:
                entity_dict["status"] = e.status
            if hasattr(e, "priority") and e.priority:
                entity_dict["priority"] = e.priority
            entities.append(entity_dict)

        return {
            "entities": entities,
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "confidence": r.confidence,
                }
                for r in result.relations
            ],
        }
    except Exception as e:
        logger.warning(
            "Vertical extraction failed for '%s': %s", database, e
        )
        return None


async def persist_vertical_entities(
    client: Any,
    message_id: str,
    extraction: dict[str, Any],
    database: str,
) -> dict[str, int]:
    """Persist vertical-extracted entities and relations to the graph.

    Creates Entity nodes with domain-specific types (e.g., ACTION_ITEM,
    MILESTONE) and links them to the source message via :MENTIONS edges.
    Relations use :RELATED_TO edges with vertical-specific relation_type
    properties (e.g., ASSIGNED_TO, DEPENDS_ON).

    Args:
        client: MemoryClient for the target vertical database.
        message_id: ID of the message these entities were extracted from.
        extraction: Dict from extract_for_vertical with 'entities' and 'relations'.
        database: Name of the vertical database (for metadata).

    Returns:
        Dict with 'entities' and 'relations' counts.
    """
    entities_stored = 0
    relations_stored = 0
    merged_count = 0
    flagged_count = 0
    entity_name_to_id: dict[str, str] = {}

    for entity in extraction.get("entities", []):
        entity_type = entity["type"]
        entity_name = entity["name"]
        confidence = max(0.0, min(1.0, entity.get("confidence", 0.8)))

        try:
            # Use base library's add_entity for dedup pipeline
            stored_entity, dedup_result = await client.long_term.add_entity(
                name=entity_name,
                entity_type=entity_type,
                resolve=True,
                generate_embedding=True,
                deduplicate=True,
                geocode=False,
                enrich=False,
                metadata={"confidence": confidence},
            )

            entity_id = str(stored_entity.id)

            if dedup_result.action == "merged":
                merged_count += 1
                logger.info(
                    "Vertical entity '%s' merged with existing '%s' (score=%.2f)",
                    entity_name,
                    dedup_result.matched_entity_name,
                    dedup_result.similarity_score,
                )
            elif dedup_result.action == "flagged":
                flagged_count += 1

            # Set domain-specific properties (status, priority, date, vertical_source)
            domain_props: dict[str, Any] = {"vertical_source": database}
            if entity.get("status"):
                domain_props["status"] = entity["status"]
            if entity.get("priority"):
                domain_props["priority"] = entity["priority"]
            if entity.get("date"):
                domain_props["date"] = entity["date"]

            set_clauses = ", ".join(f"e.{k} = ${k}" for k in domain_props)
            await client.graph.execute_write(
                f"MATCH (e:Entity {{id: $id}}) SET {set_clauses}",
                {"id": entity_id, **domain_props},
            )

            # Link to source message (MERGE prevents duplicate edges)
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

            entity_name_to_id[entity_name.lower().strip()] = entity_id
            entities_stored += 1
        except Exception as e:
            logger.warning(
                "Failed to persist vertical entity '%s': %s", entity_name, e
            )

    # Persist relations
    for relation in extraction.get("relations", []):
        source_name = relation["source"]
        target_name = relation["target"]
        relation_type = relation["relation_type"]
        confidence = max(0.0, min(1.0, relation.get("confidence", 0.7)))

        # Try ID-based linking first, fall back to name-based
        source_id = entity_name_to_id.get(source_name.lower().strip())
        target_id = entity_name_to_id.get(target_name.lower().strip())

        try:
            if source_id and target_id:
                await client.graph.execute_write(
                    """
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    MERGE (source)-[r:RELATED_TO]->(target)
                    ON CREATE SET
                        r.relation_type = $relation_type,
                        r.confidence = $confidence,
                        r.created_at = datetime()
                    ON MATCH SET
                        r.confidence = CASE WHEN $confidence > r.confidence
                            THEN $confidence ELSE r.confidence END,
                        r.updated_at = datetime()
                    """,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": relation_type,
                        "confidence": confidence,
                    },
                )
            else:
                await client.graph.execute_write(
                    """
                    MATCH (source:Entity)
                    WHERE toLower(source.name) = toLower($source_name)
                    WITH source LIMIT 1
                    MATCH (target:Entity)
                    WHERE toLower(target.name) = toLower($target_name)
                    WITH source, target LIMIT 1
                    MERGE (source)-[r:RELATED_TO]->(target)
                    ON CREATE SET
                        r.relation_type = $relation_type,
                        r.confidence = $confidence,
                        r.created_at = datetime()
                    ON MATCH SET
                        r.confidence = CASE WHEN $confidence > r.confidence
                            THEN $confidence ELSE r.confidence END,
                        r.updated_at = datetime()
                    """,
                    {
                        "source_name": source_name,
                        "target_name": target_name,
                        "relation_type": relation_type,
                        "confidence": confidence,
                    },
                )
            relations_stored += 1
        except Exception as e:
            logger.warning(
                "Failed to persist vertical relation '%s'-[%s]->'%s': %s",
                source_name, relation_type, target_name, e,
            )

    logger.info(
        "Vertical extraction for '%s': %d entities (%d merged, %d flagged), %d relations",
        database, entities_stored, merged_count, flagged_count, relations_stored,
    )

    return {
        "entities": entities_stored,
        "relations": relations_stored,
        "merged": merged_count,
        "flagged": flagged_count,
    }

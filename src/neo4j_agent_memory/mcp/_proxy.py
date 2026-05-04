"""Cross-database proxy node management."""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


async def create_proxy_reference(
    general_client,
    source_db: str,
    node_id: str,
    node_type: str,
    node_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a proxy node in the general DB referencing a vertical DB node.

    Args:
        general_client: MemoryClient for the general (neo4j) database.
        source_db: Name of the vertical database (e.g., "meetings").
        node_id: ID of the node in the vertical database.
        node_type: Type/label of the referenced node.
        node_name: Display name for the proxy.
        metadata: Optional additional metadata.

    Returns:
        ID of the created proxy node.
    """
    proxy_id = str(uuid.uuid4())

    await general_client.graph.execute_write(
        """
        CREATE (p:ProxyRef {
            id: $proxy_id,
            source_database: $source_db,
            external_id: $node_id,
            external_type: $node_type,
            name: $node_name,
            created_at: datetime()
        })
        """,
        {
            "proxy_id": proxy_id,
            "source_db": source_db,
            "node_id": node_id,
            "node_type": node_type,
            "node_name": node_name,
        },
    )

    logger.debug(
        "Created proxy ref %s -> %s:%s", proxy_id, source_db, node_id
    )
    return proxy_id


async def resolve_proxy_references(
    general_client,
    registry,
    entity_id: str,
) -> list[dict[str, Any]]:
    """Resolve proxy references for an entity, fetching data from vertical DBs.

    Args:
        general_client: MemoryClient for the general database.
        registry: ClientRegistry for accessing vertical clients.
        entity_id: Entity ID in the general database.

    Returns:
        List of resolved cross-database references.
    """
    # Find proxy refs linked to this entity
    rows = await general_client.graph.execute_read(
        """
        MATCH (e:Entity {id: $entity_id})-[:HAS_REFERENCE]->(p:ProxyRef)
        RETURN p.source_database AS db, p.external_id AS ext_id,
               p.external_type AS ext_type, p.name AS name
        """,
        {"entity_id": entity_id},
    )

    resolved = []
    for row in rows:
        db_name = row["db"]
        try:
            client = registry.get(db_name)
            # Look up the actual node in the vertical DB
            records = await client.graph.execute_read(
                """
                MATCH (n {id: $node_id})
                RETURN properties(n) AS props, labels(n) AS labels
                """,
                {"node_id": row["ext_id"]},
            )
            if records:
                resolved.append({
                    "source_database": db_name,
                    "external_id": row["ext_id"],
                    "type": row["ext_type"],
                    "name": row["name"],
                    "resolved": True,
                    "data": records[0]["props"],
                    "labels": records[0]["labels"],
                })
            else:
                resolved.append({
                    "source_database": db_name,
                    "external_id": row["ext_id"],
                    "type": row["ext_type"],
                    "name": row["name"],
                    "resolved": False,
                })
        except Exception as e:
            logger.warning("Failed to resolve proxy for %s:%s: %s", db_name, row["ext_id"], e)
            resolved.append({
                "source_database": db_name,
                "external_id": row["ext_id"],
                "resolved": False,
                "error": str(e),
            })

    return resolved

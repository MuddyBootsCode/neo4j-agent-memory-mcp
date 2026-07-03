"""MCP tool implementations for Neo4j Agent Memory.

Defines the 8 core tools as FastMCP @mcp.tool decorated functions:
- memory_search: Hybrid vector + graph search
- memory_store: Store memories (messages, facts, preferences)
- entity_lookup: Get entity with relationships
- conversation_history: Get conversation for session
- graph_query: Execute read-only Cypher queries
- add_reasoning_trace: Store procedural memory with real reasoning
- explain_reasoning: Retrieve and explain past reasoning chains
- extract_reasoning: Extract reasoning from conversation text
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from fastmcp import Context

from neo4j_agent_memory.mcp._common import get_client
from neo4j_agent_memory.mcp._logging import log_tool_call

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


async def _backfill_entity_embeddings(client: Any, message_id: str) -> int:
    """Generate embeddings for entities linked to a message that lack them.

    After add_message extracts entities, this finds any without embeddings
    and generates them using the client's embedder.

    Returns:
        Number of entities that received embeddings.
    """
    embedder = getattr(client.short_term, "_embedder", None)
    if embedder is None:
        logger.warning("No embedder available — skipping entity embedding backfill")
        return 0

    # Find entities linked to this message that have no embedding
    rows = await client.graph.execute_read(
        """
        MATCH (m:Message {id: $message_id})-[:MENTIONS]->(e:Entity)
        WHERE e.embedding IS NULL
        RETURN e.id AS id, e.name AS name
        """,
        {"message_id": message_id},
    )

    count = 0
    for row in rows:
        try:
            embedding = await embedder.embed(row["name"])
            if embedding:
                await client.graph.execute_write(
                    "MATCH (e:Entity {id: $id}) SET e.embedding = $embedding",
                    {"id": row["id"], "embedding": embedding},
                )
                count += 1
        except Exception as e:
            logger.warning("Failed to embed entity %s: %s", row["name"], e)

    if count > 0:
        logger.info("Backfilled embeddings for %d entities from message %s", count, message_id)
    return count


# Cheap first-pass reject for obvious write clauses (matched against uppercased query).
# This is DEFENSE-IN-DEPTH ONLY, not the real guard: a keyword denylist cannot catch
# every case (comments, casing tricks, string-built names, future syntax) and it does
# NOT catch write-mode APOC procedures like apoc.cypher.doIt. Actual enforcement is at
# the database layer — _execute_read_only() runs the query in a genuine READ-access-mode
# transaction, so the server itself rejects both write clauses and write-mode procedures.
WRITE_PATTERNS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\s+DELETE\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bLOAD\s+CSV\b",
    r"\bFOREACH\b",
    r"\bCALL\s+\{",
    r"\bIN\s+TRANSACTIONS\b",
]


def _is_read_only_query(query: str) -> bool:
    """Cheap first-pass check for obvious write clauses.

    This is a fast-fail nicety only; the authoritative read-only enforcement is the
    READ-access-mode transaction in _execute_read_only(). Do not rely on this alone.

    Args:
        query: The Cypher query to check.

    Returns:
        True if the query contains no obvious write keywords.
    """
    query_upper = query.upper()
    return all(not re.search(pattern, query_upper) for pattern in WRITE_PATTERNS)


async def _execute_read_only(
    client: Any, query: str, parameters: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Run a Cypher query in a genuine READ-access-mode transaction.

    Unlike the upstream ``graph.execute_read`` (which opens a session in the default
    WRITE access mode, so the server does not restrict writes), this uses a managed
    read transaction. In a read transaction the Neo4j server rejects any write clause
    AND any procedure whose declared mode is WRITE/SCHEMA (e.g. ``apoc.cypher.doIt``),
    raising a ClientError rather than mutating the graph — a catch-all the keyword
    denylist in ``_is_read_only_query`` cannot provide.

    For deployments that also provision a read-only Neo4j account (recommended,
    Enterprise RBAC), this remains correct and adds a second, independent layer.

    Args:
        client: A memory client whose ``.graph`` is the upstream Neo4j client.
        query: The Cypher query to execute.
        parameters: Optional query parameters (always passed as bound parameters).

    Returns:
        The query result rows as a list of dicts.
    """
    graph = client.graph
    driver = graph._ensure_connected()
    db_name = graph._config.database

    async def _work(tx: Any) -> list[dict[str, Any]]:
        result = await tx.run(query, parameters or {})
        return await result.data()

    async with driver.session(database=db_name) as session:
        return await session.execute_read(_work)


def register_tools(mcp: FastMCP) -> None:
    """Register all MCP tools on the server.

    Args:
        mcp: FastMCP server instance.
    """

    @mcp.tool()
    @log_tool_call
    async def memory_search(
        ctx: Context,
        query: str,
        limit: int = 10,
        memory_types: list[str] | None = None,
        session_id: str | None = None,
        threshold: float = 0.7,
        graph_augment: bool = True,
        include_expired: bool = True,
    ) -> str:
        """Search across all memory types using hybrid vector + graph search.

        Searches messages, entities, preferences, traces, and facts in the
        memory graph using vector similarity, then augments with graph neighbors.

        Note: Entity relationships in the graph use a generic RELATED_TO edge type
        with the semantic relationship stored as a property. To find specific
        relationship types, use graph_query with:
            MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
            WHERE r.relation_type = "WORKS_AT"
            RETURN a.name, r.relation_type, b.name

        Args:
            include_expired: If True (default), include expired/superseded facts
                (annotated with temporal_status). Set False to hide expired facts.
        """
        client = get_client(ctx)

        if memory_types is None:
            memory_types = ["messages", "entities", "preferences", "traces", "facts"]

        # Step 3: Search function to run per-database
        async def _search_db(client, db_name):
            results: dict[str, list[dict[str, Any]]] = {}

            if "messages" in memory_types:
                messages = await client.short_term.search_messages(
                    query=query,
                    session_id=session_id,
                    limit=limit,
                    threshold=threshold,
                )
                results["messages"] = [
                    {
                        "id": str(msg.id),
                        "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                        "similarity": msg.metadata.get("similarity") if msg.metadata else None,
                    }
                    for msg in messages
                ]

            if "entities" in memory_types:
                entities = await client.long_term.search_entities(
                    query=query,
                    limit=limit,
                )
                results["entities"] = [
                    {
                        "id": str(entity.id),
                        "name": entity.display_name,
                        "type": (
                            entity.type.value if hasattr(entity.type, "value") else str(entity.type)
                        ),
                        "description": entity.description,
                    }
                    for entity in entities
                ]

            if "preferences" in memory_types:
                preferences = await client.long_term.search_preferences(
                    query=query,
                    limit=limit,
                )
                results["preferences"] = [
                    {
                        "id": str(pref.id),
                        "category": pref.category,
                        "preference": pref.preference,
                        "context": pref.context,
                    }
                    for pref in preferences
                ]

            if "traces" in memory_types:
                traces = await client.reasoning.get_similar_traces(
                    task=query,
                    limit=limit,
                )
                results["traces"] = [
                    {
                        "id": str(trace.id),
                        "task": trace.task,
                        "outcome": trace.outcome,
                        "success": trace.success,
                        "session_id": trace.session_id,
                        "started_at": trace.started_at.isoformat() if trace.started_at else None,
                    }
                    for trace in traces
                ]

            if "facts" in memory_types:
                try:
                    from datetime import datetime as _dt
                    from datetime import timezone as _tz

                    facts = await client.long_term.search_facts(
                        query=query,
                        limit=limit,
                        threshold=threshold,
                    )

                    now_epoch = int(_dt.now(_tz.utc).timestamp() * 1000)

                    def _to_epoch(v):
                        # valid_from/valid_until are stored as epoch millis; be
                        # tolerant of a datetime from legacy data.
                        if v is None:
                            return None
                        if isinstance(v, (int, float)):
                            return int(v)
                        try:
                            return int(v.timestamp() * 1000)
                        except AttributeError:
                            return None

                    # Read temporal fields straight from the graph: the upstream
                    # Fact model mangles epoch-int valid_from/valid_until (its
                    # parser replaces unparseable values with utcnow()), which
                    # made superseded facts annotate as "active".
                    temporal_by_id: dict[str, dict[str, Any]] = {}
                    if facts:
                        temporal_rows = await client.graph.execute_read(
                            """
                            MATCH (f:Fact) WHERE f.id IN $ids
                            RETURN f.id AS id,
                                   f.valid_from AS valid_from,
                                   f.valid_until AS valid_until,
                                   f.superseded_by AS superseded_by
                            """,
                            {"ids": [str(f.id) for f in facts]},
                        )
                        temporal_by_id = {row["id"]: row for row in temporal_rows}

                    fact_results = []
                    for fact in facts:
                        temporal = temporal_by_id.get(str(fact.id), {})
                        valid_from_val = _to_epoch(temporal.get("valid_from"))
                        valid_until_val = _to_epoch(temporal.get("valid_until"))

                        # Expired when valid_until is set and in the past.
                        is_expired = (
                            valid_until_val is not None and valid_until_val <= now_epoch
                        )
                        temporal_status = "expired" if is_expired else "active"

                        superseded_by = temporal.get("superseded_by")
                        if superseded_by is None and fact.metadata:
                            superseded_by = fact.metadata.get("superseded_by")

                        fact_results.append({
                            "id": str(fact.id),
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "confidence": fact.confidence,
                            "similarity": fact.metadata.get("similarity") if fact.metadata else None,
                            "valid_from": valid_from_val,
                            "valid_until": valid_until_val,
                            "temporal_status": temporal_status,
                            "superseded_by": superseded_by,
                        })

                    # Sort: active facts first, then by confidence/similarity
                    fact_results.sort(
                        key=lambda f: (
                            0 if f["temporal_status"] == "active" else 1,
                            -(f.get("similarity") or f.get("confidence") or 0),
                        )
                    )

                    if not include_expired:
                        fact_results = [f for f in fact_results if f["temporal_status"] == "active"]

                    results["facts"] = fact_results
                except AttributeError:
                    # Older versions of the upstream library may not have search_facts
                    logger.debug("search_facts not available on this client")
                    results["facts"] = []

            # Graph augmentation — attach neighbors/mentions to vector results
            if graph_augment:
                if "entities" in results and results["entities"]:
                    results["entities"] = await _augment_entities_with_neighbors(
                        client, results["entities"]
                    )
                if "messages" in results and results["messages"]:
                    results["messages"] = await _augment_messages_with_mentions(
                        client, results["messages"]
                    )

            return results

        try:
            merged = await _search_db(client, "memory")
        except Exception as e:
            logger.error(f"Error in memory_search: {e}")
            return json.dumps({"error": str(e)})

        response = {
            "results": merged,
            "query": query,
            "graph_augmented": graph_augment,
        }

        return json.dumps(response, default=str)

    @mcp.tool()
    @log_tool_call
    async def memory_store(
        ctx: Context,
        memory_type: str,
        content: str,
        session_id: str | None = None,
        role: str = "user",
        category: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        confidence: float = 1.0,
        supersedes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory in the knowledge graph.

        Supports messages, facts (SPO triples), and user preferences. Storing a
        message runs a single unified extraction pass that pulls entities
        (POLE+O + domain types as labels), relationships, and preferences and
        persists them to the graph.

        Args:
            memory_type: Type of memory - 'message', 'fact', or 'preference'.
            content: The content to store.
            session_id: Session ID (required for message type).
            role: Message role - 'user', 'assistant', or 'system' (default: 'user').
            category: Preference category (required for preference type).
            subject: Fact subject (required for fact type).
            predicate: Fact predicate/relationship (required for fact type).
            object_value: Fact object (required for fact type).
            valid_from: When this fact became true (ISO 8601). Defaults to now.
            valid_until: When this fact stops being true (ISO 8601). None = still true.
            confidence: Fact confidence score from 0.0 to 1.0 (default: 1.0).
            supersedes: ID of an existing fact this one replaces. Sets valid_until on the old fact.
            metadata: Optional metadata to attach.
        """
        client = get_client(ctx)

        try:
            result_data: dict[str, Any] | None = None

            if memory_type == "message":
                if not session_id:
                    return json.dumps({"error": "session_id required for message storage"})

                # Store the message + embedding. Extraction is a separate unified
                # pass below, so upstream POLE+O extraction is disabled here.
                message = await client.short_term.add_message(
                    session_id=session_id,
                    role=role,
                    content=content,
                    metadata=metadata,
                    extract_entities=False,
                    generate_embedding=True,
                )

                # Unified extraction — never fatal to the stored message.
                extraction_counts = None
                try:
                    from neo4j_agent_memory.extraction.unified import (
                        extract_memory,
                        persist_memory,
                    )

                    extraction = await extract_memory(content)
                    extraction_counts = await persist_memory(
                        client, str(message.id), extraction
                    )
                except Exception as extract_err:
                    logger.warning("Unified extraction failed: %s", extract_err)

                result_data = {
                    "stored": True,
                    "type": "message",
                    "id": str(message.id),
                    "session_id": session_id,
                }
                if extraction_counts:
                    result_data["entities"] = extraction_counts["entities"]
                    result_data["relations"] = extraction_counts["relations"]
                    result_data["preferences"] = extraction_counts["preferences"]

            elif memory_type == "preference":
                if not category:
                    return json.dumps({"error": "category required for preference storage"})

                preference = await client.long_term.add_preference(
                    category=category,
                    preference=content,
                    generate_embedding=True,
                )
                result_data = {
                    "stored": True,
                    "type": "preference",
                    "id": str(preference.id),
                    "category": category,
                }

            elif memory_type == "fact":
                if not all([subject, predicate, object_value]):
                    return json.dumps(
                        {"error": "subject, predicate, and object_value required for fact storage"}
                    )

                # Parse temporal params
                from neo4j_agent_memory.temporal.lifecycle import (
                    parse_iso_datetime,
                    store_fact_atomic,
                )

                parsed_valid_from = parse_iso_datetime(valid_from)
                parsed_valid_until = parse_iso_datetime(valid_until)
                is_current_state = True

                # Attempt temporal extraction from content if valid_from not provided
                import os
                if parsed_valid_from is None and os.environ.get("NAM_TEMPORAL_EXTRACTION", "true").lower() != "false":
                    try:
                        from neo4j_agent_memory.temporal.extraction import extract_temporal_context

                        temporal_ctx = await extract_temporal_context(content)
                        if temporal_ctx["valid_at"]:
                            parsed_valid_from = parse_iso_datetime(temporal_ctx["valid_at"])
                        if temporal_ctx.get("temporal_qualifier"):
                            if metadata is None:
                                metadata = {}
                            metadata["temporal_qualifier"] = temporal_ctx["temporal_qualifier"]
                            metadata["is_current_state"] = temporal_ctx["is_current_state"]
                            is_current_state = bool(temporal_ctx["is_current_state"])
                    except Exception:
                        pass  # Non-critical — proceed without temporal extraction

                def _epoch(dt):
                    return int(dt.timestamp() * 1000) if dt is not None else None

                from_epoch = _epoch(parsed_valid_from)
                until_epoch = _epoch(parsed_valid_until)

                # Create the fact and supersede matching facts in ONE atomic
                # Cypher write (epoch-millis temporal representation throughout).
                # Supersession only runs when this fact is a current, open-ended
                # assertion — historical facts must not invalidate the current
                # one. Re-affirming an identical fact refreshes the existing
                # node (preserving valid_from) instead of duplicating it.
                outcome = await store_fact_atomic(
                    client,
                    subject=subject,
                    predicate=predicate,
                    obj=object_value,
                    confidence=confidence,
                    valid_from_epoch=from_epoch,
                    valid_until_epoch=until_epoch,
                    is_current_state=is_current_state,
                    supersedes_id=supersedes,
                    metadata=metadata,
                )
                fact_id = outcome["fact_id"]
                superseded_count = outcome["superseded_count"]
                refreshed = outcome["refreshed"]

                # Phase 2: LLM contradiction detection (semantic, beyond SPO
                # match). Skipped on a refresh — the identical fact was already
                # screened when first stored.
                contradiction_result = None
                if fact_id and not refreshed and os.environ.get("NAM_CONTRADICTION_DETECTION", "true").lower() != "false":
                    try:
                        from neo4j_agent_memory.temporal.contradiction import detect_and_invalidate

                        contradiction_result = await detect_and_invalidate(
                            client, subject, predicate, object_value, fact_id
                        )
                    except Exception as contradiction_err:
                        logger.warning("Contradiction detection failed: %s", contradiction_err)

                result_data = {
                    "stored": True,
                    "type": "fact",
                    "id": fact_id,
                    "triple": f"{subject} -> {predicate} -> {object_value}",
                    "confidence": confidence,
                    "valid_from": outcome["valid_from"],
                    "valid_until": until_epoch,
                    "superseded_facts": superseded_count,
                    "refreshed": refreshed,
                }

                if contradiction_result:
                    result_data["contradiction_detection"] = {
                        "candidates_found": contradiction_result["candidates_found"],
                        "contradictions_detected": contradiction_result["contradictions_detected"],
                        "facts_invalidated": contradiction_result["facts_invalidated"],
                        "type": contradiction_result["contradiction_type"],
                        "reasoning": contradiction_result["reasoning"],
                    }

            else:
                return json.dumps({"error": f"Unknown memory type: {memory_type}"})

            return json.dumps(result_data)

        except Exception as e:
            logger.error(f"Error in memory_store: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def entity_lookup(
        ctx: Context,
        name: str,
        entity_type: str | None = None,
        include_neighbors: bool = True,
        max_hops: int = 1,
    ) -> str:
        """Look up an entity and retrieve its relationships and neighbors.

        Searches the knowledge graph for entities by name, with optional
        graph traversal to find related entities. Neighbors include the
        semantic relationship type (e.g., WORKS_AT, CREATES, REQUIRES)
        and direction of the connection.
        """
        client = get_client(ctx)

        # Validate/normalize entity_type into a safe Neo4j label. Labels can't be
        # parameterized, so anything that isn't a bare identifier is rejected
        # (prevents Cypher injection via the entity_type argument).
        type_label: str | None = None
        if entity_type:
            from neo4j_agent_memory.graph.query_builder import sanitize_label

            type_label = sanitize_label(entity_type)
            if type_label is None:
                return json.dumps({"error": f"Invalid entity_type: {entity_type!r}"})

        try:
            # Try vector search first (requires entity embeddings)
            entities = await client.long_term.search_entities(
                query=name,
                entity_types=[entity_type] if entity_type else None,
                limit=1,
            )

            # Fallback to Cypher name match if vector search returns nothing
            if not entities:
                type_filter = f" AND e:{type_label}" if type_label else ""
                params: dict[str, Any] = {"name": name}

                cypher_results = await client.graph.execute_read(
                    f"""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name){type_filter}
                    RETURN e.id AS id, e.name AS name, e.type AS type,
                           e.description AS description, labels(e) AS labels
                    LIMIT 1
                    """,
                    params,
                )

                if not cypher_results:
                    return json.dumps({"found": False, "name": name})

                match = cypher_results[0]
                result: dict[str, Any] = {
                    "found": True,
                    "match_method": "name",
                    "entity": {
                        "id": match["id"],
                        "name": match["name"],
                        "type": match["type"],
                        "description": match["description"],
                        "labels": match["labels"],
                    },
                }

                if include_neighbors:
                    neighbors = await _get_entity_neighbors(
                        client, match["id"], max_hops
                    )
                    result["neighbors"] = neighbors

                return json.dumps(result, default=str)

            entity = entities[0]
            result = {
                "found": True,
                "match_method": "vector",
                "entity": {
                    "id": str(entity.id),
                    "name": entity.display_name,
                    "type": (
                        entity.type.value if hasattr(entity.type, "value") else str(entity.type)
                    ),
                    "description": entity.description,
                    "aliases": entity.aliases if hasattr(entity, "aliases") else [],
                },
            }

            if include_neighbors:
                neighbors = await _get_entity_neighbors(client, str(entity.id), max_hops)
                result["neighbors"] = neighbors

            return json.dumps(result, default=str)

        except Exception as e:
            logger.error(f"Error in entity_lookup: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def conversation_history(
        ctx: Context,
        session_id: str,
        limit: int = 50,
        before: str | None = None,
        include_metadata: bool = True,
    ) -> str:
        """Retrieve conversation history for a session.

        Returns messages in chronological order with role, content, and metadata.
        """
        client = get_client(ctx)

        try:
            conversation = await client.short_term.get_conversation(
                session_id=session_id,
                limit=limit,
            )

            messages = []
            for msg in conversation.messages:
                msg_data: dict[str, Any] = {
                    "id": str(msg.id),
                    "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                }
                if include_metadata and msg.metadata:
                    msg_data["metadata"] = msg.metadata
                messages.append(msg_data)

            return json.dumps(
                {
                    "session_id": session_id,
                    "message_count": len(messages),
                    "messages": messages,
                },
                default=str,
            )

        except Exception as e:
            logger.error(f"Error in conversation_history: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def graph_query(
        ctx: Context,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Execute a read-only Cypher query against the knowledge graph.

        Only read queries are allowed. The query runs in a READ-access-mode
        transaction, so the database itself rejects any write clause
        (CREATE, MERGE, DELETE, SET, REMOVE, ...) and any write-mode procedure
        (e.g. apoc.cypher.doIt) — an obvious-keyword pre-check runs first.

        Schema notes for writing effective queries:
        - Entity nodes have labels like :Entity:Person:Individual, :Entity:Organization:Company,
          :Entity:Object:Product, :Entity:Location:City, :Entity:Event
        - Entity relationships all use the edge type :RELATED_TO with the semantic
          relationship stored in the `relation_type` property (e.g., "WORKS_AT", "CREATES",
          "REQUIRES", "LIVES_IN", "OWNS"). Filter with: WHERE r.relation_type = "WORKS_AT"
        - Messages link to entities via :MENTIONS relationships
        - Conversations link to messages via :HAS_MESSAGE, :FIRST_MESSAGE, :NEXT_MESSAGE
        - Facts are stored as :Fact nodes with subject, predicate, object properties
          Fact nodes include temporal properties:
            - valid_from: When the fact became true (epoch millis or ISO string)
            - valid_until: When the fact was superseded (epoch millis, NULL = current)
            - superseded_by: ID of the fact that replaced this one (NULL = current)
            - created_at: When the fact was stored
          Example temporal query:
            MATCH (f:Fact) WHERE f.subject = 'Alice' AND f.valid_until IS NULL
            RETURN f.predicate, f.object  // Only current facts about Alice
        - Preferences are :Preference nodes with category and preference properties
        - ReasoningTrace nodes link to :ReasoningStep via :HAS_STEP, steps link to :ToolCall
        """
        # First-pass fast fail on obvious write keywords; real enforcement is the
        # READ-access-mode transaction in _execute_read_only() below.
        if not _is_read_only_query(query):
            return json.dumps(
                {
                    "error": "Only read-only queries are allowed. "
                    "Write operations (CREATE, MERGE, DELETE, SET, REMOVE) are not permitted."
                }
            )

        client = get_client(ctx)

        try:
            records = await _execute_read_only(client, query, parameters or {})
            return json.dumps(
                {
                    "success": True,
                    "row_count": len(records),
                    "rows": records,
                },
                default=str,
            )

        except Exception as e:
            logger.error(f"Error in graph_query: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def add_reasoning_trace(
        ctx: Context,
        session_id: str,
        task: str,
        steps: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        outcome: str | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a reasoning trace capturing HOW and WHY a task was solved.

        Records the task, reasoning steps with thought/action/observation,
        tool calls, and final outcome. Enables later retrieval via
        explain_reasoning to understand the agent's decision process.


        Args:
            session_id: Session ID for the reasoning trace.
            task: Description of the task being solved.
            steps: List of reasoning steps. Each step should include:
                - thought: WHY this action was chosen (hypothesis, reasoning)
                - tool_name: What tool/action was taken
                - observation: WHAT was learned from the result
                - alternatives_considered: Other approaches weighed (optional)
                - arguments: Tool arguments (optional)
                - result: Raw result (optional)
                - success: Whether this step succeeded (optional)
            tool_calls: Deprecated alias for steps (backward compatibility).
            outcome: Final outcome or result of the task.
            success: Whether the task was completed successfully.
            metadata: Optional metadata (model, latency, etc.).
        """
        from neo4j_agent_memory.memory.reasoning import ToolCallStatus

        client = get_client(ctx)
        # Use steps if provided, fall back to tool_calls for backward compat
        step_list = steps or tool_calls or []

        try:
            trace = await client.reasoning.start_trace(
                session_id=session_id,
                task=task,
                metadata=metadata or {},
            )

            for tc in step_list:
                tool_name = tc.get("tool_name", "unknown")

                # Use caller-provided thought, or generate a default
                thought = tc.get("thought") or f"Calling {tool_name}"

                # Use caller-provided observation, or fall back to raw result
                observation = tc.get("observation") or tc.get("result")
                if observation and not isinstance(observation, str):
                    observation = str(observation)

                # Include alternatives in thought if provided
                alternatives = tc.get("alternatives_considered")
                if alternatives:
                    thought = f"{thought}\n\nAlternatives considered: {alternatives}"

                step = await client.reasoning.add_step(
                    trace_id=trace.id,
                    thought=thought,
                    action=tool_name,
                    observation=observation,
                )

                # Map success field to ToolCallStatus
                tc_success = tc.get("success")
                if tc_success is False:
                    status = ToolCallStatus.FAILURE
                elif tc_success is True:
                    status = ToolCallStatus.SUCCESS
                else:
                    status = ToolCallStatus.SUCCESS

                await client.reasoning.record_tool_call(
                    step_id=step.id,
                    tool_name=tool_name,
                    arguments=tc.get("arguments", {}),
                    result=tc.get("result"),
                    status=status,
                )

            await client.reasoning.complete_trace(
                trace_id=trace.id,
                outcome=outcome,
                success=success,
            )

            return json.dumps({
                "success": True,
                "stored": True,
                "trace_id": str(trace.id),
                "session_id": session_id,
                "task": task,
                "step_count": len(step_list),
            })

        except Exception as e:
            logger.error(f"Error in add_reasoning_trace: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def explain_reasoning(
        ctx: Context,
        trace_id: str | None = None,
        query: str | None = None,
        session_id: str | None = None,
        synthesize: bool = False,
    ) -> str:
        """Retrieve and explain the reasoning behind a past decision or answer.

        Use this to answer "Why did you come up with this answer?" by retrieving
        the full reasoning chain (thought -> action -> observation for each step).

        Provide ONE of:
        - trace_id: Get reasoning for a specific trace
        - query: Find the most relevant trace by semantic similarity
        - session_id: Get the most recent trace from a session

        Set synthesize=true for a natural-language explanation powered by LLM.
        Default returns the raw reasoning chain.
        Set database to search a specific vertical.

        Args:
            trace_id: Specific trace ID to explain.
            query: Semantic search query to find relevant trace.
            session_id: Session ID to get most recent trace from.
            synthesize: If true, produce LLM-synthesized explanation (slower).
        """
        client = get_client(ctx)

        try:
            trace = None

            # Route 1: Direct trace lookup
            if trace_id:
                trace = await client.reasoning.get_trace_with_steps(trace_id)
                if not trace:
                    return json.dumps({"error": f"Trace not found: {trace_id}"})

            # Route 2: Semantic search for most relevant trace
            elif query:
                similar = await client.reasoning.get_similar_traces(
                    task=query, limit=1, success_only=False,
                )
                if not similar:
                    return json.dumps({
                        "found": False,
                        "message": "No reasoning traces found matching query",
                        "query": query,
                    })
                trace = await client.reasoning.get_trace_with_steps(similar[0].id)

            # Route 3: Most recent trace from session
            elif session_id:
                traces = await client.reasoning.list_traces(
                    session_id=session_id, limit=1,
                )
                if not traces:
                    return json.dumps({
                        "found": False,
                        "message": f"No traces found for session: {session_id}",
                    })
                trace = await client.reasoning.get_trace_with_steps(traces[0].id)

            else:
                return json.dumps({
                    "error": "Provide trace_id, query, or session_id"
                })

            if not trace:
                return json.dumps({"error": "Failed to retrieve trace details"})

            # Build the reasoning chain
            chain = {
                "trace_id": str(trace.id),
                "task": trace.task,
                "session_id": trace.session_id,
                "success": trace.success,
                "outcome": trace.outcome,
                "started_at": trace.started_at.isoformat() if trace.started_at else None,
                "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
                "steps": [
                    {
                        "step_number": step.step_number,
                        "thought": step.thought,
                        "action": step.action,
                        "observation": step.observation,
                        "tool_calls": [
                            {
                                "tool_name": tc.tool_name,
                                "arguments": tc.arguments,
                                "status": tc.status.value if hasattr(tc.status, "value") else str(tc.status),
                            }
                            for tc in (step.tool_calls or [])
                        ],
                    }
                    for step in (trace.steps or [])
                ],
            }

            # Optional: synthesize a natural-language explanation
            if synthesize and trace.steps:
                try:
                    from neo4j_agent_memory.extraction.reasoning_extractor import (
                        BamlReasoningExtractor,
                    )

                    extractor = BamlReasoningExtractor()
                    explanation = await extractor.synthesize_explanation(
                        task=trace.task,
                        steps=[
                            {
                                "thought": s.thought or "",
                                "action": s.action or "",
                                "observation": s.observation or "",
                            }
                            for s in trace.steps
                        ],
                        outcome=trace.outcome or "",
                    )
                    chain["synthesized_explanation"] = explanation
                except Exception as synth_err:
                    logger.warning(f"Synthesis failed, returning raw chain: {synth_err}")
                    chain["synthesis_error"] = str(synth_err)

            return json.dumps(chain, default=str)

        except Exception as e:
            logger.error(f"Error in explain_reasoning: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def extract_reasoning(
        ctx: Context,
        text: str,
        session_id: str,
        store: bool = True,
    ) -> str:
        """Extract structured reasoning from conversation text using LLM analysis.

        Analyzes a conversation transcript to identify the reasoning chain:
        hypotheses formed, evidence gathered, decisions made, and conclusions drawn.

        Optionally stores the extracted reasoning as a trace in memory.

        Args:
            text: Conversation transcript or text to analyze.
            session_id: Session ID to associate with the extracted trace.
            store: If true (default), store extracted reasoning as a trace.
        """
        client = get_client(ctx)

        try:
            from neo4j_agent_memory.extraction.reasoning_extractor import (
                BamlReasoningExtractor,
            )

            extractor = BamlReasoningExtractor()
            extracted = await extractor.extract_reasoning(text)

            result = {
                "extracted": True,
                "task": extracted["task"],
                "step_count": len(extracted["steps"]),
                "conclusion": extracted["final_conclusion"],
                "success": extracted["success"],
                "steps": extracted["steps"],
            }

            # Store as a reasoning trace if requested
            if store and extracted["steps"]:
                trace = await client.reasoning.start_trace(
                    session_id=session_id,
                    task=extracted["task"],
                    metadata={"source": "conversation_extraction"},
                )

                for ext_step in extracted["steps"]:
                    await client.reasoning.add_step(
                        trace_id=trace.id,
                        thought=ext_step["thought"],
                        action=ext_step["action"],
                        observation=ext_step["observation"],
                    )

                await client.reasoning.complete_trace(
                    trace_id=trace.id,
                    outcome=extracted["final_conclusion"],
                    success=extracted["success"],
                )

                result["stored"] = True
                result["trace_id"] = str(trace.id)

            return json.dumps(result, default=str)

        except Exception as e:
            logger.error(f"Error in extract_reasoning: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def temporal_query(
        ctx: Context,
        point_in_time: str,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int = 20,
    ) -> str:
        """Query facts that were valid at a specific point in time.

        Returns only facts where valid_from <= point_in_time AND
        (valid_until is NULL or valid_until > point_in_time).
        Useful for understanding what was true at a past date.

        Args:
            point_in_time: ISO 8601 datetime to query against.
            subject: Optional — filter to facts about this subject.
            predicate: Optional — filter to facts with this predicate.
            limit: Maximum results (default 20).
        """
        from neo4j_agent_memory.temporal.lifecycle import (
            parse_iso_datetime,
            temporal_fact_query,
        )

        pit = parse_iso_datetime(point_in_time)
        if not pit:
            return json.dumps({"error": f"Invalid datetime: {point_in_time}"})

        client = get_client(ctx)

        try:
            facts = await temporal_fact_query(
                client, pit, subject=subject, predicate=predicate, limit=limit
            )
            return json.dumps({
                "point_in_time": point_in_time,
                "facts": facts,
                "count": len(facts),
            }, default=str)
        except Exception as e:
            logger.error("temporal_query error: %s", e)
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def fact_evolution(
        ctx: Context,
        subject: str,
        predicate: str | None = None,
        limit: int = 50,
    ) -> str:
        """Trace how knowledge about a subject has evolved over time.

        Returns the full version history of facts about a subject,
        ordered chronologically, showing supersession chains.
        Useful for understanding how plans, statuses, or relationships changed.

        Args:
            subject: The entity/subject to trace.
            predicate: Optional — filter to a specific relationship type.
            limit: Maximum results (default 50).
        """
        from neo4j_agent_memory.temporal.lifecycle import get_fact_evolution

        client = get_client(ctx)

        try:
            evolution = await get_fact_evolution(
                client, subject, predicate=predicate, limit=limit
            )

            # Build summary
            current_facts = [f for f in evolution if f["is_current"]]
            superseded_facts = [f for f in evolution if not f["is_current"]]

            return json.dumps({
                "subject": subject,
                "predicate": predicate,
                "total_versions": len(evolution),
                "current_facts": len(current_facts),
                "superseded_facts": len(superseded_facts),
                "evolution": evolution,
            }, default=str)
        except Exception as e:
            logger.error("fact_evolution error: %s", e)
            return json.dumps({"error": str(e)})

    @mcp.tool()
    @log_tool_call
    async def knowledge_state(
        ctx: Context,
        as_of: str,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int = 20,
    ) -> str:
        """Query what the system knew at a specific point in system time.

        Uses transaction-time filtering: created_at <= as_of AND
        (expired_at is NULL or expired_at > as_of).
        Different from temporal_query which uses event time (when facts were true).

        Args:
            as_of: ISO 8601 datetime — the system time to query.
            subject: Optional — filter to facts about this subject.
            predicate: Optional — filter to facts with this predicate.
            limit: Maximum results (default 20).
            database: Target database vertical.
        """
        from neo4j_agent_memory.temporal.lifecycle import parse_iso_datetime

        as_of_dt = parse_iso_datetime(as_of)
        if not as_of_dt:
            return json.dumps({"error": f"Invalid datetime: {as_of}"})

        client = get_client(ctx)

        as_of_epoch = int(as_of_dt.timestamp() * 1000)
        params: dict[str, Any] = {"as_of": as_of_epoch, "limit": limit}

        where_clauses = [
            "f.created_at <= datetime({epochMillis: $as_of})",
            "(f.expired_at IS NULL OR f.expired_at > $as_of)",
        ]
        if subject:
            where_clauses.append("f.subject = $subject")
            params["subject"] = subject
        if predicate:
            where_clauses.append("f.predicate = $predicate")
            params["predicate"] = predicate

        where = " AND ".join(where_clauses)

        try:
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
                       f.expired_at AS expired_at,
                       f.valid_from AS valid_from,
                       f.valid_until AS valid_until
                ORDER BY f.confidence DESC, f.created_at DESC
                LIMIT $limit
                """,
                params,
            )

            facts = [
                {
                    "id": row["id"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "confidence": row["confidence"],
                    "created_at": str(row["created_at"]) if row["created_at"] else None,
                    "expired_at": str(row["expired_at"]) if row["expired_at"] else None,
                    "valid_from": str(row["valid_from"]) if row["valid_from"] else None,
                    "valid_until": str(row["valid_until"]) if row["valid_until"] else None,
                }
                for row in result
            ]

            return json.dumps({
                "as_of": as_of,
                "facts": facts,
                "count": len(facts),
            }, default=str)

        except Exception as e:
            logger.error("knowledge_state error: %s", e)
            return json.dumps({"error": str(e)})


async def _get_entity_neighbors(
    client,
    entity_id: str,
    max_hops: int = 1,
) -> list[dict[str, Any]]:
    """Get neighboring entities via graph traversal with relationship details.

    Args:
        client: MemoryClient instance.
        entity_id: Starting entity ID.
        max_hops: Maximum relationship depth (clamped to 1-3).

    Returns:
        List of neighboring entities with relationship type and direction.
    """
    max_hops = min(max(max_hops, 1), 3)
    query = """
    MATCH (e:Entity {id: $entity_id})-[r:RELATED_TO]-(neighbor:Entity)
    WHERE neighbor.id <> $entity_id
    WITH DISTINCT neighbor, r,
         CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END AS direction
    RETURN neighbor.id AS id,
           neighbor.name AS name,
           neighbor.type AS type,
           neighbor.description AS description,
           coalesce(r.relation_type, r.type, 'RELATED_TO') AS relation_type,
           r.confidence AS confidence,
           direction
    ORDER BY r.confidence DESC
    LIMIT 20
    """

    try:
        records = await client.graph.execute_read(
            query,
            {"entity_id": entity_id},
        )
        neighbors = [
            {
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "relationship": r["relation_type"],
                "direction": r["direction"],
                "confidence": r["confidence"],
            }
            for r in records
        ]

        # For multi-hop, also get second-degree connections
        if max_hops >= 2:
            query_2hop = """
            MATCH (e:Entity {id: $entity_id})-[:RELATED_TO]-(:Entity)
                  -[r2:RELATED_TO]-(hop2:Entity)
            WHERE hop2.id <> $entity_id
              AND NOT (e)-[:RELATED_TO]-(hop2)
            WITH DISTINCT hop2, r2,
                 CASE WHEN startNode(r2) = hop2 THEN 'incoming' ELSE 'outgoing' END AS direction
            RETURN hop2.id AS id,
                   hop2.name AS name,
                   hop2.type AS type,
                   hop2.description AS description,
                   coalesce(r2.relation_type, r2.type, 'RELATED_TO') AS relation_type,
                   r2.confidence AS confidence,
                   direction
            LIMIT 10
            """
            try:
                hop2_records = await client.graph.execute_read(
                    query_2hop,
                    {"entity_id": entity_id},
                )
                for r in hop2_records:
                    neighbors.append(
                        {
                            "id": r["id"],
                            "name": r["name"],
                            "type": r["type"],
                            "description": r["description"],
                            "relationship": r["relation_type"],
                            "direction": r["direction"],
                            "confidence": r["confidence"],
                            "hop": 2,
                        }
                    )
            except Exception:
                pass  # Second hop is best-effort

        return neighbors
    except Exception as e:
        logger.debug(f"Error getting neighbors: {e}")
        return []


async def _augment_entities_with_neighbors(
    client,
    entity_results: list[dict[str, Any]],
    max_neighbors: int = 5,
) -> list[dict[str, Any]]:
    """Attach 1-hop RELATED_TO neighbors to each entity result.

    Adds a 'neighbors' list to each entity dict containing related
    entities from graph traversal. This provides graph context beyond
    what vector similarity alone returns.

    Args:
        client: MemoryClient instance.
        entity_results: List of entity dicts from vector search.
        max_neighbors: Max neighbors per entity (keeps response size bounded).

    Returns:
        The same entity_results list, with 'neighbors' added to each item.
    """
    for entity in entity_results:
        entity_id = entity.get("id")
        if not entity_id:
            entity["neighbors"] = []
            continue
        try:
            neighbors = await _get_entity_neighbors(client, entity_id, max_hops=1)
            entity["neighbors"] = neighbors[:max_neighbors]
        except Exception:
            entity["neighbors"] = []
    return entity_results


async def _augment_messages_with_mentions(
    client,
    message_results: list[dict[str, Any]],
    max_mentions: int = 5,
) -> list[dict[str, Any]]:
    """Attach entities mentioned by each message via MENTIONS edges.

    Args:
        client: MemoryClient instance.
        message_results: List of message dicts from vector search.
        max_mentions: Max mentioned entities per message.

    Returns:
        The same message_results list, with 'mentioned_entities' added.
    """
    for msg in message_results:
        msg_id = msg.get("id")
        if not msg_id:
            msg["mentioned_entities"] = []
            continue
        try:
            records = await client.graph.execute_read(
                """
                MATCH (m:Message {id: $message_id})-[:MENTIONS]->(e:Entity)
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       e.description AS description
                LIMIT $limit
                """,
                {"message_id": msg_id, "limit": max_mentions},
            )
            msg["mentioned_entities"] = [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "type": r["type"],
                    "description": r["description"],
                }
                for r in records
            ]
        except Exception:
            msg["mentioned_entities"] = []
    return message_results

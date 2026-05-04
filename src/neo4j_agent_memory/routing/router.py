"""BAML-powered query router for multi-database verticals."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

from cachetools import TTLCache

from neo4j_agent_memory.verticals import get_vertical_to_db

logger = logging.getLogger(__name__)

# Mapping from BAML enum values to Neo4j database names
VERTICAL_TO_DB: dict[str, str] = get_vertical_to_db()

# Key normalization: collapse whitespace, strip punctuation, lowercase, truncate
_NORM_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_MAX_KEY_LEN = 128


def _normalize_key(*parts: str | None) -> str:
    """Build a cache key from input parts.

    Lowercases, strips punctuation, collapses whitespace, and truncates
    to _MAX_KEY_LEN chars. Readable in logs and sufficient for a 256-entry
    cache where collision risk is irrelevant.
    """
    raw = "|".join(str(p) for p in parts if p is not None)
    normed = _NORM_RE.sub("", raw.lower())
    normed = _WS_RE.sub(" ", normed).strip()
    return normed[:_MAX_KEY_LEN]


class CacheStats:
    """Simple hit/miss/eviction counters for observability."""

    __slots__ = ("hits", "misses", "coalesced")

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.coalesced = 0  # requests that awaited an in-flight LLM call

    def to_dict(self) -> dict[str, int]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "coalesced": self.coalesced,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


class QueryRouter:
    """Routes queries and storage requests to appropriate database verticals."""

    def __init__(
        self,
        available_databases: list[str],
        confidence_threshold: float = 0.3,
    ) -> None:
        self._available_dbs = set(available_databases)
        self._threshold = confidence_threshold
        self._enabled = os.environ.get(
            "NAM_ROUTING_ENABLED", "true"
        ).lower() in ("true", "1", "yes")

        # Cache configuration via env vars
        cache_enabled = os.environ.get(
            "NAM_ROUTING_CACHE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
        cache_size = int(os.environ.get("NAM_ROUTING_CACHE_SIZE", "256"))
        cache_ttl = int(os.environ.get("NAM_ROUTING_CACHE_TTL", "300"))

        self._cache_enabled = cache_enabled
        self._query_cache: TTLCache[str, RoutingResult] = TTLCache(
            maxsize=cache_size, ttl=cache_ttl
        )
        self._storage_cache: TTLCache[str, RoutingResult] = TTLCache(
            maxsize=cache_size, ttl=cache_ttl
        )

        # Pending futures for stampede prevention
        self._query_pending: dict[str, asyncio.Future[RoutingResult]] = {}
        self._storage_pending: dict[str, asyncio.Future[RoutingResult]] = {}

        # Observability
        self.query_cache_stats = CacheStats()
        self.storage_cache_stats = CacheStats()

    async def route_query(
        self,
        query: str,
        context: str | None = None,
    ) -> RoutingResult:
        """Route a search query to target database(s).

        Falls back to general DB if routing is disabled or fails.
        """
        if not self._enabled:
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

        cache_key = _normalize_key(query, context)

        # Check cache
        if self._cache_enabled:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self.query_cache_stats.hits += 1
                logger.info(
                    "routing_decision: op=query primary=%s cache=hit input=%s",
                    cached.primary, query[:80],
                )
                return cached

            # Coalesce concurrent requests for the same key
            if cache_key in self._query_pending:
                self.query_cache_stats.coalesced += 1
                logger.debug("Route cache COALESCE: %s", cache_key[:60])
                return await asyncio.shield(self._query_pending[cache_key])

        self.query_cache_stats.misses += 1

        # Set up future for stampede prevention
        future: asyncio.Future[RoutingResult] | None = None
        if self._cache_enabled:
            future = asyncio.get_event_loop().create_future()
            self._query_pending[cache_key] = future

        start_time = time.monotonic()
        try:
            from neo4j_agent_memory.baml_client.async_client import b

            decision = await b.RouteQuery(query=query, context=context)
            result = self._to_result(decision)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "routing_decision: op=query primary=%s targets=%s fanout=%s ambiguous=%s cache=miss latency_ms=%.0f input=%s",
                result.primary,
                [(db, round(conf, 2)) for db, conf in result.targets],
                result.requires_fanout,
                result.ambiguous,
                elapsed_ms,
                query[:80],
            )

            if self._cache_enabled:
                self._query_cache[cache_key] = result
                if future is not None:
                    future.set_result(result)

            return result
        except Exception as e:
            logger.warning("Route failed, defaulting to general: %s", e)
            fallback = RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )
            if future is not None and not future.done():
                future.set_result(fallback)
            return fallback
        finally:
            self._query_pending.pop(cache_key, None)

    async def route_storage(
        self,
        content: str,
        memory_type: str,
        context: str | None = None,
    ) -> RoutingResult:
        """Route a storage request to the target database.

        Falls back to general DB if routing is disabled or fails.
        """
        if not self._enabled:
            return RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )

        cache_key = _normalize_key(content, memory_type, context)

        # Check cache
        if self._cache_enabled:
            cached = self._storage_cache.get(cache_key)
            if cached is not None:
                self.storage_cache_stats.hits += 1
                logger.info(
                    "routing_decision: op=storage primary=%s cache=hit input=%s",
                    cached.primary, content[:80],
                )
                return cached

            # Coalesce concurrent requests for the same key
            if cache_key in self._storage_pending:
                self.storage_cache_stats.coalesced += 1
                logger.debug("Storage route cache COALESCE: %s", cache_key[:60])
                return await asyncio.shield(self._storage_pending[cache_key])

        self.storage_cache_stats.misses += 1

        # Set up future for stampede prevention
        future: asyncio.Future[RoutingResult] | None = None
        if self._cache_enabled:
            future = asyncio.get_event_loop().create_future()
            self._storage_pending[cache_key] = future

        start_time = time.monotonic()
        try:
            from neo4j_agent_memory.baml_client.async_client import b

            decision = await b.RouteStorage(
                content=content,
                memory_type=memory_type,
                context=context,
            )
            result = self._to_result(decision)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "routing_decision: op=storage primary=%s targets=%s fanout=%s ambiguous=%s cache=miss latency_ms=%.0f input=%s",
                result.primary,
                [(db, round(conf, 2)) for db, conf in result.targets],
                result.requires_fanout,
                result.ambiguous,
                elapsed_ms,
                content[:80],
            )

            if self._cache_enabled:
                self._storage_cache[cache_key] = result
                if future is not None:
                    future.set_result(result)

            return result
        except Exception as e:
            logger.warning("Storage route failed, defaulting to general: %s", e)
            fallback = RoutingResult(
                targets=[("neo4j", 1.0)],
                primary="neo4j",
                requires_fanout=False,
            )
            if future is not None and not future.done():
                future.set_result(fallback)
            return fallback
        finally:
            self._storage_pending.pop(cache_key, None)

    def _to_result(self, decision) -> RoutingResult:
        """Convert BAML RoutingDecision to internal RoutingResult."""
        targets = []
        for target in decision.targets:
            db_name = VERTICAL_TO_DB.get(target.vertical.value, "neo4j")
            if db_name in self._available_dbs and target.confidence >= self._threshold:
                targets.append((db_name, target.confidence))

        # Ensure at least general DB
        if not targets:
            targets = [("neo4j", 1.0)]

        primary = VERTICAL_TO_DB.get(decision.primary_vertical.value, "neo4j")
        if primary not in self._available_dbs:
            primary = "neo4j"

        return RoutingResult(
            targets=targets,
            primary=primary,
            requires_fanout=decision.requires_fanout,
            ambiguous=decision.ambiguous,
            disambiguation_options=decision.disambiguation_options or [],
        )


class ResultReranker:
    """Re-ranks results from multi-database fan-out queries."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score and filter results by relevance to the original query.

        Args:
            query: The original user query.
            results: List of result dicts with id, content, _source_db, type.

        Returns:
            Filtered and re-ordered results.
        """
        if not self._enabled or not results:
            return results

        # Skip re-ranking for small result sets (not worth the LLM call)
        if len(results) <= 3:
            return results

        try:
            from neo4j_agent_memory.baml_client.async_client import b

            items = [
                {
                    "id": r.get("id", ""),
                    "content": r.get("content") or r.get("name") or r.get("task") or str(r),
                    "source_db": r.get("_source_db", "unknown"),
                    "result_type": r.get("_result_type", "unknown"),
                }
                for r in results
            ]

            reranked = await b.RerankResults(query=query, results=items)

            # Build lookup of kept results with their scores
            kept_ids = {}
            for scored in reranked.scored_results:
                if scored.keep:
                    kept_ids[scored.id] = scored.relevance

            # Filter and re-sort original results
            filtered = [r for r in results if r.get("id", "") in kept_ids]
            filtered.sort(
                key=lambda r: kept_ids.get(r.get("id", ""), 0),
                reverse=True,
            )

            logger.info(
                "Re-ranked %d results -> %d kept for query: %s",
                len(results), len(filtered), query[:80],
            )
            return filtered

        except Exception as e:
            logger.warning("Re-ranking failed, returning unfiltered: %s", e)
            return results


class RoutingResult:
    """Result of a routing decision."""

    def __init__(
        self,
        targets: list[tuple[str, float]],
        primary: str,
        requires_fanout: bool,
        ambiguous: bool = False,
        disambiguation_options: list[str] | None = None,
    ) -> None:
        self.targets = targets  # [(db_name, confidence), ...]
        self.primary = primary
        self.requires_fanout = requires_fanout
        self.ambiguous = ambiguous
        self.disambiguation_options = disambiguation_options or []

    @property
    def target_databases(self) -> list[str]:
        """List of database names to query, ordered by confidence."""
        return [db for db, _ in self.targets]

    @property
    def primary_database(self) -> str:
        """The single most relevant database."""
        return self.primary

    def to_metadata(self) -> dict:
        """Serialize routing decision as metadata for tool responses."""
        return {
            "targets": [
                {"database": db, "confidence": round(conf, 3)}
                for db, conf in self.targets
            ],
            "primary": self.primary,
            "requires_fanout": self.requires_fanout,
            "ambiguous": self.ambiguous,
        }

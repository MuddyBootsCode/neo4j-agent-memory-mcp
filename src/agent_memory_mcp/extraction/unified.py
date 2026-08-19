"""Conversational-path extraction adapter and preference persistence.

The org ontology (``ExtractMemory``: POLE+O + domain entities and relations)
is retired. Conversational message storage now routes through the coding
extraction (:func:`agent_memory_mcp.extraction.coding.extract_coding_memory`)
with best-effort context — no branch, task, or touched files — so the
anchored coding types (decisions, gotchas, dead ends) all drop by design and
only preferences survive. The hook capture path, not message storage, is the
source of anchored coding memory.

This module keeps two pieces:

- :class:`UnifiedBamlExtractor` — the upstream ``EntityExtractor``-protocol
  adapter wired in via the extractor factory patch.
- :func:`persist_preferences` — writes extracted preferences through the
  upstream ``add_preference`` pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _clamp(value: Any, default: float = 0.8) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


class UnifiedBamlExtractor:
    """Upstream ``EntityExtractor``-protocol adapter over the coding extraction.

    Keeps the client-level extraction hook alive:
    ``MemoryClient.short_term.add_message(extract_entities=True)`` looks up its
    extractor through the upstream factory, which only knows spaCy/GLiNER/OpenAI.
    Wired in via :func:`agent_memory_mcp.mcp._extractor_patch.patch_extractor_factory`,
    this adapter runs the same ``ExtractCodingMemory`` call used by the MCP
    ``memory_store`` tool. Message text carries no session context (branch,
    task, touched files), so anchored coding memory (decisions, gotchas, dead
    ends) drops by design and the result carries preferences only — entities
    and relations are always empty. The hook capture path is the source of
    anchored coding memory.
    """

    name = "UnifiedBamlExtractor"

    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        extract_relations: bool = True,
        extract_preferences: bool = True,
    ) -> Any:
        from neo4j_agent_memory.core.exceptions import ExtractionError
        from neo4j_agent_memory.extraction.base import (
            ExtractedPreference,
            ExtractionResult,
        )

        from agent_memory_mcp.extraction.coding import extract_coding_memory

        try:
            extraction = await extract_coding_memory(
                text, branch="", task=None, files=[]
            )
        except Exception as exc:
            raise ExtractionError(
                f"Coding BAML extraction failed ({type(exc).__name__}): {exc}"
            ) from exc

        preferences = []
        if extract_preferences:
            preferences = [
                ExtractedPreference(
                    category=p["category"],
                    preference=p["preference"],
                    context=None,
                    confidence=p["confidence"],
                )
                for p in extraction["preferences"]
            ]

        return ExtractionResult(
            entities=[],
            relations=[],
            preferences=preferences,
            source_text=text,
        )


async def persist_preferences(
    client: Any, preferences: list[dict[str, Any]]
) -> int:
    """Persist extracted preferences through the upstream pipeline.

    Each preference goes through ``client.long_term.add_preference``;
    persistence is best-effort per item (a failure is logged, never raised).
    Returns the number of preferences stored.
    """
    preferences_stored = 0
    for pref in preferences:
        try:
            await client.long_term.add_preference(
                category=pref["category"],
                preference=pref["preference"],
                context=pref.get("context"),
                confidence=_clamp(pref.get("confidence"), 0.7),
            )
            preferences_stored += 1
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort per item
            logger.warning(
                "Failed to persist preference '%s': %s", pref.get("preference"), exc
            )

    logger.info("Persisted %d preferences", preferences_stored)
    return preferences_stored

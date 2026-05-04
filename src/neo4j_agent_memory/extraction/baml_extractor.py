"""BAML-based entity extraction with multi-provider support."""

import logging
from typing import Any

from neo4j_agent_memory.extraction.base import (
    ExtractedEntity,
    ExtractedPreference,
    ExtractedRelation,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

DEFAULT_BAML_CLIENT = "Bedrock"
DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "OBJECT"]


class BamlEntityExtractor:
    """Entity extractor powered by BAML with multi-provider LLM support.

    Satisfies the EntityExtractor protocol. Uses BAML's generated client
    for type-safe structured extraction with automatic retries and
    fallback chains.

    Provider selection:
        - Set ``client_name`` to choose: "Bedrock", "OpenAI", "Anthropic", "Gemini", "Resilient"
        - Pass a ``ClientRegistry`` for runtime provider switching
    """

    def __init__(
        self,
        *,
        client_name: str = DEFAULT_BAML_CLIENT,
        entity_types: list[str] | None = None,
        extract_relations: bool = True,
        extract_preferences: bool = True,
        client_registry: Any | None = None,
    ):
        self._client_name = client_name
        self._entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self._extract_relations = extract_relations
        self._extract_preferences = extract_preferences
        self._client_registry = client_registry
        self._baml_options: dict[str, Any] = {}

        if client_registry:
            self._baml_options["client_registry"] = client_registry
        elif client_name != DEFAULT_BAML_CLIENT:
            try:
                from baml_py import ClientRegistry

                registry = ClientRegistry()
                registry.set_primary(client_name)
                self._baml_options["client_registry"] = registry
            except ImportError:
                logger.warning("baml-py not installed, client_name override ignored")

    @property
    def name(self) -> str:
        return f"BamlEntityExtractor({self._client_name})"

    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        extract_relations: bool = True,
        extract_preferences: bool = True,
    ) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult(source_text=text)

        try:
            from neo4j_agent_memory.baml_client.async_client import b
        except ImportError:
            raise RuntimeError(
                "BAML client not generated. Run: uv run baml-cli generate"
            )

        types_to_use = entity_types or self._entity_types
        entity_types_str = ", ".join(types_to_use)

        try:
            result = await b.ExtractEntities(
                text=text,
                entity_types=entity_types_str,
                baml_options=self._baml_options if self._baml_options else {},
            )

            entities = [
                ExtractedEntity(
                    name=e.name,
                    type=e.type.value if hasattr(e.type, "value") else str(e.type),
                    subtype=e.subtype,
                    confidence=max(0.0, min(1.0, e.confidence)),
                    extractor="baml",
                )
                for e in result.entities
            ]

            include_relations = self._extract_relations if extract_relations is True else extract_relations
            relations = []
            if include_relations:
                entity_names = {e.name.lower() for e in entities}
                relations = [
                    ExtractedRelation(
                        source=r.source,
                        target=r.target,
                        relation_type=r.relation_type.upper(),
                        confidence=max(0.0, min(1.0, r.confidence)),
                    )
                    for r in result.relations
                    if r.source.lower() in entity_names
                    and r.target.lower() in entity_names
                ]

            include_preferences = self._extract_preferences if extract_preferences is True else extract_preferences
            preferences = []
            if include_preferences:
                preferences = [
                    ExtractedPreference(
                        category=p.category,
                        preference=p.preference,
                        context=p.context,
                        confidence=max(0.0, min(1.0, p.confidence)),
                    )
                    for p in result.preferences
                ]

            logger.debug(
                "BAML extracted %d entities, %d relations, %d preferences (client=%s)",
                len(entities),
                len(relations),
                len(preferences),
                self._client_name,
            )

            return ExtractionResult(
                entities=entities,
                relations=relations,
                preferences=preferences,
                source_text=text,
            )

        except Exception as e:
            from neo4j_agent_memory.core.exceptions import ExtractionError

            raise ExtractionError(
                f"BAML extraction failed ({type(e).__name__}): {e}"
            ) from e

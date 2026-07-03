"""Patch the upstream extractor factory with unified BAML extraction.

The base package's ``create_extractor`` only knows spaCy/GLiNER/OpenAI-LLM/
pipeline extractors — none of which are available in this deployment. When
``NAM_EXTRACTION__BAML_ENABLED=true``, route extractor creation to the
:class:`~neo4j_agent_memory.extraction.unified.UnifiedBamlExtractor` so the
client-level path (``add_message(extract_entities=True)``) runs the same
single-pass ``ExtractMemory`` call as the MCP ``memory_store`` tool.

The env var overrides regardless of the configured extractor_type, matching
the pre-refactor factory_ext behaviour [RFI-F1]. Without the flag, the base
factory is used unchanged.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _is_baml_enabled() -> bool:
    """Check if BAML extraction is enabled via env var (read at call time)."""
    return os.environ.get("NAM_EXTRACTION__BAML_ENABLED", "").lower() in (
        "true",
        "1",
        "yes",
    )


def patch_extractor_factory() -> None:
    """Replace ``extraction.factory.create_extractor`` with a BAML-aware wrapper.

    Idempotent — safe to call multiple times. ``MemoryClient._create_extractor``
    imports ``create_extractor`` from the factory module at call time, so
    patching the module attribute takes effect for every client built after
    this runs.
    """
    import neo4j_agent_memory.extraction.factory as factory_mod

    base_create = factory_mod.create_extractor
    if getattr(base_create, "_nam_baml_patched", False):
        return

    def create_extractor(extraction_config, schema_config=None, llm_config=None):
        if _is_baml_enabled():
            from neo4j_agent_memory.extraction.unified import UnifiedBamlExtractor

            logger.info(
                "BAML extraction enabled (overriding extractor_type=%s)",
                getattr(extraction_config, "extractor_type", None),
            )
            return UnifiedBamlExtractor()
        return base_create(extraction_config, schema_config, llm_config)

    create_extractor._nam_baml_patched = True  # type: ignore[attr-defined]
    factory_mod.create_extractor = create_extractor
    logger.info("Extractor factory patched with unified BAML support")

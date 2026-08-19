"""Patch the upstream extractor factory with unified BAML extraction.

The upstream ``create_extractor`` only knows spaCy/GLiNER/OpenAI-LLM/
pipeline extractors — none of which are available in this deployment. When
``NAM_EXTRACTION__BAML_ENABLED=true``, route extractor creation to the
:class:`~agent_memory_mcp.extraction.unified.UnifiedBamlExtractor` so the
client-level path (``add_message(extract_entities=True)``) runs the same
``ExtractCodingMemory`` call as the MCP ``memory_store`` tool (with no
session context, so only preferences survive).

The env var overrides regardless of the configured extractor_type, matching
the pre-refactor factory_ext behaviour [RFI-F1]. Without the flag, the base
factory is used unchanged.

Apply via :func:`agent_memory_mcp.mcp._bootstrap.bootstrap_upstream_patches`.
The patch validates its upstream target and raises
:class:`~agent_memory_mcp.mcp._embedder_patch.PatchTargetError` if upstream
changed shape — never a silent no-op.
"""

import inspect
import logging
import os

from agent_memory_mcp.mcp._embedder_patch import PatchTargetError

logger = logging.getLogger(__name__)

# Parameter names the wrapper forwards positionally; if upstream renames or
# reorders these, the patch must fail loudly instead of mis-forwarding.
_EXPECTED_FACTORY_PARAMS = ("extraction_config", "schema_config", "llm_config")


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

    Raises:
        PatchTargetError: if the upstream factory module no longer exposes a
            callable ``create_extractor`` with the expected parameters, or if
            the :class:`UnifiedBamlExtractor` replacement cannot be imported.
    """
    try:
        import neo4j_agent_memory.extraction.factory as factory_mod
    except ImportError as exc:
        raise PatchTargetError(
            "Cannot patch extractor factory: upstream module "
            "neo4j_agent_memory.extraction.factory is unavailable"
        ) from exc

    base_create = getattr(factory_mod, "create_extractor", None)
    if base_create is None or not callable(base_create):
        raise PatchTargetError(
            "Cannot patch extractor factory: upstream "
            "neo4j_agent_memory.extraction.factory has no callable "
            "'create_extractor'. The pinned neo4j-agent-memory version "
            "changed shape; update agent_memory_mcp.mcp._extractor_patch."
        )

    if getattr(base_create, "_nam_baml_patched", False):
        return  # already applied

    params = tuple(inspect.signature(base_create).parameters)
    if params[: len(_EXPECTED_FACTORY_PARAMS)] != _EXPECTED_FACTORY_PARAMS:
        raise PatchTargetError(
            "Cannot patch extractor factory: upstream create_extractor "
            f"signature changed (expected leading parameters "
            f"{_EXPECTED_FACTORY_PARAMS}, found {params}). Update "
            "agent_memory_mcp.mcp._extractor_patch to match."
        )

    # Validate the replacement is importable now, not on the first call.
    try:
        from agent_memory_mcp.extraction.unified import (  # noqa: F401
            UnifiedBamlExtractor,
        )
    except ImportError as exc:
        raise PatchTargetError(
            "Cannot patch extractor factory: "
            "agent_memory_mcp.extraction.unified.UnifiedBamlExtractor is "
            "not importable"
        ) from exc

    def create_extractor(extraction_config, schema_config=None, llm_config=None):
        if _is_baml_enabled():
            from agent_memory_mcp.extraction.unified import UnifiedBamlExtractor

            logger.info(
                "BAML extraction enabled (overriding extractor_type=%s)",
                getattr(extraction_config, "extractor_type", None),
            )
            return UnifiedBamlExtractor()
        return base_create(extraction_config, schema_config, llm_config)

    create_extractor._nam_baml_patched = True  # type: ignore[attr-defined]
    factory_mod.create_extractor = create_extractor
    logger.info("Extractor factory patched with unified BAML support")

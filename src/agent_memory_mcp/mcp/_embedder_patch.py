"""Patch upstream ``MemoryClient._create_embedder`` with Bedrock support.

The upstream factory (neo4j-agent-memory, pinned in pyproject.toml) only
handles OpenAI and SentenceTransformers — ``EmbeddingProvider.BEDROCK``
exists in its settings but falls through to ``return None``. This adds the
Bedrock branch used in production (EC2 via IAM role).

Apply via :func:`agent_memory_mcp.mcp._bootstrap.bootstrap_upstream_patches`,
NOT by calling :func:`patch_embedder_factory` from server code directly.
The patch validates its upstream targets and raises
:class:`PatchTargetError` if upstream changed shape — never a silent no-op.
"""

import logging

logger = logging.getLogger(__name__)


class PatchTargetError(RuntimeError):
    """An upstream patch target is missing or changed shape."""


def _create_embedder_extended(self):
    """Extended _create_embedder with Bedrock support."""
    from neo4j_agent_memory.config.settings import EmbeddingProvider

    config = self._settings.embedding

    if config.provider == EmbeddingProvider.OPENAI:
        from neo4j_agent_memory.embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(
            model=config.model,
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            dimensions=config.dimensions if config.dimensions != 1536 else None,
            batch_size=config.batch_size,
        )
    elif config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        from neo4j_agent_memory.embeddings.sentence_transformers import (
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder(
            model_name=config.model,
            device=config.device,
        )
    elif config.provider == EmbeddingProvider.BEDROCK:
        from neo4j_agent_memory.embeddings.bedrock import BedrockEmbedder

        logger.info(
            "Creating Bedrock embedder (model=%s, region=%s)",
            config.model,
            config.aws_region,
        )
        return BedrockEmbedder(
            model=config.model,
            region_name=config.aws_region,
            profile_name=config.aws_profile,
            batch_size=config.batch_size,
        )
    else:
        return None


# Marker so re-application and tests can detect the patched state.
_create_embedder_extended._nam_bedrock_patched = True  # type: ignore[attr-defined]


def patch_embedder_factory() -> None:
    """Replace ``MemoryClient._create_embedder`` with the Bedrock-aware version.

    Idempotent — re-applying is a no-op if the extended factory is already
    installed.

    Raises:
        PatchTargetError: if the upstream ``MemoryClient`` no longer exposes
            ``_create_embedder``, if ``EmbeddingProvider.BEDROCK`` is gone
            from upstream settings, or if the upstream ``BedrockEmbedder``
            cannot be imported. Any of these means the pinned upstream
            changed shape and the patch would otherwise silently not work.
    """
    from neo4j_agent_memory import MemoryClient

    current = getattr(MemoryClient, "_create_embedder", None)
    if current is None or not callable(current):
        raise PatchTargetError(
            "Cannot patch embedder factory: upstream MemoryClient has no "
            "callable '_create_embedder' method. The pinned neo4j-agent-memory "
            "version changed shape; update agent_memory_mcp.mcp._embedder_patch."
        )

    if getattr(current, "_nam_bedrock_patched", False):
        return  # already applied

    # Validate the upstream pieces the extended factory depends on, so a
    # missing branch fails at startup rather than on the first embed call.
    try:
        from neo4j_agent_memory.config.settings import EmbeddingProvider

        EmbeddingProvider.BEDROCK
    except (ImportError, AttributeError) as exc:
        raise PatchTargetError(
            "Cannot patch embedder factory: upstream "
            "EmbeddingProvider.BEDROCK is unavailable"
        ) from exc

    try:
        from neo4j_agent_memory.embeddings.bedrock import BedrockEmbedder  # noqa: F401
    except ImportError as exc:
        raise PatchTargetError(
            "Cannot patch embedder factory: upstream "
            "neo4j_agent_memory.embeddings.bedrock.BedrockEmbedder is "
            "unavailable"
        ) from exc

    MemoryClient._create_embedder = _create_embedder_extended
    logger.info("Embedder factory patched with Bedrock support")

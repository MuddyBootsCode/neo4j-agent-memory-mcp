"""Patch MemoryClient._create_embedder with Bedrock support.

The base package's factory only handles OpenAI and SentenceTransformers.
This adds the Bedrock branch used in production (EC2 via IAM role).
"""

import logging

logger = logging.getLogger(__name__)


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


def patch_embedder_factory():
    """Patch MemoryClient._create_embedder with Bedrock support."""
    from neo4j_agent_memory import MemoryClient

    MemoryClient._create_embedder = _create_embedder_extended
    logger.info("Embedder factory patched with Bedrock support")

"""Provider selection: use the Anthropic API directly when a key is present.

If ``ANTHROPIC_API_KEY`` is set, extraction calls route to the Anthropic API
through a runtime BAML ClientRegistry instead of the Bedrock-first
``Resilient`` chain, and embeddings default to the local
sentence-transformers model (Anthropic has no embeddings API). Explicit env
configuration always wins over these key-derived defaults.

Environment:
    ANTHROPIC_API_KEY        Enables Anthropic-direct mode.
    NAM_ANTHROPIC_MODEL      Extraction model (default claude-opus-5).
    NAM_EMBEDDING_PROVIDER   Explicit embedder choice — overrides the
                             key-derived sentence_transformers default.
    NAM_LLM_PROVIDER         Explicit LLM choice. "ollama" routes extraction
                             to a local model and wins over the key-derived
                             Anthropic default.
    NAM_OLLAMA_MODEL         Local model tag (default qwen-judge).
    NAM_OLLAMA_URL           OpenAI-compatible base URL.
"""

from __future__ import annotations

import os
from typing import Any

ANTHROPIC_CLIENT_NAME = "AnthropicDirect"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
# No temperature/top_p/top_k: sampling params are rejected (400) on
# claude-opus-5 and the rest of the Opus 4.7+ family.
# Larger than the Bedrock client's 4096: thinking is on by default on
# claude-opus-5 and thinking tokens count against max_tokens, so a 4096
# budget can be spent before the structured extraction output is emitted,
# truncating the response and failing BAML parsing.
ANTHROPIC_MAX_TOKENS = 16000

OLLAMA_CLIENT_NAME = "QwenCoder"
# The derived tag, not the upstream coder build: upstream ships a red-team
# operator persona as its SYSTEM prompt, which biases any judging or
# extraction task. See baml_src/clients.baml.
DEFAULT_OLLAMA_MODEL = "qwen-judge"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
# Output budget, not context window — the model declares num_ctx 262144
# itself. Thinking is on and thinking tokens count against this, the same
# trap ANTHROPIC_MAX_TOKENS documents above.
OLLAMA_MAX_TOKENS = 32000

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIMENSIONS = 384


def anthropic_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key or None


def anthropic_enabled() -> bool:
    return anthropic_api_key() is not None


def anthropic_client_options() -> dict[str, Any]:
    """BAML client options for the runtime Anthropic client."""
    return {
        "model": os.environ.get("NAM_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        "api_key": anthropic_api_key(),
        "max_tokens": ANTHROPIC_MAX_TOKENS,
    }


def anthropic_registry() -> Any | None:
    """ClientRegistry with Anthropic primary, or None when no key is set."""
    if not anthropic_enabled():
        return None
    from baml_py import ClientRegistry

    registry = ClientRegistry()
    registry.add_llm_client(
        name=ANTHROPIC_CLIENT_NAME,
        provider="anthropic",
        options=anthropic_client_options(),
        retry_policy="StandardRetry",
    )
    registry.set_primary(ANTHROPIC_CLIENT_NAME)
    return registry


def ollama_enabled() -> bool:
    """True only on an explicit opt-in.

    Deliberately not derived from any key or from a reachable Ollama daemon:
    a local model is a deliberate choice about cost and quality, and nothing
    in the environment can imply it the way an API key implies a provider.
    """
    return os.environ.get("NAM_LLM_PROVIDER", "").strip().lower() == "ollama"


def ollama_client_options() -> dict[str, Any]:
    """BAML client options for the runtime Ollama client."""
    return {
        "base_url": os.environ.get("NAM_OLLAMA_URL", DEFAULT_OLLAMA_URL),
        "model": os.environ.get("NAM_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        "api_key": "ollama",  # ignored by Ollama; openai-generic requires one
        "temperature": 0,
        "max_tokens": OLLAMA_MAX_TOKENS,
    }


def ollama_registry() -> Any | None:
    """ClientRegistry with the local model primary, or None when not opted in."""
    if not ollama_enabled():
        return None
    from baml_py import ClientRegistry

    registry = ClientRegistry()
    registry.add_llm_client(
        name=OLLAMA_CLIENT_NAME,
        provider="openai-generic",
        options=ollama_client_options(),
        retry_policy="StandardRetry",
    )
    registry.set_primary(OLLAMA_CLIENT_NAME)
    return registry


def default_baml_options() -> dict[str, Any]:
    """baml_options for extraction calls.

    Explicit NAM_LLM_PROVIDER=ollama wins, then Anthropic-direct when keyed.
    An empty dict leaves each BAML function on its declared client (Bedrock).
    """
    registry = ollama_registry() or anthropic_registry()
    return {"client_registry": registry} if registry else {}


def embedding_kwargs_from_env() -> dict[str, Any]:
    """EmbeddingConfig kwargs from env.

    Explicit NAM_EMBEDDING_PROVIDER wins; otherwise sentence_transformers
    (local, no cloud calls) when Anthropic-direct mode is on, else Bedrock.
    """
    default_provider = "sentence_transformers" if anthropic_enabled() else "bedrock"
    provider = os.environ.get("NAM_EMBEDDING_PROVIDER", default_provider)

    kwargs: dict[str, Any] = {"provider": provider}
    if provider == "bedrock":
        kwargs.update({
            "model": os.environ.get("NAM_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"),
            "dimensions": int(os.environ.get("NAM_EMBEDDING_DIMENSIONS", "1024")),
            "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
            "aws_profile": os.environ.get("AWS_PROFILE"),
        })
    elif provider == "openai":
        kwargs.update({
            "model": os.environ.get("NAM_EMBEDDING_MODEL", "text-embedding-3-small"),
        })
    elif provider == "sentence_transformers":
        kwargs.update({
            "model": os.environ.get("NAM_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL),
            "dimensions": int(
                os.environ.get("NAM_EMBEDDING_DIMENSIONS", str(LOCAL_EMBEDDING_DIMENSIONS))
            ),
        })
    return kwargs

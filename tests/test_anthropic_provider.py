"""Anthropic-direct provider selection: if ANTHROPIC_API_KEY is set, don't use Bedrock.

Extraction routes to the Anthropic API via a runtime BAML ClientRegistry;
embeddings default to the local sentence-transformers model (Anthropic has
no embeddings API); startup preflight accepts the key as a usable provider.
Explicit env configuration always wins over the key-derived defaults.
"""

import pytest

from agent_memory_mcp import providers


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "NAM_ANTHROPIC_MODEL",
        "NAM_EMBEDDING_PROVIDER",
        "NAM_EMBEDDING_MODEL",
        "NAM_EMBEDDING_DIMENSIONS",
        "AWS_REGION",
        "AWS_PROFILE",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


class TestAnthropicDetection:
    def test_disabled_when_key_unset(self):
        assert providers.anthropic_enabled() is False

    def test_disabled_when_key_blank(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        assert providers.anthropic_enabled() is False

    def test_enabled_when_key_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert providers.anthropic_enabled() is True


class TestBamlOptions:
    def test_empty_without_key_so_bedrock_default_stands(self):
        assert providers.default_baml_options() == {}

    def test_registry_present_with_key(self, monkeypatch):
        from baml_py import ClientRegistry

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        opts = providers.default_baml_options()
        assert isinstance(opts["client_registry"], ClientRegistry)

    def test_client_options_default_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        opts = providers.anthropic_client_options()
        assert opts["model"] == "claude-opus-5"
        assert opts["api_key"] == "sk-ant-test"

    def test_client_options_model_overridable(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("NAM_ANTHROPIC_MODEL", "claude-haiku-4-5")
        assert providers.anthropic_client_options()["model"] == "claude-haiku-4-5"

    def test_client_options_send_no_sampling_params(self, monkeypatch):
        # temperature/top_p/top_k are rejected with a 400 on claude-opus-5
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        opts = providers.anthropic_client_options()
        assert "temperature" not in opts
        assert "top_p" not in opts
        assert "top_k" not in opts


class TestEmbeddingKwargs:
    def test_bedrock_default_without_key(self):
        kwargs = providers.embedding_kwargs_from_env()
        assert kwargs["provider"] == "bedrock"
        assert kwargs["model"] == "amazon.titan-embed-text-v2:0"
        assert kwargs["dimensions"] == 1024

    def test_local_default_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        kwargs = providers.embedding_kwargs_from_env()
        assert kwargs["provider"] == "sentence_transformers"
        assert kwargs["dimensions"] == 384

    def test_explicit_provider_beats_key_derived_default(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("NAM_EMBEDDING_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        assert providers.embedding_kwargs_from_env()["provider"] == "bedrock"

    def test_openai_branch_preserved(self, monkeypatch):
        monkeypatch.setenv("NAM_EMBEDDING_PROVIDER", "openai")
        kwargs = providers.embedding_kwargs_from_env()
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "text-embedding-3-small"


class TestCallSitesThreadRegistry:
    async def test_unified_extractor_passes_baml_options(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        captured = {}

        class _StubBaml:
            async def ExtractCodingMemory(
                self, *, transcript, context, baml_options=None
            ):
                captured["baml_options"] = baml_options

                class _R:
                    decisions = []
                    gotchas = []
                    dead_ends = []
                    preferences = []

                return _R()

        import agent_memory_mcp.baml_client.async_client as ac

        monkeypatch.setattr(ac, "b", _StubBaml())

        from agent_memory_mcp.extraction.unified import UnifiedBamlExtractor

        await UnifiedBamlExtractor().extract("switched the db driver to asyncpg")
        from baml_py import ClientRegistry

        assert isinstance(captured["baml_options"].get("client_registry"), ClientRegistry)

    def test_reasoning_extractor_defaults_to_anthropic_registry(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from baml_py import ClientRegistry

        from agent_memory_mcp.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        assert isinstance(extractor._baml_options.get("client_registry"), ClientRegistry)


class TestPreflight:
    def test_anthropic_key_alone_satisfies_preflight(self, monkeypatch):
        from agent_memory_mcp.mcp.server import check_resilient_provider_credentials

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        check_resilient_provider_credentials()  # must not raise

    def test_no_provider_at_all_still_raises(self):
        from agent_memory_mcp.mcp.server import check_resilient_provider_credentials

        with pytest.raises(RuntimeError):
            check_resilient_provider_credentials()

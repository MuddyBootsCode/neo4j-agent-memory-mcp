"""Unit tests: startup preflight for the Resilient BAML fallback chain (R31).

The BAML `Resilient` client falls back Bedrock -> OpenAI -> Gemini. Bedrock is
the primary provider (AWS_REGION + resolvable boto3 credentials); OpenAI and
Gemini are fallbacks only, so their absence should be a warning — but if
*none* of the three is usable, every extraction/reasoning call would fail
outright, so that case must fail loudly at startup instead of silently at the
first request.
"""

import logging

import pytest

from neo4j_agent_memory.mcp import server as server_module


def _clear_provider_env(monkeypatch):
    for var in ("AWS_REGION", "AWS_PROFILE", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class _FakeSession:
    """Stand-in for boto3.Session with a controllable get_credentials()."""

    def __init__(self, credentials):
        self._credentials = credentials

    def __call__(self, *args, **kwargs):
        return self

    def get_credentials(self):
        return self._credentials


class TestBedrockAvailability:
    def test_unavailable_when_region_unset(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        available, reason = server_module._check_bedrock_available()
        assert available is False
        assert "AWS_REGION" in reason

    def test_unavailable_when_credentials_unresolvable(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setattr(
            "boto3.Session", _FakeSession(credentials=None), raising=True
        )
        available, reason = server_module._check_bedrock_available()
        assert available is False
        assert "credential" in reason.lower()

    def test_available_when_region_and_credentials_present(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setattr(
            "boto3.Session", _FakeSession(credentials=object()), raising=True
        )
        available, reason = server_module._check_bedrock_available()
        assert available is True
        assert reason is None


class TestFallbackAvailability:
    def test_openai_unavailable_without_key(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        available, reason = server_module._check_openai_available()
        assert available is False
        assert "OPENAI_API_KEY" in reason

    def test_openai_available_with_key(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        available, reason = server_module._check_openai_available()
        assert available is True
        assert reason is None

    def test_gemini_unavailable_without_key(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        available, reason = server_module._check_gemini_available()
        assert available is False
        assert "GEMINI_API_KEY" in reason

    def test_gemini_available_with_key(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        available, reason = server_module._check_gemini_available()
        assert available is True
        assert reason is None


class TestResilientChainPreflight:
    """check_resilient_provider_credentials(): the orchestrating entry point."""

    def test_raises_loudly_when_no_provider_is_usable(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        with pytest.raises(RuntimeError, match="No usable LLM provider"):
            server_module.check_resilient_provider_credentials()

    def test_warns_but_does_not_raise_when_only_openai_fallback_present(
        self, monkeypatch, caplog
    ):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with caplog.at_level(logging.WARNING, logger=server_module.__name__):
            server_module.check_resilient_provider_credentials()  # must not raise

        messages = "\n".join(r.message for r in caplog.records)
        assert "Bedrock" in messages

    def test_warns_but_does_not_raise_when_only_gemini_fallback_present(
        self, monkeypatch, caplog
    ):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with caplog.at_level(logging.WARNING, logger=server_module.__name__):
            server_module.check_resilient_provider_credentials()  # must not raise

        messages = "\n".join(r.message for r in caplog.records)
        assert "Bedrock" in messages

    def test_no_bedrock_warning_when_bedrock_is_fully_configured(
        self, monkeypatch, caplog
    ):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setattr(
            "boto3.Session", _FakeSession(credentials=object()), raising=True
        )

        with caplog.at_level(logging.WARNING, logger=server_module.__name__):
            server_module.check_resilient_provider_credentials()  # must not raise

        messages = "\n".join(r.message for r in caplog.records)
        assert "primary LLM provider" not in messages
        # OpenAI/Gemini are still absent — that's a fallback-only warning, not fatal.
        assert "OpenAI" in messages
        assert "Gemini" in messages

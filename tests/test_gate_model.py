"""Dedicated recall-gate model routing and warm-up (MUD-407).

The gate (ScreenRecalledMemories) timed out on 100% of live calls because
it shared NAM_OLLAMA_MODEL with extraction — a 36B model that unloads
between uses and cold-reloads in minutes against a 6s deadline. These tests
pin the fix: NAM_OLLAMA_GATE_MODEL routes only the gate, falls back to the
main model when unset, and a fail-open background task keeps the gate
model resident via Ollama's native keep_alive API.
"""

import asyncio

import pytest

from agent_memory_mcp import providers
from agent_memory_mcp.mcp import _gate_warmup


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "NAM_LLM_PROVIDER",
        "NAM_OLLAMA_MODEL",
        "NAM_OLLAMA_GATE_MODEL",
        "NAM_OLLAMA_URL",
        "NAM_OLLAMA_REASONING",
    ):
        monkeypatch.delenv(var, raising=False)


class TestGateModelSelection:
    def test_falls_back_to_main_model_when_unset(self, monkeypatch):
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        assert providers.ollama_gate_model() == "qwen-judge"

    def test_falls_back_to_default_when_nothing_set(self):
        assert providers.ollama_gate_model() == providers.DEFAULT_OLLAMA_MODEL

    def test_blank_gate_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "   ")
        assert providers.ollama_gate_model() == "qwen-judge"

    def test_gate_env_wins_when_set(self, monkeypatch):
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        assert providers.ollama_gate_model() == "qwen3:4b"

    def test_gate_client_options_swap_model_and_force_json(self, monkeypatch):
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        main = providers.ollama_client_options()
        gate = providers.ollama_gate_client_options()
        assert gate["model"] == "qwen3:4b"
        assert main["model"] == "qwen-judge"
        # Constrained JSON decoding: without it a small model free-writes
        # reasoning for ~30s and blows the gate deadline.
        assert gate["response_format"] == {"type": "json_object"}
        assert {k: v for k, v in gate.items() if k not in ("model", "response_format")} == {
            k: v for k, v in main.items() if k != "model"
        }

    def test_gate_client_options_identical_without_gate_model(self, monkeypatch):
        """Exact fallback: no response_format, no model change."""
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        assert providers.ollama_gate_client_options() == providers.ollama_client_options()


class TestGateBamlOptions:
    def test_empty_without_any_provider(self):
        assert providers.gate_baml_options() == {}

    def test_registry_present_under_ollama(self, monkeypatch):
        from baml_py import ClientRegistry

        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        opts = providers.gate_baml_options()
        assert isinstance(opts["client_registry"], ClientRegistry)

    def test_falls_back_to_anthropic_registry_when_not_ollama(self, monkeypatch):
        from baml_py import ClientRegistry

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        opts = providers.gate_baml_options()
        assert isinstance(opts["client_registry"], ClientRegistry)

    def test_default_baml_options_untouched_by_gate_env(self, monkeypatch):
        """Extraction/curation/rating stay on the main model."""
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        assert providers.ollama_client_options()["model"] == "qwen-judge"


class TestWarmupEnabled:
    def test_disabled_when_provider_not_ollama(self, monkeypatch):
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        assert _gate_warmup.warmup_enabled() is False

    def test_disabled_when_gate_model_unset(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        assert _gate_warmup.warmup_enabled() is False

    def test_disabled_when_gate_equals_main_model(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen-judge")
        assert _gate_warmup.warmup_enabled() is False

    def test_enabled_with_distinct_gate_model(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        assert _gate_warmup.warmup_enabled() is True
        assert _gate_warmup.warmup_models() == ["qwen3:4b"]

    def test_pinning_the_main_model_enables_warmup_on_its_own(self, monkeypatch):
        """MUD-407 A3: the 36B judge unloads between captures and reloads
        for minutes under contention, which is the multi-minute tail on
        every capture. Opt-in because it holds 23 GB for the server's life."""
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_PIN_MAIN_MODEL", "1")
        assert _gate_warmup.warmup_enabled() is True
        assert _gate_warmup.warmup_models() == ["qwen-judge"]

    def test_pin_main_and_gate_lists_both_once(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        monkeypatch.setenv("NAM_OLLAMA_PIN_MAIN_MODEL", "1")
        assert _gate_warmup.warmup_models() == ["qwen3:4b", "qwen-judge"]

    def test_pin_main_not_under_ollama_is_nothing(self, monkeypatch):
        monkeypatch.setenv("NAM_OLLAMA_PIN_MAIN_MODEL", "1")
        assert _gate_warmup.warmup_models() == []


class TestNativeBaseUrl:
    def test_strips_v1_suffix(self, monkeypatch):
        monkeypatch.setenv("NAM_OLLAMA_URL", "http://host.docker.internal:11434/v1")
        assert (
            _gate_warmup.native_ollama_base_url()
            == "http://host.docker.internal:11434"
        )

    def test_strips_v1_with_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("NAM_OLLAMA_URL", "http://localhost:11434/v1/")
        assert _gate_warmup.native_ollama_base_url() == "http://localhost:11434"

    def test_default_url(self):
        assert _gate_warmup.native_ollama_base_url() == "http://localhost:11434"


class TestWarmupFailOpen:
    async def test_warm_once_swallows_errors(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")

        def boom(base_url, model):
            raise ConnectionError("ollama is down")

        monkeypatch.setattr(_gate_warmup, "_post_keep_alive", boom)
        assert await _gate_warmup.warm_gate_model_once() is False

    async def test_warm_once_posts_keep_alive(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        monkeypatch.setenv("NAM_OLLAMA_URL", "http://localhost:11434/v1")
        calls = []
        monkeypatch.setattr(
            _gate_warmup, "_post_keep_alive", lambda url, model: calls.append((url, model))
        )
        assert await _gate_warmup.warm_gate_model_once() is True
        assert calls == [("http://localhost:11434", "qwen3:4b")]

    async def test_warm_once_pins_every_configured_model(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        monkeypatch.setenv("NAM_OLLAMA_PIN_MAIN_MODEL", "1")
        monkeypatch.setenv("NAM_OLLAMA_URL", "http://localhost:11434/v1")
        calls = []
        monkeypatch.setattr(
            _gate_warmup, "_post_keep_alive", lambda url, model: calls.append((url, model))
        )
        assert await _gate_warmup.warm_gate_model_once() is True
        assert calls == [("http://localhost:11434", "qwen3:4b"), ("http://localhost:11434", "qwen-judge")]


class TestStartStop:
    async def test_start_returns_none_when_not_enabled(self):
        assert _gate_warmup.start_gate_warmup() is None

    async def test_start_returns_task_and_warms(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        warmed = asyncio.Event()
        monkeypatch.setattr(
            _gate_warmup, "_post_keep_alive", lambda url, model: warmed.set()
        )
        task = _gate_warmup.start_gate_warmup()
        assert task is not None
        await asyncio.wait_for(warmed.wait(), timeout=2)
        _gate_warmup.stop_gate_warmup(task)
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_stop_none_is_noop(self):
        _gate_warmup.stop_gate_warmup(None)

    async def test_start_swallows_setup_errors(self, monkeypatch):
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")

        def boom():
            raise RuntimeError("no providers module")

        monkeypatch.setattr(_gate_warmup, "warmup_enabled", boom)
        assert _gate_warmup.start_gate_warmup() is None


class TestGateCallRouting:
    async def test_screen_memories_uses_gate_options(self, monkeypatch):
        """The recall gate call threads gate_baml_options, not the default."""
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        captured = {}

        class _Verdict:
            def __init__(self, id, keep):
                self.id = id
                self.keep = keep

        class _StubBaml:
            async def ScreenRecalledMemories(self, *, query, candidates, baml_options=None):
                captured["baml_options"] = baml_options

                class _R:
                    verdicts = [_Verdict(0, True), _Verdict(1, False)]

                return _R()

        import agent_memory_mcp.baml_client.async_client as ac

        monkeypatch.setattr(ac, "b", _StubBaml())

        from agent_memory_mcp.mcp._coding_tools import screen_memories

        memories = [
            {"kind": "Gotcha", "text": "on point"},
            {"kind": "Decision", "text": "off topic"},
        ]
        kept = await screen_memories("how do I fix the deploy?", memories)
        assert kept == [memories[0]]
        from baml_py import ClientRegistry

        assert isinstance(captured["baml_options"].get("client_registry"), ClientRegistry)

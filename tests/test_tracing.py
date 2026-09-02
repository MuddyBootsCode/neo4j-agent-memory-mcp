"""Opik tracing for the judge calls (MUD-427).

The contract under test is fail-open in the strong sense: with no
OPIK_API_KEY the module never imports the SDK, never opens a connection,
and every function is a silent no-op; with a key, any exception inside the
SDK is swallowed. A trace is never worth a broken capture or recall.
"""

import sys
import types

import pytest

from agent_memory_mcp import tracing


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch):
    """Fresh client state per test; no ambient OPIK_ config leaks in."""
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_client_failed", False)
    for var in ("OPIK_API_KEY", "OPIK_WORKSPACE", "OPIK_PROJECT_NAME"):
        monkeypatch.delenv(var, raising=False)


class _FakeOpik:
    instances: list = []

    def __init__(self, *args, **kwargs):
        _FakeOpik.instances.append(self)
        self.traces = []
        self.flushes = []

    def trace(self, **kwargs):
        self.traces.append(kwargs)

    def flush(self, timeout=None):
        self.flushes.append(timeout)


@pytest.fixture
def fake_opik(monkeypatch):
    """A stand-in ``opik`` module that records instead of networking."""
    _FakeOpik.instances = []
    mod = types.ModuleType("opik")
    mod.Opik = _FakeOpik
    monkeypatch.setitem(sys.modules, "opik", mod)
    return _FakeOpik


class TestNoOpWithoutKey:
    def test_emit_is_a_noop_and_never_touches_the_sdk(self, monkeypatch, fake_opik):
        """No key must mean zero SDK use — no client, no network."""
        tracing.emit_trace("recall-gate", input={"q": "x"}, output={"v": []})
        assert fake_opik.instances == []
        assert tracing._client is None

    def test_flush_is_a_noop_without_a_client(self, fake_opik):
        tracing.flush()
        assert fake_opik.instances == []

    def test_blank_key_counts_as_no_key(self, monkeypatch, fake_opik):
        monkeypatch.setenv("OPIK_API_KEY", "   ")
        tracing.emit_trace("curator", input={}, output={})
        assert fake_opik.instances == []


class TestEmitWithKey:
    def test_one_trace_per_call_with_the_given_fields(self, monkeypatch, fake_opik):
        monkeypatch.setenv("OPIK_API_KEY", "test-key")
        tracing.emit_trace(
            "recall-gate",
            input={"query": "q", "candidates": "0. [Gotcha] x"},
            output={"verdicts": [{"id": 0, "keep": True, "reason": "on point"}]},
            metadata={"model": "qwen3:4b", "kept": 1, "of": 1},
        )
        (client,) = fake_opik.instances
        (trace,) = client.traces
        assert trace["name"] == "recall-gate"
        assert trace["input"]["candidates"] == "0. [Gotcha] x"
        assert trace["output"]["verdicts"][0]["reason"] == "on point"
        assert trace["metadata"] == {"model": "qwen3:4b", "kept": 1, "of": 1}

    def test_none_metadata_values_are_dropped(self, monkeypatch, fake_opik):
        monkeypatch.setenv("OPIK_API_KEY", "test-key")
        tracing.emit_trace(
            "curator", input={}, output={},
            metadata={"task_key": None, "repo": "r"},
        )
        assert fake_opik.instances[0].traces[0]["metadata"] == {"repo": "r"}

    def test_client_is_shared_across_calls(self, monkeypatch, fake_opik):
        monkeypatch.setenv("OPIK_API_KEY", "test-key")
        tracing.emit_trace("curator", input={}, output={})
        tracing.emit_trace("served-rater", input={}, output={})
        assert len(fake_opik.instances) == 1
        assert len(fake_opik.instances[0].traces) == 2

    def test_flush_delegates_to_the_client(self, monkeypatch, fake_opik):
        monkeypatch.setenv("OPIK_API_KEY", "test-key")
        tracing.emit_trace("curator", input={}, output={})
        tracing.flush(timeout=3)
        assert fake_opik.instances[0].flushes == [3]


class TestFailOpen:
    def test_broken_constructor_disables_tracing_for_the_process(
        self, monkeypatch
    ):
        monkeypatch.setenv("OPIK_API_KEY", "test-key")
        mod = types.ModuleType("opik")

        def boom(*args, **kwargs):
            raise RuntimeError("sdk misconfigured")

        mod.Opik = boom
        monkeypatch.setitem(sys.modules, "opik", mod)
        tracing.emit_trace("curator", input={}, output={})  # must not raise
        assert tracing._client_failed is True
        # Later calls short-circuit instead of retrying the broken SDK.
        tracing.emit_trace("curator", input={}, output={})

    def test_trace_exception_is_swallowed(self, monkeypatch, fake_opik):
        monkeypatch.setenv("OPIK_API_KEY", "test-key")

        def boom(**kwargs):
            raise RuntimeError("backend down")

        tracing.emit_trace("curator", input={}, output={})
        fake_opik.instances[0].trace = boom
        tracing.emit_trace("curator", input={}, output={})  # must not raise

    def test_flush_exception_is_swallowed(self, monkeypatch, fake_opik):
        monkeypatch.setenv("OPIK_API_KEY", "test-key")
        tracing.emit_trace("curator", input={}, output={})

        def boom(timeout=None):
            raise RuntimeError("backend down")

        fake_opik.instances[0].flush = boom
        tracing.flush()  # must not raise


class TestHelpers:
    def test_transcript_truncates_to_the_tail(self):
        text = "x" * 10_000 + "TAIL"
        out = tracing.truncate_transcript(text)
        assert len(out) == tracing.TRANSCRIPT_TRACE_CHARS
        assert out.endswith("TAIL")
        assert tracing.truncate_transcript("") == ""

    def test_model_tag_follows_provider_selection(self, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "NAM_LLM_PROVIDER", "NAM_OLLAMA_MODEL",
                    "NAM_OLLAMA_GATE_MODEL", "NAM_ANTHROPIC_MODEL"):
            monkeypatch.delenv(var, raising=False)
        assert tracing.model_tag() == "bedrock"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert tracing.model_tag() == "claude-opus-5"
        monkeypatch.setenv("NAM_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NAM_OLLAMA_MODEL", "qwen-judge")
        monkeypatch.setenv("NAM_OLLAMA_GATE_MODEL", "qwen3:4b")
        assert tracing.model_tag() == "qwen-judge"
        assert tracing.model_tag(gate=True) == "qwen3:4b"


class TestGateSchemaStaysLean:
    """The reason field must not cost the 4B gate its latency budget:
    optional in the schema, absent for rejects, and still parseable."""

    def test_reject_verdicts_parse_without_reason(self):
        from agent_memory_mcp.baml_client.sync_client import b

        screen = b.parse.ScreenRecalledMemories(
            '{"verdicts": ['
            '{"id": 0, "keep": false},'
            '{"id": 1, "keep": true, "reason": "names the same flag"}'
            "]}"
        )
        assert [(v.id, v.keep, v.reason) for v in screen.verdicts] == [
            (0, False, None),
            (1, True, "names the same flag"),
        ]

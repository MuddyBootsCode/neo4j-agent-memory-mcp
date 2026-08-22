"""Tests for the write-time curator (MUD-404), the batched successor to the
per-item judge screen of MUD-397.

The BAML call is stubbed throughout; these exercise the wrapper's verdict
mapping, the fail-open contract, the kill switch, and the rendering the
curator sees. Real model behaviour against adversarial transcripts is
covered by tests/integration/test_extraction_injection.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent_memory_mcp.extraction.coding import candidate_line, curate_coding_memory


def _cands():
    return [
        {"kind": "Decision", "text": "chose asyncpg", "reason": "async loop", "anchor_files": ["src/db.py"],
         "concerns_task": False, "confidence": 0.9},
        {"kind": "Gotcha", "symptom": "pytest hangs", "text": "run with -m integration", "anchor_files": [],
         "concerns_task": True, "confidence": 0.8},
        {"kind": "DeadEnd", "symptom": None, "attempt": "patched pool", "why_failed": "too early",
         "anchor_files": ["src/app.py"], "concerns_task": False, "confidence": 0.6},
        {"kind": "CodingPreference", "category": "testing", "preference": "failing test first", "confidence": 0.9},
    ]


def _verdicts(*actions):
    return SimpleNamespace(verdicts=[SimpleNamespace(id=i, action=SimpleNamespace(value=a)) for i, a in enumerate(actions)])


def _stub(monkeypatch, return_value=None, side_effect=None):
    mock = AsyncMock(return_value=return_value, side_effect=side_effect)
    monkeypatch.setattr("agent_memory_mcp.baml_client.async_client.b.CurateCodingMemory", mock)
    return mock


class TestCurate:
    async def test_only_write_survives_and_counts_every_action(self, monkeypatch):
        _stub(monkeypatch, _verdicts("WRITE", "ALREADY_KNOWN", "NOT_DURABLE", "UNSUPPORTED"))
        out = await curate_coding_memory(_cands(), "transcript", ["- [Gotcha] run with -m integration"])
        assert [c["kind"] for c in out["kept"]] == ["Decision"]
        assert out["counts"] == {"write": 1, "already_known": 1, "not_durable": 1, "unsupported": 1}

    async def test_curator_sees_numbered_candidates_existing_and_transcript(self, monkeypatch):
        mock = _stub(monkeypatch, _verdicts("WRITE", "WRITE", "WRITE", "WRITE"))
        await curate_coding_memory(_cands(), "the transcript", ["[Gotcha] old lesson"])
        kwargs = mock.await_args.kwargs
        assert kwargs["candidates"].startswith("0. [Decision] chose asyncpg — async loop\n1. [Gotcha] symptom: pytest hangs | run with -m integration")
        assert kwargs["existing"] == "- [Gotcha] old lesson"
        assert kwargs["transcript"] == "the transcript"
        assert "baml_options" in kwargs

    async def test_no_existing_renders_none_marker(self, monkeypatch):
        mock = _stub(monkeypatch, _verdicts("WRITE"))
        await curate_coding_memory(_cands()[:1], "t", [])
        assert mock.await_args.kwargs["existing"] == "(none)"

    async def test_incomplete_verdicts_keep_everything(self, monkeypatch, caplog):
        _stub(monkeypatch, _verdicts("UNSUPPORTED"))  # 1 verdict for 4 candidates
        out = await curate_coding_memory(_cands(), "t", [])
        assert len(out["kept"]) == 4
        assert out["counts"]["write"] == 4

    async def test_out_of_range_ids_are_ignored_then_treated_as_incomplete(self, monkeypatch):
        v = _verdicts("UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED")
        v.verdicts[3].id = 99
        _stub(monkeypatch, v)
        out = await curate_coding_memory(_cands(), "t", [])
        assert len(out["kept"]) == 4

    async def test_curator_failure_keeps_everything(self, monkeypatch, caplog):
        _stub(monkeypatch, side_effect=RuntimeError("ollama down"))
        out = await curate_coding_memory(_cands(), "t", [])
        assert len(out["kept"]) == 4
        assert "keeping all" in caplog.text

    async def test_kill_switch_skips_the_call(self, monkeypatch):
        monkeypatch.setenv("NAM_CAPTURE_JUDGE", "off")
        mock = _stub(monkeypatch, _verdicts("UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED"))
        out = await curate_coding_memory(_cands(), "t", [])
        assert len(out["kept"]) == 4
        mock.assert_not_awaited()

    async def test_empty_candidates_skip_the_call(self, monkeypatch):
        mock = _stub(monkeypatch, _verdicts())
        out = await curate_coding_memory([], "t", [])
        assert out == {"kept": [], "counts": {"write": 0, "already_known": 0, "not_durable": 0, "unsupported": 0}}
        mock.assert_not_awaited()

    async def test_action_accepts_plain_string_enum(self, monkeypatch):
        v = SimpleNamespace(verdicts=[SimpleNamespace(id=0, action="NOT_DURABLE")])
        _stub(monkeypatch, v)
        out = await curate_coding_memory(_cands()[:1], "t", [])
        assert out["kept"] == []
        assert out["counts"]["not_durable"] == 1


class TestCandidateLine:
    def test_each_kind(self):
        c = _cands()
        assert candidate_line(c[0]) == "[Decision] chose asyncpg — async loop"
        assert candidate_line(c[1]) == "[Gotcha] symptom: pytest hangs | run with -m integration"
        assert candidate_line(c[2]) == "[DeadEnd] patched pool — failed: too early"
        assert candidate_line(c[3]) == "[Preference] testing: failing test first"

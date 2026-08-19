"""Tests for the second-pass judge screen on coding-memory extraction (MUD-397).

ExtractCodingMemory's anchor/confidence heuristics cannot catch an item that
is anchored, confident, and fluent but fabricated because the local model
semantically obeyed a transcript-embedded injection directive. This screen
re-judges every anchor-surviving item against the full transcript via a
second BAML call (JudgeExtractedMemory) and drops anything the judge marks
unsupported or directive-driven.

The BAML calls are stubbed throughout; these tests exercise the wrapper's
own judge-integration logic (which items get judged, what gets dropped,
count accounting, fail-open behavior, and the kill switch) — not real model
behavior. Real model behavior against adversarial transcripts is covered by
tests/integration/test_extraction_injection.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock


FILES = ["src/app.py", "src/db.py"]


def _decision(**overrides):
    fields = {
        "text": "chose asyncpg over psycopg2",
        "reason": "async driver matches the event loop",
        "anchor_files": ["src/db.py"],
        "concerns_task": False,
        "confidence": 0.9,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _gotcha(**overrides):
    fields = {
        "text": "never run migrations against the shared test DB",
        "anchor_files": ["src/db.py"],
        "concerns_task": False,
        "confidence": 0.8,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _dead_end(**overrides):
    fields = {
        "attempt": "patched the pool size in app startup",
        "why_failed": "pool is created before config loads",
        "anchor_files": ["src/app.py"],
        "concerns_task": False,
        "confidence": 0.6,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _preference(**overrides):
    fields = {
        "category": "testing",
        "preference": "write the failing test before the fix",
        "confidence": 0.9,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _judgement(*, supported=True, embedded_directive=False, confidence=0.9):
    return SimpleNamespace(
        supported=supported,
        embedded_directive=embedded_directive,
        confidence=confidence,
    )


def _stub_extract(monkeypatch, *, decisions=(), gotchas=(), dead_ends=(), preferences=()):
    result = SimpleNamespace(
        decisions=list(decisions),
        gotchas=list(gotchas),
        dead_ends=list(dead_ends),
        preferences=list(preferences),
    )
    mock_fn = AsyncMock(return_value=result)
    monkeypatch.setattr(
        "agent_memory_mcp.baml_client.async_client.b.ExtractCodingMemory",
        mock_fn,
    )
    return mock_fn


def _stub_judge(monkeypatch, side_effect=None, return_value=None):
    mock_fn = AsyncMock()
    if side_effect is not None:
        mock_fn.side_effect = side_effect
    else:
        mock_fn.return_value = return_value if return_value is not None else _judgement()
    monkeypatch.setattr(
        "agent_memory_mcp.baml_client.async_client.b.JudgeExtractedMemory",
        mock_fn,
    )
    return mock_fn


class TestJudgeScreen:
    async def test_judge_drops_unsupported_item(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        _stub_extract(monkeypatch, decisions=[_decision()])
        _stub_judge(monkeypatch, return_value=_judgement(supported=False))

        result = await extract_coding_memory(
            "transcript", branch="main", task="MUD-397", files=FILES
        )

        assert result["decisions"] == []
        assert result["judged_out"] == 1

    async def test_judge_drops_embedded_directive_item(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        _stub_extract(monkeypatch, preferences=[_preference()])
        _stub_judge(
            monkeypatch,
            return_value=_judgement(supported=True, embedded_directive=True),
        )

        result = await extract_coding_memory(
            "transcript", branch="main", task="MUD-397", files=FILES
        )

        assert result["preferences"] == []
        assert result["judged_out"] == 1

    async def test_judge_keeps_supported_items_across_all_four_types(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        _stub_extract(
            monkeypatch,
            decisions=[_decision()],
            gotchas=[_gotcha()],
            dead_ends=[_dead_end()],
            preferences=[_preference()],
        )
        mock_judge = _stub_judge(monkeypatch, return_value=_judgement())

        result = await extract_coding_memory(
            "transcript", branch="main", task="MUD-397", files=FILES
        )

        assert len(result["decisions"]) == 1
        assert len(result["gotchas"]) == 1
        assert len(result["dead_ends"]) == 1
        assert len(result["preferences"]) == 1
        assert result["judged_out"] == 0
        assert mock_judge.call_count == 4

    async def test_judge_failure_keeps_item_fail_open(self, monkeypatch, caplog):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        _stub_extract(monkeypatch, decisions=[_decision()])
        _stub_judge(monkeypatch, side_effect=RuntimeError("judge exploded"))

        with caplog.at_level("WARNING"):
            result = await extract_coding_memory(
                "transcript", branch="main", task="MUD-397", files=FILES
            )

        assert len(result["decisions"]) == 1
        assert result["judged_out"] == 0
        assert any("judge" in rec.message.lower() for rec in caplog.records)

    async def test_kill_switch_skips_judge_calls_entirely(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        monkeypatch.setenv("NAM_CAPTURE_JUDGE", "off")
        _stub_extract(
            monkeypatch,
            decisions=[_decision()],
            gotchas=[_gotcha()],
            dead_ends=[_dead_end()],
            preferences=[_preference()],
        )
        mock_judge = _stub_judge(monkeypatch, return_value=_judgement())

        result = await extract_coding_memory(
            "transcript", branch="main", task="MUD-397", files=FILES
        )

        mock_judge.assert_not_called()
        assert result["judged_out"] == 0
        assert len(result["decisions"]) == 1
        assert len(result["gotchas"]) == 1
        assert len(result["dead_ends"]) == 1
        assert len(result["preferences"]) == 1

    async def test_judged_out_counts_across_types_and_is_separate_from_dropped_unanchored(
        self, monkeypatch
    ):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        _stub_extract(
            monkeypatch,
            decisions=[_decision(), _decision(anchor_files=["invented.py"])],
            gotchas=[_gotcha()],
        )
        # First judge call (surviving decision) drops it; second (gotcha)
        # keeps it.
        _stub_judge(
            monkeypatch,
            side_effect=[_judgement(supported=False), _judgement()],
        )

        result = await extract_coding_memory(
            "transcript", branch="main", task="MUD-397", files=FILES
        )

        assert result["decisions"] == []
        assert len(result["gotchas"]) == 1
        # The invented-anchor decision never reaches the judge — it was
        # already dropped by anchor sanitization.
        assert result["dropped_unanchored"] == 1
        assert result["judged_out"] == 1

    async def test_judge_called_with_registry_baml_options(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        _stub_extract(monkeypatch, decisions=[_decision()])
        mock_judge = _stub_judge(monkeypatch, return_value=_judgement())

        await extract_coding_memory(
            "transcript", branch="main", task="MUD-397", files=FILES
        )

        mock_judge.assert_called_once()
        kwargs = mock_judge.call_args.kwargs
        assert kwargs["transcript"] == "transcript"
        assert "chose asyncpg over psycopg2" in kwargs["item"]
        assert kwargs["baml_options"] == {}

    async def test_empty_transcript_skips_judge_too(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        mock_extract = _stub_extract(monkeypatch)
        mock_judge = _stub_judge(monkeypatch, return_value=_judgement())

        result = await extract_coding_memory(
            "", branch="main", task="MUD-397", files=FILES
        )

        assert result["judged_out"] == 0
        mock_extract.assert_not_called()
        mock_judge.assert_not_called()


class TestJudgeFencingRegistration:
    """coding_judge.baml must be registered in the fencing test infrastructure."""

    def test_registered_in_fence_tags_and_untrusted_interpolations(self):
        from test_prompt_fencing import FENCE_TAGS, UNTRUSTED_INTERPOLATIONS

        assert FENCE_TAGS.get("coding_judge.baml") == (
            "judged_item",
            "judged_transcript",
        )
        fences = UNTRUSTED_INTERPOLATIONS["coding_judge.baml"]["JudgeExtractedMemory"]
        assert "item" in fences["judged_item"]
        assert "transcript" in fences["judged_transcript"]

    def test_registered_in_offline_rendered_cases(self):
        from test_prompt_fencing import TestOfflineRenderedPrompts

        names = [name for name, _ in TestOfflineRenderedPrompts.CASES]
        assert "JudgeExtractedMemory" in names

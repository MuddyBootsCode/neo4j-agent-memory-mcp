"""Tests for the coding-memory BAML surface (MUD-395)."""


def test_extract_coding_memory_in_generated_client():
    from agent_memory_mcp.baml_client.async_client import b

    assert hasattr(b, "ExtractCodingMemory")


def test_coding_baml_registered_in_fencing_registry():
    """coding.baml must be registered in the real fencing test infrastructure
    (tests/test_prompt_fencing.py), so it cannot silently drop out of the
    fence-position, escape-filter, and offline adversarial checks."""
    from test_prompt_fencing import FENCE_TAGS, UNTRUSTED_INTERPOLATIONS

    assert FENCE_TAGS.get("coding.baml") == (
        "session_context",
        "session_transcript",
    )
    fences = UNTRUSTED_INTERPOLATIONS["coding.baml"]["ExtractCodingMemory"]
    assert "transcript" in fences["session_transcript"]
    assert set(fences["session_context"]) == {
        "context.branch",
        "context.task",
        "file",
    }


class TestExtractCodingMemory:
    """Behavior tests for extract_coding_memory.

    The BAML call is stubbed; every assertion below exercises the wrapper's
    own anchor filtering, drop accounting, and confidence clamping.
    """

    FILES = ["src/app.py", "src/db.py"]

    EMPTY_SHAPE = {
        "decisions": [],
        "gotchas": [],
        "dead_ends": [],
        "preferences": [],
        "anchor_rate": None,
        "dropped_unanchored": 0,
    }

    def _stub(self, monkeypatch, *, decisions=(), gotchas=(), dead_ends=(), preferences=()):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

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

    @staticmethod
    def _decision(**overrides):
        from types import SimpleNamespace

        fields = {
            "text": "chose asyncpg over psycopg2",
            "reason": "async driver matches the event loop",
            "anchor_files": ["src/db.py"],
            "concerns_task": False,
            "confidence": 0.9,
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    @staticmethod
    def _gotcha(**overrides):
        from types import SimpleNamespace

        fields = {
            "text": "never run migrations against the shared test DB",
            "anchor_files": ["src/db.py"],
            "concerns_task": False,
            "confidence": 0.8,
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    @staticmethod
    def _dead_end(**overrides):
        from types import SimpleNamespace

        fields = {
            "attempt": "patched the pool size in app startup",
            "why_failed": "pool is created before config loads",
            "anchor_files": ["src/app.py"],
            "concerns_task": False,
            "confidence": 0.6,
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    @staticmethod
    def _preference(**overrides):
        from types import SimpleNamespace

        fields = {
            "category": "testing",
            "preference": "write the failing test before the fix",
            "confidence": 0.9,
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    async def test_happy_path_maps_all_types_and_clamps_confidence(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        mock_fn = self._stub(
            monkeypatch,
            decisions=[self._decision(confidence=1.5)],
            gotchas=[self._gotcha(anchor_files=[], concerns_task=True, confidence="bad")],
            dead_ends=[self._dead_end(confidence=-0.2)],
            preferences=[self._preference(confidence=2.0)],
        )

        result = await extract_coding_memory(
            "long session transcript",
            branch="feature/coding-memory-pivot",
            task="MUD-395",
            files=self.FILES,
        )

        assert result["decisions"] == [
            {
                "text": "chose asyncpg over psycopg2",
                "reason": "async driver matches the event loop",
                "anchor_files": ["src/db.py"],
                "concerns_task": False,
                "confidence": 1.0,
            }
        ]
        assert result["gotchas"] == [
            {
                "symptom": None,
                "text": "never run migrations against the shared test DB",
                "anchor_files": [],
                "concerns_task": True,
                "confidence": 0.7,
            }
        ]
        assert result["dead_ends"] == [
            {
                "symptom": None,
                "attempt": "patched the pool size in app startup",
                "why_failed": "pool is created before config loads",
                "anchor_files": ["src/app.py"],
                "concerns_task": False,
                "confidence": 0.0,
            }
        ]
        assert result["preferences"] == [
            {
                "category": "testing",
                "preference": "write the failing test before the fix",
                "confidence": 1.0,
            }
        ]
        assert result["anchor_rate"] == 1.0
        assert result["dropped_unanchored"] == 0

        mock_fn.assert_called_once()
        kwargs = mock_fn.call_args.kwargs
        assert kwargs["transcript"] == "long session transcript"
        assert kwargs["context"].branch == "feature/coding-memory-pivot"
        assert kwargs["context"].task == "MUD-395"
        assert kwargs["context"].files == self.FILES
        assert kwargs["baml_options"] == {}

    async def test_invented_anchor_filtered_but_item_kept(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        self._stub(
            monkeypatch,
            decisions=[
                self._decision(anchor_files=[" src/db.py ", "made/up.py", "src/app.py"])
            ],
        )

        result = await extract_coding_memory(
            "transcript",
            branch="main",
            task="MUD-395",
            files=self.FILES,
        )

        # Invented path removed, model order preserved for the survivors.
        assert result["decisions"][0]["anchor_files"] == ["src/db.py", "src/app.py"]
        assert result["dropped_unanchored"] == 0
        assert result["anchor_rate"] == 1.0

    async def test_unanchored_gotcha_dropped_and_counted(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        self._stub(
            monkeypatch,
            decisions=[self._decision()],
            gotchas=[self._gotcha(anchor_files=["invented.py"], concerns_task=False)],
        )

        result = await extract_coding_memory(
            "transcript",
            branch="main",
            task="MUD-395",
            files=self.FILES,
        )

        assert result["gotchas"] == []
        assert len(result["decisions"]) == 1
        assert result["dropped_unanchored"] == 1
        assert result["anchor_rate"] == 0.5

    async def test_empty_transcript_skips_baml(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        mock_fn = self._stub(monkeypatch)

        for transcript in ("", "   \n\t  "):
            result = await extract_coding_memory(
                transcript,
                branch="main",
                task="MUD-395",
                files=self.FILES,
            )
            assert result == self.EMPTY_SHAPE

        mock_fn.assert_not_called()

    async def test_anchor_rate_none_when_only_preferences(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        self._stub(monkeypatch, preferences=[self._preference()])

        result = await extract_coding_memory(
            "transcript",
            branch="main",
            task="MUD-395",
            files=self.FILES,
        )

        assert result["anchor_rate"] is None
        assert result["dropped_unanchored"] == 0
        assert len(result["preferences"]) == 1

    async def test_concerns_task_cannot_rescue_when_task_is_none(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        self._stub(
            monkeypatch,
            gotchas=[self._gotcha(anchor_files=[], concerns_task=True)],
        )

        result = await extract_coding_memory(
            "transcript",
            branch="main",
            task=None,
            files=self.FILES,
        )

        assert result["gotchas"] == []
        assert result["dropped_unanchored"] == 1
        assert result["anchor_rate"] == 0.0

    async def test_duplicate_anchor_paths_deduped_order_preserving(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        self._stub(
            monkeypatch,
            decisions=[
                self._decision(anchor_files=["src/db.py", "src/db.py", "src/app.py"])
            ],
        )

        result = await extract_coding_memory(
            "transcript",
            branch="main",
            task="MUD-395",
            files=self.FILES,
        )

        assert result["decisions"][0]["anchor_files"] == ["src/db.py", "src/app.py"]

    async def test_blank_files_entry_does_not_admit_whitespace_anchor(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        self._stub(
            monkeypatch,
            gotchas=[self._gotcha(anchor_files=["   "], concerns_task=False)],
        )

        result = await extract_coding_memory(
            "transcript",
            branch="main",
            task="MUD-395",
            files=["src/app.py", "  "],
        )

        # The whitespace-only anchor must neither match the blank files entry
        # nor rescue the item.
        assert result["gotchas"] == []
        assert result["dropped_unanchored"] == 1
        assert result["anchor_rate"] == 0.0

    async def test_kept_item_reports_concerns_task_false_when_task_is_none(self, monkeypatch):
        from agent_memory_mcp.extraction.coding import extract_coding_memory

        self._stub(
            monkeypatch,
            decisions=[self._decision(anchor_files=["src/db.py"], concerns_task=True)],
        )

        result = await extract_coding_memory(
            "transcript",
            branch="main",
            task=None,
            files=self.FILES,
        )

        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["concerns_task"] is False


class TestSymptomAndPathNormalization:
    FILES = ["src/app.py", "src/db.py"]

    async def test_symptom_passes_through_and_blank_becomes_none(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from agent_memory_mcp.extraction.coding import extract_coding_memory

        gotcha = SimpleNamespace(symptom="  pytest hangs after collection ", text="run with -m integration",
                                 anchor_files=["src/db.py"], concerns_task=False, confidence=0.8)
        dead_end = SimpleNamespace(symptom="   ", attempt="patched pool", why_failed="created before config",
                                   anchor_files=["src/app.py"], concerns_task=False, confidence=0.6)
        result = SimpleNamespace(decisions=[], gotchas=[gotcha], dead_ends=[dead_end], preferences=[])
        monkeypatch.setattr("agent_memory_mcp.baml_client.async_client.b.ExtractCodingMemory",
                            AsyncMock(return_value=result))
        out = await extract_coding_memory("t", branch="b", task=None, files=self.FILES)
        assert out["gotchas"][0]["symptom"] == "pytest hangs after collection"
        assert out["dead_ends"][0]["symptom"] is None

    async def test_anchor_paths_match_after_normalization_and_keep_caller_spelling(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from agent_memory_mcp.extraction.coding import extract_coding_memory

        gotcha = SimpleNamespace(symptom=None, text="x", anchor_files=["./src/db.py", "src/../src/app.py", "/abs/src/db.py"],
                                 concerns_task=False, confidence=0.8)
        result = SimpleNamespace(decisions=[], gotchas=[gotcha], dead_ends=[], preferences=[])
        monkeypatch.setattr("agent_memory_mcp.baml_client.async_client.b.ExtractCodingMemory",
                            AsyncMock(return_value=result))
        out = await extract_coding_memory("t", branch="b", task=None, files=["src/db.py", "./src/app.py"])
        assert out["gotchas"][0]["anchor_files"] == ["src/db.py", "./src/app.py"]
        assert out["dropped_unanchored"] == 0

    def test_normalize_path(self):
        from agent_memory_mcp.extraction.coding import normalize_path

        assert normalize_path("./a/b.py") == "a/b.py"
        assert normalize_path("a//b/../c.py") == "a/c.py"
        assert normalize_path("   ") == ""
        assert normalize_path(None) == ""

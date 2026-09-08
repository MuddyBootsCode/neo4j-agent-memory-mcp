"""Canonical lesson text vs embedding input (MUD-456, P5 E1).

Two strings that were one. ``memory_embedding_text`` is the canonical
lesson text: it identifies a lesson (the golden set hashes it into
``lesson_id``) and keys duplicate detection, so it is frozen.
``memory_embedding_input`` is what actually reaches the embedder, and may
carry a context prefix (repo, kind, anchored file basenames) that the
canonical text must never see.
"""

from unittest.mock import MagicMock

import pytest
from test_coding_tools import FakeEmbedder, FakeGraph, _extraction, _register, _stub_extract


@pytest.fixture
def mock_ctx():
    return MagicMock()


class TestEmbeddingInputSplit:
    def test_embedding_input_is_the_canonical_text_when_the_prefix_is_off(self):
        """Default: nothing about the embedded string changes, even with
        repo and files in hand. The p4-live numbers have to reproduce."""
        from agent_memory_mcp.mcp._coding_tools import (
            memory_embedding_input,
            memory_embedding_text,
        )

        props = {"text": "pin the version", "symptom": "resolves to 2.0"}
        assert memory_embedding_input(
            "Gotcha", props, repo="my-repo", files=["src/auth_server.py"]
        ) == memory_embedding_text("Gotcha", props)

    def test_full_prefix_carries_repo_kind_and_file_basenames(self, monkeypatch):
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("repo", "kind", "files"))

        assert ct.memory_embedding_input(
            "Gotcha",
            {"text": "pin the version", "symptom": "resolves to 2.0"},
            repo="gradgraph-auth-platform",
            files=["src/auth_server.py", "src/rate_limit.py"],
        ) == (
            "gradgraph-auth-platform · gotcha · auth_server.py, rate_limit.py"
            " | resolves to 2.0 | pin the version"
        )

    def test_a_selected_part_with_nothing_to_say_is_dropped(self, monkeypatch):
        """An unanchored lesson gets no empty files slot, and a prefix that
        resolves to nothing leaves the canonical text untouched."""
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("repo", "kind", "files"))
        assert ct.memory_embedding_input(
            "DeadEnd",
            {"attempt": "used a stash", "why_failed": "it was a no-op"},
            repo="my-repo",
            files=[],
        ) == "my-repo · deadend | used a stash — failed: it was a no-op"

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("files",))
        assert ct.memory_embedding_input(
            "Gotcha", {"text": "pin the version"}, repo="my-repo", files=None
        ) == "pin the version"

    def test_a_lesson_with_no_text_stays_unembeddable(self, monkeypatch):
        """Empty text can never match a vector query, and the backfill
        label-onlys it. A prefix would give it a vector made of metadata."""
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("repo", "kind", "files"))

        assert ct.memory_embedding_input(
            "Gotcha", {"text": "  "}, repo="my-repo", files=["a.py"]
        ) == ""

    def test_four_distinct_basenames_sorted_reach_the_prefix(self, monkeypatch):
        """Capped, or a lesson anchored across a tree is buried under
        filenames. Sorted and deduped, so the same lesson embeds the same
        way whether it arrives from capture (extractor order) or from the
        backfill (whatever order the graph hands back)."""
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("files",))

        assert ct.memory_embedding_input(
            "Gotcha",
            {"text": "pin the version"},
            files=["b/two.py", "a/two.py", "one.py", "four.py", "three.py", "five.py"],
        ) == "five.py, four.py, one.py, three.py | pin the version"


class TestTriggersInTheEmbeddingInput:
    """MUD-457 (E2): trigger sentences ride in the embedded string, never
    in the canonical text — a lesson that gains triggers keeps its id."""

    def test_triggers_are_appended_after_the_lesson(self, monkeypatch):
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("files",))

        assert ct.memory_embedding_input(
            "Gotcha", {"text": "pin the version"}, files=["src/deps.py"],
            triggers=["You are editing deps.py and the build resolves 2.0.",
                      "A fresh checkout installs a different version."],
        ) == (
            "deps.py | pin the version"
            " | You are editing deps.py and the build resolves 2.0."
            " A fresh checkout installs a different version."
        )

    def test_no_triggers_leaves_the_string_alone(self):
        from agent_memory_mcp.mcp._coding_tools import (
            memory_embedding_input,
            memory_embedding_text,
        )

        props = {"text": "pin the version"}
        for triggers in (None, [], ["", "   "]):
            assert memory_embedding_input(
                "Gotcha", props, triggers=triggers
            ) == memory_embedding_text("Gotcha", props)


class TestSymptomOnlyIndex:
    """MUD-460 (E5): an error-keyed query is a symptom, and matching it
    against the fix is the same mismatch the whole phase is about."""

    def test_a_lesson_with_a_symptom_embeds_only_that(self, monkeypatch):
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "SYMPTOM_ONLY", True)

        assert ct.memory_embedding_input(
            "Gotcha", {"symptom": "resolves to 2.0", "text": "pin the version"}
        ) == "resolves to 2.0"

    def test_a_lesson_without_one_falls_back_to_its_text(self, monkeypatch):
        """139 of the 287 pool lessons carry a symptom; the rest would be
        invisible to an error query if they embedded nothing."""
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "SYMPTOM_ONLY", True)

        assert ct.memory_embedding_input(
            "Gotcha", {"text": "pin the version"}
        ) == "pin the version"

    def test_the_canonical_text_is_unchanged(self, monkeypatch):
        """Whatever the index embeds, the id and the labels do not move."""
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "SYMPTOM_ONLY", True)

        assert ct.memory_embedding_text(
            "Gotcha", {"symptom": "resolves to 2.0", "text": "pin the version"}
        ) == "resolves to 2.0 | pin the version"


class TestContextPrefixSpec:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("", ()),
            ("0", ()),
            ("1", ("repo", "kind", "files")),
            ("files", ("files",)),
            ("kind, repo", ("repo", "kind")),
            ("files,bogus", ("files",)),
        ],
    )
    def test_spec_parses_to_parts_in_a_fixed_order(self, raw, expected):
        """NAM_EMBED_CONTEXT_PREFIX names the parts, so the E1 ablation is
        env-only. Order is the format's, not the operator's."""
        from agent_memory_mcp.mcp._coding_tools import context_prefix_spec

        assert context_prefix_spec(raw) == expected


class TestCaptureEmbedsWithContext:
    async def test_each_lesson_is_embedded_with_its_repo_and_its_own_anchors(
        self, monkeypatch, mock_ctx
    ):
        """The prefix is per-lesson: the files that reach it are the ones
        the extractor anchored that lesson to, not the session's file list."""
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("repo", "kind", "files"))
        embedder = FakeEmbedder()
        tools = _register(monkeypatch, FakeGraph(), embedder=embedder)
        _stub_extract(monkeypatch, _extraction(decisions=[], dead_ends=[], preferences=[]))

        await tools["capture_session_memory"](
            mock_ctx, agent_id="a", session_id="s", repo="my-repo", branch="main",
            transcript="user: do the thing", files=["unrelated.py"],
        )

        assert embedder.texts == [
            "my-repo · gotcha · a.py, b.py | pytest asyncio_mode is auto here"
        ]

    async def test_the_canonical_text_still_keys_duplicate_detection(self, monkeypatch):
        """dedupe_fused collapses the same lesson captured twice (MUD-407).
        Keyed on the embedding input, two anchorings would stop matching."""
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("repo", "kind", "files"))
        rows = [
            {"labels": ["Gotcha", "CodingMemory"], "props": {"text": "pin the version"},
             "files": ["src/auth_server.py"], "score": 0.9},
            {"labels": ["Gotcha", "CodingMemory"], "props": {"text": "pin the version"},
             "files": ["src/rate_limit.py"], "score": 0.8},
        ]

        assert ct.dedupe_fused(rows) == rows[:1]


def _load_backfill():
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "scripts", "backfill_coding_memory_embeddings.py")
    spec = importlib.util.spec_from_file_location("backfill_coding_memory_embeddings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBackfillEmbedsWithContext:
    """A store backfilled without the prefix and captured into with it
    holds two incompatible vector spaces, and nothing says so."""

    def test_the_select_carries_the_repo_and_the_anchored_files(self):
        mod = _load_backfill()
        query = mod._select_query()

        assert "MADE_IN" in query and "AS repo" in query
        assert "ABOUT" in query and "AS files" in query

    def test_a_row_embeds_with_its_repo_and_its_anchors(self, monkeypatch):
        import agent_memory_mcp.mcp._coding_tools as ct

        monkeypatch.setattr(ct, "CONTEXT_PREFIX", ("repo", "kind", "files"))
        mod = _load_backfill()

        assert mod.embedding_input_for({
            "labels": ["Gotcha", "CodingMemory"],
            "props": {"text": "pin the version"},
            "repo": "my-repo",
            "files": ["src/auth_server.py"],
        }) == "my-repo · gotcha · auth_server.py | pin the version"

    def test_a_row_of_an_unknown_kind_embeds_nothing(self):
        mod = _load_backfill()

        assert mod.embedding_input_for({"labels": ["Change"], "props": {"text": "x"}}) == ""

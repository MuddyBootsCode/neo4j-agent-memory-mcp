"""Tests for the UserPromptSubmit recall hook (push-mode memory injection).

The hook is a thin client over the memory_search MCP tool: it must be
deterministic, fast, and fail-open — a broken server may never block a
prompt submit. The search transport is injected so these tests exercise
the real formatting/dispatch code without a live server.
"""

import json
import io
from datetime import datetime, timedelta, timezone

import pytest

from agent_memory_mcp.hook.recall_hook import (
    build_hook_output,
    format_coding_memories,
    format_context,
    format_overlap_block,
    run,
)


SAMPLE_RESPONSE = {
    "results": {
        "facts": [
            {
                "subject": "Sarah Chen",
                "predicate": "delegates_to",
                "object": "Marcus Webb",
                "confidence": 0.95,
                "similarity": 0.88,
                "temporal_status": "active",
                "valid_from": 1740787200000,
                "valid_until": 1743465600000,
                "superseded_by": None,
            },
            {
                "subject": "Refund approvals",
                "predicate": "approved_by",
                "object": "Ops Manager",
                "confidence": 0.9,
                "similarity": 0.82,
                "temporal_status": "expired",
                "valid_from": None,
                "valid_until": 1735689600000,
                "superseded_by": "fact-123",
            },
        ],
        "entities": [
            {
                "id": "e1",
                "name": "Sarah Chen",
                "type": "Person",
                "description": "Ops Manager, on leave 1-31 March",
                "neighbors": [
                    {
                        "name": "Ops Manager",
                        "relationship": "holds_role",
                        "direction": "outgoing",
                        "type": "Role",
                        "id": "e2",
                        "description": None,
                        "confidence": 0.9,
                    }
                ],
            }
        ],
        "preferences": [
            {
                "id": "p1",
                "category": "tooling",
                "preference": "Prefers uv over pip",
                "context": "python projects",
            }
        ],
    },
    "query": "who signs off refunds in March",
    "graph_augmented": True,
}


def _run(prompt_payload, search):
    stdin = io.StringIO(json.dumps(prompt_payload))
    stdout = io.StringIO()
    code = run(stdin=stdin, stdout=stdout, search=search)
    return code, stdout.getvalue()


class TestFormatContext:
    def test_renders_facts_as_triples(self):
        text = format_context(SAMPLE_RESPONSE, ms=12.0)
        assert "Sarah Chen --[delegates_to]--> Marcus Webb" in text

    def test_header_counts_items_and_reports_ms(self):
        text = format_context(SAMPLE_RESPONSE, ms=12.0)
        header = text.split("\n")[0]
        assert header.startswith("memory:")
        assert "12 ms" in header
        # 2 facts + 1 entity + 1 preference
        assert "4" in header

    def test_expired_facts_are_annotated(self):
        text = format_context(SAMPLE_RESPONSE, ms=1.0)
        line = next(ln for ln in text.split("\n") if "approved_by" in ln)
        assert "expired" in line

    def test_active_facts_not_annotated_as_expired(self):
        text = format_context(SAMPLE_RESPONSE, ms=1.0)
        line = next(ln for ln in text.split("\n") if "delegates_to" in ln)
        assert "expired" not in line

    def test_entity_descriptions_carried_in_where_block(self):
        text = format_context(SAMPLE_RESPONSE, ms=1.0)
        assert "where:" in text
        assert "on leave 1-31 March" in text

    def test_entity_neighbors_rendered_as_triples(self):
        text = format_context(SAMPLE_RESPONSE, ms=1.0)
        assert "Sarah Chen --[holds_role]--> Ops Manager" in text

    def test_preferences_rendered_with_category(self):
        text = format_context(SAMPLE_RESPONSE, ms=1.0)
        assert "[tooling] Prefers uv over pip" in text

    def test_empty_results_reports_no_matches(self):
        empty = {"results": {"facts": [], "entities": [], "preferences": []}}
        text = format_context(empty, ms=3.0)
        assert "no memory matches" in text

    def test_header_ignores_entities_that_render_nothing(self):
        response = {
            "results": {
                "facts": [],
                "entities": [
                    {"id": "e1", "name": "Ghost", "description": None, "neighbors": []},
                    {"id": "e2", "name": "Sarah Chen", "description": "Ops Manager"},
                ],
                "preferences": [],
            }
        }
        text = format_context(response, ms=1.0)
        assert text.split("\n")[0] == "memory: 1 items recalled in 1 ms"
        assert "Ghost" not in text

    def test_entity_with_edges_and_description_counted_once(self):
        text = format_context(SAMPLE_RESPONSE, ms=12.0)
        # 2 facts + 1 entity (renders an edge and a where: note) + 1 preference
        assert text.split("\n")[0] == "memory: 4 items recalled in 12 ms"

    def test_truncated_header_counts_only_surviving_items(self):
        response = {
            "results": {
                "facts": [
                    {
                        "subject": f"Entity {i}",
                        "predicate": "related_to",
                        "object": "x" * 200,
                        "temporal_status": "active",
                    }
                    for i in range(50)
                ],
                "entities": [],
                "preferences": [],
            }
        }
        text = format_context(response, ms=1.0, max_chars=1000)
        lines = text.split("\n")
        assert lines[-1] == "… (truncated)"
        shown = sum(1 for ln in lines if "related_to" in ln)
        assert lines[0] == f"memory: {shown} items recalled in 1 ms"
        assert shown < 50

    def test_output_capped_at_max_chars(self):
        big = {
            "results": {
                "facts": [
                    {
                        "subject": f"Entity {i}",
                        "predicate": "related_to",
                        "object": f"Other {i}" + "x" * 200,
                        "temporal_status": "active",
                    }
                    for i in range(100)
                ],
                "entities": [],
                "preferences": [],
            }
        }
        text = format_context(big, ms=1.0, max_chars=2000)
        assert len(text) <= 2000
        assert text.startswith("memory:")


class TestSearchArgs:
    def test_default_threshold_is_hook_default_not_server_default(self):
        from agent_memory_mcp.hook.recall_hook import build_search_args

        args = build_search_args("who signs off refunds")
        assert args["query"] == "who signs off refunds"
        assert args["threshold"] == 0.5
        assert args["memory_types"] == ["facts", "entities", "preferences"]

    def test_threshold_overridable_via_env(self, monkeypatch):
        from agent_memory_mcp.hook.recall_hook import build_search_args

        monkeypatch.setenv("NAM_HOOK_THRESHOLD", "0.35")
        assert build_search_args("q")["threshold"] == 0.35


class TestBuildHookOutput:
    def test_shape_matches_userpromptsubmit_contract(self):
        text = "memory: 4 items recalled in 2 ms\nA --[b]--> C"
        out = build_hook_output(text)
        assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert out["hookSpecificOutput"]["additionalContext"] == text
        assert out["systemMessage"] == "memory: 4 items recalled in 2 ms"


class TestRun:
    def test_emits_valid_hook_json_on_success(self):
        code, out = _run(
            {"prompt": "who signs off refunds in March"},
            search=lambda q: json.dumps(SAMPLE_RESPONSE),
        )
        assert code == 0
        payload = json.loads(out)
        ctx = payload["hookSpecificOutput"]["additionalContext"]
        assert "Sarah Chen --[delegates_to]--> Marcus Webb" in ctx

    def test_fail_open_when_search_raises(self):
        def boom(q):
            raise ConnectionError("server down")

        code, out = _run({"prompt": "anything"}, search=boom)
        assert code == 0
        assert out == ""

    def test_fail_open_when_search_returns_error_payload(self):
        code, out = _run(
            {"prompt": "anything"},
            search=lambda q: json.dumps({"error": "neo4j unavailable"}),
        )
        assert code == 0
        assert out == ""

    def test_empty_prompt_skips_search(self):
        calls = []

        def spy(q):
            calls.append(q)
            return json.dumps(SAMPLE_RESPONSE)

        code, out = _run({"prompt": "   "}, search=spy)
        assert code == 0
        assert out == ""
        assert calls == []

    def test_malformed_stdin_fails_open(self):
        stdin = io.StringIO("not json{{{")
        stdout = io.StringIO()
        code = run(stdin=stdin, stdout=stdout, search=lambda q: "{}")
        assert code == 0
        assert stdout.getvalue() == ""


NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


def _seen(**delta):
    """ISO-8601 timestamp `delta` before the fixed NOW."""
    return (NOW - timedelta(**delta)).isoformat()


def _overlap(**overrides):
    row = {
        "agent": "codex-a",
        "files": ["src/a.py"],
        "task": "MUD-395",
        "last_seen": _seen(seconds=10),
    }
    row.update(overrides)
    return row


class TestFormatOverlapBlock:
    def test_empty_overlaps_return_none(self):
        assert format_overlap_block([]) is None
        assert format_overlap_block(None) is None

    def test_header_counts_rows(self):
        text = format_overlap_block([_overlap(), _overlap(agent="codex-b")], now=NOW)
        lines = text.split("\n")
        assert lines[0] == "agents: 2 active nearby"
        assert len(lines) == 3

    @pytest.mark.parametrize(
        ("last_seen", "expected"),
        [
            (_seen(seconds=89), "just now"),
            (_seen(seconds=91), "1m ago"),
            (_seen(minutes=89), "89m ago"),
            (_seen(minutes=91), "1h ago"),
            (_seen(hours=35), "35h ago"),
            (_seen(hours=37), "1d ago"),
        ],
    )
    def test_relative_time_buckets(self, last_seen, expected):
        text = format_overlap_block([_overlap(task=None, last_seen=last_seen)], now=NOW)
        assert f"({expected})" in text

    @pytest.mark.parametrize("last_seen", ["not-a-date", None, 12345])
    def test_malformed_last_seen_degrades_to_recently(self, last_seen):
        text = format_overlap_block([_overlap(task=None, last_seen=last_seen)], now=NOW)
        assert "(recently)" in text

    def test_naive_timestamp_treated_as_utc(self):
        naive = (NOW - timedelta(hours=2)).replace(tzinfo=None).isoformat()
        text = format_overlap_block([_overlap(task=None, last_seen=naive)], now=NOW)
        assert "(2h ago)" in text

    def test_files_capped_at_three_with_more_marker(self):
        files = ["src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py"]
        text = format_overlap_block([_overlap(files=files)], now=NOW)
        assert "src/a.py, src/b.py, src/c.py, +2 more" in text
        assert "src/d.py" not in text

    def test_line_with_task_and_files(self):
        text = format_overlap_block([_overlap()], now=NOW)
        assert "  codex-a is editing src/a.py (MUD-395, just now)" in text

    def test_null_task_is_omitted(self):
        text = format_overlap_block([_overlap(task=None)], now=NOW)
        assert "  codex-a is editing src/a.py (just now)" in text

    def test_empty_files_with_task_says_working_on(self):
        text = format_overlap_block([_overlap(files=[])], now=NOW)
        assert "  codex-a is working on MUD-395 (just now)" in text

    def test_thoroughly_malformed_row_does_not_raise(self):
        rows = [
            {"agent": 42, "files": "not-a-list", "task": 7, "last_seen": None},
            {"files": [1, None]},
            "not-even-a-dict",
        ]
        text = format_overlap_block(rows, now=NOW)
        assert text.startswith("agents: 3 active nearby")


class TestFormatCodingMemories:
    def test_empty_memories_return_none(self):
        assert format_coding_memories([]) is None
        assert format_coding_memories(None) is None

    def test_line_with_files_and_task(self):
        memories = [
            {
                "kind": "Decision",
                "text": "Use fastmcp streamable HTTP",
                "files": ["src/server.py"],
                "task": "MUD-395",
            }
        ]
        text = format_coding_memories(memories)
        assert text == "[decision] Use fastmcp streamable HTTP (src/server.py, MUD-395)"

    def test_line_with_files_only(self):
        memories = [{"kind": "Gotcha", "text": "Neo4j needs tz", "files": ["a.py"], "task": None}]
        assert format_coding_memories(memories) == "[gotcha] Neo4j needs tz (a.py)"

    def test_line_with_task_only(self):
        memories = [{"kind": "Gotcha", "text": "Neo4j needs tz", "files": [], "task": "MUD-1"}]
        assert format_coding_memories(memories) == "[gotcha] Neo4j needs tz (MUD-1)"

    def test_line_with_neither_files_nor_task(self):
        memories = [{"kind": "Gotcha", "text": "Neo4j needs tz", "files": [], "task": None}]
        assert format_coding_memories(memories) == "[gotcha] Neo4j needs tz"

    def test_dead_end_kind_maps_to_two_words(self):
        memories = [{"kind": "DeadEnd", "text": "Tried X", "files": [], "task": None}]
        assert format_coding_memories(memories) == "[dead end] Tried X"

    def test_unknown_kind_lowercased_as_is(self):
        memories = [{"kind": "Insight", "text": "Y", "files": [], "task": None}]
        assert format_coding_memories(memories) == "[insight] Y"

    def test_files_capped_at_three_with_more_marker(self):
        memories = [
            {"kind": "Decision", "text": "Z", "files": ["a", "b", "c", "d"], "task": None}
        ]
        assert format_coding_memories(memories) == "[decision] Z (a, b, c, +1 more)"

    def test_row_missing_text_is_skipped(self):
        memories = [
            {"kind": "Decision", "files": ["a.py"], "task": None},
            {"kind": "Gotcha", "text": "kept", "files": [], "task": None},
        ]
        assert format_coding_memories(memories) == "[gotcha] kept"

    def test_all_rows_skipped_returns_none(self):
        memories = [{"kind": "Decision"}, {"kind": "Gotcha", "text": ""}]
        assert format_coding_memories(memories) is None

    def test_thoroughly_malformed_row_does_not_raise(self):
        memories = [
            {"kind": 3, "text": "still shown", "files": 7, "task": ["x"]},
            "not-a-dict",
        ]
        text = format_coding_memories(memories)
        assert "still shown" in text

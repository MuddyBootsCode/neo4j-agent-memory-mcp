"""Tests for the UserPromptSubmit recall hook (push-mode memory injection).

The hook is a thin client over the memory_search MCP tool: it must be
deterministic, fast, and fail-open — a broken server may never block a
prompt submit. The search transport is injected so these tests exercise
the real formatting/dispatch code without a live server.
"""

import json
import io
import re
import time
from datetime import datetime, timedelta, timezone

import pytest

from agent_memory_mcp.hook import recall_hook
from agent_memory_mcp.hook.recall_hook import (
    _lookback_since_iso,
    build_hook_output,
    format_coding_memories,
    format_context,
    format_overlap_block,
    gather_session_context,
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


class TestCodingRecallArgs:
    def test_default_overlap_window(self, monkeypatch):
        from agent_memory_mcp.hook.recall_hook import build_coding_recall_args

        monkeypatch.delenv("NAM_OVERLAP_WINDOW_HOURS", raising=False)
        args = build_coding_recall_args("p", dict(CTX))
        assert args == {
            "prompt": "p",
            "agent_id": "agent-1",
            "session_id": CTX.get("session_id") or "agent-1",
            "repo": "neo4j-agent-memory-mcp",
            "files": ["src/a.py"],
            "task_key": "MUD-395",
            "overlap_window_hours": 24.0,
        }

    def test_overlap_window_overridable_via_env(self, monkeypatch):
        from agent_memory_mcp.hook.recall_hook import build_coding_recall_args

        monkeypatch.setenv("NAM_OVERLAP_WINDOW_HOURS", "6")
        assert build_coding_recall_args("p", dict(CTX))["overlap_window_hours"] == 6.0


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
            (_seen(seconds=-7200), "just now"),  # future-dated clock skew
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
        memories = [
            {
                "kind": "Gotcha",
                "text": "Neo4j needs tz",
                "files": ["a.py"],
                "task": None,
            }
        ]
        assert format_coding_memories(memories) == "[gotcha] Neo4j needs tz (a.py)"

    def test_line_with_task_only(self):
        memories = [
            {"kind": "Gotcha", "text": "Neo4j needs tz", "files": [], "task": "MUD-1"}
        ]
        assert format_coding_memories(memories) == "[gotcha] Neo4j needs tz (MUD-1)"

    def test_line_with_neither_files_nor_task(self):
        memories = [
            {"kind": "Gotcha", "text": "Neo4j needs tz", "files": [], "task": None}
        ]
        assert format_coding_memories(memories) == "[gotcha] Neo4j needs tz"

    def test_dead_end_kind_maps_to_two_words(self):
        memories = [{"kind": "DeadEnd", "text": "Tried X", "files": [], "task": None}]
        assert format_coding_memories(memories) == "[dead end] Tried X"

    def test_unknown_kind_lowercased_as_is(self):
        memories = [{"kind": "Insight", "text": "Y", "files": [], "task": None}]
        assert format_coding_memories(memories) == "[insight] Y"

    def test_files_capped_at_three_with_more_marker(self):
        memories = [
            {
                "kind": "Decision",
                "text": "Z",
                "files": ["a", "b", "c", "d"],
                "task": None,
            }
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


# ---------------------------------------------------------------------------
# Coding path (Task 7): session context gathering + coding_recall composition.
# All transports are faked; nothing here touches git or a server.
# ---------------------------------------------------------------------------

CTX = {
    "cwd": "/repo",
    "repo": "neo4j-agent-memory-mcp",
    "branch": "feature/MUD-395",
    "files": ["src/a.py"],
    "task_key": "MUD-395",
    "agent_id": "agent-1",
    "session_id": "sess-1",
}


def _sample_search(q):
    return json.dumps(SAMPLE_RESPONSE)


def _coding_response(**overrides):
    resp = {"memories": [], "fallback": False, "overlaps": []}
    resp.update(overrides)
    return resp


def _run_b(payload, search=None, coding_recall=None, gather=None):
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    code = run(
        stdin=stdin,
        stdout=stdout,
        search=search
        if search is not None
        else (lambda q: json.dumps(SAMPLE_RESPONSE)),
        coding_recall=coding_recall,
        gather=gather,
    )
    return code, stdout.getvalue()


class TestGatherSessionContext:
    def _stub_git(self, monkeypatch, repo="neo", branch="feature/MUD-395", files=None):
        monkeypatch.setattr(recall_hook, "repo_name", lambda c: repo)
        monkeypatch.setattr(recall_hook, "current_branch", lambda c: branch)
        monkeypatch.setattr(
            recall_hook, "edited_files", lambda c: list(files or ["a.py"])
        )
        monkeypatch.delenv("NAM_AGENT_ID", raising=False)
        monkeypatch.delenv("NAM_TASK_KEY", raising=False)

    def test_missing_cwd_returns_none(self):
        assert gather_session_context({}) is None
        assert gather_session_context({"cwd": ""}) is None

    def test_non_repo_cwd_returns_none(self, monkeypatch):
        self._stub_git(monkeypatch, repo=None)
        monkeypatch.setattr(recall_hook, "repo_name", lambda c: None)
        assert gather_session_context({"cwd": "/not/a/repo"}) is None

    def test_context_fields_from_git(self, monkeypatch):
        self._stub_git(monkeypatch)
        ctx = gather_session_context({"cwd": "/x", "session_id": "sess-9"})
        assert ctx == {
            "cwd": "/x",
            "repo": "neo",
            "branch": "feature/MUD-395",
            "files": ["a.py"],
            "task_key": "MUD-395",
            "agent_id": "sess-9",
            "session_id": "sess-9",
            "git_budget_spent": False,
        }

    def test_env_agent_id_override(self, monkeypatch):
        self._stub_git(monkeypatch)
        monkeypatch.setenv("NAM_AGENT_ID", "codex-7")
        ctx = gather_session_context({"cwd": "/x", "session_id": "sess-9"})
        assert ctx["agent_id"] == "codex-7"
        assert ctx["session_id"] == "sess-9"

    def test_agent_id_defaults_when_no_env_and_no_session(self, monkeypatch):
        self._stub_git(monkeypatch)
        ctx = gather_session_context({"cwd": "/x"})
        assert ctx["agent_id"] == "unknown-agent"
        assert ctx["session_id"] == "unknown-agent"

    def test_files_capped_at_fifty(self, monkeypatch):
        self._stub_git(monkeypatch, files=[f"f{i}.py" for i in range(60)])
        ctx = gather_session_context({"cwd": "/x"})
        assert len(ctx["files"]) == 50

    def test_gather_exception_returns_none(self, monkeypatch):
        monkeypatch.setattr(recall_hook, "repo_name", lambda c: 1 / 0)
        assert gather_session_context({"cwd": "/x"}) is None

    def test_budget_skips_later_collectors(self, monkeypatch):
        calls = []

        def slow_repo(c):
            time.sleep(0.05)
            return "neo"

        monkeypatch.setenv("NAM_HOOK_GIT_BUDGET", "0.01")
        monkeypatch.delenv("NAM_AGENT_ID", raising=False)
        monkeypatch.delenv("NAM_TASK_KEY", raising=False)
        monkeypatch.setattr(recall_hook, "repo_name", slow_repo)
        monkeypatch.setattr(
            recall_hook, "current_branch", lambda c: calls.append("branch") or "b"
        )
        monkeypatch.setattr(
            recall_hook, "edited_files", lambda c: calls.append("files") or ["f"]
        )
        ctx = gather_session_context({"cwd": "/x", "session_id": "s"})
        assert calls == []
        assert ctx is not None
        assert ctx["branch"] is None
        assert ctx["files"] == []
        assert ctx["git_budget_spent"] is True


class TestLookbackSinceIso:
    def test_default_is_24_hours(self, monkeypatch):
        monkeypatch.delenv("NAM_CAPTURE_LOOKBACK_HOURS", raising=False)
        assert _lookback_since_iso(NOW) == (NOW - timedelta(hours=24)).isoformat()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NAM_CAPTURE_LOOKBACK_HOURS", "2.5")
        assert _lookback_since_iso(NOW) == (NOW - timedelta(hours=2.5)).isoformat()


class TestRunCodingPath:
    def _ctx_text(self, out):
        return json.loads(out)["hookSpecificOutput"]["additionalContext"]

    def _path_a_reference(self, prompt, search):
        # No cwd in the payload: the real gather bails, giving pure Path A.
        return _run({"prompt": prompt}, search=search)

    def test_overlap_block_before_memory_lines(self):
        resp = _coding_response(
            memories=[
                {"kind": "Gotcha", "text": "Neo4j needs tz", "files": [], "task": None}
            ],
            overlaps=[_overlap()],
        )
        code, out = _run_b(
            {"prompt": "p", "cwd": "/repo"},
            coding_recall=lambda p, c: resp,
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        payload = json.loads(out)
        ctx_text = payload["hookSpecificOutput"]["additionalContext"]
        assert ctx_text.index("agents: 1 active nearby") < ctx_text.index("[gotcha]")
        sm = payload["systemMessage"]
        assert "1 items recalled" in sm
        assert "1 agents nearby" in sm

    def test_fallback_true_routes_to_injected_search(self):
        searched = []

        def spy(q):
            searched.append(q)
            return json.dumps(SAMPLE_RESPONSE)

        resp = _coding_response(fallback=True, overlaps=[_overlap()])
        code, out = _run_b(
            {"prompt": "p", "cwd": "/repo"},
            search=spy,
            coding_recall=lambda p, c: resp,
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        assert searched == ["p"]
        ctx_text = self._ctx_text(out)
        assert "Sarah Chen --[delegates_to]--> Marcus Webb" in ctx_text
        assert ctx_text.index("agents: 1 active nearby") < ctx_text.index(
            "Sarah Chen --[delegates_to]"
        )
        sm = json.loads(out)["systemMessage"]
        assert "4 items recalled" in sm
        assert "1 agents nearby" in sm

    def test_coding_recall_none_matches_path_a(self):
        search = _sample_search
        ref_code, ref_out = self._path_a_reference("who approves", search)
        code, out = _run_b(
            {"prompt": "who approves", "cwd": "/repo"},
            search=search,
            coding_recall=lambda p, c: None,
            gather=lambda pl: dict(CTX),
        )
        assert (code, out) == (ref_code, ref_out)

    def test_coding_recall_raises_matches_path_a(self):
        def boom(p, c):
            raise ConnectionError("server down")

        search = _sample_search
        _, ref_out = self._path_a_reference("q", search)
        code, out = _run_b(
            {"prompt": "q", "cwd": "/repo"},
            search=search,
            coding_recall=boom,
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        assert out == ref_out

    def test_gather_none_skips_coding_recall(self):
        calls = []

        def cr(p, c):
            calls.append((p, c))
            return _coding_response()

        search = _sample_search
        _, ref_out = self._path_a_reference("q", search)
        code, out = _run_b(
            {"prompt": "q", "cwd": "/repo"},
            search=search,
            coding_recall=cr,
            gather=lambda pl: None,
        )
        assert code == 0
        assert calls == []
        assert out == ref_out

    def test_empty_memories_run_fallback_search(self):
        # fallback=false with memories=[] must still reach embedding
        # recall: the anchor graph having nothing is not proof the
        # classic index has nothing.
        searched = []

        def spy(q):
            searched.append(q)
            return json.dumps(SAMPLE_RESPONSE)

        code, out = _run_b(
            {"prompt": "q", "cwd": "/repo"},
            search=spy,
            coding_recall=lambda p, c: _coding_response(),
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        assert searched == ["q"]
        payload = json.loads(out)
        ctx_text = payload["hookSpecificOutput"]["additionalContext"]
        assert "Sarah Chen --[delegates_to]--> Marcus Webb" in ctx_text
        assert "4 items recalled" in payload["systemMessage"]

    def test_empty_memories_no_match_search_matches_path_a(self):
        # When the fallback search also finds nothing, the classic
        # no-match output appears, byte-identical to Path A modulo ms.
        empty = json.dumps(
            {"results": {"facts": [], "entities": [], "preferences": []}}
        )
        _, ref_out = self._path_a_reference("q", lambda q: empty)
        code, out = _run_b(
            {"prompt": "q", "cwd": "/repo"},
            search=lambda q: empty,
            coding_recall=lambda p, c: _coding_response(),
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        norm = lambda text: re.sub(r"in \d+ ms", "in N ms", text)  # noqa: E731
        assert norm(out) == norm(ref_out)
        assert "(no memory matches for this prompt)" in out

    def test_empty_memories_failing_search_no_overlaps_emit_nothing(self):
        # Silent exit survives only when the fallback search yields
        # nothing renderable and there is no overlap block either.
        def boom(q):
            raise ConnectionError("server down")

        code, out = _run_b(
            {"prompt": "q", "cwd": "/repo"},
            search=boom,
            coding_recall=lambda p, c: _coding_response(),
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        assert out == ""

    def test_empty_memories_with_overlaps_render_overlaps_plus_search(self):
        resp = _coding_response(overlaps=[_overlap()])
        code, out = _run_b(
            {"prompt": "p", "cwd": "/repo"},
            coding_recall=lambda p, c: resp,
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        payload = json.loads(out)
        ctx_text = payload["hookSpecificOutput"]["additionalContext"]
        assert ctx_text.index("agents: 1 active nearby") < ctx_text.index(
            "Sarah Chen --[delegates_to]"
        )
        sm = payload["systemMessage"]
        assert "4 items recalled" in sm
        assert "1 agents nearby" in sm

    @pytest.mark.parametrize("bad", [["not-a-dict"], "junk", 42, {}, {"memories": []}])
    def test_malformed_coding_response_falls_back_to_path_a(self, bad):
        search = _sample_search
        _, ref_out = self._path_a_reference("q", search)
        code, out = _run_b(
            {"prompt": "q", "cwd": "/repo"},
            search=search,
            coding_recall=lambda p, c: bad,
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        assert out == ref_out

    def test_char_cap_applies_to_memory_section_only(self, monkeypatch):
        monkeypatch.setenv("NAM_HOOK_MAX_CHARS", "40")
        resp = _coding_response(
            memories=[{"kind": "Gotcha", "text": "x" * 200, "files": [], "task": None}],
            overlaps=[_overlap()],
        )
        code, out = _run_b(
            {"prompt": "p", "cwd": "/repo"},
            coding_recall=lambda p, c: resp,
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        ctx_text = self._ctx_text(out)
        assert "codex-a is editing src/a.py" in ctx_text
        assert "x" * 200 not in ctx_text
        assert "… (truncated)" in ctx_text


class TestFormatterRideAlongs:
    def test_coding_memory_list_task_dropped(self):
        memories = [{"kind": "Decision", "text": "T", "files": [], "task": ["MUD-1"]}]
        assert format_coding_memories(memories) == "[decision] T"

    def test_overlap_non_string_task_dropped(self):
        text = format_overlap_block([_overlap(task=["MUD-1"])], now=NOW)
        assert "  codex-a is editing src/a.py (just now)" in text

    def test_multiline_memory_text_flattened_to_one_line(self):
        memories = [
            {"kind": "Gotcha", "text": "line1\nline2", "files": [], "task": None}
        ]
        assert format_coding_memories(memories) == "[gotcha] line1 line2"


class TestFallbackSearchFailure:
    def test_overlap_block_survives_raising_search(self):
        def boom(q):
            raise ConnectionError("server down")

        resp = _coding_response(fallback=True, overlaps=[_overlap()])
        code, out = _run_b(
            {"prompt": "p", "cwd": "/repo"},
            search=boom,
            coding_recall=lambda p, c: resp,
            gather=lambda pl: dict(CTX),
        )
        assert code == 0
        payload = json.loads(out)
        ctx_text = payload["hookSpecificOutput"]["additionalContext"]
        assert "agents: 1 active nearby" in ctx_text
        assert "Sarah Chen" not in ctx_text
        sm = payload["systemMessage"]
        assert "0 items recalled" in sm
        assert "1 agents nearby" in sm


class TestParseContextHeader:
    def test_pinned_to_header_wording(self):
        # Regression guard: if _header's wording changes, this breaks and
        # _CONTEXT_HEADER_RE must change with it.
        from agent_memory_mcp.hook.recall_hook import _header, _parse_context_header

        count, body = _parse_context_header(_header(7, 3.0) + "\n\nA --[b]--> C")
        assert count == 7
        assert body == "A --[b]--> C"

    def test_unrecognized_header_degrades_to_zero(self):
        from agent_memory_mcp.hook.recall_hook import _parse_context_header

        count, body = _parse_context_header("something else\nbody")
        assert count == 0
        assert body == "body"


class TestCodingRecallViaMcp:
    """Transport internals with a faked fastmcp Client (no network)."""

    def _fake_client(self, monkeypatch, payload_text, tool_calls=None, tool_args=None):
        import fastmcp

        class FakeResult:
            data = payload_text

        class FakeClient:
            def __init__(self, transport, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def call_tool(self, name, args):
                if tool_calls is not None:
                    tool_calls.append(name)
                if tool_args is not None:
                    tool_args[name] = args
                return FakeResult()

        monkeypatch.setattr(fastmcp, "Client", FakeClient)

    def test_hanging_capture_does_not_destroy_recall(self, monkeypatch):
        import asyncio as aio

        recall_json = json.dumps({"memories": [], "fallback": True, "overlaps": []})
        self._fake_client(monkeypatch, recall_json)

        async def hang(client, ctx, commits):
            await aio.sleep(10)

        monkeypatch.setattr(recall_hook, "commits_since", lambda cwd, since: [])
        monkeypatch.setattr(recall_hook, "record_activity_via_mcp", hang)
        monkeypatch.setattr(recall_hook, "CAPTURE_TIMEOUT", 0.05)
        result = recall_hook.coding_recall_via_mcp("p", dict(CTX))
        assert result == {"memories": [], "fallback": True, "overlaps": []}

    def test_overlap_window_env_reaches_tool_args(self, monkeypatch):
        recall_json = json.dumps({"memories": [], "fallback": False, "overlaps": []})
        seen = {}
        self._fake_client(monkeypatch, recall_json, tool_args=seen)
        monkeypatch.setattr(recall_hook, "commits_since", lambda cwd, since: [])
        monkeypatch.setenv("NAM_OVERLAP_WINDOW_HOURS", "6")
        result = recall_hook.coding_recall_via_mcp("p", dict(CTX))
        assert result == {"memories": [], "fallback": False, "overlaps": []}
        assert seen["coding_recall"]["overlap_window_hours"] == 6.0

    def test_budget_spent_skips_commit_scan(self, monkeypatch):
        recall_json = json.dumps({"memories": [], "fallback": False, "overlaps": []})
        calls = []
        self._fake_client(monkeypatch, recall_json, tool_calls=calls)
        scans = []
        monkeypatch.setattr(
            recall_hook, "commits_since", lambda cwd, since: scans.append(cwd) or []
        )
        result = recall_hook.coding_recall_via_mcp(
            "p", dict(CTX, git_budget_spent=True)
        )
        assert scans == []
        assert result == {"memories": [], "fallback": False, "overlaps": []}
        assert calls == ["coding_recall", "record_coding_activity"]

"""Tests for the UserPromptSubmit recall hook (push-mode memory injection).

The hook is a thin client over the memory_search MCP tool: it must be
deterministic, fast, and fail-open — a broken server may never block a
prompt submit. The search transport is injected so these tests exercise
the real formatting/dispatch code without a live server.
"""

import json
import io

from agent_memory_mcp.hook.recall_hook import (
    build_hook_output,
    format_context,
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

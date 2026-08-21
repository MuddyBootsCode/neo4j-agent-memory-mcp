"""Unit tests for the deterministic-plane Cypher builders.

Pure-function tests: every builder returns ``(query, params)`` and is asserted
textually — no Neo4j required. Injection safety is checked by passing sentinel
argument values and asserting they never appear in the query text.
"""

from __future__ import annotations

import pytest

from agent_memory_mcp.capture.cypher import (
    RECALL_KINDS,
    anchored_memory_write,
    commit_upsert,
    editing_upsert,
    session_upsert,
)

TS = "2026-08-18T12:00:00Z"
SENTINEL = "INJECT'ED\"--"


def _assert_no_arg_leaks(query: str, values: list[str]) -> None:
    for value in values:
        assert value not in query, f"argument value leaked into query: {value!r}"


# ---------------------------------------------------------------------------
# session_upsert
# ---------------------------------------------------------------------------


class TestSessionUpsert:
    def test_params_without_task_key(self):
        query, params = session_upsert(
            agent_id="agent-1",
            session_id="sess-1",
            repo="repo-a",
            branch="main",
            task_key=None,
            ts=TS,
        )
        assert params == {
            "agent_id": "agent-1",
            "session_id": "sess-1",
            "repo": "repo-a",
            "branch": "main",
            "ts": TS,
        }
        assert "WorkTask" not in query
        assert "WORKING_ON" not in query

    def test_params_with_task_key(self):
        query, params = session_upsert(
            agent_id="agent-1",
            session_id="sess-1",
            repo="repo-a",
            branch="main",
            task_key="MUD-395",
            ts=TS,
        )
        assert params == {
            "agent_id": "agent-1",
            "session_id": "sess-1",
            "repo": "repo-a",
            "branch": "main",
            "task_key": "MUD-395",
            "ts": TS,
        }
        assert "MERGE (t:WorkTask {key: $task_key})" in query
        assert "MERGE (s)-[r:WORKING_ON]->(t)" in query
        assert "SET r.at = datetime($ts)" in query

    def test_merge_patterns_and_timestamps(self):
        query, _ = session_upsert("a", "s", "r", "b", None, TS)
        assert "MERGE (a:CodeAgent {id: $agent_id})" in query
        assert "MERGE (s:CodingSession {id: $session_id})" in query
        assert "MERGE (a)-[:RUNS]->(s)" in query
        assert "ON CREATE SET" in query
        assert "s.started_at = datetime($ts), s.last_seen = datetime($ts)" in query
        assert "ON MATCH SET" in query
        assert "s.last_seen = datetime($ts)" in query
        assert "datetime()" not in query.replace("datetime($ts)", "")

    def test_only_merge_clauses(self):
        for task_key in (None, "MUD-1"):
            query, _ = session_upsert("a", "s", "r", "b", task_key, TS)
            assert "CREATE (" not in query

    def test_no_argument_interpolation(self):
        args = [f"{SENTINEL}{i}" for i in range(6)]
        query, _ = session_upsert(*args)
        _assert_no_arg_leaks(query, args)


# ---------------------------------------------------------------------------
# editing_upsert
# ---------------------------------------------------------------------------


class TestEditingUpsert:
    def test_empty_paths_returns_none(self):
        assert editing_upsert("sess-1", "repo-a", [], TS) is None

    def test_params(self):
        query, params = editing_upsert("sess-1", "repo-a", ["a.py", "b.py"], TS)
        assert params == {
            "session_id": "sess-1",
            "repo": "repo-a",
            "paths": ["a.py", "b.py"],
            "ts": TS,
        }

    def test_unwind_single_list_param(self):
        query, _ = editing_upsert("s", "r", ["a.py"], TS)
        assert "UNWIND $paths AS path" in query
        assert "MERGE (f:CodeFile {repo: $repo, path: path})" in query

    def test_set_refreshes_relationship_timestamp(self):
        query, _ = editing_upsert("s", "r", ["a.py"], TS)
        assert "MERGE (s)-[r:EDITING]->(f)" in query
        assert "SET r.at = datetime($ts)" in query
        assert "ON CREATE SET r.at" not in query
        assert "ON MATCH SET r.at" not in query

    def test_only_merge_clauses(self):
        query, _ = editing_upsert("s", "r", ["a.py"], TS)
        assert "CREATE (" not in query

    def test_no_argument_interpolation(self):
        session_id, repo, path = f"{SENTINEL}0", f"{SENTINEL}1", f"{SENTINEL}2"
        query, _ = editing_upsert(session_id, repo, [path], TS)
        _assert_no_arg_leaks(query, [session_id, repo, path])


# ---------------------------------------------------------------------------
# commit_upsert
# ---------------------------------------------------------------------------


class TestCommitUpsert:
    def test_params(self):
        query, params = commit_upsert(
            session_id="sess-1",
            repo="repo-a",
            sha="abc123",
            message="fix: thing",
            paths=["a.py"],
            ts=TS,
        )
        assert params == {
            "session_id": "sess-1",
            "repo": "repo-a",
            "sha": "abc123",
            "message": "fix: thing",
            "paths": ["a.py"],
            "ts": TS,
        }

    def test_merge_patterns(self):
        query, _ = commit_upsert("s", "r", "sha", "m", ["a.py"], TS)
        assert "MERGE (c:Change {sha: $sha})" in query
        assert "ON CREATE SET" in query
        assert "c.message = $message" in query
        assert "c.repo = $repo" in query
        assert "c.at = datetime($ts)" in query
        assert "MERGE (s)-[:PERFORMED]->(c)" in query
        assert "UNWIND $paths AS path" in query
        assert "MERGE (f:CodeFile {repo: $repo, path: path})" in query
        assert "MERGE (c)-[:TOUCHED]->(f)" in query

    def test_idempotent_only_merge_clauses(self):
        query, _ = commit_upsert("s", "r", "sha", "m", ["a.py"], TS)
        # "CREATE (" cannot false-positive on "ON CREATE SET": the SET form
        # is never followed by an open paren.
        assert "CREATE (" not in query

    def test_no_argument_interpolation(self):
        session_id, repo, sha, message, path = (f"{SENTINEL}{i}" for i in range(5))
        query, _ = commit_upsert(session_id, repo, sha, message, [path], TS)
        _assert_no_arg_leaks(query, [session_id, repo, sha, message, path])


# ---------------------------------------------------------------------------
# anchored_memory_write
# ---------------------------------------------------------------------------

ALLOWED_KINDS = ["Decision", "Gotcha", "DeadEnd", "CodingPreference"]


class TestAnchoredMemoryWrite:
    def test_rejects_bad_kind(self):
        with pytest.raises(ValueError, match="EvilLabel"):
            anchored_memory_write(
                kind="EvilLabel",
                props={"text": "x"},
                session_id="s",
                repo="r",
                anchor_paths=[],
                task_key=None,
                ts=TS,
            )

    def test_rejects_injection_in_kind(self):
        bad = "Decision) DETACH DELETE n //"
        with pytest.raises(ValueError):
            anchored_memory_write(bad, {}, "s", "r", [], None, TS)

    @pytest.mark.parametrize("kind", ALLOWED_KINDS)
    def test_accepts_each_allowlisted_kind(self, kind):
        query, _ = anchored_memory_write(kind, {"text": "x"}, "s", "r", [], None, TS)
        # Recall kinds carry the shared label so one vector index spans them.
        expected = (
            f"CREATE (m:{kind}:CodingMemory)"
            if kind in RECALL_KINDS
            else f"CREATE (m:{kind})"
        )
        assert expected in query

    @pytest.mark.parametrize(
        "bad_value", [{"nested": 1}, [1, 2], (1,), {1, 2}]
    )
    def test_rejects_nested_props(self, bad_value):
        with pytest.raises(ValueError):
            anchored_memory_write(
                "Decision", {"bad": bad_value}, "s", "r", [], None, TS
            )

    def test_props_map_param_no_per_key_interpolation(self):
        props = {"text": f"{SENTINEL}text", "confidence": 0.9, "flag": True}
        query, params = anchored_memory_write(
            "Gotcha", props, "s", "r", [], None, TS
        )
        assert "SET m = $props" in query
        assert "m.created_at = datetime($ts)" in query
        assert params["props"] == props
        _assert_no_arg_leaks(query, [f"{SENTINEL}text"])
        assert "$text" not in query
        assert "$confidence" not in query

    def test_params_without_task_key(self):
        query, params = anchored_memory_write(
            "Decision", {"text": "x"}, "sess-1", "repo-a", ["a.py"], None, TS
        )
        assert params == {
            "props": {"text": "x"},
            "session_id": "sess-1",
            "repo": "repo-a",
            "anchor_paths": ["a.py"],
            "ts": TS,
        }
        assert "CONCERNS" not in query
        assert "WorkTask" not in query

    def test_params_with_task_key(self):
        query, params = anchored_memory_write(
            "Decision", {"text": "x"}, "sess-1", "repo-a", ["a.py"], "MUD-395", TS
        )
        assert params == {
            "props": {"text": "x"},
            "session_id": "sess-1",
            "repo": "repo-a",
            "anchor_paths": ["a.py"],
            "task_key": "MUD-395",
            "ts": TS,
        }
        assert "MERGE (t:WorkTask {key: $task_key})" in query
        assert "MERGE (m)-[:CONCERNS]->(t)" in query

    def test_anchor_edges_and_session_link(self):
        query, _ = anchored_memory_write(
            "DeadEnd", {"text": "x"}, "s", "r", ["a.py"], None, TS
        )
        assert "MATCH (s:CodingSession {id: $session_id})" in query
        assert "MERGE (m)-[:MADE_IN]->(s)" in query
        assert "UNWIND $anchor_paths AS path" in query
        assert "MERGE (f:CodeFile {repo: $repo, path: path})" in query
        assert "MERGE (m)-[:ABOUT]->(f)" in query

    def test_single_create_clause(self):
        query, _ = anchored_memory_write(
            "Decision", {"text": "x"}, "s", "r", ["a.py"], "MUD-1", TS
        )
        assert query.count("CREATE (") == 1

    def test_no_argument_interpolation(self):
        session_id, repo, path, task_key = (f"{SENTINEL}{i}" for i in range(4))
        query, _ = anchored_memory_write(
            "Decision",
            {"text": f"{SENTINEL}p"},
            session_id,
            repo,
            [path],
            task_key,
            TS,
        )
        _assert_no_arg_leaks(
            query, [session_id, repo, path, task_key, f"{SENTINEL}p"]
        )

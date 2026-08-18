"""Pure Cypher builders for the deterministic coding-memory plane.

Every function returns ``(query, params)`` for the caller to run through
``client.graph.execute_write`` — nothing here touches Neo4j. All argument
values travel as query parameters; the single exception is the node label in
:func:`anchored_memory_write`, which is validated against a fixed allowlist
before interpolation.

Timestamps: callers supply ``ts`` as an ISO 8601 string and queries use
``datetime($ts)``, never bare ``datetime()``, so writes are reproducible and
testable.
"""

from __future__ import annotations

from typing import Any

# Labels permitted for anchored extracted-plane memories. The label is the
# only string interpolated into any query in this module.
ANCHORED_KINDS = frozenset({"Decision", "Gotcha", "DeadEnd", "CodingPreference"})

_PRIMITIVES = (str, int, float, bool)


def session_upsert(
    agent_id: str,
    session_id: str,
    repo: str,
    branch: str,
    task_key: str | None,
    ts: str,
) -> tuple[str, dict[str, Any]]:
    """Upsert an agent, its coding session, and (optionally) the task link.

    On create the session records repo, branch, ``started_at``, and
    ``last_seen``; on match it refreshes branch and ``last_seen``.
    ``repo`` is intentionally not refreshed on match — it is immutable per
    session id. When ``task_key`` is given, the ``WorkTask`` is MERGEd and
    the ``WORKING_ON`` edge timestamp is refreshed on every call. ``ts`` is an
    ISO 8601 string.
    """
    query = """
        MERGE (a:CodeAgent {id: $agent_id})
        MERGE (s:CodingSession {id: $session_id})
        ON CREATE SET s.repo = $repo, s.branch = $branch,
            s.started_at = datetime($ts), s.last_seen = datetime($ts)
        ON MATCH SET s.branch = $branch, s.last_seen = datetime($ts)
        MERGE (a)-[:RUNS]->(s)
    """
    params: dict[str, Any] = {
        "agent_id": agent_id,
        "session_id": session_id,
        "repo": repo,
        "branch": branch,
        "ts": ts,
    }
    if task_key is not None:
        query += """
        MERGE (t:WorkTask {key: $task_key})
        MERGE (s)-[r:WORKING_ON]->(t)
        SET r.at = datetime($ts)
        """
        params["task_key"] = task_key
    return query, params


def editing_upsert(
    session_id: str,
    repo: str,
    paths: list[str],
    ts: str,
) -> tuple[str, dict[str, Any]] | None:
    """Record that a session is editing the given files.

    Returns ``None`` when ``paths`` is empty — there is nothing to write and
    callers skip the call entirely. MATCHes the session; if it does not
    exist the whole write is a silent no-op — run session_upsert first.
    The ``EDITING`` edge's ``r.at`` is SET
    (not ON CREATE SET) so it refreshes on every call; recency-window reads
    depend on that. ``ts`` is an ISO 8601 string.
    """
    if not paths:
        return None
    query = """
        MATCH (s:CodingSession {id: $session_id})
        UNWIND $paths AS path
        MERGE (f:CodeFile {repo: $repo, path: path})
        MERGE (s)-[r:EDITING]->(f)
        SET r.at = datetime($ts)
    """
    params: dict[str, Any] = {
        "session_id": session_id,
        "repo": repo,
        "paths": paths,
        "ts": ts,
    }
    return query, params


def commit_upsert(
    session_id: str,
    repo: str,
    sha: str,
    message: str,
    paths: list[str],
    ts: str,
) -> tuple[str, dict[str, Any]]:
    """Upsert a commit and its touched files, linked to the session.

    MERGE everywhere, so re-sending the same commit is idempotent: the
    ``Change`` is keyed on ``sha`` and its message/repo/timestamp are only
    written on create. MATCHes the session; if it does not
    exist the whole write is a silent no-op — run session_upsert first.
    ``ts`` is an ISO 8601 string.
    """
    query = """
        MATCH (s:CodingSession {id: $session_id})
        MERGE (c:Change {sha: $sha})
        ON CREATE SET c.message = $message, c.repo = $repo,
            c.at = datetime($ts)
        MERGE (s)-[:PERFORMED]->(c)
        WITH c
        UNWIND $paths AS path
        MERGE (f:CodeFile {repo: $repo, path: path})
        MERGE (c)-[:TOUCHED]->(f)
    """
    params: dict[str, Any] = {
        "session_id": session_id,
        "repo": repo,
        "sha": sha,
        "message": message,
        "paths": paths,
        "ts": ts,
    }
    return query, params


def anchored_memory_write(
    kind: str,
    props: dict[str, Any],
    session_id: str,
    repo: str,
    anchor_paths: list[str],
    task_key: str | None,
    ts: str,
) -> tuple[str, dict[str, Any]]:
    """Persist an extracted memory node anchored to files, session, and task.

    ``kind`` must be one of :data:`ANCHORED_KINDS`; it becomes the node label
    and is the only interpolated value in this module, so anything else
    raises ``ValueError``. The node is CREATEd, not MERGEd — repeated
    extraction of similar text may produce duplicate nodes, which is
    acceptable in v1. ``props`` values must be primitives (str/int/float/
    bool); nested containers raise ``ValueError`` (Neo4j property
    constraint). Anchor edges come from ``anchor_paths`` (an empty list
    creates the node with no ``ABOUT`` edges); the ``CONCERNS`` edge is only
    built when ``task_key`` is given. MATCHes the session; if it does not
    exist the whole write is a silent no-op — run session_upsert first.
    ``ts`` is an ISO 8601 string.
    """
    if kind not in ANCHORED_KINDS:
        raise ValueError(
            f"kind {kind!r} is not an allowed anchored-memory label; "
            f"expected one of {sorted(ANCHORED_KINDS)}"
        )
    for key, value in props.items():
        if not isinstance(value, _PRIMITIVES):
            raise ValueError(
                f"props[{key!r}] must be a primitive (str/int/float/bool), "
                f"got {type(value).__name__}"
            )

    query = f"""
        MATCH (s:CodingSession {{id: $session_id}})
        CREATE (m:{kind})
        SET m = $props, m.created_at = datetime($ts)
        MERGE (m)-[:MADE_IN]->(s)
    """
    params: dict[str, Any] = {
        "props": props,
        "session_id": session_id,
        "repo": repo,
        "anchor_paths": anchor_paths,
        "ts": ts,
    }
    if task_key is not None:
        query += """
        MERGE (t:WorkTask {key: $task_key})
        MERGE (m)-[:CONCERNS]->(t)
        """
        params["task_key"] = task_key
    query += """
        WITH m
        UNWIND $anchor_paths AS path
        MERGE (f:CodeFile {repo: $repo, path: path})
        MERGE (m)-[:ABOUT]->(f)
    """
    return query, params

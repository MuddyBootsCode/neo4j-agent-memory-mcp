"""Pure Cypher builders for the deterministic coding-memory plane.

Every function returns ``(query, params)`` for the caller to run through
``client.graph.execute_write`` — nothing here touches Neo4j. All argument
values travel as query parameters; the single exception is the node label in
:func:`anchored_memory_write`, which is validated against a fixed allowlist
before interpolation.

Timestamps: callers supply ``ts`` as an ISO 8601 string and queries in this
module use ``datetime($ts)``, never bare ``datetime()``, so writes are
reproducible and testable.
"""

from __future__ import annotations

from typing import Any

# Labels permitted for anchored extracted-plane memories. The label is the
# only string interpolated into any query in this module.
ANCHORED_KINDS = frozenset({"Decision", "Gotcha", "DeadEnd", "CodingPreference"})

# Kinds coding_recall serves. They carry a shared :CodingMemory label in
# addition to their own so a single vector index spans all three -- Neo4j
# vector indexes are per-label, and three indexes would mean three queries to
# merge by hand. CodingPreference is excluded from recall, so it stays
# unlabelled and unindexed.
SHARED_RECALL_LABEL = "CodingMemory"
RECALL_KINDS = frozenset({"Decision", "Gotcha", "DeadEnd"})

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
    embedding: list[float] | None = None,
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

    ``embedding`` travels outside ``props`` because it is a list, which the
    primitive guard above rejects — Neo4j stores a float list fine, the guard
    exists to catch nested containers. It is set only for
    :data:`RECALL_KINDS`, which also gain the :data:`SHARED_RECALL_LABEL` so
    one vector index serves all of them.
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

    labels = kind
    if kind in RECALL_KINDS:
        labels = f"{kind}:{SHARED_RECALL_LABEL}"

    # Lifecycle defaults (MUD-405): a lesson is valid from the session that
    # produced it until something expires it; evidence_count counts the
    # sessions that asserted it, served/helpful/harmful are read-side
    # counters for ranking.
    query = f"""
        MATCH (s:CodingSession {{id: $session_id}})
        CREATE (m:{labels})
        SET m = $props, m.created_at = datetime($ts),
            m.valid_from = datetime($ts), m.evidence_count = 1,
            m.served_count = 0, m.helpful = 0, m.harmful = 0
        MERGE (m)-[:MADE_IN]->(s)
    """
    params: dict[str, Any] = {
        "props": props,
        "session_id": session_id,
        "repo": repo,
        "anchor_paths": anchor_paths,
        "ts": ts,
    }
    if embedding is not None and kind in RECALL_KINDS:
        query += """
        SET m.embedding = $embedding
        """
        params["embedding"] = embedding
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
        RETURN DISTINCT elementId(m) AS eid
    """
    return query, params


# --- Lesson lifecycle (MUD-405) ---------------------------------------------
#
# Lessons accumulate instead of duplicating. The curator's ALREADY_KNOWN
# verdict names an existing node, which gains evidence and a REASSERTED_IN
# edge; SUPERSEDES writes the new lesson and expires the old one, keeping it
# reachable for "why did we stop doing X" but out of recall. Recall records
# what it served (SERVED_TO), and a later commit in that session touching
# the lesson's file closes the loop with RESOLVED_BY.


def reassert_write(eid: str, session_id: str, ts: str) -> tuple[str, dict[str, Any]]:
    """An existing lesson was stated again by another session."""
    query = """
        MATCH (m) WHERE elementId(m) = $eid
        MATCH (s:CodingSession {id: $session_id})
        SET m.evidence_count = coalesce(m.evidence_count, 1) + 1,
            m.last_asserted_at = datetime($ts)
        MERGE (m)-[r:REASSERTED_IN]->(s)
        SET r.at = datetime($ts)
    """
    return query, {"eid": eid, "session_id": session_id, "ts": ts}


def expire_write(eid: str, superseded_by: str | None, ts: str) -> tuple[str, dict[str, Any]]:
    """Retire a lesson. ``superseded_by`` is the elementId of its successor,
    or None when it simply stopped being true. Never deletes."""
    query = """
        MATCH (m) WHERE elementId(m) = $eid
        SET m.expired_at = datetime($ts), m.superseded_by = $superseded_by
    """
    if superseded_by is not None:
        query += """
        WITH m
        MATCH (n) WHERE elementId(n) = $superseded_by
        MERGE (n)-[:SUPERSEDES]->(m)
        """
    return query, {"eid": eid, "superseded_by": superseded_by, "ts": ts}


def served_write(eids: list[str], session_id: str, ts: str) -> tuple[str, dict[str, Any]] | None:
    """Recall injected these lessons into a session. None when nothing was."""
    if not eids:
        return None
    query = """
        MATCH (s:CodingSession {id: $session_id})
        UNWIND $eids AS eid
        MATCH (m) WHERE elementId(m) = eid
        SET m.served_count = coalesce(m.served_count, 0) + 1,
            m.last_served_at = datetime($ts)
        MERGE (m)-[r:SERVED_TO]->(s)
        ON CREATE SET r.at = datetime($ts), r.count = 1
        ON MATCH SET r.at = datetime($ts), r.count = coalesce(r.count, 0) + 1
    """
    return query, {"eids": eids, "session_id": session_id, "ts": ts}


def resolve_write(session_id: str) -> tuple[str, dict[str, Any]]:
    """Link lessons served to a session to the session's later commits that
    touched one of the lesson's files. Idempotent; returns the number of
    new RESOLVED_BY edges as ``resolved``."""
    query = """
        MATCH (m)-[sv:SERVED_TO]->(s:CodingSession {id: $session_id})
        MATCH (s)-[:PERFORMED]->(c:Change)-[:TOUCHED]->(f:CodeFile)<-[:ABOUT]-(m)
        WHERE (m:Gotcha OR m:DeadEnd) AND c.at >= sv.at
          AND NOT EXISTS { MATCH (m)-[:RESOLVED_BY]->(c) }
        MERGE (m)-[:RESOLVED_BY]->(c)
        RETURN count(*) AS resolved
    """
    return query, {"session_id": session_id}

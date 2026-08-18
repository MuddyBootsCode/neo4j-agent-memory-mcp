"""MCP tools for the coding-memory planes.

Three push-model tools. The first two run inside the recall hook before the
model round trip, so they must be cheap and deterministic — no LLM calls.
The third runs at session end, where latency is free, and is the one place
the extracted plane gets written.

- record_coding_activity: write the session/editing/commit facts for a
  coding session via the pure builders in ``capture/cypher.py``.
- coding_recall: anchor-first read of extracted memories (Decision, Gotcha,
  DeadEnd) plus cross-agent overlap detection.
- capture_session_memory: run the transcript through ExtractCodingMemory
  (BAML) and persist the anchored results via ``anchored_memory_write``.

All Cypher in this module is fully parameterized; no user-supplied value is
ever interpolated into query text.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastmcp import Context

from agent_memory_mcp.capture.cypher import (
    anchored_memory_write,
    commit_upsert,
    editing_upsert,
    session_upsert,
)
from agent_memory_mcp.extraction.coding import extract_coding_memory
from agent_memory_mcp.mcp._common import get_client
from agent_memory_mcp.mcp._logging import log_tool_call

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Input caps applied silently; the returned counts reflect what was written.
_MAX_EDITED_FILES = 100
_MAX_COMMITS = 50
_MAX_TRANSCRIPT_CHARS = 80_000
_MAX_ANCHOR_FILES = 100

# Write order for extracted-plane items; also the by_kind key order.
_CAPTURE_KINDS = ("Decision", "Gotcha", "DeadEnd", "CodingPreference")

# Extracted-memory kinds served by coding_recall (CodingPreference is
# deliberately excluded from anchor-first recall in v1).
_RECALL_KINDS = ("Decision", "Gotcha", "DeadEnd")

# The label disjunction is interpolated from _RECALL_KINDS — a fixed module
# constant, never user input — so this is not an injection surface. It keeps
# the query and the rendering logic on a single source of truth.
_KIND_DISJUNCTION = " OR ".join(f"m:{kind}" for kind in _RECALL_KINDS)

# Anchor-first memory read. ``$files`` is always a list (possibly empty) and
# ``$task_key`` may be null — a null comparison never matches, so each anchor
# clause degrades to a no-op when its input is absent.
_MEMORIES_QUERY = """
    MATCH (m)
    WHERE (__KIND_DISJUNCTION__)
      AND (
        EXISTS {
          MATCH (m)-[:ABOUT]->(f:CodeFile)
          WHERE f.repo = $repo AND f.path IN $files
        }
        OR EXISTS {
          MATCH (m)-[:CONCERNS]->(t:WorkTask)
          WHERE t.key = $task_key
        }
      )
    WITH DISTINCT m
    ORDER BY m.created_at DESC
    LIMIT 10
    OPTIONAL MATCH (m)-[:ABOUT]->(af:CodeFile)
    OPTIONAL MATCH (m)-[:CONCERNS]->(wt:WorkTask)
    WITH m,
         [p IN collect(DISTINCT af.path) WHERE p IS NOT NULL] AS files,
         [k IN collect(DISTINCT wt.key) WHERE k IS NOT NULL] AS tasks
    ORDER BY m.created_at DESC
    RETURN labels(m) AS labels,
           properties(m) AS props,
           files,
           head(tasks) AS task,
           toString(m.created_at) AS at
""".replace("__KIND_DISJUNCTION__", _KIND_DISJUNCTION)

# Cross-agent overlap read, anchored from the data: start at the given
# CodeFiles / WorkTask and traverse to sessions and agents, so the match never
# scans every (agent)-[:RUNS]->(session) pair. Recent EDITING / WORKING_ON
# edges (inside the recency window) of OTHER agents qualify. Aggregated to ONE
# row per agent: the union of overlapping files across that agent's qualifying
# sessions, with `last_seen` the overall newest qualifying edge.
_OVERLAPS_QUERY = """
    CALL {
        MATCH (f:CodeFile {repo: $repo})<-[e:EDITING]-(s2:CodingSession)
        WHERE f.path IN $files
          AND e.at >= datetime() - duration({hours: $window})
        RETURN s2, f.path AS path, e.at AS seen
      UNION ALL
        MATCH (t:WorkTask)<-[w:WORKING_ON]-(s2:CodingSession)
        WHERE t.key = $task_key
          AND w.at >= datetime() - duration({hours: $window})
        RETURN s2, null AS path, w.at AS seen
    }
    MATCH (a2:CodeAgent)-[:RUNS]->(s2)
    WHERE a2.id <> $agent_id
    WITH a2,
         [p IN collect(DISTINCT path) WHERE p IS NOT NULL] AS files,
         max(CASE WHEN path IS NULL THEN 1 ELSE 0 END) AS task_hit,
         max(seen) AS last_seen
    ORDER BY last_seen DESC
    LIMIT 20
    RETURN a2.id AS agent,
           files,
           CASE WHEN task_hit = 1 THEN $task_key ELSE null END AS task,
           toString(last_seen) AS last_seen
"""


def _render_memory(row: dict[str, Any]) -> dict[str, Any]:
    """Map a memory-query row to the tool's output shape."""
    labels = row.get("labels") or []
    kind = next((label for label in labels if label in _RECALL_KINDS), None)
    props = row.get("props") or {}
    if kind == "DeadEnd":
        text = f"{props.get('attempt', '')} — failed: {props.get('why_failed', '')}"
    else:
        text = props.get("text", "")
    return {
        "kind": kind,
        "text": text,
        "files": row.get("files") or [],
        "task": row.get("task"),
        "at": row.get("at"),
    }


def register_coding_tools(mcp: FastMCP) -> None:
    """Register the coding-memory tools on the FastMCP server."""

    @mcp.tool()
    @log_tool_call
    async def record_coding_activity(
        ctx: Context,
        agent_id: str,
        session_id: str,
        repo: str,
        branch: str,
        task_key: str | None = None,
        edited_files: list[str] | None = None,
        commits: list[dict] | None = None,
    ) -> str:
        """Record a coding session's activity: session, edited files, commits.

        Push-model capture tool — the session hook calls this directly (before
        any model round trip) to persist deterministic coding facts. One
        timestamp is generated for the whole call and shared by every write,
        and the session upsert always runs first (the other writes MATCH the
        session and would be silent no-ops without it).

        Inputs are capped silently (100 edited files, 50 commits); the
        returned counts reflect what was written. A commit dict without a
        ``sha`` is skipped and counted in ``skipped_commits``.

        Each write runs in its own transaction, so a failure mid-sequence
        leaves the earlier writes committed. Every write is MERGE-idempotent,
        so the recovery contract is: retry the whole call. On error the
        payload carries the partial progress made before the failure:
        {"error", "edited_files", "commits", "skipped_commits"}.

        Returns JSON: {"session", "edited_files", "commits",
        "skipped_commits"}.
        """
        client = get_client(ctx)
        ts = datetime.now(timezone.utc).isoformat()

        files_written = 0
        commits_written = 0
        commits_skipped = 0

        try:
            files = list(edited_files or [])[:_MAX_EDITED_FILES]
            commit_dicts = list(commits or [])[:_MAX_COMMITS]

            # Session first — later writes MATCH the session node.
            query, params = session_upsert(
                agent_id, session_id, repo, branch, task_key, ts
            )
            await client.graph.execute_write(query, params)

            built = editing_upsert(session_id, repo, files, ts)
            if built is not None:
                query, params = built
                await client.graph.execute_write(query, params)
                files_written = len(files)

            for commit in commit_dicts:
                sha = commit.get("sha")
                if not sha:
                    commits_skipped += 1
                    continue
                query, params = commit_upsert(
                    session_id,
                    repo,
                    sha,
                    commit.get("message", ""),
                    list(commit.get("files") or []),
                    ts,
                )
                await client.graph.execute_write(query, params)
                commits_written += 1

            return json.dumps(
                {
                    "session": session_id,
                    "edited_files": files_written,
                    "commits": commits_written,
                    "skipped_commits": commits_skipped,
                }
            )

        except Exception as e:
            logger.error(f"Error in record_coding_activity: {e}")
            return json.dumps(
                {
                    "error": str(e),
                    "edited_files": files_written,
                    "commits": commits_written,
                    "skipped_commits": commits_skipped,
                }
            )

    @mcp.tool()
    @log_tool_call
    async def capture_session_memory(
        ctx: Context,
        agent_id: str,
        session_id: str,
        repo: str,
        branch: str,
        transcript: str,
        task_key: str | None = None,
        files: list[str] | None = None,
    ) -> str:
        """Extract anchored memories from a session transcript and store them.

        Session-end capture tool — the SessionEnd hook calls this with the
        rendered transcript and the session's git context. The transcript
        goes through ExtractCodingMemory (a BAML/LLM call, so this tool is
        slow); every kept item is written via ``anchored_memory_write``:
        Decision/Gotcha/DeadEnd with their surviving anchor files as ABOUT
        edges and a CONCERNS edge only when the item concerns the task,
        CodingPreference always, session-anchored (MADE_IN) with no file or
        task edges. Stored props follow coding_recall's read contract:
        Decision {text, reason, confidence}, Gotcha {text, confidence},
        DeadEnd {attempt, why_failed, confidence}, CodingPreference
        {category, preference, confidence}.

        Inputs are capped silently: the transcript at 80,000 chars keeping
        the TAIL (recent turns matter most), ``files`` at 100 entries. An
        empty or whitespace transcript returns immediately without calling
        the extractor.

        One timestamp is shared by every write. The session upsert runs
        first — but only when extraction kept at least one item, so an
        empty extraction (or an extraction failure) writes nothing at all.
        Each write is its own transaction; on error the payload carries
        the partial progress: {"error", "stored", "by_kind",
        "dropped_unanchored", "anchor_rate"}. Memory nodes are CREATEd,
        not MERGEd, so the retry contract is NOT idempotent — a retry
        after a partial failure may duplicate already-stored items.

        Returns JSON: {"stored", "by_kind", "dropped_unanchored",
        "anchor_rate"}.
        """
        client = get_client(ctx)

        if not transcript or not transcript.strip():
            return json.dumps(
                {"stored": 0, "dropped_unanchored": 0, "anchor_rate": None}
            )

        ts = datetime.now(timezone.utc).isoformat()
        by_kind = {kind: 0 for kind in _CAPTURE_KINDS}
        dropped_unanchored = 0
        anchor_rate = None

        try:
            extracted = await extract_coding_memory(
                transcript[-_MAX_TRANSCRIPT_CHARS:],
                branch=branch,
                task=task_key,
                files=list(files or [])[:_MAX_ANCHOR_FILES],
            )
            dropped_unanchored = extracted["dropped_unanchored"]
            anchor_rate = extracted["anchor_rate"]

            # (kind, props, anchor_paths, task_key-or-None), in write order.
            # concerns_task/anchor_files become edges, never node props.
            items: list[tuple[str, dict[str, Any], list[str], str | None]] = []
            for d in extracted["decisions"]:
                items.append(
                    (
                        "Decision",
                        {
                            "text": d["text"],
                            "reason": d["reason"],
                            "confidence": d["confidence"],
                        },
                        d["anchor_files"],
                        task_key if d.get("concerns_task") else None,
                    )
                )
            for g in extracted["gotchas"]:
                items.append(
                    (
                        "Gotcha",
                        {"text": g["text"], "confidence": g["confidence"]},
                        g["anchor_files"],
                        task_key if g.get("concerns_task") else None,
                    )
                )
            for de in extracted["dead_ends"]:
                items.append(
                    (
                        "DeadEnd",
                        {
                            "attempt": de["attempt"],
                            "why_failed": de["why_failed"],
                            "confidence": de["confidence"],
                        },
                        de["anchor_files"],
                        task_key if de.get("concerns_task") else None,
                    )
                )
            for p in extracted["preferences"]:
                items.append(
                    (
                        "CodingPreference",
                        {
                            "category": p["category"],
                            "preference": p["preference"],
                            "confidence": p["confidence"],
                        },
                        [],
                        None,
                    )
                )

            if items:
                # Session first — the memory writes MATCH the session node.
                query, params = session_upsert(
                    agent_id, session_id, repo, branch, task_key, ts
                )
                await client.graph.execute_write(query, params)

                for kind, props, anchor_paths, item_task_key in items:
                    query, params = anchored_memory_write(
                        kind, props, session_id, repo, anchor_paths,
                        item_task_key, ts,
                    )
                    await client.graph.execute_write(query, params)
                    by_kind[kind] += 1

            return json.dumps(
                {
                    "stored": sum(by_kind.values()),
                    "by_kind": by_kind,
                    "dropped_unanchored": dropped_unanchored,
                    "anchor_rate": anchor_rate,
                }
            )

        except Exception as e:
            logger.error(f"Error in capture_session_memory: {e}")
            return json.dumps(
                {
                    "error": str(e),
                    "stored": sum(by_kind.values()),
                    "by_kind": by_kind,
                    "dropped_unanchored": dropped_unanchored,
                    "anchor_rate": anchor_rate,
                }
            )

    @mcp.tool()
    @log_tool_call
    async def coding_recall(
        ctx: Context,
        prompt: str,
        agent_id: str,
        repo: str,
        files: list[str] | None = None,
        task_key: str | None = None,
        overlap_window_hours: float = 24.0,
    ) -> str:
        """Recall coding memories anchored to files or a task, plus overlaps.

        Push-model recall tool — the prompt hook calls this before the model
        round trip. Memories (Decision, Gotcha, DeadEnd) are fetched
        anchor-first: reached from the given files (ABOUT) or task
        (CONCERNS), newest first, limit 10. ``prompt`` is accepted for future
        relevance ranking but unused in v1.

        Overlaps report sessions of OTHER agents that touched the same files
        or task within ``overlap_window_hours``.

        When neither ``files`` nor ``task_key`` is given, no queries run and
        ``fallback`` is true — callers should fall back to memory_search.

        Returns JSON: {"memories": [...], "fallback": bool, "overlaps":
        [...]}.
        """
        client = get_client(ctx)
        file_list = list(files or [])
        fallback = not file_list and task_key is None

        try:
            memories: list[dict[str, Any]] = []
            overlaps: list[dict[str, Any]] = []

            if not fallback:
                rows = await client.graph.execute_read(
                    _MEMORIES_QUERY,
                    {"repo": repo, "files": file_list, "task_key": task_key},
                )
                memories = [_render_memory(row) for row in rows]

                overlap_rows = await client.graph.execute_read(
                    _OVERLAPS_QUERY,
                    {
                        "agent_id": agent_id,
                        "repo": repo,
                        "files": file_list,
                        "task_key": task_key,
                        "window": overlap_window_hours,
                    },
                )
                overlaps = [
                    {
                        "agent": row.get("agent"),
                        "files": row.get("files") or [],
                        "task": row.get("task"),
                        "last_seen": row.get("last_seen"),
                    }
                    for row in overlap_rows
                ]

            return json.dumps(
                {"memories": memories, "fallback": fallback, "overlaps": overlaps}
            )

        except Exception as e:
            logger.error(f"Error in coding_recall: {e}")
            return json.dumps({"error": str(e)})

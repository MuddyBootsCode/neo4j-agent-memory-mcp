"""MCP tools for the deterministic coding-memory plane.

Two push-model tools: the recall hook calls them directly before the model
round trip, so both must be cheap and deterministic — no LLM calls here.

- record_coding_activity: write the session/editing/commit facts for a
  coding session via the pure builders in ``capture/cypher.py``.
- coding_recall: anchor-first read of extracted memories (Decision, Gotcha,
  DeadEnd) plus cross-agent overlap detection.

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
    commit_upsert,
    editing_upsert,
    session_upsert,
)
from agent_memory_mcp.mcp._common import get_client
from agent_memory_mcp.mcp._logging import log_tool_call

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Input caps applied silently; the returned counts reflect what was written.
_MAX_EDITED_FILES = 100
_MAX_COMMITS = 50

# Extracted-memory kinds served by coding_recall (CodingPreference is
# deliberately excluded from anchor-first recall in v1).
_RECALL_KINDS = ("Decision", "Gotcha", "DeadEnd")

# Anchor-first memory read. ``$files`` is always a list (possibly empty) and
# ``$task_key`` may be null — a null comparison never matches, so each anchor
# clause degrades to a no-op when its input is absent.
_MEMORIES_QUERY = """
    MATCH (m)
    WHERE (m:Decision OR m:Gotcha OR m:DeadEnd)
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
"""

# Cross-agent overlap read: recent sessions of OTHER agents touching the same
# files (EDITING) or the same task (WORKING_ON) inside the recency window.
# Grouped per (agent, session); `last_seen` is the newest qualifying edge.
_OVERLAPS_QUERY = """
    MATCH (a2:CodeAgent)-[:RUNS]->(s2:CodingSession)
    WHERE a2.id <> $agent_id
    OPTIONAL MATCH (s2)-[e:EDITING]->(f:CodeFile)
    WHERE f.repo = $repo AND f.path IN $files
      AND e.at >= datetime() - duration({hours: $window})
    OPTIONAL MATCH (s2)-[w:WORKING_ON]->(t:WorkTask)
    WHERE t.key = $task_key
      AND w.at >= datetime() - duration({hours: $window})
    WITH a2, s2,
         [p IN collect(DISTINCT f.path) WHERE p IS NOT NULL] AS files,
         max(e.at) AS file_seen,
         max(w.at) AS task_seen,
         count(t) AS task_hits
    WHERE size(files) > 0 OR task_hits > 0
    RETURN a2.id AS agent,
           files,
           CASE WHEN task_hits > 0 THEN $task_key ELSE null END AS task,
           toString(
             CASE
               WHEN file_seen IS NULL THEN task_seen
               WHEN task_seen IS NULL THEN file_seen
               WHEN file_seen >= task_seen THEN file_seen
               ELSE task_seen
             END
           ) AS last_seen
    ORDER BY agent, last_seen DESC
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

        Returns JSON: {"session", "edited_files", "commits",
        "skipped_commits"}.
        """
        client = get_client(ctx)
        ts = datetime.now(timezone.utc).isoformat()

        try:
            files = list(edited_files or [])[:_MAX_EDITED_FILES]
            commit_dicts = list(commits or [])[:_MAX_COMMITS]

            # Session first — later writes MATCH the session node.
            query, params = session_upsert(
                agent_id, session_id, repo, branch, task_key, ts
            )
            await client.graph.execute_write(query, params)

            files_written = 0
            built = editing_upsert(session_id, repo, files, ts)
            if built is not None:
                query, params = built
                await client.graph.execute_write(query, params)
                files_written = len(files)

            commits_written = 0
            commits_skipped = 0
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
            return json.dumps({"error": str(e)})

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

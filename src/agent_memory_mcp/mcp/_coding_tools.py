"""MCP tools for the coding-memory planes.

Three push-model tools. The first two run inside the recall hook before the
model round trip, so their cost is paid on every prompt. record_coding_activity
is deterministic; coding_recall makes ONE batched LLM call to screen what it
retrieved (~1.5s at the default depth of 10), because cosine ranking alone
could not tell relevant lessons from irrelevant ones. The third runs at
session end, where latency is free, and is the one place the extracted plane
gets written.

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
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastmcp import Context

from agent_memory_mcp.capture.cypher import (
    RECALL_KINDS,
    SHARED_RECALL_LABEL,
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
# Caps the same payload as MAX_TRANSCRIPT_CHARS in hook/capture_hook.py —
# one on the receiving side, one on the sending side. Move them together.
_MAX_TRANSCRIPT_CHARS = 80_000
_MAX_ANCHOR_FILES = 100

# Write order for extracted-plane items; also the by_kind key order.
_CAPTURE_KINDS = ("Decision", "Gotcha", "DeadEnd", "CodingPreference")

# Extracted-memory kinds served by coding_recall (CodingPreference is
# deliberately excluded from anchor-first recall in v1).
_RECALL_KINDS = ("Decision", "Gotcha", "DeadEnd")

# Memories returned per recall call. Matches the anchor-first LIMIT it
# replaces, so the hook's context budget is unchanged.
_RECALL_LIMIT = 10

# Candidates fetched for the relevance gate to screen. Measured on 20
# queries at depth 20: cosine ordering barely discriminates (a relevant
# lesson is about as likely at rank 19 as rank 1), so the top-5 cut reached
# only 33% of the relevant lessons the retriever could see. Ranks 1-10 hold
# 55% of them, and a batched gate over 10 candidates costs ~1.5s against
# ~1.0s for 5 and ~4.3s for 20. 10 is the knee of that curve.
GATE_DEPTH = int(os.environ.get("NAM_RECALL_GATE_DEPTH", "10"))
# Set NAM_RECALL_GATE=0 to skip the gate and return the top _RECALL_LIMIT by
# score -- the MUD-401 behaviour, ~33% precision instead of ~69%.
GATE_ENABLED = os.environ.get("NAM_RECALL_GATE", "1") != "0"

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

# --- Hybrid recall (MUD-401) ------------------------------------------
#
# The anchor-first query above is what MUD-395 shipped and MUD-401 measured:
# on a corpus with real file continuity it scored 12% precision against 52%
# for plain embedding search, because sharing a file with a past session says
# the two are NEAR each other, not that the old lesson answers the new
# question. The prompt is the signal, and v1 threw it away.
#
# So: retrieve by prompt similarity, and use the anchor as a boost rather
# than the gate. A lesson anchored to a file the caller is editing outranks
# an equally-similar one that is not, but an unanchored lesson can still be
# returned, and an anchored-but-irrelevant one no longer crowds the list.
CODING_MEMORY_INDEX = "coding_memory_embedding_idx"

# Added to cosine similarity when a candidate is anchored to one of the
# caller's files or their task. Cosine over MiniLM lands most real
# neighbours in 0.3-0.7, so 0.15 reorders within a band without letting a
# weak anchored match jump a strong unanchored one.
ANCHOR_BOOST = 0.15
# Floor for the vector leg. Deliberately below the server's 0.7 default:
# prompts are questions and lessons are statements, so honest matches score
# lower than statement-to-statement pairs would.
HYBRID_THRESHOLD = 0.45
# Vector candidates fetched before boosting and trimming to the caller's
# limit. Oversampling is what gives the boost something to reorder.
HYBRID_CANDIDATES = 40

_HYBRID_QUERY = """
    CALL db.index.vector.queryNodes($index, $candidates, $embedding)
    YIELD node AS m, score
    WHERE score >= $threshold
    MATCH (m)-[:MADE_IN]->(s:CodingSession {repo: $repo})
    WITH DISTINCT m, score
    WITH m, score,
         CASE WHEN EXISTS {
                  MATCH (m)-[:ABOUT]->(f:CodeFile)
                  WHERE f.repo = $repo AND f.path IN $files
              }
              OR EXISTS {
                  MATCH (m)-[:CONCERNS]->(t:WorkTask)
                  WHERE t.key = $task_key
              }
         THEN $anchor_boost ELSE 0.0 END AS boost
    WITH m, score, score + boost AS ranked, boost > 0 AS anchored
    ORDER BY ranked DESC
    LIMIT $limit
    OPTIONAL MATCH (m)-[:ABOUT]->(af:CodeFile)
    OPTIONAL MATCH (m)-[:CONCERNS]->(wt:WorkTask)
    WITH m, score, ranked, anchored,
         [p IN collect(DISTINCT af.path) WHERE p IS NOT NULL] AS files,
         [k IN collect(DISTINCT wt.key) WHERE k IS NOT NULL] AS tasks
    ORDER BY ranked DESC
    RETURN labels(m) AS labels,
           properties(m) AS props,
           files,
           head(tasks) AS task,
           toString(m.created_at) AS at,
           score,
           anchored
"""

_CREATE_INDEX = (
    f"CREATE VECTOR INDEX {CODING_MEMORY_INDEX} IF NOT EXISTS "
    f"FOR (m:{SHARED_RECALL_LABEL}) ON (m.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
    "`vector.similarity_function`: 'cosine'}}"
)


def memory_embedding_text(kind: str, props: dict[str, Any]) -> str:
    """The text embedded for a lesson node.

    DeadEnd carries its meaning across two properties, so both go in; the
    others embed the one field a reader would recognise the lesson by.
    """
    if kind == "DeadEnd":
        return f"{props.get('attempt', '')} — failed: {props.get('why_failed', '')}".strip()
    return str(props.get("text", "")).strip()


def _embedder(client: Any) -> Any:
    """The client's embedder, or None when embeddings are unavailable."""
    return getattr(client.long_term, "_embedder", None)


async def _embed(client: Any, text: str) -> list[float] | None:
    """Embed one string, or return None if that is not possible.

    Never raises: every caller has a working non-embedded path, so an
    embedder that is missing, misconfigured, or failing degrades to the
    anchor-first behaviour instead of failing the write or the read.
    """
    embedder = _embedder(client)
    if embedder is None or not text:
        return None
    try:
        return await embedder.embed(text)
    except Exception as e:  # pragma: no cover - provider-specific failures
        logger.warning(f"embedding failed, falling back to anchor-only: {e}")
        return None


async def ensure_coding_memory_index(client: Any) -> bool:
    """Create the lesson vector index if it does not exist yet.

    Idempotent (``IF NOT EXISTS``) and best-effort: returns False when the
    embedder reports no usable integer dimension count or the DDL fails,
    which sends recall down the anchor-first path rather than erroring.
    """
    embedder = _embedder(client)
    dims = getattr(embedder, "dimensions", None) if embedder else None
    if not isinstance(dims, int) or dims <= 0:
        # Silence here is how a whole run ends up with embedded nodes and no
        # index to find them through. Embedders are created lazily, so this
        # usually means the caller ran before anything was embedded.
        logger.warning(
            f"skipping {CODING_MEMORY_INDEX}: embedder "
            f"{type(embedder).__name__ if embedder else None} reports "
            f"dimensions={dims!r}; embed something first"
        )
        return False
    try:
        # $dims cannot be a query parameter in index DDL, so it is formatted
        # in -- it is an int from the embedder config, never user input.
        await client.graph.execute_write(
            _CREATE_INDEX.replace("$dims", str(dims)), {}
        )
        return True
    except Exception as e:  # pragma: no cover - DDL permissions vary
        logger.warning(f"could not ensure {CODING_MEMORY_INDEX}: {e}")
        return False


def _candidate_block(memories: list[dict[str, Any]]) -> str:
    """Numbered candidate lines for the gate, ids matching list position."""
    return "\n".join(
        f"{i}. [{m.get('kind')}] {m.get('text', '')}"
        for i, m in enumerate(memories)
    )


async def screen_memories(
    prompt: str, memories: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep only the candidates a judge finds genuinely on point.

    Fail-open in the strong sense: any error, timeout, or malformed verdict
    set returns the input untouched. A gate that cannot run must degrade to
    MUD-401 behaviour, never to an empty recall -- the hook injects whatever
    comes back, so an exception here would silently blind the model.

    A verdict set that does not cover every candidate is discarded whole
    rather than treated as "the missing ones are rejects": a truncated or
    confused judge would otherwise look exactly like a decisive one.
    """
    if not memories:
        return memories

    from agent_memory_mcp.baml_client.async_client import b
    from agent_memory_mcp.providers import default_baml_options

    try:
        screen = await b.ScreenRecalledMemories(
            query=prompt,
            candidates=_candidate_block(memories),
            baml_options=default_baml_options(),
        )
        verdicts = {
            v.id: bool(v.keep)
            for v in screen.verdicts
            if 0 <= v.id < len(memories)
        }
    except Exception:
        logger.warning("recall gate failed; returning ungated", exc_info=True)
        return memories

    if len(verdicts) != len(memories):
        logger.warning(
            f"recall gate covered {len(verdicts)}/{len(memories)} candidates; "
            "returning ungated"
        )
        return memories

    kept = [m for i, m in enumerate(memories) if verdicts.get(i)]
    logger.info(f"recall gate kept {len(kept)}/{len(memories)}")
    return kept


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
    rendered = {
        "kind": kind,
        "text": text,
        "files": row.get("files") or [],
        "task": row.get("task"),
        "at": row.get("at"),
    }
    # Only the hybrid query returns these; the anchor-first one has no score.
    if row.get("score") is not None:
        rendered["score"] = round(float(row["score"]), 4)
        rendered["anchored"] = bool(row.get("anchored"))
    return rendered


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

        Every Decision/Gotcha/DeadEnd is embedded before it is written so
        coding_recall can rank it against a prompt; ``embedded`` reports how
        many got a vector. An unavailable embedder is not an error — the
        item is stored without one and is simply unreachable by the vector
        leg of recall.

        Returns JSON: {"stored", "by_kind", "dropped_unanchored",
        "anchor_rate", "embedded"}.
        """
        client = get_client(ctx)

        if not transcript or not transcript.strip():
            return json.dumps(
                {
                    "stored": 0,
                    "by_kind": {kind: 0 for kind in _CAPTURE_KINDS},
                    "dropped_unanchored": 0,
                    "anchor_rate": None,
                }
            )

        ts = datetime.now(timezone.utc).isoformat()
        by_kind = {kind: 0 for kind in _CAPTURE_KINDS}
        dropped_unanchored = 0
        anchor_rate = None
        embedded = 0

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
                # Embed before writing anything: the index only needs to
                # exist if at least one vector is going in, so an absent or
                # broken embedder costs no DDL and no behaviour change.
                embeddings: list[list[float] | None] = []
                for kind, props, _anchor_paths, _task in items:
                    vector = None
                    if kind in RECALL_KINDS:
                        vector = await _embed(
                            client, memory_embedding_text(kind, props)
                        )
                        if vector is not None:
                            embedded += 1
                    embeddings.append(vector)

                # Session first — the memory writes MATCH the session node.
                query, params = session_upsert(
                    agent_id, session_id, repo, branch, task_key, ts
                )
                await client.graph.execute_write(query, params)
                if embedded:
                    await ensure_coding_memory_index(client)

                for (kind, props, anchor_paths, item_task_key), vector in zip(
                    items, embeddings
                ):
                    query, params = anchored_memory_write(
                        kind, props, session_id, repo, anchor_paths,
                        item_task_key, ts, embedding=vector,
                    )
                    await client.graph.execute_write(query, params)
                    by_kind[kind] += 1

            return json.dumps(
                {
                    "stored": sum(by_kind.values()),
                    "by_kind": by_kind,
                    "dropped_unanchored": dropped_unanchored,
                    "anchor_rate": anchor_rate,
                    "embedded": embedded,
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
                    "embedded": embedded,
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
        """Recall coding memories relevant to the prompt, plus overlaps.

        Push-model recall tool — the prompt hook calls this before the model
        round trip. Memories (Decision, Gotcha, DeadEnd) are ranked by
        similarity to ``prompt``, with a boost for candidates anchored to
        ``files`` or ``task_key``, then screened by a relevance judge
        (strategy "hybrid+gate").

        The gate exists because ranking alone does not discriminate:
        measured at depth 20, a relevant lesson was about as likely to land
        at rank 19 as rank 1, so a top-5 cut reached only 33% of the
        relevant lessons the retriever could see. GATE_DEPTH candidates are
        fetched and screened in one batched call; the judge scored 69%
        precision where cosine scored 33%. Set NAM_RECALL_GATE=0 to skip it
        and take the ungated top ``_RECALL_LIMIT`` instead.

        v1 ranked anchor-first and ignored the prompt entirely. Measured on
        a corpus with real file continuity, that scored 12% precision to
        plain embedding search's 52% — sharing a file with a past session
        means the two are near each other, not that the old lesson answers
        the new question. The anchor is now a tiebreaker, not the gate.

        Falls back to the v1 anchor-first read (strategy "anchor") only when
        the hybrid leg cannot run at all — no embedder, no vector index, or
        an empty prompt — never merely because it matched nothing. An
        anchored candidate the prompt does not match is the 88% this change
        exists to stop injecting.

        Overlaps report sessions of OTHER agents that touched the same files
        or task within ``overlap_window_hours``; they are anchor-based and
        unaffected.

        ``fallback`` is true when neither leg could run, and callers should
        fall back to memory_search.

        Returns JSON: {"memories": [...], "fallback": bool, "strategy":
        "hybrid"|"anchor"|null, "overlaps": [...], "timing_ms": {...}}.
        ``timing_ms`` holds per-stage wall time (embed, vector, gate,
        overlaps) so the hook's headline number can be read stage by stage.
        """
        client = get_client(ctx)
        file_list = list(files or [])
        has_anchor = bool(file_list) or task_key is not None
        timing: dict[str, float] = {}

        def _lap(stage: str, started: float) -> None:
            timing[stage] = round((time.perf_counter() - started) * 1000, 1)

        try:
            memories: list[dict[str, Any]] = []
            overlaps: list[dict[str, Any]] = []
            strategy = None

            t0 = time.perf_counter()
            embedding = await _embed(client, (prompt or "").strip())
            _lap("embed", t0)
            if embedding is not None:
                try:
                    t0 = time.perf_counter()
                    rows = await client.graph.execute_read(
                        _HYBRID_QUERY,
                        {
                            "index": CODING_MEMORY_INDEX,
                            "embedding": embedding,
                            "candidates": HYBRID_CANDIDATES,
                            "threshold": HYBRID_THRESHOLD,
                            "anchor_boost": ANCHOR_BOOST,
                            "limit": (
                                max(GATE_DEPTH, _RECALL_LIMIT)
                                if GATE_ENABLED
                                else _RECALL_LIMIT
                            ),
                            "repo": repo,
                            "files": file_list,
                            "task_key": task_key,
                        },
                    )
                    _lap("vector", t0)
                    memories = [_render_memory(row) for row in rows]
                    strategy = "hybrid"
                    if GATE_ENABLED:
                        t0 = time.perf_counter()
                        memories = await screen_memories(prompt, memories)
                        _lap("gate", t0)
                        strategy = "hybrid+gate"
                    memories = memories[:_RECALL_LIMIT]
                except Exception as e:
                    # Most likely the index does not exist yet (nothing has
                    # been captured since the upgrade). Anchor-first still
                    # works on un-embedded nodes, so use it.
                    logger.warning(f"hybrid recall unavailable: {e}")

            if strategy is None and has_anchor:
                rows = await client.graph.execute_read(
                    _MEMORIES_QUERY,
                    {"repo": repo, "files": file_list, "task_key": task_key},
                )
                memories = [_render_memory(row) for row in rows]
                strategy = "anchor"

            fallback = strategy is None

            if has_anchor:
                t0 = time.perf_counter()
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
                _lap("overlaps", t0)
                overlaps = [
                    {
                        "agent": row.get("agent"),
                        "files": row.get("files") or [],
                        "task": row.get("task"),
                        "last_seen": row.get("last_seen"),
                    }
                    for row in overlap_rows
                ]

            logger.info(
                f"coding_recall strategy={strategy} memories={len(memories)} "
                f"timing_ms={timing}"
            )
            return json.dumps(
                {
                    "memories": memories,
                    "fallback": fallback,
                    "strategy": strategy,
                    "overlaps": overlaps,
                    "timing_ms": timing,
                }
            )

        except Exception as e:
            logger.error(f"Error in coding_recall: {e}")
            return json.dumps({"error": str(e)})

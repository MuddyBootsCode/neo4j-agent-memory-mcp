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

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastmcp import Context

from agent_memory_mcp import tracing
from agent_memory_mcp.capture.cypher import (
    expire_write,
    outcome_write,
    reassert_write,
    resolve_write,
    served_lessons_read,
    served_unused_write,
    served_write,
    RECALL_KINDS,
    SHARED_RECALL_LABEL,
    anchored_memory_write,
    commit_upsert,
    editing_upsert,
    session_upsert,
)
from agent_memory_mcp.extraction.coding import (
    UNUSED,
    candidate_line,
    curate_coding_memory,
    extract_coding_memory,
    rate_served_lessons,
)
from agent_memory_mcp.extraction.unified import persist_preferences
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
_MAX_TRANSCRIPT_CHARS = 400_000
_MAX_ANCHOR_FILES = 100
_MAX_ERROR_STEPS = 40

# The transcript is cut into windows (MUD-404) instead of a single 80k tail,
# so a long session's early decisions are not lost. Windows are ranked by
# how much failure/correction evidence they carry and the top
# NAM_CAPTURE_MAX_WINDOWS get an extraction call each; the newest window is
# always included.
WINDOW_CHARS = int(os.environ.get("NAM_CAPTURE_WINDOW_CHARS", "60000"))
MAX_WINDOWS = int(os.environ.get("NAM_CAPTURE_MAX_WINDOWS", "4"))
_SIGNAL_RE = re.compile(
    r"(?i)(\[error from |traceback|\bfailed\b|no, |instead|actually|revert|"
    r"doesn't work|does not work|didn't work|still fails|wrong)"
)
# Existing lessons shown to the curator per candidate: nearest by cosine,
# inside the repo, above this similarity.
NEIGHBOR_LIMIT = 5
NEIGHBOR_THRESHOLD = 0.6

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
# Wall-clock cap on one gate call. BAML's own request timeout is 900 s; a
# hung local model would otherwise hold the hook until Claude Code kills
# it, which reads as an empty recall rather than a slow one (seen in the
# MUD-403/404 golden runs: one query per run stalled for the full 900 s).
GATE_TIMEOUT_S = float(os.environ.get("NAM_RECALL_GATE_TIMEOUT", "6"))

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

# --- Fused recall (MUD-401, MUD-406) --------------------------------------
#
# The anchor-first query above is what MUD-395 shipped and MUD-401 measured:
# on a corpus with real file continuity it scored 12% precision against 52%
# for plain embedding search, because sharing a file with a past session says
# the two are NEAR each other, not that the old lesson answers the new
# question. The prompt is the signal, and v1 threw it away.
#
# MUD-401 ranked by cosine with an anchor boost. The golden set (MUD-403)
# measured that boost at zero effect on P@5 and cosine alone leaving 56-61%
# of relevant lessons outside the top 20. MUD-406 adds a second leg: BM25
# over the lesson text (symptom, text, reason, attempt, why_failed), which
# matches the file paths, env vars and error strings a prompt carries
# verbatim and an embedding blurs. The two legs are fused with reciprocal
# rank fusion; the anchor is reported, not scored.
CODING_MEMORY_INDEX = "coding_memory_embedding_idx"
CODING_MEMORY_TEXT_INDEX = "coding_memory_text_idx"

# Kept for the anchored flag and for experiments that still read it; no
# longer added to any score.
ANCHOR_BOOST = 0.15
# Floor for the vector leg. Deliberately below the server's 0.7 default:
# prompts are questions and lessons are statements, so honest matches score
# lower than statement-to-statement pairs would.
HYBRID_THRESHOLD = 0.45
# Candidates fetched per leg. The vector index spans every repo, so the
# repo filter runs after the index read and the fetch must oversample or a
# busy store starves a quiet repo of candidates.
HYBRID_CANDIDATES = 100
LEG_LIMIT = 40
# RRF constant. 60 is the value from the original paper and what Graphiti
# uses; it keeps a rank-1 hit on one leg from dominating rank-3 on both.
RRF_K = 60

# --- Outcome prior (MUD-407) ------------------------------------------------
#
# A lesson several sessions have asserted, and one later sessions acted on,
# is worth more than a one-off at the same cosine. The fused score is scaled
# by 1 + OUTCOME_WEIGHT*(w - 0.5)*2 + EVIDENCE_WEIGHT*min(evidence, cap)/cap,
# where w is the EMA the session-end pass maintains (0.5 when unrated, so an
# unrated lesson is neither promoted nor buried).
#
# Both terms ship non-zero, because the alternative is Cognee's: influence
# 0.0 that nobody ever turns on. They are not gentle. RRF scores across a
# 20-candidate pool span about a third (1/61 to 1/81), so the default
# swing -- 0.8 for a consistently harmful lesson up to 1.3 for a proven,
# repeatedly asserted one -- can move a lesson roughly ten ranks. That is
# deliberate while cosine ordering is measured not to discriminate (a
# relevant lesson is about as likely at rank 19 as rank 1, MUD-403), and
# it is the first thing to tune on the first golden run whose pool
# actually carries counters. Set both to 0 to restore pure RRF.
OUTCOME_WEIGHT = float(os.environ.get("NAM_RECALL_OUTCOME_WEIGHT", "0.2"))
EVIDENCE_WEIGHT = float(os.environ.get("NAM_RECALL_EVIDENCE_WEIGHT", "0.1"))
EVIDENCE_CAP = 5

_LEG_TAIL = """
    MATCH (m)-[:MADE_IN]->(s:CodingSession {repo: $repo})
    WHERE m.expired_at IS NULL
    WITH DISTINCT m, score
    ORDER BY score DESC
    LIMIT $limit
    WITH m, score,
         EXISTS {
             MATCH (m)-[:ABOUT]->(f:CodeFile)
             WHERE f.repo = $repo AND f.path IN $files
         }
         OR EXISTS {
             MATCH (m)-[:CONCERNS]->(t:WorkTask)
             WHERE t.key = $task_key
         } AS anchored
    OPTIONAL MATCH (m)-[:ABOUT]->(af:CodeFile)
    OPTIONAL MATCH (m)-[:CONCERNS]->(wt:WorkTask)
    WITH m, score, anchored,
         [p IN collect(DISTINCT af.path) WHERE p IS NOT NULL] AS files,
         [k IN collect(DISTINCT wt.key) WHERE k IS NOT NULL] AS tasks
    ORDER BY score DESC
    RETURN elementId(m) AS eid,
           labels(m) AS labels,
           properties(m) AS props,
           files,
           head(tasks) AS task,
           toString(m.created_at) AS at,
           score,
           anchored
"""

_VECTOR_LEG_QUERY = """
    CALL db.index.vector.queryNodes($index, $candidates, $embedding)
    YIELD node AS m, score
    WHERE score >= $threshold
""" + _LEG_TAIL

_FULLTEXT_LEG_QUERY = """
    CALL db.index.fulltext.queryNodes($index, $query, {limit: $candidates})
    YIELD node AS m, score
""" + _LEG_TAIL

# Kept under its old name for the recall sweep and probe, which import it:
# the vector leg alone, ranked by cosine.
_HYBRID_QUERY = _VECTOR_LEG_QUERY

_CREATE_INDEX = (
    f"CREATE VECTOR INDEX {CODING_MEMORY_INDEX} IF NOT EXISTS "
    f"FOR (m:{SHARED_RECALL_LABEL}) ON (m.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
    "`vector.similarity_function`: 'cosine'}}"
)
_CREATE_TEXT_INDEX = (
    f"CREATE FULLTEXT INDEX {CODING_MEMORY_TEXT_INDEX} IF NOT EXISTS "
    f"FOR (m:{SHARED_RECALL_LABEL}) "
    "ON EACH [m.symptom, m.text, m.reason, m.attempt, m.why_failed]"
)

# Lucene query construction for the BM25 leg. Prompts are free text; every
# token becomes an OR term so a path, an env var, or an error fragment in
# the prompt can match on its own. Characters with Lucene meaning are
# dropped rather than escaped -- a prompt is never a query language.
_LUCENE_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*")
_LUCENE_STOP = frozenset(
    "a an and are as at be by can do for from how i if in is it its me my of on or "
    "so that the this to we what when where which why will with you your".split()
)
MAX_LUCENE_TERMS = 64


def lucene_query(prompt: str) -> str:
    """OR-query of the prompt's tokens, safe to hand to queryNodes."""
    terms: list[str] = []
    seen: set[str] = set()
    for tok in _LUCENE_TOKEN_RE.findall(prompt or ""):
        tok = tok.strip("./-")
        low = tok.lower()
        if len(tok) < 2 or low in _LUCENE_STOP or low in seen:
            continue
        seen.add(low)
        # Quote anything with punctuation so "/", "." and "-" stay literal.
        terms.append(f'"{tok}"' if re.search(r"[./-]", tok) else tok)
        if len(terms) >= MAX_LUCENE_TERMS:
            break
    return " ".join(terms)


def memory_embedding_text(kind: str, props: dict[str, Any]) -> str:
    """The text embedded for a lesson node.

    Symptom first where there is one (MUD-404): it is what a later prompt
    will contain — the error, the failing command — while ``text`` is the
    fix, which only matches once you already know it. Decision embeds its
    reason too; the reason carries the discriminating nouns.
    """
    symptom = str(props.get("symptom") or "").strip()
    if kind == "DeadEnd":
        body = f"{props.get('attempt', '')} — failed: {props.get('why_failed', '')}".strip()
    elif kind == "Decision":
        reason = str(props.get("reason") or "").strip()
        body = f"{props.get('text', '')} — {reason}".strip(" —") if reason else str(props.get("text", "")).strip()
    else:
        body = str(props.get("text", "")).strip()
    return f"{symptom} | {body}" if symptom else body


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
    except Exception as e:  # pragma: no cover - DDL permissions vary
        logger.warning(f"could not ensure {CODING_MEMORY_INDEX}: {e}")
        return False
    try:
        await client.graph.execute_write(_CREATE_TEXT_INDEX, {})
    except Exception as e:  # pragma: no cover - DDL permissions vary
        # The BM25 leg is optional: recall degrades to the vector leg.
        logger.warning(f"could not ensure {CODING_MEMORY_TEXT_INDEX}: {e}")
    return True


async def vector_leg(
    client: Any, embedding: list[float], *, repo: str, files: list[str],
    task_key: str | None, limit: int = LEG_LIMIT, threshold: float = HYBRID_THRESHOLD,
) -> list[dict[str, Any]]:
    """Top ``limit`` lessons in ``repo`` by cosine to ``embedding``."""
    return await client.graph.execute_read(
        _VECTOR_LEG_QUERY,
        {"index": CODING_MEMORY_INDEX, "candidates": max(HYBRID_CANDIDATES, limit),
         "embedding": embedding, "threshold": threshold, "limit": limit,
         "repo": repo, "files": files, "task_key": task_key},
    )


async def fulltext_leg(
    client: Any, prompt: str, *, repo: str, files: list[str],
    task_key: str | None, limit: int = LEG_LIMIT,
) -> list[dict[str, Any]]:
    """Top ``limit`` lessons in ``repo`` by BM25 over the lesson text, or []
    when the prompt has no usable terms or the index is missing."""
    query = lucene_query(prompt)
    if not query:
        return []
    try:
        return await client.graph.execute_read(
            _FULLTEXT_LEG_QUERY,
            {"index": CODING_MEMORY_TEXT_INDEX, "query": query,
             "candidates": max(HYBRID_CANDIDATES, limit), "limit": limit,
             "repo": repo, "files": files, "task_key": task_key},
        )
    except Exception as e:
        logger.warning(f"fulltext leg unavailable: {e}")
        return []


def outcome_prior(props: dict[str, Any] | None) -> float:
    """Multiplier for a lesson's fused score from its outcome history.

    1.0 for a lesson with no history, so a fresh store ranks exactly as it
    did before MUD-407. Reads ``outcome_weight`` (the session-end EMA) and
    ``evidence_count`` (how many sessions asserted it); a malformed value is
    treated as absent rather than raising inside the ranker.
    """
    props = props or {}
    prior = 1.0
    try:
        weight = props.get("outcome_weight")
        if weight is not None:
            prior += OUTCOME_WEIGHT * (float(weight) - 0.5) * 2
        evidence = props.get("evidence_count")
        if evidence is not None:
            prior += EVIDENCE_WEIGHT * min(float(evidence), EVIDENCE_CAP) / EVIDENCE_CAP
    except (TypeError, ValueError):
        return 1.0
    return max(prior, 0.0)


def rrf_fuse(legs: list[list[dict[str, Any]]], k: int = RRF_K) -> list[dict[str, Any]]:
    """Reciprocal rank fusion over leg result lists keyed by ``eid``, scaled
    by each lesson's outcome prior (MUD-407).

    Each row's ``score`` becomes the fused score; the per-leg ranks are kept
    under ``ranks`` for diagnostics. Rows missing an ``eid`` (older fakes)
    fall back to their rendered text as the key.
    """
    fused: dict[str, dict[str, Any]] = {}
    for li, leg in enumerate(legs):
        for rank, row in enumerate(leg):
            key = row.get("eid") or json.dumps(row.get("props"), sort_keys=True, default=str)
            entry = fused.get(key)
            if entry is None:
                entry = dict(row)
                entry["leg_scores"] = {}
                entry["ranks"] = {}
                entry["score"] = 0.0
                fused[key] = entry
            entry["score"] += 1.0 / (k + rank + 1)
            entry["ranks"][li] = rank
            entry["leg_scores"][li] = row.get("score")
    for entry in fused.values():
        prior = outcome_prior(entry.get("props"))
        entry["prior"] = round(prior, 4)
        entry["score"] *= prior
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)


def _lesson_dedup_key(row: dict[str, Any]) -> str:
    """Normalized lesson text for duplicate detection: the same string the
    lesson embeds (``memory_embedding_text``), casefolded with whitespace
    collapsed. Empty when the row has no text at all."""
    labels = row.get("labels") or []
    kind = next((label for label in labels if label in _RECALL_KINDS), None)
    text = memory_embedding_text(kind or "", row.get("props") or {})
    return " ".join(text.casefold().split())


def dedupe_fused(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop fused rows whose normalized lesson text duplicates a
    higher-ranked row (MUD-407).

    The same lesson captured in several sessions is several :CodingMemory
    nodes with distinct elementIds, so RRF's per-eid keying serves the
    identical text repeatedly. Keyed on the normalized embedding text, first
    occurrence wins — ``rows`` arrives sorted by fused score, so that is the
    highest-ranked instance. Rows with no text are kept as-is rather than
    collapsed with each other. Pure list filter; empty and single-item
    inputs pass through unchanged.
    """
    if len(rows) < 2:
        return rows
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = _lesson_dedup_key(row)
        if key:
            if key in seen:
                continue
            seen.add(key)
        kept.append(row)
    return kept


async def retrieve_candidates(
    client: Any, *, prompt: str, repo: str, files: list[str],
    task_key: str | None, limit: int, embedding: list[float] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fused candidate rows for ``prompt`` and the strategy that produced
    them: "fused" when both legs ran, "vector" / "fulltext" when only one
    could, None when neither (no embedder and no usable terms)."""
    if embedding is None:
        embedding = await _embed(client, (prompt or "").strip())
    legs: list[list[dict[str, Any]]] = []
    names: list[str] = []
    if embedding is not None:
        legs.append(await vector_leg(client, embedding, repo=repo, files=files, task_key=task_key))
        names.append("vector")
    text_rows = await fulltext_leg(client, prompt, repo=repo, files=files, task_key=task_key)
    if text_rows or not legs:
        if text_rows:
            legs.append(text_rows)
            names.append("fulltext")
    if not legs:
        return [], None
    # Dedup before truncating so freed slots backfill with the next-ranked
    # distinct lessons instead of shrinking the recall.
    fused = dedupe_fused(rrf_fuse(legs))[:limit]
    return fused, ("fused" if len(names) == 2 else names[0])


def _candidate_block(memories: list[dict[str, Any]]) -> str:
    """Numbered candidate lines for the gate, ids matching list position."""
    return "\n".join(
        f"{i}. [{m.get('kind')}] {m.get('text', '')}"
        for i, m in enumerate(memories)
    )


async def screen_memories(
    prompt: str, memories: list[dict[str, Any]],
    trace_meta: dict[str, Any] | None = None,
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
    from agent_memory_mcp.providers import gate_baml_options

    candidate_block = _candidate_block(memories)
    started = time.perf_counter()
    try:
        screen = await asyncio.wait_for(
            b.ScreenRecalledMemories(
                query=prompt,
                candidates=candidate_block,
                baml_options=gate_baml_options(),
            ),
            timeout=GATE_TIMEOUT_S,
        )
        verdicts = {
            v.id: bool(v.keep)
            for v in screen.verdicts
            if 0 <= v.id < len(memories)
        }
        verdict_log = [
            {"id": v.id, "keep": bool(v.keep), "reason": getattr(v, "reason", None)}
            for v in screen.verdicts
        ]
    except asyncio.TimeoutError:
        logger.warning(f"recall gate exceeded {GATE_TIMEOUT_S}s; returning ungated")
        return memories
    except Exception:
        logger.warning("recall gate failed; returning ungated", exc_info=True)
        return memories

    # One trace per judge call (MUD-427), covered or not — the coverage
    # discard is itself worth observing. Silent no-op without OPIK_API_KEY.
    tracing.emit_trace(
        "recall-gate",
        input={"query": prompt, "candidates": candidate_block},
        output={"verdicts": verdict_log},
        metadata={
            "model": tracing.model_tag(gate=True),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "kept": sum(1 for i in range(len(memories)) if verdicts.get(i)),
            "of": len(memories),
            **(trace_meta or {}),
        },
    )

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
    # Only the ranked queries return these; the anchor-first one has no score.
    if row.get("score") is not None:
        rendered["score"] = round(float(row["score"]), 4)
        rendered["anchored"] = bool(row.get("anchored"))
    if row.get("ranks"):
        rendered["ranks"] = {str(k): v for k, v in row["ranks"].items()}
    if row.get("eid"):
        rendered["eid"] = row["eid"]
    for key in ("evidence_count", "served_count", "helpful", "harmful"):
        if props.get(key) is not None:
            rendered[key] = props[key]
    return rendered


def transcript_windows(transcript: str, size: int = WINDOW_CHARS, max_windows: int = MAX_WINDOWS) -> list[str]:
    """Cut a rendering into line-aligned windows and pick the ones worth an
    extraction call: the newest always, then the highest-signal others."""
    lines = transcript.splitlines()
    windows: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        if current and length + len(line) + 1 > size:
            windows.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        windows.append("\n".join(current))
    if len(windows) <= max_windows:
        return windows
    newest = len(windows) - 1
    scored = sorted(
        (i for i in range(newest)),
        key=lambda i: len(_SIGNAL_RE.findall(windows[i])),
        reverse=True,
    )
    chosen = sorted(set(scored[: max_windows - 1]) | {newest})
    return [windows[i] for i in chosen]


def error_step_candidates(steps: list[dict], task_key: str | None) -> list[dict[str, Any]]:
    """DeadEnd candidates from errored tool steps, no LLM involved. The
    curator decides whether each is a durable lesson or a one-off."""
    out: list[dict[str, Any]] = []
    for step in steps[:_MAX_ERROR_STEPS]:
        error = str(step.get("error") or "").strip()
        if not error:
            continue
        tool = str(step.get("tool") or "tool")
        attempt = f"{tool}: {step.get('input') or ''}".strip(": ")
        file = step.get("file")
        out.append({
            "kind": "DeadEnd",
            "symptom": error[:300],
            "attempt": attempt[:200],
            "why_failed": error[:300],
            "anchor_files": [file] if isinstance(file, str) and file else [],
            "concerns_task": task_key is not None,
            "confidence": 0.5,
            "source": "error_step",
        })
    return out


_NEIGHBORS_QUERY = """
    CALL db.index.vector.queryNodes($index, $limit, $embedding)
    YIELD node AS m, score
    WHERE score >= $threshold AND m.expired_at IS NULL
    MATCH (m)-[:MADE_IN]->(:CodingSession {repo: $repo})
    RETURN elementId(m) AS eid, labels(m) AS labels, properties(m) AS props, score
    ORDER BY score DESC
"""


async def _neighbors(client: Any, repo: str, embedding: list[float] | None) -> list[tuple[str, str]]:
    """``(elementId, rendered line)`` of the nearest live stored lessons, or
    [] when the index is missing or the lookup fails (a fresh store has no
    index yet)."""
    if embedding is None:
        return []
    try:
        rows = await client.graph.execute_read(
            _NEIGHBORS_QUERY,
            {"index": CODING_MEMORY_INDEX, "limit": NEIGHBOR_LIMIT,
             "embedding": embedding, "threshold": NEIGHBOR_THRESHOLD, "repo": repo},
        )
    except Exception as e:
        logger.debug(f"neighbor lookup skipped: {e}")
        return []
    out = []
    for row in rows:
        kind = next((label for label in (row.get("labels") or []) if label in _RECALL_KINDS), None)
        props = {k: v for k, v in (row.get("props") or {}).items() if k != "embedding"}
        if kind and row.get("eid"):
            out.append((row["eid"], candidate_line({"kind": kind, **props})))
    return out


def _candidates_from(extracted: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for d in extracted["decisions"]:
        items.append({"kind": "Decision", **d})
    for g in extracted["gotchas"]:
        items.append({"kind": "Gotcha", **g})
    for de in extracted["dead_ends"]:
        items.append({"kind": "DeadEnd", **de})
    for p in extracted["preferences"]:
        items.append({"kind": "CodingPreference", **p})
    return items


def _node_props(item: dict[str, Any]) -> dict[str, Any]:
    kind = item["kind"]
    if kind == "Decision":
        return {"text": item["text"], "reason": item["reason"], "confidence": item["confidence"]}
    if kind == "Gotcha":
        props = {"text": item["text"], "confidence": item["confidence"]}
    elif kind == "DeadEnd":
        props = {"attempt": item["attempt"], "why_failed": item["why_failed"], "confidence": item["confidence"]}
    else:
        return {"category": item["category"], "preference": item["preference"], "confidence": item["confidence"]}
    if item.get("symptom"):
        props["symptom"] = item["symptom"]
    return props


async def capture_transcript(
    client: Any,
    *,
    transcript: str,
    agent_id: str,
    session_id: str,
    repo: str,
    branch: str,
    task_key: str | None,
    files: list[str],
    error_steps: list[dict] | None = None,
    ts: str | None = None,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The capture pipeline behind ``capture_session_memory``, callable
    directly (the golden harness builds its pool through this so it
    measures exactly what production writes).

    Per window: extract → add error-step candidates (first window only) →
    embed → nearest stored lessons → curate → write survivors. The session
    upsert runs once, before the first write, and only if something is
    written. Returns the tool's result dict.
    """
    by_kind = {kind: 0 for kind in _CAPTURE_KINDS}
    # ``progress`` (if given) is updated in place so a caller that catches
    # an exception mid-run still sees the counts written so far.
    result: dict[str, Any] = progress if progress is not None else {}
    result.update({
        "stored": 0, "by_kind": by_kind, "dropped_unanchored": 0,
        "anchor_rate": None, "embedded": 0, "windows": 0,
        "curated": {"write": 0, "already_known": 0, "supersedes": 0, "not_durable": 0, "unsupported": 0},
        "reasserted": 0, "superseded": 0, "preferences": 0,
        "rated": {"served": 0, "helpful": 0, "harmful": 0, "unused": 0},
        "rated_windows": 0,
    })
    if not transcript or not transcript.strip():
        return result

    ts = ts or datetime.now(timezone.utc).isoformat()
    files = list(files)[:_MAX_ANCHOR_FILES]
    windows = transcript_windows(transcript[-_MAX_TRANSCRIPT_CHARS:])
    result["windows"] = len(windows)
    anchor_rates: list[float] = []
    session_written = False
    index_ensured = False

    trace_meta = {"session_id": session_id, "repo": repo, "task_key": task_key}
    for wi, window in enumerate(windows):
        extracted = await extract_coding_memory(
            window, branch=branch, task=task_key, files=files, trace_meta=trace_meta
        )
        result["dropped_unanchored"] += extracted["dropped_unanchored"]
        if extracted["anchor_rate"] is not None:
            anchor_rates.append(extracted["anchor_rate"])
        candidates = _candidates_from(extracted)
        if wi == 0 and error_steps:
            candidates += error_step_candidates(error_steps, task_key)
        if not candidates:
            continue

        # Embed before curating: the embedding finds the dedup context and,
        # for survivors, is what gets stored. ``existing`` is numbered by
        # position so the curator's known_as maps back to an elementId.
        embeddings: list[list[float] | None] = []
        existing_lines: list[str] = []
        existing_eids: list[str] = []
        seen_eids: set[str] = set()
        for c in candidates:
            vector = None
            if c["kind"] in RECALL_KINDS:
                vector = await _embed(client, memory_embedding_text(c["kind"], _node_props(c)))
                for eid, line in await _neighbors(client, repo, vector):
                    if eid not in seen_eids:
                        seen_eids.add(eid)
                        existing_eids.append(eid)
                        existing_lines.append(line)
            embeddings.append(vector)

        curated = await curate_coding_memory(
            candidates, window, existing_lines, trace_meta=trace_meta
        )
        for key, n in curated["counts"].items():
            result["curated"][key] = result["curated"].get(key, 0) + n
        kept_by_id = {id(c): c for c in curated["kept"]}
        vectors_by_id = {id(c): v for c, v in zip(candidates, embeddings)}

        async def _ensure_session() -> None:
            nonlocal session_written
            if not session_written:
                q, p = session_upsert(agent_id, session_id, repo, branch, task_key, ts)
                await client.graph.execute_write(q, p)
                session_written = True

        # ALREADY_KNOWN: the stored lesson gains evidence instead of a twin.
        for _c, known in curated["known"]:
            if known is None or known >= len(existing_eids):
                continue
            await _ensure_session()
            q, p = reassert_write(existing_eids[known], session_id, ts)
            await client.graph.execute_write(q, p)
            result["reasserted"] += 1

        for c in candidates:
            kept = kept_by_id.get(id(c))
            if kept is None:
                continue
            if c["kind"] == "CodingPreference":
                # Preferences go through the upstream store (embedded,
                # served by memory_search); CodingPreference nodes were
                # write-only (MUD-405).
                result["preferences"] += await persist_preferences(client, [c])
                continue
            vector = vectors_by_id.get(id(c))
            await _ensure_session()
            if vector is not None and not index_ensured:
                await ensure_coding_memory_index(client)
                index_ensured = True
            if vector is not None:
                result["embedded"] += 1
            q, p = anchored_memory_write(
                c["kind"], _node_props(c), session_id, repo,
                list(c.get("anchor_files") or []),
                task_key if c.get("concerns_task") else None,
                ts, embedding=vector,
            )
            rows = await client.graph.execute_write(q, p)
            by_kind[c["kind"]] += 1
            result["stored"] += 1
            new_eid = rows[0].get("eid") if rows and isinstance(rows[0], dict) else None
            old = kept.get("supersedes")
            if old is not None and old < len(existing_eids):
                q, p = expire_write(existing_eids[old], new_eid, ts)
                await client.graph.execute_write(q, p)
                result["superseded"] += 1

    result["stored"] = sum(by_kind.values())
    result["anchor_rate"] = (sum(anchor_rates) / len(anchor_rates)) if anchor_rates else None
    rated = await rate_session_outcomes(
        client, session_id=session_id, transcript=transcript, ts=ts
    )
    result["rated_windows"] = rated.pop("windows", 0)
    result["rated"] = rated
    # Latency is free at session end; a bounded flush here means capture-side
    # traces survive even an unclean server shutdown. No-op without a key.
    tracing.flush()
    return result


def window_after(transcript: str, prompt: str | None, size: int = WINDOW_CHARS) -> str:
    """The transcript from ``prompt`` onward, ``size`` chars at most; the
    tail when there is no prompt or it cannot be found (a serving from a
    resumed session whose earlier turns are not in this transcript).

    Whitespace is collapsed on both sides before matching because the
    served prompt is stored collapsed and the rendered transcript joins
    text blocks with single spaces. The last occurrence wins: a prompt
    submitted twice is rated on its latest run, which is the one the edge
    timestamp refers to."""
    if not prompt:
        return transcript[-size:]
    needle = " ".join(prompt.split())[:60]
    hay = " ".join(transcript.split())
    idx = hay.rfind(needle) if needle else -1
    if idx < 0:
        return transcript[-size:]
    # Include the "user: " prefix when the prompt starts a rendered line.
    prefix = "user: "
    if hay[max(0, idx - len(prefix)):idx] == prefix:
        idx -= len(prefix)
    return hay[idx:idx + size]


def _max_outcome_windows() -> int:
    try:
        return max(1, int(os.environ.get("NAM_OUTCOME_MAX_WINDOWS", "3")))
    except ValueError:
        return 3


async def rate_session_outcomes(
    client: Any, *, session_id: str, transcript: str, ts: str
) -> dict[str, int]:
    """Session-end outcome pass (MUD-407): rate the lessons recall served to
    this session and move their counters.

    Runs whether or not the session produced new lessons — a session that
    learned nothing can still have acted on what it was given. Servings are
    grouped by the prompt they were served for and each group is rated
    against the transcript window that followed that prompt, newest group
    first, at most ``NAM_OUTCOME_MAX_WINDOWS`` (default 3) rater calls per
    capture. Groups past the cap stay unrated and are read again by the
    next capture of this session. A serving whose prompt is not in the
    transcript is rated against the tail.

    UNUSED verdicts stamp the edge without touching the lesson: the
    serving is judged and done with. Unrated servings (rater failed or
    skipped the id) are left for the next capture.

    Best-effort. Any failure returns the counts so far and leaves the store
    otherwise untouched; capture has already committed its writes by this
    point and must not be undone by a rater problem.
    """
    counts = {"served": 0, "helpful": 0, "harmful": 0, "unused": 0}
    try:
        rows = await client.graph.execute_read(*served_lessons_read(session_id))
    except Exception as e:
        logger.debug(f"served lookup skipped: {e}")
        return counts
    # Rows arrive newest first; dict insertion order keeps the groups so.
    groups: dict[str | None, list[tuple[str, str]]] = {}
    for row in rows or []:
        kind = next((label for label in (row.get("labels") or []) if label in _RECALL_KINDS), None)
        props = {k: v for k, v in (row.get("props") or {}).items() if k != "embedding"}
        if kind and row.get("eid"):
            prompt = row.get("prompt") or None
            groups.setdefault(prompt, []).append((row["eid"], candidate_line({"kind": kind, **props})))
    counts["served"] = sum(len(g) for g in groups.values())
    if not groups:
        return counts

    for wi, (prompt, served) in enumerate(groups.items()):
        if wi >= _max_outcome_windows():
            break
        try:
            verdicts = await rate_served_lessons(
                [line for _, line in served], window_after(transcript, prompt),
                trace_meta={"session_id": session_id, "window": wi, "prompt": (prompt or "")[:80]},
            )
            ratings = [
                {"eid": eid, "helpful": bool(verdict), "reason": reason}
                for (eid, _line), (verdict, reason) in zip(served, verdicts)
                if isinstance(verdict, bool)
            ]
            unused = [eid for (eid, _line), (verdict, _r) in zip(served, verdicts) if verdict == UNUSED]
            built = outcome_write(ratings, session_id, ts)
            if built is not None:
                await client.graph.execute_write(*built)
            stamp = served_unused_write(unused, session_id, ts)
            if stamp is not None:
                await client.graph.execute_write(*stamp)
        except Exception as e:
            logger.warning(f"outcome rating failed: {e}")
            return counts
        counts["helpful"] += sum(1 for r in ratings if r["helpful"])
        counts["harmful"] += sum(1 for r in ratings if not r["helpful"])
        counts["unused"] += len(unused)
        counts["windows"] = wi + 1
    return counts


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

            resolved = 0
            if commits_written:
                # A lesson served to this session whose file a later commit
                # touched is, as far as the graph can tell, acted on.
                rows = await client.graph.execute_write(*resolve_write(session_id))
                if rows and isinstance(rows[0], dict):
                    resolved = int(rows[0].get("resolved") or 0)

            return json.dumps(
                {
                    "session": session_id,
                    "edited_files": files_written,
                    "commits": commits_written,
                    "skipped_commits": commits_skipped,
                    "resolved": resolved,
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
        error_steps: list[dict] | None = None,
    ) -> str:
        """Extract anchored memories from a session transcript and store them.

        Session-end capture tool — the SessionEnd hook calls this with the
        rendered transcript (tool output included) and the session's git
        context. The work is :func:`capture_transcript`: the transcript is
        cut into windows, each window goes through ExtractCodingMemory,
        errored tool steps become DeadEnd candidates for free, every
        candidate is embedded and its nearest stored lessons fetched, and
        one CurateCodingMemory call per window decides WRITE /
        ALREADY_KNOWN / NOT_DURABLE / UNSUPPORTED. Survivors are written
        via ``anchored_memory_write``: Decision/Gotcha/DeadEnd with their
        anchor files as ABOUT edges and a CONCERNS edge when they concern
        the task, CodingPreference session-anchored only. Stored props:
        Decision {text, reason, confidence}, Gotcha {symptom?, text,
        confidence}, DeadEnd {symptom?, attempt, why_failed, confidence},
        CodingPreference {category, preference, confidence}.

        Inputs are capped silently: transcript 400,000 chars (tail),
        ``files`` 100, ``error_steps`` 40. Each write is its own
        transaction; nodes are CREATEd, so a retry after a partial failure
        may duplicate items the curator did not yet know about.

        Returns JSON: {"stored", "by_kind", "dropped_unanchored",
        "anchor_rate", "embedded", "windows", "curated"}.
        """
        client = get_client(ctx)
        progress: dict[str, Any] = {}
        try:
            result = await capture_transcript(
                client,
                transcript=transcript,
                agent_id=agent_id,
                session_id=session_id,
                repo=repo,
                branch=branch,
                task_key=task_key,
                files=list(files or []),
                error_steps=list(error_steps or []),
                progress=progress,
            )
            return json.dumps(result)
        except Exception as e:
            # The payload carries the partial progress made before the
            # failure, so a caller can see what landed.
            logger.error(f"Error in capture_session_memory: {e}")
            return json.dumps({"error": str(e), **progress})

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
        session_id: str | None = None,
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
            try:
                t0 = time.perf_counter()
                rows, strategy = await retrieve_candidates(
                    client, prompt=prompt, repo=repo, files=file_list,
                    task_key=task_key, embedding=embedding,
                    limit=max(GATE_DEPTH, _RECALL_LIMIT) if GATE_ENABLED else _RECALL_LIMIT,
                )
                _lap("vector", t0)
                if strategy is not None:
                    memories = [_render_memory(row) for row in rows]
                    if GATE_ENABLED:
                        t0 = time.perf_counter()
                        memories = await screen_memories(
                            prompt, memories,
                            trace_meta={
                                "session_id": session_id or agent_id,
                                "repo": repo, "task_key": task_key,
                            },
                        )
                        _lap("gate", t0)
                        strategy = f"{strategy}+gate"
                    memories = memories[:_RECALL_LIMIT]
                    # What was injected, so a later commit on the lesson's
                    # file can close the loop (MUD-405). Best-effort.
                    served = [m.pop("eid") for m in memories if m.get("eid")]
                    built = served_write(
                        served, session_id or agent_id,
                        datetime.now(timezone.utc).isoformat(), repo,
                        prompt=prompt,
                    )
                    if built is not None:
                        try:
                            await client.graph.execute_write(*built)
                        except Exception as e:
                            logger.warning(f"served write failed: {e}")
            except Exception as e:
                # Most likely the index does not exist yet (nothing has
                # been captured since the upgrade). Anchor-first still
                # works on un-embedded nodes, so use it.
                logger.warning(f"ranked recall unavailable: {e}")
                strategy = None

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

"""Coding-session memory extraction with code-enforced anchoring.

Coding memory has two planes: a deterministic plane (branches, commits, files
touched) written directly by git and hooks, and an anchored plane — decisions,
gotchas, dead ends — that this module extracts from the session transcript via
BAML. Every anchored item must point at files the session actually touched or
at the stated task. The prompt asks the model for that, but the rule is
enforced here in code: invented anchor paths are removed, and items left with
no anchor are dropped and counted in ``dropped_unanchored``.

Two functions, two calls per transcript window:

- :func:`extract_coding_memory` — ``ExtractCodingMemory`` plus anchor
  sanitisation. Returns candidates.
- :func:`curate_coding_memory` — ``CurateCodingMemory`` (MUD-404), one
  batched call that sees every candidate, the nearest lessons already
  stored, and the window, and returns WRITE / ALREADY_KNOWN / NOT_DURABLE /
  UNSUPPORTED per candidate. It replaces the per-item judge of MUD-397,
  which re-sent the whole transcript once per candidate, and it is also
  the write-time dedup gate. ``NAM_CAPTURE_JUDGE=off`` skips it.

Both are fail-open where a model failure would otherwise destroy capture:
a curator call that raises or returns an incomplete verdict set keeps every
candidate and logs a warning.
"""

from __future__ import annotations

import logging
import os
import posixpath
import time
from typing import Any

from agent_memory_mcp import tracing

from .unified import _clamp

logger = logging.getLogger(__name__)

_JUDGE_KILL_SWITCH_ENV = "NAM_CAPTURE_JUDGE"
# Below this share of candidates covered by verdicts the curator is treated
# as broken and everything is kept (fail-open). At or above it, the
# verdicts given are applied and the few uncovered candidates are handled
# individually — a local model that drops one id out of twenty must not
# turn into twenty writes.
MIN_VERDICT_COVERAGE = 0.8

# Candidate kinds in the order the curator sees them and the tool writes them.
CANDIDATE_KINDS = ("Decision", "Gotcha", "DeadEnd", "CodingPreference")


def _curator_enabled() -> bool:
    return os.environ.get(_JUDGE_KILL_SWITCH_ENV, "").strip().lower() != "off"


def normalize_path(path: Any) -> str:
    """Canonical form for anchor comparison: stripped, ``./`` and ``..``
    segments collapsed, forward slashes. Empty string for non-paths."""
    if not isinstance(path, str):
        return ""
    p = path.strip().replace("\\", "/")
    if not p:
        return ""
    norm = posixpath.normpath(p)
    return "" if norm == "." else norm


def candidate_line(item: dict[str, Any]) -> str:
    """One-line rendering of a candidate, the same text the curator and the
    embedder see so a verdict is about exactly what gets stored."""
    kind = item.get("kind")
    if kind == "Decision":
        return f"[Decision] {item['text']} — {item.get('reason', '')}".rstrip(" —")
    if kind == "Gotcha":
        sym = f"symptom: {item['symptom']} | " if item.get("symptom") else ""
        return f"[Gotcha] {sym}{item['text']}"
    if kind == "DeadEnd":
        sym = f"symptom: {item['symptom']} | " if item.get("symptom") else ""
        return f"[DeadEnd] {sym}{item['attempt']} — failed: {item['why_failed']}"
    return f"[Preference] {item.get('category', '')}: {item.get('preference', '')}"


def _empty_result() -> dict[str, Any]:
    return {
        "decisions": [],
        "gotchas": [],
        "dead_ends": [],
        "preferences": [],
        "anchor_rate": None,
        "dropped_unanchored": 0,
    }


async def extract_coding_memory(
    transcript: str,
    *,
    branch: str,
    task: str | None,
    files: list[str],
    trace_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the coding-memory BAML extraction over a session transcript.

    Returns a dict with ``decisions``, ``gotchas``, ``dead_ends`` and
    ``preferences`` lists plus anchoring metrics. Each item's
    ``anchor_files`` is reduced to its intersection with ``files`` (compared
    after :func:`normalize_path`, reported in the caller's spelling, model
    order preserved, duplicates removed); a decision/gotcha/dead-end left
    with no anchor files and a falsy ``concerns_task`` is dropped and
    counted in ``dropped_unanchored``. When ``task`` is None,
    ``concerns_task`` cannot rescue an item. ``anchor_rate`` is
    kept/(kept+dropped) over the three anchored types, or None when none
    were extracted. Preferences are session-anchored by definition and never
    affect the rate.

    No screening happens here; see :func:`curate_coding_memory`.
    """
    if not transcript or not transcript.strip():
        return _empty_result()

    import agent_memory_mcp.baml_client.async_client as _async_client
    from agent_memory_mcp.baml_client.types import CodingSessionContext
    from agent_memory_mcp.providers import default_baml_options

    context = CodingSessionContext(branch=branch, task=task, files=files)
    started = time.perf_counter()
    result = await _async_client.b.ExtractCodingMemory(
        transcript=transcript,
        context=context,
        baml_options=default_baml_options(),
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    # Normalised form -> the caller's spelling, so stored anchors match the
    # CodeFile nodes the deterministic plane already wrote.
    allowed: dict[str, str] = {}
    for f in files:
        norm = normalize_path(f)
        if norm:
            allowed.setdefault(norm, f.strip())
    kept_anchored = 0
    dropped_unanchored = 0

    def _sanitize_anchors(anchor_files: list[str]) -> list[str]:
        seen: set[str] = set()
        sanitized: list[str] = []
        for raw in anchor_files:
            norm = normalize_path(raw)
            if norm and norm in allowed and norm not in seen:
                seen.add(norm)
                sanitized.append(allowed[norm])
        return sanitized

    def _admit(anchor_files: list[str], concerns_task: bool) -> bool:
        """Count and admit one anchored-type item; False means drop it."""
        nonlocal kept_anchored, dropped_unanchored
        if anchor_files or (concerns_task and task is not None):
            kept_anchored += 1
            return True
        dropped_unanchored += 1
        return False

    def _symptom(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    decisions: list[dict[str, Any]] = []
    for d in result.decisions:
        anchors = _sanitize_anchors(d.anchor_files)
        if _admit(anchors, d.concerns_task):
            decisions.append(
                {
                    "text": d.text,
                    "reason": d.reason,
                    "anchor_files": anchors,
                    "concerns_task": bool(d.concerns_task) and task is not None,
                    "confidence": _clamp(d.confidence, 0.7),
                }
            )

    gotchas: list[dict[str, Any]] = []
    for g in result.gotchas:
        anchors = _sanitize_anchors(g.anchor_files)
        if _admit(anchors, g.concerns_task):
            gotchas.append(
                {
                    "symptom": _symptom(getattr(g, "symptom", None)),
                    "text": g.text,
                    "anchor_files": anchors,
                    "concerns_task": bool(g.concerns_task) and task is not None,
                    "confidence": _clamp(g.confidence, 0.7),
                }
            )

    dead_ends: list[dict[str, Any]] = []
    for de in result.dead_ends:
        anchors = _sanitize_anchors(de.anchor_files)
        if _admit(anchors, de.concerns_task):
            dead_ends.append(
                {
                    "symptom": _symptom(getattr(de, "symptom", None)),
                    "attempt": de.attempt,
                    "why_failed": de.why_failed,
                    "anchor_files": anchors,
                    "concerns_task": bool(de.concerns_task) and task is not None,
                    "confidence": _clamp(de.confidence, 0.7),
                }
            )

    preferences = [
        {
            "category": p.category,
            "preference": p.preference,
            "confidence": _clamp(p.confidence, 0.7),
        }
        for p in result.preferences
    ]

    total_anchored = kept_anchored + dropped_unanchored
    anchor_rate = kept_anchored / total_anchored if total_anchored else None

    logger.debug(
        "ExtractCodingMemory: %d decisions, %d gotchas, %d dead ends, "
        "%d preferences, anchor_rate=%s, dropped_unanchored=%d",
        len(decisions), len(gotchas), len(dead_ends),
        len(preferences), anchor_rate, dropped_unanchored,
    )
    out = {
        "decisions": decisions,
        "gotchas": gotchas,
        "dead_ends": dead_ends,
        "preferences": preferences,
        "anchor_rate": anchor_rate,
        "dropped_unanchored": dropped_unanchored,
    }
    tracing.emit_trace(
        "extraction",
        input={
            "transcript": tracing.truncate_transcript(transcript),
            "branch": branch,
            "task": task,
            "files": list(files),
        },
        output=out,
        metadata={
            "model": tracing.model_tag(),
            "elapsed_ms": elapsed_ms,
            "extracted": len(decisions) + len(gotchas) + len(dead_ends) + len(preferences),
            "dropped_unanchored": dropped_unanchored,
            **(trace_meta or {}),
        },
    )
    return out


def _verdict_counts() -> dict[str, int]:
    return {"write": 0, "already_known": 0, "supersedes": 0, "not_durable": 0, "unsupported": 0}


async def curate_coding_memory(
    candidates: list[dict[str, Any]],
    transcript: str,
    existing: list[str],
    trace_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Screen candidates in one batched call; return what to write.

    ``candidates`` are dicts with a ``kind`` (see :data:`CANDIDATE_KINDS`)
    plus that kind's fields. ``existing`` are rendered lines of lessons
    already stored near these candidates (the dedup context), shown to the
    model numbered by list position. Returns ``{"kept": [...], "counts":
    {write, already_known, supersedes, not_durable, unsupported},
    "known": [...]}``: ``kept`` are candidates to write (WRITE and
    SUPERSEDES, the latter carrying ``supersedes`` = index into
    ``existing``); ``known`` are ``(candidate, index into existing)`` pairs
    the model matched to a stored lesson (MUD-405 turns those into
    evidence).

    Fail-open in the strong sense: the curator disabled, raising, or
    returning a verdict set that does not cover every candidate keeps all
    of them — a truncated or confused model must not look like a decisive
    one, and a broken curator must degrade to "no screening", never to
    "capture nothing".
    """
    counts = _verdict_counts()
    if not candidates:
        return {"kept": [], "counts": counts, "known": []}
    if not _curator_enabled():
        counts["write"] = len(candidates)
        return {"kept": list(candidates), "counts": counts, "known": []}

    import agent_memory_mcp.baml_client.async_client as _async_client
    from agent_memory_mcp.providers import default_baml_options

    block = "\n".join(f"{i}. {candidate_line(c)}" for i, c in enumerate(candidates))
    existing_block = "\n".join(f"{i}. {line}" for i, line in enumerate(existing)) or "(none)"
    started = time.perf_counter()
    try:
        curated = await _async_client.b.CurateCodingMemory(
            candidates=block,
            existing=existing_block,
            transcript=transcript,
            baml_options=default_baml_options(),
        )
        verdicts: dict[int, tuple[str, int | None]] = {}
        verdict_log: list[dict[str, Any]] = []
        for v in curated.verdicts:
            if not (isinstance(v.id, int) and 0 <= v.id < len(candidates)):
                continue
            action = str(getattr(v.action, "value", v.action)).lower()
            known = getattr(v, "known_as", None)
            if not (isinstance(known, int) and 0 <= known < len(existing)):
                known = None
            # A match verdict without a valid target is just a WRITE or a
            # reject: never merge into a lesson the model did not name.
            if action == "already_known" and known is None:
                action = "write"
            if action == "supersedes" and known is None:
                action = "write"
            verdicts[v.id] = (action, known)
            verdict_log.append({
                "id": v.id, "action": action, "known_as": known,
                "reason": getattr(v, "reason", None),
            })
    except Exception:
        logger.warning("CurateCodingMemory failed; keeping all candidates", exc_info=True)
        counts["write"] = len(candidates)
        return {"kept": list(candidates), "counts": counts, "known": []}

    tracing.emit_trace(
        "curator",
        input={
            "candidates": block,
            "existing": existing_block,
            "transcript": tracing.truncate_transcript(transcript),
        },
        output={"verdicts": verdict_log},
        metadata={
            "model": tracing.model_tag(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "kept": sum(1 for a, _ in verdicts.values() if a in ("write", "supersedes")),
            "of": len(candidates),
            **(trace_meta or {}),
        },
    )

    coverage = len(verdicts) / len(candidates)
    if coverage < MIN_VERDICT_COVERAGE:
        logger.warning(
            "CurateCodingMemory covered %d/%d candidates; keeping all",
            len(verdicts), len(candidates),
        )
        counts["write"] = len(candidates)
        return {"kept": list(candidates), "counts": counts, "known": []}

    # A nearly complete verdict set is applied as given. Candidates the
    # model skipped are kept only when they came from the extractor; a raw
    # error step with no verdict is not a lesson anyone asked for.
    kept: list[dict[str, Any]] = []
    known_pairs: list[tuple[dict[str, Any], int]] = []
    for i, c in enumerate(candidates):
        verdict = verdicts.get(i)
        if verdict is None:
            counts["uncovered"] = counts.get("uncovered", 0) + 1
            if c.get("source") != "error_step":
                kept.append(c)
            continue
        action, known = verdict
        counts[action] = counts.get(action, 0) + 1
        if action == "write":
            kept.append(c)
        elif action == "supersedes":
            # Tag in place: the pipeline matches kept entries by identity.
            c["supersedes"] = known
            kept.append(c)
        elif action == "already_known":
            known_pairs.append((c, known))
    if counts.get("uncovered"):
        logger.warning(
            "CurateCodingMemory covered %d/%d candidates; applied the verdicts given",
            len(verdicts), len(candidates),
        )
    logger.info("CurateCodingMemory: %s", counts)
    return {"kept": kept, "counts": counts, "known": known_pairs}


# --- Outcome rating (MUD-407) ------------------------------------------------

_RATER_KILL_SWITCH_ENV = "NAM_OUTCOME_RATER"


def _rater_enabled() -> bool:
    return os.environ.get(_RATER_KILL_SWITCH_ENV, "").strip().lower() != "off"


async def rate_served_lessons(
    lessons: list[str],
    transcript: str,
    trace_meta: dict[str, Any] | None = None,
) -> list[tuple[bool | None, str | None]]:
    """Rate lessons recall served to a finished session, one per input line.

    ``lessons`` are rendered lesson lines shown to the model numbered by
    list position. Returns a list the same length of ``(verdict, reason)``
    pairs: True for helpful, False for harmful, None for unused or unrated
    (MUD-427: the reason is the judge's one-sentence citation, None when
    the model gave none or the lesson went unrated).

    Fail-open: the rater disabled, raising, or returning nothing leaves
    every lesson unrated. An unrated lesson keeps the neutral weight it
    already has, so a broken rater degrades to "no outcome signal", never
    to a store full of lessons marked harmful.
    """
    unrated: list[tuple[bool | None, str | None]] = [(None, None)] * len(lessons)
    if not lessons or not transcript.strip() or not _rater_enabled():
        return unrated

    import agent_memory_mcp.baml_client.async_client as _async_client
    from agent_memory_mcp.providers import default_baml_options

    block = "\n".join(f"{i}. {line}" for i, line in enumerate(lessons))
    started = time.perf_counter()
    try:
        rated = await _async_client.b.RateServedLessons(
            lessons=block,
            transcript=transcript,
            baml_options=default_baml_options(),
        )
    except Exception:
        logger.warning("RateServedLessons failed; leaving lessons unrated", exc_info=True)
        return unrated

    out = list(unrated)
    counts = {"helpful": 0, "harmful": 0, "unused": 0}
    verdict_log: list[dict[str, Any]] = []
    for v in getattr(rated, "verdicts", []) or []:
        if not (isinstance(v.id, int) and 0 <= v.id < len(lessons)):
            continue
        outcome = str(getattr(v.outcome, "value", v.outcome)).lower()
        reason = getattr(v, "reason", None)
        if outcome == "helpful":
            out[v.id] = (True, reason)
        elif outcome == "harmful":
            out[v.id] = (False, reason)
        else:
            outcome = "unused"
        counts[outcome] = counts.get(outcome, 0) + 1
        verdict_log.append({"id": v.id, "outcome": outcome, "reason": reason})
    logger.info("RateServedLessons: %s", counts)
    tracing.emit_trace(
        "served-rater",
        input={
            "lessons": block,
            "transcript": tracing.truncate_transcript(transcript),
        },
        output={"verdicts": verdict_log},
        metadata={
            "model": tracing.model_tag(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "of": len(lessons),
            **counts,
            **(trace_meta or {}),
        },
    )
    return out

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
from typing import Any

from .unified import _clamp

logger = logging.getLogger(__name__)

_JUDGE_KILL_SWITCH_ENV = "NAM_CAPTURE_JUDGE"

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
    result = await _async_client.b.ExtractCodingMemory(
        transcript=transcript,
        context=context,
        baml_options=default_baml_options(),
    )

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
    return {
        "decisions": decisions,
        "gotchas": gotchas,
        "dead_ends": dead_ends,
        "preferences": preferences,
        "anchor_rate": anchor_rate,
        "dropped_unanchored": dropped_unanchored,
    }


def _verdict_counts() -> dict[str, int]:
    return {"write": 0, "already_known": 0, "not_durable": 0, "unsupported": 0}


async def curate_coding_memory(
    candidates: list[dict[str, Any]],
    transcript: str,
    existing: list[str],
) -> dict[str, Any]:
    """Screen candidates in one batched call; return what to write.

    ``candidates`` are dicts with a ``kind`` (see :data:`CANDIDATE_KINDS`)
    plus that kind's fields. ``existing`` are rendered lines of lessons
    already stored near these candidates (the dedup context). Returns
    ``{"kept": [...], "counts": {write, already_known, not_durable,
    unsupported}}``.

    Fail-open in the strong sense: the curator disabled, raising, or
    returning a verdict set that does not cover every candidate keeps all
    of them — a truncated or confused model must not look like a decisive
    one, and a broken curator must degrade to "no screening", never to
    "capture nothing".
    """
    counts = _verdict_counts()
    if not candidates:
        return {"kept": [], "counts": counts}
    if not _curator_enabled():
        counts["write"] = len(candidates)
        return {"kept": list(candidates), "counts": counts}

    import agent_memory_mcp.baml_client.async_client as _async_client
    from agent_memory_mcp.providers import default_baml_options

    block = "\n".join(f"{i}. {candidate_line(c)}" for i, c in enumerate(candidates))
    existing_block = "\n".join(f"- {line}" for line in existing) or "(none)"
    try:
        curated = await _async_client.b.CurateCodingMemory(
            candidates=block,
            existing=existing_block,
            transcript=transcript,
            baml_options=default_baml_options(),
        )
        verdicts = {
            v.id: str(getattr(v.action, "value", v.action)).lower()
            for v in curated.verdicts
            if isinstance(v.id, int) and 0 <= v.id < len(candidates)
        }
    except Exception:
        logger.warning("CurateCodingMemory failed; keeping all candidates", exc_info=True)
        counts["write"] = len(candidates)
        return {"kept": list(candidates), "counts": counts}

    if len(verdicts) != len(candidates):
        logger.warning(
            "CurateCodingMemory covered %d/%d candidates; keeping all",
            len(verdicts), len(candidates),
        )
        counts["write"] = len(candidates)
        return {"kept": list(candidates), "counts": counts}

    kept: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        action = verdicts[i]
        counts[action] = counts.get(action, 0) + 1
        if action == "write":
            kept.append(c)
    logger.info("CurateCodingMemory: %s", counts)
    return {"kept": kept, "counts": counts}

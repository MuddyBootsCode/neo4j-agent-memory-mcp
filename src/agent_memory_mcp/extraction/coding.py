"""Coding-session memory extraction with code-enforced anchoring.

Coding memory has two planes: a deterministic plane (branches, commits, files
touched) written directly by git and hooks, and an anchored plane — decisions,
gotchas, dead ends — that this module extracts from the session transcript via
BAML. Every anchored item must point at files the session actually touched or
at the stated task. The prompt asks the model for that, but the rule is
enforced here in code: invented anchor paths are removed, and items left with
no anchor are dropped and counted in ``dropped_unanchored``.
"""

from __future__ import annotations

import logging
from typing import Any

from .unified import _clamp

logger = logging.getLogger(__name__)


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
    ``preferences`` lists plus two anchoring metrics. Each item's
    ``anchor_files`` is reduced to its intersection with ``files`` (exact match
    after ``.strip()``, model order preserved, duplicates removed); a decision/gotcha/dead-end left
    with no anchor files and a falsy ``concerns_task`` is dropped and counted
    in ``dropped_unanchored``. When ``task`` is None, ``concerns_task`` cannot
    rescue an item. ``anchor_rate`` is kept/(kept+dropped) over the three
    anchored types, or None when none were extracted. Preferences are
    session-anchored by definition and never affect the rate.
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

    allowed = {f.strip() for f in files}
    kept_anchored = 0
    dropped_unanchored = 0

    def _sanitize_anchors(anchor_files: list[str]) -> list[str]:
        seen: set[str] = set()
        sanitized: list[str] = []
        for path in (a.strip() for a in anchor_files):
            if path and path in allowed and path not in seen:
                seen.add(path)
                sanitized.append(path)
        return sanitized

    def _admit(anchor_files: list[str], concerns_task: bool) -> bool:
        """Count and admit one anchored-type item; False means drop it."""
        nonlocal kept_anchored, dropped_unanchored
        if anchor_files or (concerns_task and task is not None):
            kept_anchored += 1
            return True
        dropped_unanchored += 1
        return False

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

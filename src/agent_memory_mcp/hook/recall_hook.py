"""UserPromptSubmit recall hook: push-mode memory injection.

Reads the Claude Code hook payload from stdin, queries the running MCP
server's ``memory_search`` tool over streamable HTTP, and emits the
results as ``additionalContext`` so retrieval happens before the model
sees the prompt (push, not pull — the model never asks).

Fail-open contract: any error, timeout, or malformed payload exits 0
with no output. Memory must never block a prompt submit.

Configuration (environment):
    NAM_HOOK_URL        MCP server URL (default http://127.0.0.1:8080/mcp)
    NAM_HTTP_TOKEN      Bearer token, if the server requires one
    NAM_HOOK_TIMEOUT    Whole-call budget in seconds (default 5)
    NAM_HOOK_MAX_CHARS  Cap on injected context size (default 4000)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, TextIO

DEFAULT_URL = "http://127.0.0.1:8080/mcp"
DEFAULT_TIMEOUT = 5.0
DEFAULT_MAX_CHARS = 4000
SEARCH_LIMIT = 5
# Lower than the server's 0.7 default: a prompt rarely shares vocabulary
# with the facts that answer it (measured: the 3-hop refund chain scores
# 0.53-0.67 against its own question). Precision is recovered by the
# formatter's small cap, not by the similarity cutoff.
DEFAULT_THRESHOLD = 0.5
# Messages and traces are deliberately excluded: they are bulky, rarely
# load-bearing for a fresh prompt, and available on demand via the pull
# tools. The injection carries only compact, factual memory.
MEMORY_TYPES = ["facts", "entities", "preferences"]

TRUNCATION_MARKER = "… (truncated)"


def _header(count: int, ms: float) -> str:
    return f"memory: {count} items recalled in {ms:.0f} ms"


def format_context(
    response: dict[str, Any],
    ms: float,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Render a memory_search response as compact triples + notes.

    Mirrors the graph-memory-starter output shape: a counts header, one
    line per fact, entity descriptions in a ``where:`` block (conditions
    live on entities, not edges), preferences last.

    The header counts what the body actually renders, not what the
    response contained: an entity with no description and no neighbors
    produces no line, and truncation drops trailing records.
    """
    results = response.get("results") or {}
    facts = results.get("facts") or []
    entities = results.get("entities") or []
    prefs = results.get("preferences") or []

    lines: list[str] = []
    # Line index at which each rendered record first appears. A record
    # that renders nothing never lands here, and truncation counts only
    # the indices that survive.
    item_at: list[int] = []

    for f in facts:
        line = f"{f.get('subject')} --[{f.get('predicate')}]--> {f.get('object')}"
        status = f.get("temporal_status")
        if status and status != "active":
            line += f"   ({status})"
        item_at.append(len(lines))
        lines.append(line)

    # An entity can render both edges and a where: note; count it once,
    # at whichever comes first.
    entity_at: dict[int, int] = {}
    for i, e in enumerate(entities):
        for n in e.get("neighbors") or []:
            rel = n.get("relationship")
            entity_at.setdefault(i, len(lines))
            if n.get("direction") == "incoming":
                lines.append(f"{n.get('name')} --[{rel}]--> {e.get('name')}")
            else:
                lines.append(f"{e.get('name')} --[{rel}]--> {n.get('name')}")

    notes = [(i, e) for i, e in enumerate(entities) if e.get("description")]
    if notes:
        lines.append("")
        lines.append("where:")
        for i, e in notes:
            entity_at.setdefault(i, len(lines))
            lines.append(f"  {e.get('name')}: {e.get('description')}")
    item_at.extend(entity_at.values())

    if prefs:
        lines.append("")
        for p in prefs:
            item_at.append(len(lines))
            lines.append(f"[{p.get('category')}] {p.get('preference')}")

    if not item_at:
        return _header(0, ms) + "\n(no memory matches for this prompt)"

    header = _header(len(item_at), ms)
    text = "\n".join([header, ""] + lines)
    if len(text) <= max_chars:
        return text

    # Header and blank line are kept verbatim; the recount below can only
    # shorten the header, so the budget stays conservative.
    budget = max_chars - len(header) - 1 - 1 - len(TRUNCATION_MARKER) - 1
    kept: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > budget:
            break
        kept.append(line)
        size += len(line) + 1
    shown = sum(1 for idx in item_at if idx < len(kept))
    kept.append(TRUNCATION_MARKER)
    return "\n".join([_header(shown, ms), ""] + kept)


# Coding-memory rendering (coding_recall tool). Overlaps render first and
# ungated; memory lines follow under the existing char cap. Both formatters
# share the hook's fail-open contract: malformed rows degrade, never raise.
MAX_FILES_SHOWN = 3
KIND_LABELS = {"Decision": "decision", "Gotcha": "gotcha", "DeadEnd": "dead end"}
# Relative-time bucket edges, in seconds. Each edge sits past the unit it
# closes (90s, 90min, 36h) so a value like "89 minutes" still reads in the
# finer unit instead of rounding down to "1h".
JUST_NOW_CUTOFF = 90
MINUTES_CUTOFF = 90 * 60
HOURS_CUTOFF = 36 * 3600


def _shown_files(files: Any) -> str:
    """Join up to MAX_FILES_SHOWN paths, noting how many were elided.

    Best-effort on malformed input: a non-list renders as no files, and
    non-string entries are coerced with str().
    """
    if not isinstance(files, list) or not files:
        return ""
    shown = [str(f) for f in files[:MAX_FILES_SHOWN]]
    if len(files) > MAX_FILES_SHOWN:
        shown.append(f"+{len(files) - MAX_FILES_SHOWN} more")
    return ", ".join(shown)


def _relative_time(last_seen: Any, now: datetime) -> str:
    """Humanize an ISO-8601 timestamp relative to ``now``.

    Naive timestamps are treated as UTC (Neo4j datetimes usually carry an
    offset, but the contract does not require one). Anything unparseable
    degrades to "recently" rather than raising.
    """
    try:
        seen = datetime.fromisoformat(last_seen)
    except (TypeError, ValueError):
        return "recently"
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    seconds = (now - seen).total_seconds()
    if seconds < JUST_NOW_CUTOFF:
        return "just now"
    if seconds < MINUTES_CUTOFF:
        return f"{int(seconds // 60)}m ago"
    if seconds < HOURS_CUTOFF:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def format_overlap_block(
    overlaps: list[dict], now: datetime | None = None
) -> str | None:
    """Render coding_recall overlaps as a compact awareness block.

    None when there is nothing to show. Otherwise a counts header and one
    line per row: who, what they touch, and how recently. Never raises —
    the hook is fail-open, and a formatter crash would kill the whole
    injection — so malformed rows degrade to a best-effort line.
    """
    if not overlaps or not isinstance(overlaps, list):
        return None
    now = now if now is not None else datetime.now(timezone.utc)

    lines = [f"agents: {len(overlaps)} active nearby"]
    for row in overlaps:
        if not isinstance(row, dict):
            row = {}
        agent = str(row.get("agent") or "another agent")
        task = row.get("task")
        task = str(task) if task else None
        files = _shown_files(row.get("files"))
        when = _relative_time(row.get("last_seen"), now)
        if files:
            detail = f"{task}, {when}" if task else when
            lines.append(f"  {agent} is editing {files} ({detail})")
        elif task:
            lines.append(f"  {agent} is working on {task} ({when})")
        else:
            lines.append(f"  {agent} is working nearby ({when})")
    return "\n".join(lines)


def format_coding_memories(memories: list[dict]) -> str | None:
    """Render coding_recall memories, one line per record.

    None when there is nothing to show — including when every row is
    missing its text. Kinds map to lowercase labels (DeadEnd → "dead
    end"); an unknown kind passes through lowercased. Never raises;
    malformed rows degrade or are skipped.
    """
    if not memories or not isinstance(memories, list):
        return None

    lines: list[str] = []
    for row in memories:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        kind = row.get("kind")
        if isinstance(kind, str) and kind:
            label = KIND_LABELS.get(kind, kind.lower())
        else:
            label = "note"
        task = row.get("task")
        task = str(task) if task else None
        parts = [p for p in (_shown_files(row.get("files")), task) if p]
        suffix = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"[{label}] {text.strip()}{suffix}")
    return "\n".join(lines) if lines else None


def build_hook_output(text: str) -> dict[str, Any]:
    """Wrap formatted context in the UserPromptSubmit hook contract."""
    return {
        "systemMessage": text.split("\n")[0],
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        },
    }


def _extract_text(result: Any) -> str:
    """Pull the JSON string out of a fastmcp CallToolResult."""
    data = getattr(result, "data", None)
    if isinstance(data, str):
        return data
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            return text
    raise ValueError("memory_search returned no text content")


def build_search_args(prompt: str) -> dict[str, Any]:
    """Arguments for the memory_search tool call."""
    return {
        "query": prompt,
        "limit": SEARCH_LIMIT,
        "memory_types": list(MEMORY_TYPES),
        "threshold": float(os.environ.get("NAM_HOOK_THRESHOLD", DEFAULT_THRESHOLD)),
    }


def search_via_mcp(prompt: str) -> str:
    """Call memory_search on the running server; returns its JSON string."""
    import asyncio

    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    url = os.environ.get("NAM_HOOK_URL", DEFAULT_URL)
    token = os.environ.get("NAM_HTTP_TOKEN")
    timeout = float(os.environ.get("NAM_HOOK_TIMEOUT", DEFAULT_TIMEOUT))

    headers = {"Authorization": f"Bearer {token}"} if token else None
    transport = StreamableHttpTransport(url, headers=headers)

    async def _call() -> str:
        async with Client(transport, timeout=timeout) as client:
            result = await client.call_tool("memory_search", build_search_args(prompt))
            return _extract_text(result)

    return asyncio.run(asyncio.wait_for(_call(), timeout=timeout))


def run(
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    search: Callable[[str], str] | None = None,
) -> int:
    """Hook entry point. Always returns 0: recall is best-effort."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    search = search if search is not None else search_via_mcp

    try:
        payload = json.load(stdin)
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return 0

        t0 = time.perf_counter()
        raw = search(prompt)
        ms = (time.perf_counter() - t0) * 1000

        response = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(response, dict) or "error" in response:
            return 0

        max_chars = int(os.environ.get("NAM_HOOK_MAX_CHARS", DEFAULT_MAX_CHARS))
        text = format_context(response, ms=ms, max_chars=max_chars)
        print(json.dumps(build_hook_output(text)), file=stdout)
        return 0
    except Exception:
        return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()

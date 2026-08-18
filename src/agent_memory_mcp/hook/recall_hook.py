"""UserPromptSubmit recall hook: push-mode memory injection.

Reads the Claude Code hook payload from stdin and injects memory as
``additionalContext`` so retrieval happens before the model sees the
prompt (push, not pull — the model never asks). Two paths:

- Coding path: when the payload's ``cwd`` is a git repo, a fast git
  sweep builds a session context (repo, branch, edited files, task key)
  and the hook calls ``coding_recall`` — anchor-first coding memories
  plus warnings about other agents working nearby. The same connection
  also makes a best-effort ``record_coding_activity`` call (push capture
  of the session's files and recent commits); its errors and stalls are
  swallowed and it is bounded by its own small timeout.
- Classic path: outside a repo, or whenever the coding path fails or the
  server says ``fallback``, the hook queries ``memory_search`` and
  renders compact triples + notes, exactly as before.

Fail-open contract: any error, timeout, or malformed payload exits 0
with no output. Memory must never block a prompt submit. Worst-case
coding-path latency stacks the git sweep (NAM_HOOK_GIT_BUDGET plus up to
one 3 s commit scan) with up to two MCP sessions of NAM_HOOK_TIMEOUT
each (coding_recall, then a fallback memory_search).

Configuration (environment):
    NAM_HOOK_URL          MCP server URL (default http://127.0.0.1:8080/mcp)
    NAM_HTTP_TOKEN        Bearer token, if the server requires one
    NAM_HOOK_TIMEOUT      Per-MCP-session budget in seconds (default 5)
    NAM_HOOK_MAX_CHARS    Cap on injected context size (default 4000)
    NAM_AGENT_ID          Stable agent identity; falls back to the
                          payload's session_id, then "unknown-agent"
    NAM_HOOK_GIT_BUDGET   Wall-clock budget in seconds for the git sweep
                          (default 4.0); collectors past it are skipped
    NAM_CAPTURE_LOOKBACK_HOURS
                          How far back the push capture looks for
                          commits (default 24)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TextIO

from agent_memory_mcp.capture.git_sweep import (
    commits_since,
    current_branch,
    edited_files,
    infer_task_key,
    repo_name,
)

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


def _clean_task(task: Any) -> str | None:
    """A task is rendered only when it is a non-empty string.

    Lists, ints, and other reprs would leak noise into the injection.
    """
    return task if isinstance(task, str) and task else None


def format_overlap_block(
    overlaps: list[dict] | None, now: datetime | None = None
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
        task = _clean_task(row.get("task"))
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


def format_coding_memories(memories: list[dict] | None) -> str | None:
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
        task = _clean_task(row.get("task"))
        parts = [p for p in (_shown_files(row.get("files")), task) if p]
        suffix = f" ({', '.join(parts)})" if parts else ""
        # One record, one line: embedded newlines would let truncation
        # split a record and make the header count dishonest.
        flat = text.strip().replace("\n", " ")
        lines.append(f"[{label}] {flat}{suffix}")
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


# Coding path: git-derived session context feeding the coding_recall /
# record_coding_activity tools.
DEFAULT_GIT_BUDGET = 4.0
DEFAULT_LOOKBACK_HOURS = 24.0
# Hard ceiling on the piggybacked push capture: past this, the capture is
# abandoned so a stalled write can never cost the recall its result.
CAPTURE_TIMEOUT = 1.0
MAX_FILES_SENT = 50
# A coding_recall response must carry all of these to be trusted; anything
# thinner is treated as malformed and the hook falls back to memory_search.
_CODING_KEYS = frozenset({"memories", "fallback", "overlaps"})


def gather_session_context(payload: dict) -> dict | None:
    """Build the coding-session context from the hook payload's ``cwd``.

    None disables the coding path entirely: missing/empty ``cwd``, a cwd
    that is not a git repo, or any unexpected error. Each git collector
    already carries a per-subprocess timeout; NAM_HOOK_GIT_BUDGET bounds
    the whole sweep — collectors past the budget are skipped, leaving a
    thinner context rather than a stalled hook.
    """
    try:
        cwd = payload.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            return None
        budget = float(os.environ.get("NAM_HOOK_GIT_BUDGET", DEFAULT_GIT_BUDGET))
        deadline = time.perf_counter() + budget

        repo = repo_name(cwd)
        if repo is None:
            return None
        branch = current_branch(cwd) if time.perf_counter() < deadline else None
        files = edited_files(cwd) if time.perf_counter() < deadline else []
        budget_spent = time.perf_counter() >= deadline

        agent_id = (
            os.environ.get("NAM_AGENT_ID")
            or payload.get("session_id")
            or "unknown-agent"
        )
        return {
            "cwd": cwd,
            "repo": repo,
            "branch": branch,
            "files": list(files)[:MAX_FILES_SENT],
            "task_key": infer_task_key(branch),
            "agent_id": agent_id,
            "session_id": payload.get("session_id") or agent_id,
            "git_budget_spent": budget_spent,
        }
    except Exception:
        return None


def _lookback_since_iso(now: datetime | None = None) -> str:
    """Push-capture commit horizon: a known-good timestamp computed here.

    Always derived from the current clock — never a stored value, which
    could be garbage and make git return the whole history.
    """
    hours = float(os.environ.get("NAM_CAPTURE_LOOKBACK_HOURS", DEFAULT_LOOKBACK_HOURS))
    now = now if now is not None else datetime.now(timezone.utc)
    return (now - timedelta(hours=hours)).isoformat()


async def record_activity_via_mcp(client: Any, ctx: dict, commits: list) -> None:
    """Best-effort, error-swallowed push capture over an open client session.

    Not fire-and-forget: the caller awaits it, but bounds it with
    CAPTURE_TIMEOUT and swallows everything it raises, so it can never
    cost the recall its result. Skipped on a detached HEAD (no branch) —
    the server requires one.
    """
    if not ctx.get("branch"):
        return
    try:
        await client.call_tool(
            "record_coding_activity",
            {
                "agent_id": ctx["agent_id"],
                "session_id": ctx["session_id"],
                "repo": ctx["repo"],
                "branch": ctx["branch"],
                "task_key": ctx["task_key"],
                "edited_files": list(ctx.get("files") or [])[:MAX_FILES_SENT],
                "commits": commits,
            },
        )
    except Exception:
        pass


def coding_recall_via_mcp(prompt: str, ctx: dict) -> dict | None:
    """Call coding_recall, then record_coding_activity, on one connection.

    Returns the parsed coding_recall dict, or None on any failure (server
    down, error payload, timeout) so the caller falls back to
    memory_search. The push capture piggybacks on the same client session
    to avoid a second connection setup; its errors are swallowed
    separately and never affect the returned recall.
    """
    import asyncio

    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    url = os.environ.get("NAM_HOOK_URL", DEFAULT_URL)
    token = os.environ.get("NAM_HTTP_TOKEN")
    timeout = float(os.environ.get("NAM_HOOK_TIMEOUT", DEFAULT_TIMEOUT))

    headers = {"Authorization": f"Bearer {token}"} if token else None
    transport = StreamableHttpTransport(url, headers=headers)

    # The gather already spent its git budget: don't stack another git
    # subprocess on top; capture just goes out without commits.
    if ctx.get("git_budget_spent"):
        commits: list = []
    else:
        commits = commits_since(ctx["cwd"], _lookback_since_iso())

    async def _call() -> str | None:
        async with Client(transport, timeout=timeout) as client:
            text: str | None = None
            try:
                result = await client.call_tool(
                    "coding_recall",
                    {
                        "prompt": prompt,
                        "agent_id": ctx["agent_id"],
                        "repo": ctx["repo"],
                        "files": ctx["files"],
                        "task_key": ctx["task_key"],
                    },
                )
                text = _extract_text(result)
            except Exception:
                text = None
            # The capture gets its own small budget, and even the
            # CancelledError from the outer timeout is swallowed here:
            # a stalled capture must never destroy a recall already won.
            try:
                await asyncio.wait_for(
                    record_activity_via_mcp(client, ctx, commits),
                    timeout=CAPTURE_TIMEOUT,
                )
            except (asyncio.CancelledError, Exception):
                pass
            return text

    try:
        raw = asyncio.run(asyncio.wait_for(_call(), timeout=timeout))
        if raw is None:
            return None
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or "error" in parsed:
            return None
        return parsed
    except Exception:
        return None


# Pinned to _header's wording by TestParseContextHeader; change both
# together.
_CONTEXT_HEADER_RE = re.compile(r"^memory: (\d+) items recalled")


def _parse_context_header(formatted: str) -> tuple[int, str | None]:
    """Split a format_context result into (item count, body).

    The count is read back from the header line so the combined coding
    header can fold it in; an unrecognized header degrades to 0.
    """
    first, _, rest = formatted.partition("\n")
    match = _CONTEXT_HEADER_RE.match(first)
    count = int(match.group(1)) if match else 0
    return count, (rest.lstrip("\n") or None)


def _coding_header(count: int, agents: int, ms: float) -> str:
    if agents:
        return f"memory: {count} items recalled, {agents} agents nearby in {ms:.0f} ms"
    return _header(count, ms)


def _truncate_section(text: str, max_chars: int) -> str:
    """Cap one section by cutting whole lines and appending the marker."""
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(TRUNCATION_MARKER) - 1
    kept: list[str] = []
    size = 0
    for line in text.split("\n"):
        if size + len(line) + 1 > budget:
            break
        kept.append(line)
        size += len(line) + 1
    kept.append(TRUNCATION_MARKER)
    return "\n".join(kept)


def _emit_memory_search(
    prompt: str, search: Callable[[str], str], stdout: TextIO
) -> None:
    """Classic path: memory_search + format_context, unchanged behavior."""
    t0 = time.perf_counter()
    raw = search(prompt)
    ms = (time.perf_counter() - t0) * 1000

    response = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(response, dict) or "error" in response:
        return

    max_chars = int(os.environ.get("NAM_HOOK_MAX_CHARS", DEFAULT_MAX_CHARS))
    text = format_context(response, ms=ms, max_chars=max_chars)
    print(json.dumps(build_hook_output(text)), file=stdout)


def run(
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    search: Callable[[str], str] | None = None,
    coding_recall: Callable[[str, dict], dict | None] | None = None,
    gather: Callable[[dict], dict | None] | None = None,
) -> int:
    """Hook entry point. Always returns 0: recall is best-effort.

    In a git repo the coding path runs first (coding_recall + push
    capture); everywhere else — and on any coding-path failure or a
    ``fallback`` response — the classic memory_search path answers.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    search = search if search is not None else search_via_mcp
    coding_recall = (
        coding_recall if coding_recall is not None else coding_recall_via_mcp
    )
    gather = gather if gather is not None else gather_session_context

    try:
        payload = json.load(stdin)
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return 0

        t0 = time.perf_counter()
        ctx = gather(payload)
        response: Any = None
        if ctx is not None:
            try:
                response = coding_recall(prompt, ctx)
            except Exception:
                response = None

        if not isinstance(response, dict) or not _CODING_KEYS <= response.keys():
            _emit_memory_search(prompt, search, stdout)
            return 0

        max_chars = int(os.environ.get("NAM_HOOK_MAX_CHARS", DEFAULT_MAX_CHARS))
        overlaps = response.get("overlaps") or []
        overlap_text = format_overlap_block(overlaps)
        n_agents = len(overlaps) if overlap_text else 0

        memory_body: str | None
        count = 0
        if response.get("fallback"):
            # Anchor-less recall: the coding graph has nothing to offer,
            # so the classic search supplies the memory section. Its
            # header is folded into the combined one.
            # A search failure only costs the memory section — the
            # overlap block is deterministic and already in hand.
            try:
                raw = search(prompt)
                sresp = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                sresp = None
            ms = (time.perf_counter() - t0) * 1000
            memory_body = None
            if isinstance(sresp, dict) and "error" not in sresp:
                formatted = format_context(sresp, ms=ms, max_chars=max_chars)
                count, memory_body = _parse_context_header(formatted)
        else:
            ms = (time.perf_counter() - t0) * 1000
            memory_body = format_coding_memories(response.get("memories"))
            if memory_body is not None:
                # The char cap applies to the memory section only; the
                # overlap block is an awareness signal and rides ungated.
                memory_body = _truncate_section(memory_body, max_chars)
                count = sum(
                    1 for ln in memory_body.split("\n") if ln != TRUNCATION_MARKER
                )

        if overlap_text is None and memory_body is None:
            return 0

        header = _coding_header(count, n_agents, ms)
        sections = [header] + [s for s in (overlap_text, memory_body) if s]
        print(json.dumps(build_hook_output("\n\n".join(sections))), file=stdout)
        return 0
    except Exception:
        return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()

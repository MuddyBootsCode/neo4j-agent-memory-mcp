"""SessionEnd capture hook: push the session transcript into extracted memory.

Reads the Claude Code SessionEnd/Stop hook payload from stdin, renders the
session transcript (a JSONL file named by ``transcript_path``) into plain
``user:`` / ``assistant:`` lines, gathers the session's git context (repo,
branch, edited files, task key), and calls the server's
``capture_session_memory`` tool — which runs the BAML extraction and writes
the anchored plane (Decision, Gotcha, DeadEnd, CodingPreference). The hook
itself stays thin: no LLM calls, no Neo4j, one MCP round trip.

Fail-open contract, identical to the recall hook: ANY error, timeout, or
malformed payload exits 0 with no output. Nothing is printed on success
either — SessionEnd output is not injected anywhere.

Configuration (environment):
    NAM_CAPTURE_DISABLED  "1" disables the hook entirely (kill switch)
    NAM_HOOK_URL          MCP server URL (default http://127.0.0.1:8080/mcp)
    NAM_HTTP_TOKEN        Bearer token, if the server requires one
    NAM_CAPTURE_TIMEOUT   Whole-call budget in seconds (default 30.0 —
                          extraction is an LLM call and needs more room
                          than the recall hook's NAM_HOOK_TIMEOUT)
    NAM_AGENT_ID          Stable agent identity; falls back to the
                          payload's session_id, then "unknown-agent"
    NAM_TASK_KEY          Explicit task key; overrides the ticket or
                          branch inferred from git
"""

from __future__ import annotations

import json
import os
import re
import sys
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
DEFAULT_CAPTURE_TIMEOUT = 30.0
# Caps the same payload as _MAX_TRANSCRIPT_CHARS in mcp/_coding_tools.py —
# one on the sending side, one on the receiving side. Move them together.
# The server windows the rendering (MUD-404), so this is the whole-session
# ceiling, not one LLM call's worth.
MAX_TRANSCRIPT_CHARS = 400_000
MAX_FILES_SENT = 100
MAX_ERROR_STEPS = 40
# How far back commits count as "this session's" when anchoring lessons.
CAPTURE_LOOKBACK_HOURS = 24.0

# Tool output that signals a failure. An errored step is the evidence a
# Gotcha or DeadEnd is made of, so it is kept nearly whole; successful
# output is summarised to its first line.
_ERROR_RE = re.compile(
    r"(?i)(traceback|exception|\berror\b|\bfailed\b|\bfatal\b|panic:|"
    r"exit code [1-9]|exit=[1-9]|assertionerror|command not found|"
    r"permission denied|no such file)"
)
MAX_ERROR_RESULT_CHARS = 2_000
MAX_OK_RESULT_CHARS = 200
MAX_TOOL_INPUT_CHARS = 200
# Tools whose call is worth a line in the rendering. Read/Grep/Glob are
# navigation, not evidence.
_RENDERED_TOOLS = ("Edit", "Write", "Bash", "MultiEdit")

# Linked-worktree layouts; order matters (".claude/worktrees" contains
# "/worktrees/").
_WORKTREE_MARKERS = ("/.claude/worktrees/", "/.worktrees/")


def _block_text(content: Any) -> str:
    """Plain text of a string or a block list's text blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return " ".join(parts)
    return ""


def _render_content(content: Any) -> str:
    """Flatten a transcript message's content to plain text.

    A string passes through; a block list keeps only ``text`` blocks
    (tool_use / tool_result / thinking blocks are skipped) joined with a
    space. Anything else renders as "".
    """
    return _block_text(content)


def _tool_call_line(block: dict) -> str | None:
    """One-line rendering of a tool_use block, or None for tools not worth it."""
    name = block.get("name")
    if name not in _RENDERED_TOOLS:
        return None
    args = block.get("input") if isinstance(block.get("input"), dict) else {}
    if name in ("Edit", "Write", "MultiEdit"):
        path = args.get("file_path")
        return f"[{name.lower()} {path}]" if isinstance(path, str) else None
    command = args.get("command")
    if not isinstance(command, str):
        return None
    command = " ".join(command.split())[:MAX_TOOL_INPUT_CHARS]
    return f"[bash: {command}]"


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]} … {text[-tail:]}"


def _tool_result_line(block: dict, tool_name: str | None) -> str | None:
    """Render a tool_result: errors nearly whole, successes as a stub."""
    text = _block_text(block.get("content"))
    if not text:
        return None
    text = " ".join(text.split())
    name = tool_name or "tool"
    if block.get("is_error") or _ERROR_RE.search(text):
        return f"[error from {name}] {_truncate_middle(text, MAX_ERROR_RESULT_CHARS)}"
    return f"[{name} ok] {text[:MAX_OK_RESULT_CHARS]}"


def _iter_records(path: str):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            try:
                record = json.loads(raw)
            except Exception:
                continue
            if isinstance(record, dict):
                yield record


def _render_lines(path: str) -> list[str]:
    """Role-prefixed lines for every user/assistant turn, tool calls and
    results included."""
    rendered: list[str] = []
    tool_names: dict[str, str] = {}
    for record in _iter_records(path):
        if record.get("isMeta"):
            continue
        role = record.get("type")
        if role not in ("user", "assistant"):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                rendered.append(f"{role}: {content.strip()}")
            continue
        if not isinstance(content, list):
            continue
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            elif kind == "tool_use":
                if isinstance(block.get("id"), str) and isinstance(block.get("name"), str):
                    tool_names[block["id"]] = block["name"]
                line = _tool_call_line(block)
                if line:
                    text_parts.append(line)
            elif kind == "tool_result":
                line = _tool_result_line(block, tool_names.get(block.get("tool_use_id")))
                if line:
                    rendered.append(f"tool: {line}")
        if text_parts:
            rendered.append(f"{role}: {' '.join(text_parts)}")
    return rendered


def extract_transcript_text(path: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Render a Claude Code transcript JSONL file as role-prefixed lines.

    Each relevant line is ``{"type": "user"|"assistant", "message":
    {"content": ...}}``; everything else — other line types, malformed
    JSON, empty content — is skipped. Entries marked ``isMeta`` are
    skipped too: they are injected machinery (local-command caveats,
    skill files), not conversation. ``isSidechain`` entries are
    deliberately INCLUDED — subagent work carries decisions worth
    extracting.

    Tool activity is rendered as well (MUD-404): Edit/Write/Bash calls as
    one-line markers inside the assistant turn, and tool results as
    ``tool:`` lines — errors nearly whole (they are the evidence a Gotcha
    or DeadEnd is made of), successful output as a short stub.

    Returns the TAIL of the rendering capped at ``max_chars``, cut at a
    line boundary, except when a single message alone exceeds the cap, in
    which case its tail is returned. Never raises; a missing or unreadable
    file yields "".
    """
    try:
        rendered = _render_lines(path)
        # Tail cap: keep whole lines from the end while they fit.
        kept: list[str] = []
        size = 0
        for line in reversed(rendered):
            cost = len(line) + (1 if kept else 0)
            if size + cost > max_chars:
                break
            kept.append(line)
            size += cost
        if not kept and rendered:
            # A single oversized message: keep its tail rather than nothing.
            return rendered[-1][-max_chars:]
        return "\n".join(reversed(kept))
    except Exception:
        return ""


def normalize_repo_path(path: Any, repo_dir: str) -> str | None:
    """A repo-relative path for a tool_input file_path, or None.

    Absolute paths under ``repo_dir`` are made relative to it; paths inside
    a linked worktree are made relative to that worktree (git worktrees
    share one repo identity, see git_sweep.repo_name); relative paths are
    normalised (``./src/x.py`` → ``src/x.py``). Anything outside the repo
    is None.
    """
    if not isinstance(path, str) or not path.strip():
        return None
    path = path.strip()
    for marker in _WORKTREE_MARKERS:
        at = path.find(marker)
        if at == -1:
            continue
        rest = path[at + len(marker):].split("/", 1)
        return os.path.normpath(rest[1]) if len(rest) == 2 and rest[1] else None
    prefix = repo_dir.rstrip("/") + "/"
    if path.startswith(prefix):
        path = path[len(prefix):]
    elif path.startswith("/"):
        return None
    norm = os.path.normpath(path)
    if norm.startswith("..") or norm == ".":
        return None
    return norm


def transcript_touched_files(path: str, repo_dir: str, cap: int = MAX_FILES_SENT) -> list[str]:
    """Repo-relative files the session edited via Edit/Write/MultiEdit,
    in first-touch order. Never raises."""
    seen: dict[str, None] = {}
    try:
        for record in _iter_records(path):
            if record.get("type") != "assistant":
                continue
            content = (record.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                if block.get("name") not in ("Edit", "Write", "MultiEdit"):
                    continue
                args = block.get("input") if isinstance(block.get("input"), dict) else {}
                rel = normalize_repo_path(args.get("file_path"), repo_dir)
                if rel:
                    seen.setdefault(rel, None)
                if len(seen) >= cap:
                    return list(seen)
    except Exception:
        pass
    return list(seen)


def error_steps(path: str, repo_dir: str, cap: int = MAX_ERROR_STEPS) -> list[dict]:
    """Tool calls whose result was an error: ``{"tool", "input", "error",
    "file"}`` in transcript order, newest last. Zero-LLM DeadEnd candidates
    (MUD-404); the curator decides whether each is worth keeping."""
    calls: dict[str, dict] = {}
    steps: list[dict] = []
    try:
        for record in _iter_records(path):
            content = (record.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    args = block.get("input") if isinstance(block.get("input"), dict) else {}
                    summary = args.get("command") or args.get("file_path") or args.get("description") or ""
                    calls[block["id"]] = {
                        "tool": block.get("name") or "tool",
                        "input": " ".join(str(summary).split())[:MAX_TOOL_INPUT_CHARS],
                        "file": normalize_repo_path(args.get("file_path"), repo_dir),
                    }
                elif block.get("type") == "tool_result":
                    text = " ".join(_block_text(block.get("content")).split())
                    if not text or not (block.get("is_error") or _ERROR_RE.search(text)):
                        continue
                    call = calls.get(block.get("tool_use_id"), {"tool": "tool", "input": "", "file": None})
                    steps.append({**call, "error": _truncate_middle(text, 600)})
    except Exception:
        pass
    return steps[-cap:]


def gather_capture_context(payload: dict) -> dict | None:
    """Build the capture context from the hook payload's ``cwd``.

    None disables the capture entirely: missing ``cwd``, a cwd that is
    not a git repo, a detached HEAD (the server requires a branch), or
    any unexpected error. Each git collector carries its own subprocess
    timeout; a SessionEnd hook has no latency budget to enforce beyond
    that.
    """
    try:
        cwd = payload.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            return None
        repo = repo_name(cwd)
        if repo is None:
            return None
        branch = current_branch(cwd)
        if branch is None:
            return None
        agent_id = (
            os.environ.get("NAM_AGENT_ID")
            or payload.get("session_id")
            or "unknown-agent"
        )
        return {
            "agent_id": agent_id,
            "session_id": payload.get("session_id") or agent_id,
            "repo": repo,
            "branch": branch,
            "files": session_files(cwd, payload.get("transcript_path")),
            "error_steps": error_steps(payload.get("transcript_path") or "", cwd),
            "task_key": infer_task_key(branch, repo=repo),
        }
    except Exception:
        return None


def session_files(cwd: str, transcript_path: Any) -> list[str]:
    """Files the session touched, from every source available at SessionEnd.

    ``git status`` alone misses a session that committed its work — the
    tree is clean and every lesson anchors to nothing (the MUD-401 corpus
    dropped 90 lessons that way). So the union of: uncommitted edits,
    files in commits from the lookback window, and Edit/Write paths in
    the transcript. Order: transcript first (what the agent actually
    worked on), then commits, then the working tree. Capped.
    """
    seen: dict[str, None] = {}
    if isinstance(transcript_path, str):
        for p in transcript_touched_files(transcript_path, cwd):
            seen.setdefault(p, None)
    since = (datetime.now(timezone.utc) - timedelta(hours=CAPTURE_LOOKBACK_HOURS)).isoformat()
    for commit in commits_since(cwd, since):
        for p in commit.get("files") or []:
            seen.setdefault(p, None)
    for p in edited_files(cwd, cap=MAX_FILES_SENT):
        seen.setdefault(p, None)
    return list(seen)[:MAX_FILES_SENT]


def capture_via_mcp(transcript: str, ctx: dict) -> None:
    """Call capture_session_memory on the running server.

    The response is ignored — there is nowhere to surface it — so the
    call only needs to complete. Raises on failure; run() swallows it.
    """
    import asyncio

    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    url = os.environ.get("NAM_HOOK_URL", DEFAULT_URL)
    token = os.environ.get("NAM_HTTP_TOKEN")
    timeout = float(os.environ.get("NAM_CAPTURE_TIMEOUT", DEFAULT_CAPTURE_TIMEOUT))

    headers = {"Authorization": f"Bearer {token}"} if token else None
    transport = StreamableHttpTransport(url, headers=headers)

    async def _call() -> None:
        async with Client(transport, timeout=timeout) as client:
            await client.call_tool(
                "capture_session_memory",
                {
                    "agent_id": ctx["agent_id"],
                    "session_id": ctx["session_id"],
                    "repo": ctx["repo"],
                    "branch": ctx["branch"],
                    "transcript": transcript,
                    "task_key": ctx["task_key"],
                    "files": ctx["files"],
                    "error_steps": ctx.get("error_steps") or [],
                },
            )

    asyncio.run(asyncio.wait_for(_call(), timeout=timeout))


def run(
    stdin: TextIO | None = None,
    capture: Callable[[str, dict], None] | None = None,
    gather: Callable[[dict], dict | None] | None = None,
) -> int:
    """Hook entry point. Always returns 0 and prints nothing: capture is
    best-effort and SessionEnd output is not injected anywhere."""
    if os.environ.get("NAM_CAPTURE_DISABLED") == "1":
        return 0

    stdin = stdin if stdin is not None else sys.stdin
    capture = capture if capture is not None else capture_via_mcp
    gather = gather if gather is not None else gather_capture_context

    try:
        payload = json.load(stdin)
        if not isinstance(payload, dict):
            return 0
        transcript_path = payload.get("transcript_path")
        if not isinstance(transcript_path, str) or not os.path.isfile(
            transcript_path
        ):
            return 0
        transcript = extract_transcript_text(transcript_path)
        if not transcript.strip():
            return 0
        ctx = gather(payload)
        if ctx is None:
            return 0
        capture(transcript, ctx)
        return 0
    except Exception:
        return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()

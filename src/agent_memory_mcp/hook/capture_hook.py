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
import sys
from typing import Any, Callable, TextIO

from agent_memory_mcp.capture.git_sweep import (
    current_branch,
    edited_files,
    infer_task_key,
    repo_name,
)

DEFAULT_URL = "http://127.0.0.1:8080/mcp"
DEFAULT_CAPTURE_TIMEOUT = 30.0
# Caps the same payload as _MAX_TRANSCRIPT_CHARS in mcp/_coding_tools.py —
# one on the sending side, one on the receiving side. Move them together.
MAX_TRANSCRIPT_CHARS = 80_000
MAX_FILES_SENT = 100


def _render_content(content: Any) -> str:
    """Flatten a transcript message's content to plain text.

    A string passes through; a block list keeps only ``text`` blocks
    (tool_use / tool_result / thinking blocks are skipped) joined with a
    space. Anything else renders as "".
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return " ".join(parts)
    return ""


def extract_transcript_text(path: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Render a Claude Code transcript JSONL file as role-prefixed lines.

    Each relevant line is ``{"type": "user"|"assistant", "message":
    {"content": ...}}``; everything else — other line types, malformed
    JSON, empty content — is skipped. Entries marked ``isMeta`` are
    skipped too: they are injected machinery (local-command caveats,
    skill files), not conversation, and a multi-KB skill file would
    evict genuine late-session turns from the tail. ``isSidechain``
    entries are deliberately INCLUDED — subagent work carries decisions
    worth extracting. Returns the TAIL of the rendering
    capped at ``max_chars``, cut at a line boundary (whole rendered
    messages), except when a single message alone exceeds the cap, in
    which case its tail is returned. Never raises; a missing or
    unreadable file yields "".
    """
    try:
        rendered: list[str] = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                try:
                    record = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("isMeta"):
                    continue
                role = record.get("type")
                if role not in ("user", "assistant"):
                    continue
                message = record.get("message")
                if not isinstance(message, dict):
                    continue
                text = _render_content(message.get("content"))
                if text:
                    rendered.append(f"{role}: {text}")

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
            "files": edited_files(cwd, cap=MAX_FILES_SENT),
            "task_key": infer_task_key(branch, repo=repo),
        }
    except Exception:
        return None


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

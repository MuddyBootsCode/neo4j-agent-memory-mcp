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


def format_context(
    response: dict[str, Any],
    ms: float,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Render a memory_search response as compact triples + notes.

    Mirrors the graph-memory-starter output shape: a counts header, one
    line per fact, entity descriptions in a ``where:`` block (conditions
    live on entities, not edges), preferences last.
    """
    results = response.get("results") or {}
    facts = results.get("facts") or []
    entities = results.get("entities") or []
    prefs = results.get("preferences") or []

    total = len(facts) + len(entities) + len(prefs)
    header = f"memory: {total} items recalled in {ms:.0f} ms"
    if total == 0:
        return header + "\n(no memory matches for this prompt)"

    lines = [header, ""]

    for f in facts:
        line = f"{f.get('subject')} --[{f.get('predicate')}]--> {f.get('object')}"
        status = f.get("temporal_status")
        if status and status != "active":
            line += f"   ({status})"
        lines.append(line)

    for e in entities:
        for n in e.get("neighbors") or []:
            rel = n.get("relationship")
            if n.get("direction") == "incoming":
                lines.append(f"{n.get('name')} --[{rel}]--> {e.get('name')}")
            else:
                lines.append(f"{e.get('name')} --[{rel}]--> {n.get('name')}")

    notes = [(e.get("name"), e.get("description")) for e in entities if e.get("description")]
    if notes:
        lines.append("")
        lines.append("where:")
        lines.extend(f"  {name}: {desc}" for name, desc in notes)

    if prefs:
        lines.append("")
        lines.extend(f"[{p.get('category')}] {p.get('preference')}" for p in prefs)

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text

    budget = max_chars - len(TRUNCATION_MARKER) - 1
    kept: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > budget:
            break
        kept.append(line)
        size += len(line) + 1
    kept.append(TRUNCATION_MARKER)
    return "\n".join(kept)


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

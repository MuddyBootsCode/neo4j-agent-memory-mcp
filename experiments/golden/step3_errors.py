"""Step 3-errors: an error-keyed query set (MUD-460, P5 E5).

A second trigger and a second query set. Every other P5 experiment replays
a human prompt; this replays a failure. ExpeRepair keys retrieval on the
latest test feedback (+6.8 on SWE-bench Verified); cue-anchored memory
fires on a non-zero exit rather than on a prompt. Nothing published
measures recall keyed by error text against stored symptoms.

Harvested from the same query sessions as the prompt set, so the pool's
holdout still holds. The query text is the failure as ``error_steps``
returns it — middle-truncated to 600 characters, head and tail, so a long
traceback keeps the exception line that diagnoses it.

Query ids start at ERROR_ID_BASE, out of the prompt set's range, so a
labels.json copied from the prompt run can never be mistaken for these.

    GOLDEN_RUN=p5-e5 GOLDEN_SPLIT_FROM=results/p4-live/session_split.json \\
    uv run --no-sync --project ../.. python -u step3_errors.py

Writes queries.json in the same shape step4 and step5 expect, so the rest
of the pipeline is unchanged apart from step4's rubric.
"""

from __future__ import annotations

import json
import os
import re

from lib import HERE, load_json, sample_evenly, save_json, touched_files

MAX_QUERIES = int(os.environ.get("GOLDEN_MAX_QUERIES", "100"))
ERROR_ID_BASE = 1000

# Sandbox and harness noise: the tool never ran against the repo, so no
# lesson could have helped. Anything matching is dropped before dedup.
_NOISE_RE = re.compile(
    r"(?i)(command not found|no such file or directory: /|permission denied.*\.claude|"
    r"operation not permitted|killed by signal|user rejected|tool use was rejected|"
    r"claude requested permissions|api error: 5\d\d|context deadline exceeded)"
)
# Two failures of the same command in a row are one lesson's worth of
# evidence, not two queries. Keyed on the first line, normalized.
_DEDUP_CHARS = 120


def _first_line(error: str) -> str:
    return " ".join(error.split())[:_DEDUP_CHARS].casefold()


def main() -> None:
    src = os.environ.get("GOLDEN_SPLIT_FROM")
    if src:
        src = os.path.join(HERE, src) if not os.path.isabs(src) else src
        with open(src, encoding="utf-8") as fh:
            split = json.load(fh)
    else:
        split = load_json("session_split.json")
    if not split:
        raise SystemExit("set GOLDEN_SPLIT_FROM=results/<run>/session_split.json")

    from agent_memory_mcp.hook.capture_hook import error_steps

    seen: set[str] = set()
    candidates: list[dict] = []
    noise = dupes = 0
    for s in split["query_sessions"]:
        steps = error_steps(s["path"], s.get("repo_root") or "", cap=10_000, with_index=True)
        for step in steps:
            error = " ".join(str(step.get("error") or "").split())
            if not error:
                continue
            if _NOISE_RE.search(error):
                noise += 1
                continue
            key = _first_line(error)
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            candidates.append({
                # The whole failure as the reader gives it. error_steps has
                # already middle-truncated to 600 characters, keeping head
                # AND tail, because a traceback's diagnosis is its last
                # line; cutting the head off again would throw away exactly
                # what production preserved (Codex review, MUD-460).
                "session": s["session"], "repo": s["repo"], "line": step["index"],
                "prompt": error,
                "tool": step.get("tool"), "attempt": step.get("input"),
                "path": s["path"], "repo_root": s.get("repo_root") or "",
            })

    print(f"{len(candidates)} distinct failures across {len(split['query_sessions'])} sessions "
          f"({noise} dropped as sandbox/harness noise, {dupes} as repeats)")
    picked = sample_evenly(candidates, MAX_QUERIES)

    queries = []
    for i, c in enumerate(picked):
        files = touched_files(c["path"], c["repo_root"], before_line=c["line"])
        queries.append({
            "query_id": ERROR_ID_BASE + i,
            "session": c["session"], "repo": c["repo"], "line": c["line"],
            "prompt": c["prompt"], "files": files,
            "anchorable": bool(files), "short": len(c["prompt"].split()) < 15,
            "tool": c["tool"], "attempt": c["attempt"],
        })
    save_json("queries.json", queries)
    with_files = sum(1 for q in queries if q["files"])
    print(f"{len(queries)} error queries, ids {ERROR_ID_BASE}..{ERROR_ID_BASE + len(queries) - 1}; "
          f"{with_files} had edited files before the failure")
    for q in queries[:5]:
        print(f"  q{q['query_id']} [{q['tool']}] {q['prompt'][:80]!r}")


if __name__ == "__main__":
    main()

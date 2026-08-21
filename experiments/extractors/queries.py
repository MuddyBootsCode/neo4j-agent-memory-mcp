"""Evaluation queries — real user prompts from Claude Code transcripts.

The recall hook fires on every prompt, so the query set deliberately keeps
conversational continuations ("ok lets move on", "whats next?"). Filtering to
prompts that happen to have answers would measure a workload that does not
exist and hide how much noise gets injected when nothing relevant is stored.
"""

from __future__ import annotations

import glob
import os
import random
import re
import json

SKIP_PREFIX = (
    "<local-command-caveat>", "<command-name>", "<local-command-stdout>",
    "<command-message>", "<system-reminder>", "[Request interrupted", "Caveat:",
)
SKIP_SUBSTR = (
    "<local-command-stdout>", "<task-notification>", "<bash-stdout>",
    "<bash-stderr>", "This session is being continued from a previous",
)


def sample(n, seed=23):
    seen, rows = set(), []
    for path in sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))):
        project = os.path.basename(os.path.dirname(path))
        for line in open(path, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "user":
                continue
            c = (d.get("message") or {}).get("content")
            if not isinstance(c, str):
                continue
            t = re.sub(r"<system-reminder>.*?</system-reminder>", "", c, flags=re.S).strip()
            if not t or t.startswith(SKIP_PREFIX) or any(s in t for s in SKIP_SUBSTR):
                continue
            if not (15 <= len(t) <= 400) or t.count("\n") > 6:
                continue
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            rows.append({"prompt": t, "project": project})

    # Round-robin across projects so one busy repo cannot dominate.
    rng = random.Random(seed)
    by_project = {}
    for r in rows:
        by_project.setdefault(r["project"], []).append(r)
    buckets = [rng.sample(v, len(v)) for v in by_project.values()]
    rng.shuffle(buckets)
    picked = []
    while len(picked) < n and buckets:
        buckets = [b for b in buckets if b]
        for b in list(buckets):
            if len(picked) >= n:
                break
            picked.append(b.pop())
    rng.shuffle(picked)
    for i, p in enumerate(picked):
        p["split"] = "calibrate" if i < len(picked) // 2 else "report"
    return picked

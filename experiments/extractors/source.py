"""Source text for extraction — real Claude Code session transcripts.

Chunk selection is deterministic (fixed seed, fixed ordering) so every variant
sees byte-identical input. That is what makes the comparison paired: any
difference in the resulting subgraph comes from the extractor, not the sample.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re

CHUNK_CHARS = 4000
TRANSCRIPTS = os.path.expanduser("~/.claude/projects/*/*.jsonl")


def _clean(text):
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = re.sub(
        r"<local-command-[a-z]+>.*?</local-command-[a-z]+>", "", text, flags=re.S
    )
    return text.strip()


def _turns(path):
    out = []
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        content = (d.get("message") or {}).get("content")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if not isinstance(content, str):
            continue
        text = _clean(content)
        if not text or text.startswith(("<command-name>", "Caveat:")):
            continue
        out.append(f"{d['type']}: {text[:1500]}")
    return out


def chunks(limit):
    """Deterministic list of {id, text} chunks, identical across variants."""
    files = sorted(f for f in glob.glob(TRANSCRIPTS) if os.path.getsize(f) > 80_000)
    random.Random(11).shuffle(files)

    out = []
    for path in files:
        turns = _turns(path)
        if len(turns) < 10:
            continue
        sid = os.path.basename(path)[:8]
        buf, size, n = [], 0, 0
        for t in turns:
            if size + len(t) > CHUNK_CHARS and buf:
                out.append({"id": f"{sid}-{n}", "text": "\n".join(buf)})
                buf, size, n = [], 0, n + 1
                if len(out) >= limit:
                    return out
            buf.append(t)
            size += len(t)
        if buf:
            out.append({"id": f"{sid}-{n}", "text": "\n".join(buf)})
        if len(out) >= limit:
            return out
    return out[:limit]

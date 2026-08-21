"""Shared helpers for the recall-precision sweep (MUD-395).

Everything here is read-only with respect to src/ and tests/ — it imports
repo code, never copies it. All paths are absolute; the sweep writes only
under experiments/recall_sweep/results/.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REPO_NAME = "neo4j-agent-memory-mcp"
TRANSCRIPTS_DIR = (
    "/Users/muddybootscode/.claude/projects/"
    "-Users-muddybootscode-Projects-neo4j-agent-memory-mcp"
)
SWEEP_DB = "sweepcorpus"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen-judge"

# Repo-root prefix stripped from absolute tool_input.file_path values to get
# repo-relative CodeFile paths, matching the convention edited_files() (git
# status --porcelain -C repo_dir) already produces.
_REPO_PREFIX = REPO_ROOT.rstrip("/") + "/"

# Synthetic content injected into the transcript that is NOT something a
# human typed: task notifications, system reminders, command-name/message
# expansions, local-command output. These arrive as type:"user" turns and
# would otherwise pass the "real prompt" filter.
_SYNTHETIC_PREFIX_RE = re.compile(
    r"^<(task-notification|system-reminder|command-message|command-name|"
    r"local-command-stdout|user-prompt-submit-hook|automated-reminder|"
    r"command-args)\b"
)


def result_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


def load_json(name: str, default=None):
    path = result_path(name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(name: str, data) -> None:
    path = result_path(name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def list_sessions() -> list[dict]:
    """All session transcripts directly under TRANSCRIPTS_DIR, oldest first.

    Ignores the memory/ subdirectory (per instructions) and anything not a
    top-level *.jsonl file.
    """
    paths = glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.jsonl"))
    sessions = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        sessions.append({"session": stem, "path": path, "mtime": os.path.getmtime(path)})
    sessions.sort(key=lambda s: s["mtime"])
    return sessions


def split_sessions(sessions: list[dict], corpus_fraction: float = 0.6) -> tuple[list[dict], list[dict]]:
    """Chronological split: earlier corpus_fraction -> corpus, rest -> query.

    With very few sessions this still produces at least one of each side
    when n >= 2, per the "run at whatever size exists" instruction.
    """
    n = len(sessions)
    if n == 0:
        return [], []
    if n == 1:
        return sessions, []
    k = round(n * corpus_fraction)
    k = max(1, min(k, n - 1))
    return sessions[:k], sessions[k:]


_WORKTREE_MARKER = "/.worktrees/"


def _repo_relative(path: str) -> str | None:
    """CodeFile.path for an absolute tool_input.file_path.

    Production's edited_files() runs `git status --porcelain -C <cwd>`, whose
    paths are relative to the session's cwd — the worktree root when the
    session ran inside a linked worktree, not the main checkout root. A path
    under .worktrees/<name>/... is therefore normalized relative to that
    worktree, not to REPO_ROOT, so a file edited in a worktree lines up with
    the same file edited in the main checkout (git worktrees share one repo
    identity — see git_sweep.repo_name). Anything else falls back to
    stripping REPO_ROOT.
    """
    if not isinstance(path, str) or not path:
        return None
    marker = path.find(_WORKTREE_MARKER)
    if marker != -1:
        rest = path[marker + len(_WORKTREE_MARKER):]  # "<worktree-name>/rel/path"
        parts = rest.split("/", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]
        return None
    if path.startswith(_REPO_PREFIX):
        return path[len(_REPO_PREFIX):]
    if not path.startswith("/"):
        return path
    return None  # outside the repo entirely — not a CodeFile we can anchor to


def touched_files(transcript_path: str, *, before_line: int | None = None) -> list[str]:
    """Repo-relative files touched via Edit/Write tool_use in this transcript.

    Scans assistant tool_use blocks for name in {Edit, Write} and reads
    tool_input.file_path. Order-preserving dedup. When before_line is given,
    only tool_use entries at a JSONL line index strictly less than
    before_line count — this is the "ordering is easy" reconstruction the
    task calls for, since the transcript is already chronological.
    """
    seen: dict[str, None] = {}
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as fh:
            for idx, raw in enumerate(fh):
                if before_line is not None and idx >= before_line:
                    break
                try:
                    record = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(record, dict) or record.get("type") != "assistant":
                    continue
                message = record.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    if block.get("name") not in ("Edit", "Write"):
                        continue
                    tool_input = block.get("input")
                    if not isinstance(tool_input, dict):
                        continue
                    rel = _repo_relative(tool_input.get("file_path"))
                    if rel:
                        seen.setdefault(rel, None)
    except Exception:
        return []
    return list(seen)


def _content_text(content) -> str | None:
    """Plain text of a message.content, or None if it is not real prose.

    A string passes through. A list is real prose only when every block is a
    text block (a tool_result block anywhere means this is synthetic
    machinery, not something a human typed).
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        if not content:
            return None
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        texts = [
            b.get("text", "").strip()
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        texts = [t for t in texts if t]
        return " ".join(texts) if texts else None
    return None


def real_user_prompt(record: dict) -> str | None:
    """The human-typed text of a transcript line, or None.

    Filters isMeta, non-"user" types, tool-result turns, synthetic injected
    content (task notifications, system reminders, command expansions), and
    slash commands. Does NOT filter by word count — callers do that.
    """
    if not isinstance(record, dict) or record.get("isMeta"):
        return None
    if record.get("type") != "user":
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    text = _content_text(message.get("content"))
    if not text:
        return None
    if text.startswith("/"):
        return None
    if _SYNTHETIC_PREFIX_RE.match(text):
        return None
    return text


def iter_transcript_lines(transcript_path: str):
    """Yield (line_index, record_dict) for every parseable JSON line."""
    with open(transcript_path, encoding="utf-8", errors="replace") as fh:
        for idx, raw in enumerate(fh):
            try:
                record = json.loads(raw)
            except Exception:
                continue
            if isinstance(record, dict):
                yield idx, record


def sample_evenly(items: list, k: int) -> list:
    """Evenly-spaced sample of up to k items, preserving relative order."""
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return list(items)
    step = n / k
    idxs = sorted({int(i * step) for i in range(k)})
    # Rounding can collide and undershoot k; top up from the remaining pool.
    if len(idxs) < k:
        remaining = [i for i in range(n) if i not in idxs]
        idxs.extend(remaining[: k - len(idxs)])
        idxs = sorted(idxs)
    return [items[i] for i in idxs[:k]]

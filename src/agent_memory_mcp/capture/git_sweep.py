"""Git collectors for the deterministic coding-memory plane.

Every function here runs inside the UserPromptSubmit recall hook, which is
fail-open: memory must never block a prompt. So every git call carries a
short timeout and ANY failure — nonzero exit, timeout, missing ``git``
binary, not a repo — degrades to the documented empty value instead of
raising. Subprocess calls are kept minimal because these run on every
prompt submit.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

GIT_TIMEOUT = 3.0
"""Seconds before any single git subprocess is abandoned."""

# Matches a ticket key like MUD-395 inside a branch name.
_TASK_KEY_RE = re.compile(r"[A-Z]{2,}-\d+")

# A commit header line from ``commits_since``: full sha, then the \x1f
# unit separator. File lines never match — git quotes control characters
# in ``--name-only`` output.
_COMMIT_HEADER_RE = re.compile(r"^[0-9a-f]{40}\x1f")


def _git(repo_dir: str, *args: str) -> str | None:
    """Run one git command; stdout on success, None on any failure."""
    # Drop ambient repo overrides so ``-C repo_dir`` always wins, and decode
    # lossily so non-UTF8 output degrades per character, not per call.
    env = {
        k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_INDEX_FILE")
    }
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, *args],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT,
            env=env,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def current_branch(repo_dir: str) -> str | None:
    """The checked-out branch name, or None when detached or not a repo."""
    out = _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    if out is None:
        return None
    branch = out.strip()
    if not branch or branch == "HEAD":
        return None
    return branch


def repo_name(repo_dir: str) -> str | None:
    """Basename of the repository's top-level directory, or None."""
    out = _git(repo_dir, "rev-parse", "--show-toplevel")
    if out is None:
        return None
    toplevel = out.strip()
    if not toplevel:
        return None
    return os.path.basename(toplevel)


def edited_files(repo_dir: str, cap: int = 50) -> list[str]:
    """Files touched in the working tree: dirty, staged, and untracked.

    One ``git status --porcelain`` call: entries whose X or Y column marks a
    staged or unstaged change, plus ``??`` untracked entries. Renames yield
    the new path. Order-preserving dedup, truncated to ``cap`` entries.
    Empty list on failure.
    """
    out = _git(repo_dir, "status", "--porcelain")
    if out is None:
        return []

    seen: dict[str, None] = {}
    for line in out.splitlines():
        if len(line) < 4:
            continue
        x, y, path = line[0], line[1], line[3:]
        untracked = x == "?" and y == "?"
        changed = x not in " ?" or y not in " ?"
        if not (untracked or changed):
            continue
        # Rename entries read "old -> new"; the new path is the live one.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            seen.setdefault(path, None)
    return list(seen)[:cap]


def commits_since(repo_dir: str, since_iso: str, cap: int = 20) -> list[dict[str, Any]]:
    """Commits newer than ``since_iso``, newest first, with touched files.

    Each entry is ``{"sha", "ts", "message", "files"}``; merge and empty
    commits carry ``files: []``. Empty list on any git failure.
    """
    out = _git(
        repo_dir,
        "log",
        "--no-show-signature",
        f"--since={since_iso}",
        f"--max-count={cap}",
        "--pretty=format:%H%x1f%cI%x1f%s",
        "--name-only",
    )
    if out is None:
        return []

    commits: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in out.splitlines():
        if _COMMIT_HEADER_RE.match(line):
            # maxsplit keeps any \x1f inside the subject in the message.
            sha, ts, message = line.split("\x1f", 2)
            current = {"sha": sha, "ts": ts, "message": message, "files": []}
            commits.append(current)
        elif line.strip() and current is not None:
            current["files"].append(line)
    return commits


def infer_task_key(branch: str | None, env: dict | None = None) -> str | None:
    """Task key for the current work: env override, ticket in branch, branch.

    ``NAM_TASK_KEY`` (from ``env``, default ``os.environ``) wins when set and
    non-empty; else the first ``ABC-123``-style match in the branch; else the
    branch itself; None when the branch is None/empty with no override.
    """
    source = os.environ if env is None else env
    override = source.get("NAM_TASK_KEY")
    if override:
        return override
    if not branch:
        return None
    match = _TASK_KEY_RE.search(branch)
    if match:
        return match.group(0)
    return branch

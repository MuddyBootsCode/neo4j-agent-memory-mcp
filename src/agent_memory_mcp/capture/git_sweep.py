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
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
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

    Order-preserving union of ``git diff --name-only HEAD``,
    ``git diff --name-only --cached``, and untracked files, truncated to
    ``cap`` entries. Empty list when any of the three calls fails.
    """
    outputs = []
    for args in (
        ("diff", "--name-only", "HEAD"),
        ("diff", "--name-only", "--cached"),
        ("ls-files", "--others", "--exclude-standard"),
    ):
        out = _git(repo_dir, *args)
        if out is None:
            return []
        outputs.append(out)

    seen: dict[str, None] = {}
    for out in outputs:
        for line in out.splitlines():
            path = line.strip()
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

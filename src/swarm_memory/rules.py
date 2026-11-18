"""Path, signature and record rules, ported from the Phase 1 replay.

The hook cannot import ``experiments/replay/replay.py``: that script pulls in the
old package and would cost startup time on every tool call. The functions here
are copies, and ``tests/test_swarm_memory.py`` checks them against the replay on
the same inputs so the two cannot drift apart.

Two rules differ from the replay, each because the hook runs live:

* A file counts when it exists on disk now. The replay checked ``git ls-files``
  plus history.
* The repo comes from reading ``.git`` and the git dir's ``commondir`` file, with
  no git subprocess.

Neither copy reads shell command text: what a command touched cannot be told
from its text.

The failure signature is the replay's, unchanged. The store keeps only its
keyed hash (``store.signature_hash``), so no text from a tool's output is kept.
"""

from __future__ import annotations

import functools
import os
import re

FILE_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"})
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
MAX_NAMED_FILES = 25
MAX_TOKENS = 400

# ---------------------------------------------------------------- repos


@functools.lru_cache(maxsize=512)
def locate(path: str):
    """``(repo_key, repo_name, checkout_root, git_dir)`` for an absolute path
    inside a git checkout, else None.

    The repo key is the real path of the git common directory, so the main
    checkout and every linked worktree of one repo share it. A ``.git``
    directory marks a main checkout. A ``.git`` file names a linked worktree's
    git dir, whose ``commondir`` file points at the common directory.
    """
    if not isinstance(path, str) or not os.path.isabs(path):
        return None
    d = os.path.normpath(path)
    if not os.path.isdir(d):
        d = os.path.dirname(d)
    for _ in range(64):
        dot = os.path.join(d, ".git")
        if os.path.isdir(dot):
            gitdir = common = dot
            break
        if os.path.isfile(dot):
            try:
                with open(dot, encoding="utf-8") as fh:
                    first = fh.readline().strip()
            except OSError:
                return None
            if not first.startswith("gitdir:"):
                return None
            gitdir = first.split(":", 1)[1].strip()
            if not os.path.isabs(gitdir):
                gitdir = os.path.normpath(os.path.join(d, gitdir))
            common = gitdir
            try:
                with open(os.path.join(gitdir, "commondir"), encoding="utf-8") as fh:
                    rel = fh.read().strip()
                if rel:
                    common = os.path.normpath(os.path.join(gitdir, rel))
            except OSError:
                pass  # a submodule's git dir is its own common dir
            break
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent
    else:
        return None
    key = os.path.realpath(common)
    name = os.path.basename(os.path.dirname(key)) if os.path.basename(key) == ".git" else os.path.basename(key)
    return key, name, d, gitdir


def branch(git_dir: str) -> str | None:
    """The checked-out branch from ``<git_dir>/HEAD``, or None when detached."""
    try:
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
    except OSError:
        return None
    prefix = "ref: refs/heads/"
    return head[len(prefix):] if head.startswith(prefix) else None


def repo_rel(path: str, root: str) -> str | None:
    """Path relative to a checkout root, or None outside it or inside ``.git``."""
    p = os.path.normpath(path)
    prefix = root.rstrip("/") + "/"
    if not p.startswith(prefix):
        return None
    rel = p[len(prefix):]
    if not rel or rel == ".git" or rel.startswith(".git/"):
        return None
    return rel


def file_location(path: str):
    """``(repo_key, repo_name, worktree, rel)`` for a file inside a checkout, else None.

    The path is first resolved to the file it names, symlinks included, so a
    link and its target are one file. ``worktree`` is the real path of the
    checkout root. ``rel`` is relative to that root, so the same file in two
    worktrees of one repo gets one ``rel``.
    """
    if not isinstance(path, str) or not os.path.isabs(path):
        return None
    path = os.path.realpath(path)
    loc = locate(path)
    if loc is None:
        return None
    key, name, root, _ = loc
    rel = repo_rel(path, root)
    if rel is None:
        return None
    return key, name, os.path.realpath(root), rel


def file_key(repo_key: str, rel: str) -> str:
    return f"{repo_key}::{rel}"


_KEY = re.compile(r"^[A-Z][A-Z0-9]{0,9}$")


def parse_ticket_keys(keys) -> tuple:
    """Upper-case prefixes of one to ten letters or digits, starting with a
    letter. Ticket prefixes are a local setting (``store.ticket_keys``), never
    part of the repo. The same rule as the replay's ``parse_ticket_keys``."""
    if isinstance(keys, str):
        keys = keys.split(",")
    if not isinstance(keys, (list, tuple)):
        return ()
    clean = (k.strip() for k in keys if isinstance(k, str))
    return tuple(dict.fromkeys(k for k in clean if _KEY.match(k)))


@functools.lru_cache(maxsize=32)
def _ticket_patterns(keys: tuple):
    if not keys:
        return None, None
    alt = r"\b(?:%s)-\d+\b" % "|".join(map(re.escape, keys))
    return re.compile(alt), re.compile(alt, re.I)


def prompt_tickets(text: str, keys=()) -> list[str]:
    """Ticket keys with a configured prefix that a prompt names, upper case
    only (replay Limit 7). No prefixes, no tickets."""
    upper, _ = _ticket_patterns(tuple(keys))
    out: list[str] = []
    if upper is None:
        return out
    for t in upper.findall(text or ""):
        if t not in out:
            out.append(t)
    return out


def branch_ticket(name: str | None, keys=()) -> str | None:
    """The first ticket key with a configured prefix in a branch name, which
    is usually lower case."""
    _, any_case = _ticket_patterns(tuple(keys))
    m = any_case.search(name or "") if any_case else None
    return m.group(0).upper() if m else None


# ---------------------------------------------------------------- names in prompt and brief text

_TOKEN_SPLIT = re.compile(r"[^\s`'\"()<>\[\]{},;|*!?=]+")
_HAS_EXT = re.compile(r"\.[A-Za-z][\w]{0,7}$")


def _clean_token(tok: str) -> str:
    # A leading dot belongs to the path (./x, ../x, .env); only trailing dots are punctuation.
    tok = tok.lstrip(",:;)]}>'\"`").rstrip(".,:;)]}>'\"`")
    tok = re.sub(r"(:\d+){1,2}$", "", tok)
    tok = re.sub(r"#L\d+(-L?\d+)?$", "", tok)
    tok = tok.lstrip("@")
    while tok.startswith("./"):
        tok = tok[2:]
    return tok


def _candidates(tok: str, bases: list[str]) -> list[str]:
    if tok.startswith("~"):
        return [os.path.normpath(os.path.expanduser(tok))]
    if tok.startswith("/"):
        return [os.path.normpath(tok)]
    return [os.path.normpath(os.path.join(b, tok)) for b in bases]


def named_files(text: str, bases: list[str]) -> list[str]:
    """Absolute paths of existing files that a prompt or a brief names.

    A token counts when it has a ``/`` or a file extension and resolves to a
    regular file, absolute or relative to one of ``bases`` in order.
    """
    found: list[str] = []
    for n, raw in enumerate(_TOKEN_SPLIT.findall(text or "")):
        if n >= MAX_TOKENS or len(found) >= MAX_NAMED_FILES:
            break
        tok = _clean_token(raw)
        if not tok or not ("/" in tok or _HAS_EXT.search(tok)):
            continue
        for cand in _candidates(tok, bases):
            if os.path.isfile(cand):
                if cand not in found:
                    found.append(cand)
                break
    return found


# A stored value that must be one bare name, such as an exception's type.
_BARE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")


def bare_name(value) -> str | None:
    """``value`` when it is one bare name (letters, digits, ``.``, ``_``,
    ``+``, ``-``, at most 64), else None."""
    return value if isinstance(value, str) and _BARE_NAME.fullmatch(value) else None


# ---------------------------------------------------------------- failures (F4, unchanged)

NOISE = re.compile(r"(permission|user doesn.t want|rejected|interrupted|command not found|rate limit|overloaded|blocked: sleep|tool_use_error|no longer exists|tabs_context|browser-internal|leave site|screenshot|monitor failed|shell cwd was reset|this session is isolated)", re.I)
ERRPAT = re.compile(r"(error|exception|traceback|failed|fatal|cannot|can't|not found|no such|denied|refused|timeout|timed out|invalid|unexpected|missing|undefined|assert|panic|abort|conflict|rejected|unauthorized|forbidden|does not exist|already exists|not a git|unbound|syntax|typeerror|keyerror|valueerror|attributeerror|importerror|modulenotfound)", re.I)
HEX = re.compile(r"\b[0-9a-f]{6,}\b"); NUM = re.compile(r"\d+"); PATH = re.compile(r"(/[\w.\-@]+){2,}"); WS = re.compile(r"\s+")  # noqa: E702
QUOTED = re.compile(r"'[^']{0,80}'|\"[^\"]{0,80}\"")


def norm(line):
    l = line.strip().lower(); l = PATH.sub("<path>", l); l = HEX.sub("<hex>", l); l = QUOTED.sub("<q>", l); l = NUM.sub("<n>", l); l = WS.sub(" ", l)  # noqa: E702,E741
    return l[:110]


def diag_line(t):
    lines = [ln for ln in t.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.lower().startswith("exit code")]
    if not lines:
        return ""
    for ln in reversed(lines):
        if ERRPAT.search(ln) and len(ln.strip()) > 12:
            return ln
    return lines[-1]
# --- end of the F4 code ---


def failure_signature(error_text: str) -> str | None:
    """The F4 signature of a failed call's output, or None for F4 noise: the
    replay's rule, unchanged. The store keeps only its keyed hash."""
    ln = diag_line(error_text or "")
    if not ln or NOISE.search(ln):
        return None
    sig = norm(ln)
    return sig if len(sig) >= 15 else None


_HOOK_BLOCK = re.compile(r"(?i)(?:pre|post)tooluse(?::[\w*]+)? hook (?:error|blocked)|\bhook\b.{0,40}\b(?:blocked|denied)")
_INPUT_REJECTED = re.compile(r"(?i)inputvalidationerror|does not match this schema|invalid tool input|tool_use_error")


def failure_kind(error_text: str) -> str:
    """A hook that refused the call, a tool input the tool rejected, or a real failure."""
    if _HOOK_BLOCK.search(error_text or ""):
        return "hook block"
    if _INPUT_REJECTED.search(error_text or ""):
        return "tool input rejected"
    return "command or tool failure"


_EXIT_LINE = re.compile(r"^\s*exit code (-?\d+)\s*$", re.I)


def exit_code(error_text: str) -> int | None:
    first = (error_text or "").lstrip().split("\n", 1)[0]
    m = _EXIT_LINE.match(first)
    return int(m.group(1)) if m else None

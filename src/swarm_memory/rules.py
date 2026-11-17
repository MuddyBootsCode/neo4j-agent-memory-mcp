"""Path, signature and record rules, ported from the Phase 1 replay.

The hook cannot import ``experiments/replay/replay.py``: that script pulls in the
old package and would cost startup time on every tool call. The functions here
are copies, and ``tests/test_swarm_memory.py`` checks them against the replay on
the same inputs so the two cannot drift apart.

Four rules differ from the replay, each because the hook runs live:

* A file counts when it exists on disk now. The replay checked ``git ls-files``
  plus history.
* The repo comes from reading ``.git`` and the git dir's ``commondir`` file, with
  no git subprocess.
* In a shell command, a relative path after ``cd <dir>`` also resolves from that
  directory. The replay could only match such a path as a unique suffix.
* Signatures pass through ``scrub()`` as well, so a long token never reaches the
  store. That changes a signature only when its error line holds a token of 24 or
  more letters, digits, ``_`` or ``-``.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import shlex

TEAM_KEYS = ("MUD", "GRA")
FILE_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"})
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
SHELL_TOOLS = frozenset({"Bash", "PowerShell"})
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

    ``worktree`` is the real path of the checkout root. ``rel`` is relative to
    that root, so the same file in two worktrees of one repo gets one ``rel``.
    """
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


# ---------------------------------------------------------------- tickets

_TICKET = re.compile(r"\b(?:%s)-\d+\b" % "|".join(TEAM_KEYS))
_TICKET_ANY_CASE = re.compile(r"\b(?:%s)-\d+\b" % "|".join(TEAM_KEYS), re.I)


def prompt_tickets(text: str) -> list[str]:
    """Ticket keys a prompt names, upper case only (replay Limit 7)."""
    out: list[str] = []
    for t in _TICKET.findall(text or ""):
        if t not in out:
            out.append(t)
    return out


def branch_ticket(name: str | None) -> str | None:
    """The first ticket key in a branch name, which is usually lower case."""
    m = _TICKET_ANY_CASE.search(name or "")
    return m.group(0).upper() if m else None


# ---------------------------------------------------------------- shell commands (replay rules)

_SHELL_OPS = {"&&", "||", ";", "|", "&", ";;", "|&", "(", ")"}
_SKIP_HEADS = {"cd", "export", "source", ".", "set", "pushd", "popd", "unset", "true", ":", "trap", "ulimit"}
_WRAPPERS = {"sudo", "time", "env", "command", "nohup", "exec", "npx", "bunx", "pnpx", "xargs", "timeout"}
_RUNNERS = {("uv", "run"), ("poetry", "run"), ("pnpm", "exec"), ("npm", "exec"), ("pipenv", "run"),
            ("bundle", "exec"), ("agent-test-budget", "run"), ("yarn", "dlx"), ("pdm", "run"),
            ("hatch", "run"), ("rye", "run")}
_SHELLS = {"bash", "zsh", "sh"}
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_INTERPRETERS = re.compile(r"(?:python[\d.]*|node|bash|sh|zsh|tsx|ts-node|deno|bun|ruby|perl)$")


def _segments(command: str) -> list[list[str]]:
    segs: list[list[str]] = []
    for line in command.replace("\\\n", " ").splitlines():
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        try:
            toks = list(lex)
        except ValueError:
            toks = line.split()
        cur: list[str] = []
        for t in toks:
            if t in _SHELL_OPS:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            segs.append(cur)
    return segs


def _strip_wrappers(toks: list[str]) -> list[str]:
    changed = True
    while toks and changed:
        changed = False
        head = os.path.basename(toks[0])
        if head in _WRAPPERS:
            toks, changed = toks[1:], True
        elif len(toks) >= 2 and (head, toks[1]) in _RUNNERS:
            toks, changed = toks[2:], True
        elif re.fullmatch(r"python[\d.]*", head) and len(toks) >= 3 and toks[1] == "-m":
            toks, changed = toks[2:], True
        if changed:
            while toks and (toks[0].startswith("-") or _ENV_ASSIGN.match(toks[0])):
                toks = toks[1:]
    return toks


def cmd_prefix(command: str) -> str:
    """Program plus first non-flag argument of the first command that is not
    a cd, export or similar; wrappers such as ``uv run`` are stripped."""
    if not isinstance(command, str) or not command.strip():
        return ""
    first_head = ""
    for toks in _segments(command):
        while toks and _ENV_ASSIGN.match(toks[0]):
            toks = toks[1:]
        if not toks:
            continue
        head = os.path.basename(toks[0])
        first_head = first_head or head
        if head in _SKIP_HEADS:
            continue
        toks = _strip_wrappers(toks)
        if not toks:
            continue
        prog, args = os.path.basename(toks[0]), toks[1:]
        if prog in _SHELLS and "-c" in args and args.index("-c") + 1 < len(args):
            inner = cmd_prefix(args[args.index("-c") + 1])
            if inner:
                return inner
        if prog == "git":
            kept, skip = [], False
            for a in args:
                if skip:
                    skip = False
                elif a in ("-C", "-c"):
                    skip = True
                else:
                    kept.append(a)
            args = kept
        rest = [a for a in args if not a.startswith("-")]
        return " ".join([prog] + [r[:60] for r in rest[:1]])
    return first_head


def command_program(command: str) -> str:
    """The program a command runs, without its arguments."""
    prefix = cmd_prefix(command)
    return prefix.split(" ", 1)[0] if prefix else ""


def prefix_key(command: str) -> str:
    """An opaque key for the command prefix. Failures are matched to later
    successes on it, so the prefix itself, which can hold a secret such as a
    connection string, is never stored."""
    return hashlib.sha256(cmd_prefix(command).encode("utf-8", "replace")).hexdigest()[:16]


_TOKEN_SPLIT = re.compile(r"[^\s`'\"()<>\[\]{},;|*!?=]+")
_HAS_EXT = re.compile(r"\.[A-Za-z][\w]{0,7}$")


def _clean_token(tok: str) -> str:
    tok = tok.strip(".,:;)]}>'\"`")
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


def named_files(text: str, bases: list[str], exclude: set | None = None) -> list[str]:
    """Absolute paths of existing files that a prompt or command names.

    A token counts when it has a ``/`` or a file extension and resolves to a
    regular file, absolute or relative to one of ``bases`` in order.
    """
    found: list[str] = []
    exclude = exclude or set()
    for n, raw in enumerate(_TOKEN_SPLIT.findall(text or "")):
        if n >= MAX_TOKENS or len(found) >= MAX_NAMED_FILES:
            break
        tok = _clean_token(raw)
        if not tok or not ("/" in tok or _HAS_EXT.search(tok)):
            continue
        for cand in _candidates(tok, bases):
            if os.path.isfile(cand):
                if cand not in found and cand not in exclude:
                    found.append(cand)
                break
    return found


def _command_bases(command: str, cwd: str) -> list[str]:
    """The cwd, then each directory a ``cd`` in the command moves to."""
    bases = [cwd]
    current = cwd
    for toks in _segments(command):
        if len(toks) >= 2 and toks[0] == "cd":
            current = os.path.normpath(os.path.join(current, os.path.expanduser(toks[1])))
            if current not in bases:
                bases.append(current)
    return bases


def executed_files(command: str, bases: list[str]) -> set[str]:
    """Files a command runs as its program (``./x.sh``, ``node x.mjs``,
    ``bash x.sh``), which is using a tool rather than working on the file."""
    out: set[str] = set()
    for toks in _segments(command or ""):
        while toks and _ENV_ASSIGN.match(toks[0]):
            toks = toks[1:]
        toks = _strip_wrappers(toks)
        if not toks:
            continue
        prog = toks[0]
        if _INTERPRETERS.match(os.path.basename(prog)):
            args = [a for a in toks[1:] if not a.startswith("-")]
            if not args:
                continue
            prog = args[0]
        prog = _clean_token(prog)
        if not prog or not ("/" in prog or _HAS_EXT.search(prog)):
            continue
        for cand in _candidates(prog, bases):
            if os.path.isfile(cand):
                out.add(cand)
                break
    return out


def command_files(command: str, cwd: str) -> list[str]:
    """Absolute paths of existing files a shell command names, except a
    script the command runs as its program (MUD-850 D18, replay rule)."""
    if not isinstance(command, str) or not command.strip() or not cwd:
        return []
    try:
        bases = _command_bases(command, cwd)
        executed = executed_files(command, bases)
    except Exception:
        bases, executed = [cwd], set()
    return named_files(command, bases, exclude=executed)


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


_SECRETISH = re.compile(r"[A-Za-z0-9_\-]{24,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def scrub(text: str) -> str:
    """Drop anything token-shaped (replay rule)."""
    return _EMAIL.sub("<email>", _SECRETISH.sub("<tok>", text))


def failure_signature(error_text: str) -> str | None:
    """The F4 signature of a failed call's output, or None for F4 noise."""
    ln = diag_line(error_text or "")
    if not ln or NOISE.search(ln):
        return None
    sig = norm(ln)
    return scrub(sig) if len(sig) >= 15 else None


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
_SECRET_VALUE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|auth[_-]?token|authorization)"
    r"(\s*[=:]\s*)\S+")
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_URL_CREDENTIALS = re.compile(r"(\b[a-z][\w+.-]*://)[^/\s:@]+:[^/\s@]+@", re.I)
HEAD_CHARS = 160


def exit_code(error_text: str) -> int | None:
    first = (error_text or "").lstrip().split("\n", 1)[0]
    m = _EXIT_LINE.match(first)
    return int(m.group(1)) if m else None


def error_head(error_text: str, limit: int = HEAD_CHARS) -> str:
    """A short redacted head of a failed call's output: the ``Exit code`` line
    dropped, whitespace flattened, the home directory shortened to ``~``, and
    credentials, long tokens and emails replaced."""
    lines = [ln for ln in (error_text or "").splitlines() if ln.strip()]
    if lines and _EXIT_LINE.match(lines[0]):
        lines = lines[1:]
    head = " ".join(" ".join(lines).split())[: limit * 3]
    home = os.path.expanduser("~")
    if home and home != "/":
        head = head.replace(home, "~")
    head = _URL_CREDENTIALS.sub(r"\1<redacted>@", head)
    head = _SECRET_VALUE.sub(lambda m: m.group(1) + m.group(2) + "<redacted>", head)
    head = _BEARER.sub("bearer <redacted>", head)
    return scrub(head)[:limit]

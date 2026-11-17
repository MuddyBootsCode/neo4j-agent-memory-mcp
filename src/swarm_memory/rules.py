"""Path, signature and record rules, ported from the Phase 1 replay.

The hook cannot import ``experiments/replay/replay.py``: that script pulls in the
old package and would cost startup time on every tool call. The functions here
are copies, and ``tests/test_swarm_memory.py`` checks them against the replay on
the same inputs so the two cannot drift apart.

Three rules differ from the replay, each because the hook runs live:

* A file counts when it exists on disk now. The replay checked ``git ls-files``
  plus history.
* The repo comes from reading ``.git`` and the git dir's ``commondir`` file, with
  no git subprocess.
* In a shell command, each segment's relative paths resolve from the directory
  a ``cd`` earlier in the command moved to, as the shell would. The replay
  could only match such a path as a unique suffix.

The failure signature is the replay's, unchanged. The store keeps only its
keyed hash (``store.signature_hash``), so no text from a tool's output is kept.
"""

from __future__ import annotations

import functools
import os
import re
import shlex

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
_JS_INLINE = {"-e", "--eval", "-p", "--print"}
_INLINE_CODE = {"python": {"-c"}, "node": _JS_INLINE, "bun": _JS_INLINE, "tsx": _JS_INLINE, "ts-node": _JS_INLINE,
                "ruby": {"-e"}, "perl": {"-e", "-E"}}


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
    return _unwrap(toks)[0]


def _unwrap(toks: list[str]) -> tuple[list[str], bool]:
    """``toks`` without its wrappers, and whether the word now first came
    right after a wrapper option, so it may be that option's value
    (``sudo -u alice id``) rather than the program."""
    after_option = False
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
            after_option = False
            while toks and (toks[0].startswith("-") or _ENV_ASSIGN.match(toks[0])):
                after_option = toks[0].startswith("-") and toks[0] != "--"
                toks = toks[1:]
            if head == "timeout" and toks and _DURATION.match(toks[0]):
                toks, after_option = toks[1:], False
    return toks, after_option


_DURATION = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")
_SHELL_OPT_VALUE = {"-o", "-O", "+o", "+O", "--rcfile", "--init-file"}
_SHORT_OPTS = re.compile(r"-[A-Za-z]+")


def _shell_command_string(args: list[str]) -> str | None:
    """The command string a shell runs with ``-c``, also when ``c`` sits in an
    option cluster such as ``-lc`` or ``-ec``: the first argument after the
    options. None when the shell runs a script or reads its input."""
    want, skip = False, False
    for a in args:
        if skip:
            skip = False
        elif a in _SHELL_OPT_VALUE:
            skip = True
        elif a.startswith(("-", "+")) and len(a) > 1:
            want = want or bool(_SHORT_OPTS.fullmatch(a) and "c" in a)
        else:
            return a if want else None
    return None


def program_tokens(command: str) -> list[str]:
    """Tokens a command runs as programs rather than works on: each segment's
    first word, its program once wrappers such as ``env``, ``timeout 60`` or
    ``uv run`` are stripped, and the script an interpreter runs (``node x.mjs``,
    ``bash x.sh``). A shell's ``-c '...'`` string is parsed as commands, and
    code passed inline (``python -c``, ``node -e``) names no script. A bare
    name and an absolute path such as ``/opt/homebrew/bin/python3`` are alike."""
    out: list[str] = []
    for toks in _segments(command or ""):
        while toks and _ENV_ASSIGN.match(toks[0]):
            toks = toks[1:]
        if not toks:
            continue
        out.append(toks[0])
        toks = _strip_wrappers(toks)
        if not toks:
            continue
        out.append(toks[0])
        prog, args = os.path.basename(toks[0]), toks[1:]
        inner = _shell_command_string(args) if prog in _SHELLS else None
        if inner is not None:
            out += program_tokens(inner)
        elif _INTERPRETERS.match(prog):
            inline = _INLINE_CODE.get(re.sub(r"[\d.]+$", "", prog), set())
            script = next((a for a in args if a in inline or not a.startswith("-")), None)
            if script and script not in inline:
                out.append(script)
    return out


# A stored program is one bare name. Anything else, such as a whole command a
# wrapper option carried (a curl command and its --user value, after env -S), is
# not stored at all.
_BARE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")


def bare_name(value) -> str | None:
    """``value`` when it is one bare name (letters, digits, ``.``, ``_``,
    ``+``, ``-``, at most 64), else None."""
    return value if isinstance(value, str) and _BARE_NAME.fullmatch(value) else None


def command_program(command: str) -> str:
    """The program a command runs, without its arguments: the first command
    that is not a cd, export or similar, with wrappers such as ``uv run``
    stripped and a shell's ``-c`` string read as commands of its own. Empty
    when that is not one bare name, or when the word follows a wrapper option
    and may be its value, so a wrapper argument is never stored."""
    return bare_name(_first_program(command)) or ""


def _first_program(command: str) -> str:
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
        toks, after_option = _unwrap(toks)
        if not toks:
            continue
        if after_option:
            return ""
        prog = os.path.basename(toks[0])
        inner = _shell_command_string(toks[1:]) if prog in _SHELLS else None
        if inner is not None and _first_program(inner):
            return _first_program(inner)
        return prog
    return first_head


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


def executed_files(command: str, bases: list[str]) -> set[str]:
    """Existing files a command runs as a program (``program_tokens``), which
    is using a tool rather than working on the file."""
    out: set[str] = set()
    for tok in program_tokens(command):
        tok = _clean_token(tok)
        if not tok or not ("/" in tok or _HAS_EXT.search(tok)):
            continue
        for cand in _candidates(tok, bases):
            if os.path.isfile(cand):
                out.add(cand)
                break
    return out


MAX_SEGMENTS = 200
MAX_SHELL_DEPTH = 3


def _shell_payload(toks: list[str]) -> str | None:
    """The ``-c`` string of a segment that runs a shell, wrappers stripped."""
    while toks and _ENV_ASSIGN.match(toks[0]):
        toks = toks[1:]
    toks = _strip_wrappers(toks)
    if not toks or os.path.basename(toks[0]) not in _SHELLS:
        return None
    return _shell_command_string(toks[1:])


# git options before the subcommand that take a separate value.
_GIT_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"}


def _segment_dir(toks: list[str], current: str) -> str:
    """The directory a segment resolves its paths from: ``current``, or for
    git each ``-C <path>`` before the subcommand in turn, a relative one from
    the one before, as git does. It holds for that segment only."""
    while toks and _ENV_ASSIGN.match(toks[0]):
        toks = toks[1:]
    toks = _strip_wrappers(toks)
    if not toks or os.path.basename(toks[0]) != "git":
        return current
    here, i = current, 1
    while i < len(toks) and toks[i].startswith("-"):
        if toks[i] in _GIT_VALUE_OPTS and i + 1 < len(toks):
            if toks[i] == "-C" and toks[i + 1]:
                here = os.path.normpath(os.path.join(here, os.path.expanduser(toks[i + 1])))
            i += 2
        else:
            i += 1
    return here


def command_files(command: str, cwd: str, _depth: int = 0) -> list[str]:
    """Absolute paths of existing files a shell command names, except what it
    runs as a program (MUD-850 D18, replay rule). Each segment resolves its
    relative paths from the directory the command's ``cd`` had moved to by
    then, as the shell would, and a git segment from where its ``-C`` options
    point. A shell's ``-c`` string is read the same way, from the directory
    that shell starts in, so its ``cd`` moves only its own commands."""
    if not isinstance(command, str) or not command.strip() or not cwd:
        return []
    found: list[str] = []
    current = cwd
    for toks in _segments(command)[:MAX_SEGMENTS]:
        if toks and toks[0] == "cd":
            if len(toks) >= 2:
                current = os.path.normpath(os.path.join(current, os.path.expanduser(toks[1])))
            continue
        inner = _shell_payload(toks) if _depth < MAX_SHELL_DEPTH else None
        if inner is not None:
            names = command_files(inner, current, _depth + 1)
        else:
            text = shlex.join(toks)
            here = _segment_dir(toks, current)
            try:
                executed = executed_files(text, [here])
            except Exception:
                executed = set()
            names = named_files(text, [here], exclude=executed)
        for f in names:
            if f not in found:
                found.append(f)
                if len(found) >= MAX_NAMED_FILES:
                    return found
    return found


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

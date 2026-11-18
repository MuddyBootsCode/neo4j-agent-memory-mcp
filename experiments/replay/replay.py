#!/usr/bin/env python3
"""Phase 1 of MUD-850: replay past Claude Code transcripts and count how often
each injection moment would have fired. It builds no hook and no store.

Run from the repository root:

    python3 experiments/replay/replay.py                 # sessions since 2026-08-01
    python3 experiments/replay/replay.py --since 2026-10-01 --run oct

Read-only on ~/.claude/projects and on every repo the sessions ran in (git
ls-files, git log, git cat-file and plain file reads). It runs no code from
those repos: ontology files are parsed with the installed ontology-observer
plugin's library, never with a copy inside the repo. Output goes to
experiments/replay/results/<run>/, which is gitignored: report.md,
summary.json, runs.json and examples.json. Nothing leaves the laptop.

Sessions. A record belongs to the session its ``sessionId`` names, whichever
transcript holds it: resumed and forked transcripts repeat earlier records,
and subagent transcripts (``<session>/subagents/**``) carry their parent's id.
Each tool call and prompt is counted once. The repo and worktree come from
the session's cwd: the path before ``/.claude/worktrees/<name>`` (or the cwd
itself) walked up to the directory that holds ``.git``. The main checkout
counts as one worktree. An agent is a session, and two agents are different
when they ran in different worktrees. Each touch is attributed to the worktree
of the file it touched. A session whose cwd moves into another worktree of its
repo is left out of runs, since a run places each session in one worktree. So
is a session whose cwd moves into another repo: its events are not split by
repo, so its work in one repo would count as overlap and failures in the other.
The summary and the report count the two apart, and a session that did both
counts as having moved to another repo.

Runs. A session's activity is split into spans wherever two consecutive
events are more than ``--idle-gap`` minutes apart. Two sessions in different
worktrees of one repo are linked when spans of theirs intersect, so a session
that sat idle while another worked does not count as overlap. A run is a
connected set of linked sessions.

Moment 3 (with a tool result). A touch by agent B is a hit when another agent
A in the same run had touched the same file, or a related file, in the window
before it. Touches are timed by their result, when a hook would see them, and
a call whose result never came back is left out. A touch is a file tool call
(Read, Edit, Write, MultiEdit, NotebookEdit) on a repo file. Shell commands are
not read, since what a command touched cannot be told from its text. Related
is tested three ways: same directory, import
neighbour (one hop either way, static parse of Python and JS/TS imports), and
same ontology rule (a gate in ``ontology/*.md`` naming the file, or
``ontology/gates/*.json``). Paths are repo-relative, so worktrees compare,
including linked worktrees outside the main checkout, which a file tool call
reaches through the worktree's ``.git`` file. A symlink inside a checkout
counts as the file it points to. Relations come from
today's files and branch tips, not the version each agent saw, so related-file
counts can drift as repos change. Same-file counts do not depend on them, and
MUD-850 D16 logs related files without delivering them.

Moment 4 (with a failure). Failures are the ``error_steps()`` rules (an
``is_error`` result that is not a blocked or rejected call) and the signature
is the F4 normaliser (MUD-478 attachment ``failure-recurrence-baseline-
2026-09-09.py``), both reused unchanged. A failure is a hit when another agent
in the same run had failed with that signature within the window before. Only
that exact part is measured: whether an agent got past a failure is not, since
no rule for it held up (MUD-850). Hook blocks and rejected tool inputs are
counted apart from command and tool failures. A session whose cwd moved to
another repo is left out of every repo's failure counts and of the signature
index that Moment 2 matches pasted errors against, as it is left out of runs:
which repo each of its failures belongs to is not tracked. The summary counts
those sessions and their failures.

Moment 2 (with the prompt). A prompt is a hit when it contains a path that
exists in the repo, a pasted error line, or a ticket key such as ``ABC-123``.
Ticket prefixes are a local setting and never part of the repo:
``--ticket-keys``, else ``$SWARM_MEMORY_TICKET_KEYS``, else ``ticket_keys`` in
the ``config.json`` beside the swarm_memory store. With none set, no prompt
names a ticket. Human prompts and the briefs another session writes when it
starts one are counted separately. A session whose cwd moved to another repo
is left out of every Moment 2 count, as it is left out of runs and failure
counts: which repo and directory each prompt was written in is not tracked.
The summary counts those sessions and their prompts and briefs.

A hit means the trigger fired, not that a delivery would have helped (MUD-850
R15).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import functools
import glob
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone

# Importing the ontology libraries below must not leave __pycache__ in other repos.
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from agent_memory_mcp.hook.capture_hook import (  # noqa: E402  (the error_steps() rules)
    _NOT_AN_ATTEMPT_RE,
    _block_text,
    error_steps,
    normalize_repo_path,
)

SINCE_DEFAULT = "2026-08-01"
IDLE_GAP_S = 15 * 60
WINDOWS_MIN = (15, 60, 240)
TICKET_KEYS_ENV = "SWARM_MEMORY_TICKET_KEYS"
TICKET_KEYS: tuple = ()  # set by main() from load_ticket_keys()
FILE_TOOLS = {"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"}
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
WORKTREE_MARKERS = ("/.claude/worktrees/", "/.worktrees/")
PLUGIN_ONTOLOGY_LIB = os.path.expanduser(
    "~/.claude/plugins/cache/muddybootscode-odscai-2026/ontology-observer/0.1.0/skills/ontology-driven-dev/scripts"
)
# The first pass on 2026-11-16 (MUD-850 Finding 15).
FIRST_PASS = {"runs": 54, "sessions": 200, "repo_paths": 9}

# --- F4 recurrence script, unchanged (MUD-478 attachment failure-recurrence-baseline-2026-09-09.py) ---
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
    # prefer last error-looking line (tracebacks diagnose at the tail), else first error-looking, else last line
    for ln in reversed(lines):
        if ERRPAT.search(ln) and len(ln.strip()) > 12:
            return ln
    return lines[-1]
# --- end of the F4 code ---


def failure_signature(error_text: str) -> str | None:
    """The F4 signature of a failed call's output, or None for F4 noise."""
    ln = diag_line(error_text or "")
    if not ln or NOISE.search(ln):
        return None
    sig = norm(ln)
    return sig if len(sig) >= 15 else None


def parse_ts(value) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------- runs

def activity_spans(times, gap: float = IDLE_GAP_S) -> list[tuple[float, float]]:
    """Merge event times into spans; a gap longer than ``gap`` starts a new one."""
    spans: list[tuple[float, float]] = []
    for t in sorted(times):
        if spans and t - spans[-1][1] <= gap:
            spans[-1] = (spans[-1][0], t)
        else:
            spans.append((t, t))
    return spans


def group_runs(sessions) -> list[list[str]]:
    """Connected sets of sessions whose spans intersect across worktrees of one repo.

    ``sessions``: dicts with sid, repo, worktree, spans. Returns sorted sid
    lists, only for sets that hold two or more worktrees.
    """
    parent = {s["sid"]: s["sid"] for s in sessions}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_repo = collections.defaultdict(list)
    for s in sessions:
        by_repo[s["repo"]].append(s)
    for members in by_repo.values():
        items = sorted((a, b, s["sid"], s["worktree"]) for s in members for a, b in s["spans"])
        active: list[tuple[float, str, str]] = []
        for a, b, sid, wt in items:
            active = [x for x in active if x[0] >= a]
            for _end, sid2, wt2 in active:
                if wt2 != wt:
                    ra, rb = find(sid), find(sid2)
                    if ra != rb:
                        parent[ra] = rb
            active.append((b, sid, wt))
    comps = collections.defaultdict(list)
    for s in sessions:
        comps[find(s["sid"])].append(s)
    runs = [sorted(m["sid"] for m in members) for members in comps.values()
            if len({m["worktree"] for m in members}) >= 2]
    return sorted(runs)


# ---------------------------------------------------------------- failures

_HOOK_BLOCK = re.compile(r"(?i)(?:pre|post)tooluse(?::[\w*]+)? hook (?:error|blocked)|\bhook\b.{0,40}\b(?:blocked|denied)")
_INPUT_REJECTED = re.compile(r"(?i)inputvalidationerror|does not match this schema|invalid tool input|tool_use_error")


def failure_kind(error_text: str) -> str:
    """A hook that refused the call, a tool input the tool rejected, or a real failure."""
    if _HOOK_BLOCK.search(error_text or ""):
        return "hook block"
    if _INPUT_REJECTED.search(error_text or ""):
        return "tool input rejected"
    return "command or tool failure"


def repeat_failure_hits(failures: list[dict], window_s: float | None = None) -> list[dict]:
    """Failures whose signature another worktree in the run had already hit.

    ``failures``: dicts with ts, sid, worktree and sig. With ``window_s``,
    only earlier failures within that many seconds count. The earlier failure
    reported is the latest one.
    """
    by_sig = collections.defaultdict(list)
    for f in sorted(failures, key=lambda f: f["ts"]):
        by_sig[f["sig"]].append(f)
    hits = []
    for group in by_sig.values():
        for k, f in enumerate(group):
            earlier = [g for g in group[:k] if g["worktree"] != f["worktree"] and g["ts"] < f["ts"]
                       and (window_s is None or f["ts"] - g["ts"] <= window_s)]
            if earlier:
                hits.append({"failure": f, "earlier": earlier[-1]})
    return sorted(hits, key=lambda h: h["failure"]["ts"])


# ---------------------------------------------------------------- moment 3

def call_ran(call: dict) -> bool:
    """Whether a call ran at all. One a hook refused before it started, or
    whose input the tool rejected, touched nothing. A command that ran and
    failed did run."""
    if call.get("blocked"):
        return False
    return call.get("ok") is not False or failure_kind(call.get("error") or "") == "command or tool failure"


def touch_ts(call: dict) -> float | None:
    """When a hook would have seen a call's touch: when its result came back.
    A call with no result never reaches a hook, so it is no touch."""
    return call.get("rts")


def file_window_hits(events: list[dict], window_s: float, related=None) -> list[dict]:
    """Events that another session's touch of the same (or a related) file, in
    another worktree, preceded within the window. Only a strictly earlier
    touch counts, so two touches at one moment do not hit each other.

    ``events``: dicts with ts, sid, worktree, file, all from one run.
    ``related``: None for the same file, else a function from a file to the
    set of files related to it. One hit per event; ``other`` is the most
    recent qualifying touch.
    """
    evs = sorted(events, key=lambda e: e["ts"])
    by_file = collections.defaultdict(list)
    for k, e in enumerate(evs):
        by_file[e["file"]].append((e["ts"], k))
    hits = []
    for e in evs:
        targets = related(e["file"]) if related else (e["file"],)
        best = None
        for f in targets:
            lst = by_file.get(f)
            if not lst:
                continue
            lo = bisect.bisect_left(lst, (e["ts"] - window_s, -1))
            hi = bisect.bisect_left(lst, (e["ts"], -1))
            for _ts, k2 in reversed(lst[lo:hi]):
                o = evs[k2]
                if o["worktree"] == e["worktree"] or o["sid"] == e["sid"]:
                    continue
                if best is None or o["ts"] > best["ts"]:
                    best = o
                break
        if best is not None:
            hits.append({"event": e, "other": best, "gap_s": e["ts"] - best["ts"]})
    return hits


def edit_after_edit_calls(events: list[dict], window_s: float) -> set:
    """Edit calls that another worktree's edit of the same file preceded
    within the window, whatever reads came between: the hook's same-file
    edit rule. ``events`` also carry ``edit`` and ``call``."""
    return {h["event"]["call"] for h in file_window_hits([e for e in events if e["edit"]], window_s)}


def same_directory_relation(files):
    by_dir = collections.defaultdict(set)
    for f in files:
        by_dir[os.path.dirname(f)].add(f)
    return lambda f: by_dir.get(os.path.dirname(f), set()) - {f}


def build_py_index(files) -> dict:
    """Module-path suffix (tuple of parts) to the .py files that end with it."""
    idx = collections.defaultdict(set)
    for f in files:
        if not f.endswith(".py"):
            continue
        parts = f[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for k in range(1, len(parts) + 1):
            idx[tuple(parts[-k:])].add(f)
    return idx


_PY_FROM = re.compile(r"^[ \t]*from[ \t]+(\.*)([\w.]*)[ \t]+import[ \t]+(\([^)]*\)|[^\n#]+)", re.M)
_PY_IMPORT = re.compile(r"^[ \t]*import[ \t]+([\w.]+(?:[ \t]*,[ \t]*[\w.]+)*)", re.M)


def py_imports(rel: str, text: str, py_index: dict, files) -> set[str]:
    """Repo files a Python file imports. A one-part absolute import resolves
    only next to the importer; a dotted one by suffix, if at most 3 files match."""
    here = os.path.dirname(rel)

    def resolve_abs(mod):
        parts = tuple(p for p in mod.split(".") if p)
        if not parts:
            return set()
        if len(parts) == 1:
            return {f for f in (os.path.join(here, parts[0] + ".py"), os.path.join(here, parts[0], "__init__.py"))
                    if f in files}
        found = py_index.get(parts, set())
        return set(found) if len(found) <= 3 else set()

    def resolve_rel(dots, mod):
        base = here
        for _ in range(dots - 1):
            base = os.path.dirname(base)
        path = os.path.join(base, *mod.split(".")) if mod else base
        return {f for f in (path + ".py", os.path.join(path, "__init__.py")) if f in files}

    out: set[str] = set()
    for m in _PY_FROM.finditer(text):
        dots, mod = len(m.group(1)), m.group(2)
        names = [n.strip().split(" as ")[0].strip() for n in m.group(3).strip("()").replace("\n", ",").split(",")]
        names = [n for n in names if re.fullmatch(r"\w+", n)]
        if dots:
            out |= resolve_rel(dots, mod)
            for n in names:
                out |= resolve_rel(dots, f"{mod}.{n}" if mod else n)
        else:
            out |= resolve_abs(mod)
            for n in names:
                out |= resolve_abs(f"{mod}.{n}")
    for m in _PY_IMPORT.finditer(text):
        for mod in m.group(1).split(","):
            out |= resolve_abs(mod.strip())
    out.discard(rel)
    return out


_JS_SPEC = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*|^\s*import\s+)['"]([^'"\n]+)['"]""", re.M)
_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".d.ts", ".vue", ".svelte", ".json")
JS_SOURCE = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".vue", ".svelte")


def _resolve_js_path(path: str, files) -> str | None:
    path = os.path.normpath(path)
    cands = [path] + [path + e for e in _JS_EXTS] + [os.path.join(path, "index" + e) for e in _JS_EXTS]
    stem, ext = os.path.splitext(path)
    if ext in (".js", ".jsx", ".mjs", ".cjs"):
        cands += [stem + e for e in (".ts", ".tsx", ".mts", ".cts")]
    for c in cands:
        if c in files:
            return c
    return None


def _resolve_package(spec: str, packages: dict, files) -> str | None:
    parts = spec.split("/")
    name, sub = ("/".join(parts[:2]), parts[2:]) if spec.startswith("@") else (parts[0], parts[1:])
    root = packages.get(name)
    if root is None:
        return None
    bases = [os.path.join(root, *sub), os.path.join(root, "src", *sub)] if sub else \
        [os.path.join(root, "src", "index"), os.path.join(root, "index")]
    for base in bases:
        target = _resolve_js_path(base, files)
        if target:
            return target
    return None


def js_imports(rel: str, text: str, files, packages: dict) -> set[str]:
    """Repo files a JS/TS file imports: relative specifiers, ``@/`` and ``~/``
    aliases to the nearest ``src/``, and workspace packages by package.json name."""
    here = os.path.dirname(rel)
    out: set[str] = set()
    for m in _JS_SPEC.finditer(text):
        spec = m.group(1)
        target = None
        if spec.startswith("."):
            target = _resolve_js_path(os.path.join(here, spec), files)
        elif spec.startswith(("@/", "~/")):
            d = here
            while True:
                target = _resolve_js_path(os.path.join(d, "src", spec[2:]), files)
                if target or not d:
                    break
                d = os.path.dirname(d)
        else:
            target = _resolve_package(spec, packages, files)
        if target and target != rel:
            out.add(target)
    return out


# ---------------------------------------------------------------- moment 2

_SYNTHETIC_PREFIX_RE = re.compile(  # experiments/golden/lib.py, for records that predate ``origin``
    r"^<(task-notification|system-reminder|command-message|command-name|"
    r"local-command-stdout|user-prompt-submit-hook|automated-reminder|"
    r"command-args|bash-input|bash-stdout|bash-stderr|"
    r"[a-z0-9-]+-command)\b"
)
_REMINDER_BLOCK = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _content_text(content) -> str | None:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        if not content or any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        texts = [b.get("text", "").strip() for b in content if isinstance(b, dict) and b.get("type") == "text"]
        texts = [t for t in texts if t]
        return "\n".join(texts) if texts else None
    return None


def human_prompt(record: dict) -> str | None:
    """Text Michael typed or pasted, or None. Records carrying origin,
    promptSource or turnOrigin must say ``origin.kind == "human"``."""
    if (record.get("type") != "user" or record.get("isMeta") or record.get("isSidechain")
            or record.get("isCompactSummary")):
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    text = _content_text(message.get("content"))
    if not text:
        return None
    origin = record.get("origin")
    kind = origin.get("kind") if isinstance(origin, dict) else None
    marked = kind is not None or record.get("promptSource") is not None or record.get("turnOrigin") is not None
    if marked and kind != "human":
        return None
    text = _REMINDER_BLOCK.sub("", text).strip()
    if not text or text.startswith(("<bash-input>", "<bash-stdout>", "<bash-stderr>", "[Request interrupted")):
        return None
    if text.startswith("This session is being continued from a previous conversation"):
        return None
    if text.startswith(("<command-name>", "<command-message>")):
        m = re.search(r"<command-args>(.*?)</command-args>", text, re.S)
        return (m.group(1).strip() if m else "") or None
    if not marked and _SYNTHETIC_PREFIX_RE.match(text):
        return None
    return text


_BRIEF_PREFIXES = ("Background (from the session that started this one)",
                   "Task (from the session that started this one)")


def session_brief(record: dict) -> str | None:
    """The task text another Claude session wrote when it started this one,
    which is how a desktop side session or swarm worker gets its prompt."""
    if record.get("type") != "user" or record.get("isMeta") or record.get("isSidechain"):
        return None
    message = record.get("message")
    text = _content_text(message.get("content")) if isinstance(message, dict) else None
    if not text:
        return None
    text = _REMINDER_BLOCK.sub("", text).strip()
    return text if text.startswith(_BRIEF_PREFIXES) else None


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


def build_suffix_index(files) -> dict:
    """Last three path components to the files that end with them."""
    idx = collections.defaultdict(set)
    for f in files:
        idx["/".join(f.split("/")[-3:])].add(f)
    return idx


def prompt_paths(text: str, files, cwd_prefix: str = "", repo_root: str | None = None,
                 suffix_index: dict | None = None, checkout_root: str | None = None) -> list[str]:
    """Repo-relative paths of existing repo files named in a prompt.

    Absolute paths are normalised through worktree markers, or taken relative
    to ``checkout_root``, the session's own checkout, when it is a linked
    worktree outside the main checkout. Relative ones are tried from the repo
    root, then from the session's cwd. A relative path of three or more parts
    that matches no file there counts when it is the unique suffix of one repo
    file.
    """
    found: list[str] = []
    for raw in _TOKEN_SPLIT.findall(text):
        tok = _clean_token(raw)
        if not tok or not ("/" in tok or _HAS_EXT.search(tok)):
            continue
        if tok.startswith(("/", "~")):
            path = os.path.expanduser(tok)
            rel = (repo_rel(path, repo_root) if repo_root else None) or checkout_rel(path, checkout_root)
            cands = [rel] if rel else []
        else:
            cands = [os.path.normpath(tok)]
            if cwd_prefix:
                cands.append(os.path.normpath(os.path.join(cwd_prefix, tok)))
        hit = next((c for c in cands if c in files), None)
        if hit is None and suffix_index is not None and tok.count("/") >= 2 and not tok.startswith(("/", "~")):
            key = "/".join(os.path.normpath(tok).split("/")[-3:])
            matches = [f for f in suffix_index.get(key, ()) if f.endswith("/" + os.path.normpath(tok))]
            hit = matches[0] if len(matches) == 1 else None
        if hit and hit not in found:
            found.append(hit)
    return found


_KEY = re.compile(r"^[A-Z][A-Z0-9]{0,9}$")


def store_config_path() -> str:
    """The ``config.json`` beside the swarm_memory store (MUD-850 Phase 2)."""
    db = os.environ.get("SWARM_MEMORY_DB") or os.path.expanduser("~/.local/share/swarm-memory/observe.sqlite3")
    return os.path.join(os.path.dirname(db), "config.json")


def parse_ticket_keys(keys) -> tuple:
    """Upper-case prefixes of one to ten letters or digits, starting with a letter."""
    if isinstance(keys, str):
        keys = keys.split(",")
    if not isinstance(keys, (list, tuple)):
        return ()
    clean = (k.strip() for k in keys if isinstance(k, str))
    return tuple(dict.fromkeys(k for k in clean if _KEY.match(k)))


def load_ticket_keys(flag: str | None = None) -> tuple:
    """Ticket prefixes from ``--ticket-keys``, else ``$SWARM_MEMORY_TICKET_KEYS``,
    else ``ticket_keys`` in the store's ``config.json``, else none."""
    if flag:
        return parse_ticket_keys(flag)
    env = os.environ.get(TICKET_KEYS_ENV)
    if env:
        return parse_ticket_keys(env)
    try:
        with open(store_config_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return ()
    return parse_ticket_keys(data.get("ticket_keys") if isinstance(data, dict) else None)


@functools.lru_cache(maxsize=None)
def _ticket_re(keys: tuple):
    return re.compile(r"\b(?:%s)-\d+\b" % "|".join(map(re.escape, keys))) if keys else None


def prompt_tickets(text: str, keys=None) -> list[str]:
    """Ticket keys a prompt names, upper case only, in order. ``keys``
    defaults to the ones main() loaded."""
    rx = _ticket_re(tuple(TICKET_KEYS if keys is None else keys))
    out: list[str] = []
    if rx is None:
        return out
    for t in rx.findall(text):
        if t not in out:
            out.append(t)
    return out


_ERR_LINE = re.compile("|".join([
    r"Traceback \(most recent call last\)",
    r"^\s*(?:[\w$]+\.)*[A-Z]\w*(?:Error|Exception)\b(?:\s*\[[\w-]+\])?:",
    r"^\s*(?:error|fatal|panic)(?:\[[\w-]+\])?:\s",
    r"^\s*(?:ERROR|FATAL|FAILED|CRITICAL)\b",
    r"\bnpm ERR!",
    r"\b(?:[Ee]xit code|exited with code|exit status)\s*[1-9]\d*\b",
    r"\b(?:ENOENT|EACCES|EPERM|ECONNREFUSED|EADDRINUSE|ETIMEDOUT|ECONNRESET)\b",
    r"\b(?:command not found|No such file or directory|Permission denied|Segmentation fault)\b",
    r"^\s*at .+\(.+:\d+:\d+\)\s*$",
    r"^\s*File \".+\", line \d+",
    r"[\w./-]+:\d+:\d+: (?:error|warning)\b",
    r"\berror TS\d{3,5}\b",
    r"\b[45]\d\d (?:Bad Request|Unauthorized|Forbidden|Not Found|Internal Server Error|Bad Gateway|"
    r"Service Unavailable|Gateway Timeout)\b",
    r"\b\w+: error: ",
]))


_QUOTE_PREFIX = re.compile(r"^[\s>|│$#•*]+")


def prompt_error_lines(text: str) -> list[str]:
    """Lines of a prompt that look like pasted tool or program output
    reporting an error, without quote or prompt markers in front."""
    out: list[str] = []
    for line in text.splitlines():
        clean = _QUOTE_PREFIX.sub("", line).strip()
        if clean and _ERR_LINE.search(clean) and clean not in out:
            out.append(clean)
    return out


# ---------------------------------------------------------------- transcript scan



def _new_bucket() -> dict:
    return {"times": [], "cwds": [], "branches": collections.Counter(), "calls": [], "prompts": [], "briefs": []}


def scan_transcript(path: str, since_ts: float, want_prompts: bool = True) -> dict:
    """One pass over a transcript: activity times, tool calls with results, prompts.

    Records go to the session named by their own ``sessionId``. A resumed or
    forked transcript repeats earlier records under the id of the session
    that made them, so they land there and ``merge_scans`` drops the repeats.
    """
    is_sub = "/subagents/" in path
    stem = os.path.basename(path)[:-6]
    default_sid = os.path.basename(path.split("/subagents/", 1)[0]) if is_sub else stem
    out = {"path": path, "err_steps": 0, "sessions": {}}
    calls_by_id: dict[str, dict] = {}
    seen_prompt_ids: set = set()
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return out
    with fh:
        for line_no, raw in enumerate(fh):
            if raw.startswith('{"type":"') and not raw.startswith(('{"type":"user"', '{"type":"assistant"')):
                continue
            if '"user"' not in raw and '"assistant"' not in raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            kind = rec.get("type")
            if kind not in ("user", "assistant"):
                continue
            message = rec.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            ts = parse_ts(rec.get("timestamp"))
            # error_steps() parity is counted over the whole file, whatever the dates.
            if kind == "user" and isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                        flat = " ".join(_block_text(b.get("content")).split())
                        if flat and not _NOT_AN_ATTEMPT_RE.search(flat):
                            out["err_steps"] += 1
            if ts is None or ts < since_ts:
                continue
            sid = rec.get("sessionId") if isinstance(rec.get("sessionId"), str) else default_sid
            bucket = out["sessions"].get(sid)
            if bucket is None:
                bucket = out["sessions"][sid] = _new_bucket()
            bucket["times"].append(ts)
            cwd = rec.get("cwd")
            if isinstance(cwd, str) and cwd and (not bucket["cwds"] or bucket["cwds"][-1][1] != cwd):
                bucket["cwds"].append((ts, cwd))
            branch = rec.get("gitBranch")
            if isinstance(branch, str) and branch and branch != "HEAD":
                bucket["branches"][branch] += 1
            if not isinstance(content, list):
                content = None
            if kind == "assistant" and content:
                for b in content:
                    if not isinstance(b, dict) or b.get("type") != "tool_use":
                        continue
                    args = b.get("input") if isinstance(b.get("input"), dict) else {}
                    name = b.get("name") or "tool"
                    fpath = args.get("file_path") or args.get("notebook_path")
                    call = {"id": b.get("id"), "ts": ts, "line": line_no, "tool": name, "sub": is_sub,
                            "path": fpath if isinstance(fpath, str) else None,
                            "cwd": cwd if isinstance(cwd, str) and cwd else None,
                            "ok": None, "blocked": False, "rts": None, "error": None}
                    bucket["calls"].append(call)
                    if isinstance(b.get("id"), str):
                        calls_by_id[b["id"]] = call
            elif kind == "user":
                if content:
                    for b in content:
                        if not isinstance(b, dict) or b.get("type") != "tool_result":
                            continue
                        call = calls_by_id.get(b.get("tool_use_id"))
                        if call is None:
                            continue
                        text = _block_text(b.get("content"))
                        flat = " ".join(text.split())
                        call["rts"] = ts
                        if b.get("is_error"):
                            if flat and _NOT_AN_ATTEMPT_RE.search(flat):
                                call["blocked"] = True
                                continue
                            call["ok"] = False
                            if flat:
                                call["error"] = text if len(text) <= 4000 else text[:1500] + "\n" + text[-2500:]
                        else:
                            call["ok"] = True
                if want_prompts:
                    pid = rec.get("promptId") or rec.get("uuid") or f"{path}:{line_no}"
                    if pid in seen_prompt_ids:
                        continue
                    text = human_prompt(rec)
                    if text:
                        seen_prompt_ids.add(pid)
                        bucket["prompts"].append({"id": pid, "ts": ts, "text": text[:20000]})
                        continue
                    text = session_brief(rec)
                    if text:
                        seen_prompt_ids.add(pid)
                        bucket["briefs"].append({"id": pid, "ts": ts, "text": text[:20000]})
    return out


def _scan_job(args):
    path, since_ts, want_prompts = args
    return scan_transcript(path, since_ts, want_prompts)


def merge_scans(scans: list[dict]) -> dict[str, dict]:
    """Per-session union of the scans, each tool call and prompt once."""
    merged: dict[str, dict] = {}
    for scan in scans:
        is_sub = "/subagents/" in scan["path"]
        for sid, b in scan["sessions"].items():
            m = merged.get(sid)
            if m is None:
                m = merged[sid] = {"times": set(), "cwds": [], "branches": collections.Counter(), "calls": {},
                                   "prompts": {}, "briefs": {}, "files": set(), "subagent_files": set()}
            m["times"].update(b["times"])
            m["cwds"] += b["cwds"]
            m["branches"].update(b["branches"])
            for c in b["calls"]:
                key = c["id"] or (c["ts"], c["tool"], c["path"])
                prev = m["calls"].get(key)
                if prev is None or (prev["ok"] is None and not prev["blocked"]):
                    m["calls"][key] = c
            for p in b["prompts"]:
                m["prompts"].setdefault(p["id"], p)
            for p in b["briefs"]:
                m["briefs"].setdefault(p["id"], p)
            (m["subagent_files"] if is_sub else m["files"]).add(scan["path"])
    for m in merged.values():
        m["times"] = sorted(m["times"])
        m["cwds"].sort(key=lambda x: x[0])
        m["calls"] = sorted(m["calls"].values(), key=lambda c: (c["ts"], c["sub"], c["line"]))
        m["prompts"] = sorted(m["prompts"].values(), key=lambda p: p["ts"])
        m["briefs"] = sorted(m["briefs"].values(), key=lambda p: p["ts"])
    return merged


# ---------------------------------------------------------------- repos

# A repo's own config can make a read run its code: a core.fsmonitor hook runs on
# any index read, log.showSignature runs gpg.program on git log, and a partial
# clone fetches missing objects over the network. All three are turned off.
_GIT_SAFE = ("-c", "core.fsmonitor=false", "-c", "log.showSignature=false")


def _git(repo_root: str, *args: str, input_bytes: bytes | None = None) -> bytes:
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        res = subprocess.run(["git", *_GIT_SAFE, "-C", repo_root, *args], input=input_bytes, capture_output=True,
                             timeout=300, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return b""
    return res.stdout if res.returncode == 0 else b""


@functools.lru_cache(maxsize=None)
def find_git_root(path: str) -> tuple[str | None, str | None]:
    """(main checkout root, checkout root) for a path, walking up to ``.git``.
    The two differ inside a linked worktree, whose ``.git`` is a file."""
    p = path.rstrip("/") or "/"
    while True:
        dot_git = os.path.join(p, ".git")
        if os.path.isdir(dot_git):
            return p, p
        if os.path.isfile(dot_git):
            try:
                with open(dot_git, encoding="utf-8") as fh:
                    first = fh.readline().strip()
            except OSError:
                return None, None
            if first.startswith("gitdir:"):
                gitdir = first.split(":", 1)[1].strip()
                if "/.git/worktrees/" in gitdir:
                    return gitdir.split("/.git/worktrees/")[0], p
            return None, None
        parent = os.path.dirname(p)
        if parent == p:
            return None, None
        p = parent


@functools.lru_cache(maxsize=None)
def resolve_location(cwd: str) -> tuple[str | None, str | None]:
    """(repo root, worktree root) for a session cwd."""
    if not cwd:
        return None, None
    wt_root = None
    base = cwd
    for marker in WORKTREE_MARKERS:
        at = cwd.find(marker)
        if at != -1:
            name = cwd[at + len(marker):].split("/", 1)[0]
            wt_root = cwd[:at + len(marker)] + name
            base = cwd[:at]
            break
    repo, checkout = find_git_root(base)
    if repo is None:
        return None, None
    return repo, wt_root or checkout


def worktree_of(path: str, repo_root: str, checkout: str | None) -> str | None:
    """The checkout an absolute path lies in, lower-cased like session
    worktrees: a linked worktree under the repo by its marker, the main
    checkout, or the session's own checkout outside it."""
    low = path.lower()
    root = repo_root.lower().rstrip("/") + "/"
    if low.startswith(root):
        for marker in WORKTREE_MARKERS:
            at = low.find(marker)
            if at != -1:
                return low[:at + len(marker)] + low[at + len(marker):].split("/", 1)[0]
        return repo_root.lower()
    if checkout and low.startswith(checkout.lower().rstrip("/") + "/"):
        return checkout.lower()
    sibling = sibling_checkout(path, repo_root)
    return sibling.lower() if sibling else None


@functools.lru_cache(maxsize=None)
def sibling_checkout(path: str, repo_root: str) -> str | None:
    """The linked worktree of ``repo_root`` an absolute path lies in when
    that worktree sits outside the main checkout, found from its ``.git``
    file, else None."""
    if not path or not path.startswith("/"):
        return None
    main, checkout = find_git_root(os.path.dirname(path))
    if not main or checkout == main or repo_identity(main) != repo_identity(repo_root):
        return None
    return checkout


def sibling_rel(path: str | None, repo_root: str) -> str | None:
    """Path relative to a linked worktree of the repo outside its main
    checkout, which neither ``repo_rel`` nor ``checkout_rel`` reaches."""
    return checkout_rel(path, sibling_checkout(path, repo_root)) if path else None


@functools.lru_cache(maxsize=None)
def canonical_path(path: str, roots: tuple) -> str:
    """``path`` with its symlinks resolved when the file it names lies in the
    same checkout root, written under that root as recorded, so a link and
    its target are one file (``alias.py -> src/app.py``). Unchanged when the
    target leaves the root or the path is in none of ``roots``."""
    if not path or not path.startswith("/"):
        return path
    for root in roots:
        if not root:
            continue
        prefix = root.rstrip("/") + "/"
        if not path.lower().startswith(prefix.lower()):
            continue
        real, real_root = os.path.realpath(path), os.path.realpath(root).rstrip("/") + "/"
        if not real.lower().startswith(real_root.lower()):
            return path
        return prefix + real[len(real_root):]
    return path


def attribute_call(c: dict, repo: str, checkout: str) -> None:
    """Sets a call's ``path`` to the file it names, symlinks resolved inside
    its checkout, and ``file`` to its repo-relative path. A file tool's path
    in a linked worktree outside the main checkout counts as well."""
    if c["path"]:
        c["path"] = canonical_path(c["path"], (repo, checkout))
    c["file"] = repo_rel(c["path"], repo) or checkout_rel(c["path"], checkout) or (
        sibling_rel(c["path"], repo) if c["tool"] in FILE_TOOLS else None)


def worktree_label(worktree: str, repo_root: str) -> str:
    return "main checkout" if worktree == repo_root.lower() else os.path.basename(worktree)


def repo_rel(path: str | None, repo_root: str) -> str | None:
    """Repo-relative path, or None when the path lies outside this repo,
    including inside another repo's worktree."""
    if not path:
        return None
    prefix = repo_root.rstrip("/") + "/"
    if path.startswith("/"):
        if not path.lower().startswith(prefix.lower()):
            return None
        path = prefix + path[len(prefix):]  # the repo's own letter case, so the prefix strips
    return normalize_repo_path(path, repo_root)


def checkout_rel(path: str | None, checkout: str | None) -> str | None:
    """Path relative to a session's own checkout, for a linked worktree that
    lives outside the main checkout, where ``repo_rel`` cannot reach it."""
    if not path or not checkout or not path.startswith("/"):
        return None
    prefix = checkout.rstrip("/") + "/"
    if not path.lower().startswith(prefix.lower()):
        return None
    rel = os.path.normpath(path[len(prefix):])
    if rel in (".", ".git") or rel.startswith(("../", ".git/")):
        return None
    return rel


def repo_identity(repo_root: str):
    try:
        st = os.stat(os.path.join(repo_root, ".git"))
        return (st.st_dev, st.st_ino)
    except OSError:
        return repo_root.lower()


def repo_file_set(repo_root: str, since: str) -> set[str]:
    files = set(_git(repo_root, "ls-files", "-z").decode("utf-8", "replace").split("\0"))
    log = _git(repo_root, "log", "--all", f"--since={since}", "--name-only", "--pretty=format:", "--no-renames")
    files |= {ln.strip() for ln in log.decode("utf-8", "replace").splitlines()}
    files.discard("")
    return files


def read_repo_files(repo_root: str, wanted: dict[str, set]) -> dict[str, list[str]]:
    """Every version of each wanted file an agent could have seen: the main
    checkout's working tree and the tip of each branch a session that touched
    it was on (git cat-file), or HEAD when neither exists."""
    texts: dict[str, list[str]] = collections.defaultdict(list)
    specs: list[tuple[str, str]] = []
    for rel, branches in wanted.items():
        p = os.path.join(repo_root, rel)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    texts[rel].append(fh.read(400_000))
            except OSError:
                pass
        for branch in sorted(b for b in branches if b):
            specs.append((rel, f"refs/heads/{branch}:{rel}"))
        specs.append((rel, f"HEAD:{rel}"))
    blob = _git(repo_root, "cat-file", "--batch", input_bytes=("\n".join(s for _, s in specs) + "\n").encode())
    pos, answers = 0, []
    while pos < len(blob) and len(answers) < len(specs):
        nl = blob.find(b"\n", pos)
        if nl == -1:
            break
        header = blob[pos:nl].decode("utf-8", "replace").split()
        pos = nl + 1
        if len(header) == 3 and header[2].isdigit():
            size = int(header[2])
            answers.append(blob[pos:pos + size].decode("utf-8", "replace"))
            pos += size + 1
        else:
            answers.append(None)
    head_text = {}
    for (rel, spec), text in zip(specs, answers):
        if text is None:
            continue
        if spec.startswith("HEAD:"):
            head_text[rel] = text[:400_000]
        elif text[:400_000] not in texts[rel]:
            texts[rel].append(text[:400_000])
    for rel, text in head_text.items():
        if not texts[rel]:
            texts[rel].append(text)
    return {rel: versions for rel, versions in texts.items() if versions}


def workspace_packages(repo_root: str, files) -> dict[str, str]:
    packages = {}
    for f in files:
        if os.path.basename(f) != "package.json" or "node_modules/" in f:
            continue
        try:
            with open(os.path.join(repo_root, f), encoding="utf-8") as fh:
                name = json.load(fh).get("name")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(name, str) and name:
            packages[name] = os.path.dirname(f)
    return packages


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GATE_BASE = r"(?:^|[\s/`'\"(\[])%s(?=$|[\s`'\")\],.:;])"


def ontology_rules(repo_root: str, files) -> tuple[dict[str, set], str] | None:
    """File to rule keys for a repo with ``ontology/``, the way the 2026-11-16
    ontology session linked them: a governed file whose name a rule's gate
    names, or a file in an ``ontology/gates/*.json`` function list."""
    od = os.path.join(repo_root, "ontology")
    if not os.path.isdir(od):
        return None
    # The parser is always the installed plugin's, never a copy inside the repo:
    # the replay reads the repos it analyses and runs no code from them.
    lib_path = os.path.join(PLUGIN_ONTOLOGY_LIB, "ontology_lib.py")
    if not os.path.isfile(lib_path):
        return None
    from pathlib import Path
    lib = _load_module(lib_path, f"ontology_lib_{abs(hash(repo_root))}")
    onts = lib.load_all(Path(repo_root))
    invs = [i for o in onts for i in o.get("invariants", [])]
    rules = collections.defaultdict(set)
    for f in files:
        if not lib.path_governed_by(f, onts):
            continue
        pat = re.compile(_GATE_BASE % re.escape(os.path.basename(f)))
        for inv in invs:
            if inv.get("gate") and pat.search(inv["gate"]):
                rules[f].add(" ".join(str(inv.get("text", "")).lower().split()))
    for p in glob.glob(os.path.join(od, "gates", "*.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                gate = json.load(fh)
        except (OSError, ValueError):
            continue
        key = " ".join(str(gate.get("text") or gate.get("id") or p).lower().split())
        for fn in gate.get("functions", []) or []:
            if isinstance(fn, dict) and isinstance(fn.get("file"), str):
                rules[fn["file"]].add(key)
    return dict(rules), "plugin 0.1.0"


# ---------------------------------------------------------------- report helpers

_SECRETISH = re.compile(r"[A-Za-z0-9_\-]{24,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def scrub(text: str) -> str:
    """Signatures go into a Linear comment: drop anything token-shaped."""
    return _EMAIL.sub("<email>", _SECRETISH.sub("<tok>", text))


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def frac(n: int, d: int) -> str:
    return f"{n:,} / {d:,} ({pct(n, d)})"


def iso_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def iso_minute(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _median(values):
    return statistics.median(values) if values else None


def round_robin(items: list[dict], key, k: int = 5) -> list[dict]:
    """Up to k items, taking one per key value in turn, in the given order."""
    groups = collections.OrderedDict()
    for it in items:
        groups.setdefault(key(it), []).append(it)
    out = []
    while len(out) < k and any(groups.values()):
        for g in groups.values():
            if g and len(out) < k:
                out.append(g.pop(0))
    return out


# ---------------------------------------------------------------- main

def discover(projects: str) -> tuple[list[str], dict[str, list[str]]]:
    """Top-level session transcripts, and subagent transcripts keyed by
    (project dir, parent session id). A subagent's parent file may be gone."""
    tops = sorted(glob.glob(os.path.join(projects, "*", "*.jsonl")))
    subs = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(projects, "*", "*", "subagents", "**", "*.jsonl"), recursive=True)):
        if os.path.basename(p) == "journal.jsonl":  # a workflow's journal, not a transcript
            continue
        session_dir = p.split("/subagents/", 1)[0]
        subs[(os.path.dirname(session_dir), os.path.basename(session_dir))].append(p)
    return tops, subs


def build_sessions(merged: dict, gap_s: float) -> tuple[dict, collections.Counter]:
    """Sessions with the repo and worktree of their first cwd in a git repo,
    and counts of the sessions that have none."""
    sessions: dict[str, dict] = {}
    excluded = collections.Counter()
    repo_roots: dict = {}
    for sid, m in merged.items():
        if not m["times"]:
            excluded["no events since the start date"] += 1
            continue
        repo = wt = cwd = None
        for _ts, c in m["cwds"]:
            repo, wt = resolve_location(c)
            if repo:
                cwd = c
                break
        if repo is None:
            excluded["cwd not in a git repo"] += 1
            continue
        ident = repo_identity(repo)
        repo = repo_roots.setdefault(ident, repo)
        located = [(r, w) for r, w in (resolve_location(c) for _t, c in m["cwds"]) if r]
        worktrees = {w.lower() for r, w in located if repo_identity(r) == ident}
        repos = {repo_identity(r) for r, _w in located}
        for c in m["calls"]:
            attribute_call(c, repo, wt)
        own = sorted(f for f in m["files"] if os.path.basename(f)[:-6] == sid)
        home = (own or sorted(m["files"]) or [f.split("/subagents/", 1)[0] for f in sorted(m["subagent_files"])])[0]
        cwd_prefix = os.path.relpath(cwd, wt) if cwd.lower().startswith(wt.lower()) else ""
        sessions[sid] = {
            "sid": sid, "repo": repo, "worktree": wt.lower(), "worktree_name": (
                "main checkout" if wt == repo else os.path.basename(wt)),
            "cwd": cwd, "moved": len(worktrees) > 1, "switched_repo": len(repos) > 1,
            "project_dir": os.path.basename(os.path.dirname(home)),
            "cwd_prefix": "" if cwd_prefix in (".", "") or cwd_prefix.startswith("..") else cwd_prefix,
            "times": m["times"], "spans": activity_spans(m["times"], gap_s),
            "calls": m["calls"], "prompts": m["prompts"], "briefs": m["briefs"],
            "branch": m["branches"].most_common(1)[0][0] if m["branches"] else None,
            "subagents": len(m["subagent_files"]), "copied_into": len(m["files"]) - len(own),
        }
    names = {}
    for root in set(repo_roots.values()):
        base = os.path.basename(root)
        names[root] = base if sum(1 for r in repo_roots.values() if os.path.basename(r) == base) == 1 else \
            f"{os.path.basename(os.path.dirname(root))}/{base}"
    for s in sessions.values():
        s["repo_name"] = names[s["repo"]]
    return sessions, excluded


def left_out_counts(sessions: dict) -> dict:
    """Sessions left out of runs, by reason. A session whose cwd moved to
    another repo counts there, whether or not it also moved between worktrees."""
    return {"cwd moved between worktrees": sum(1 for s in sessions.values() if s["moved"] and not s["switched_repo"]),
            "cwd moved to another repo": sum(1 for s in sessions.values() if s["switched_repo"])}


def place_in_runs(sessions: dict, gap_s: float) -> tuple[list, list]:
    """The sessions that can be placed in one worktree, and the runs they form."""
    placed = [s for s in sessions.values() if not s["moved"] and not s["switched_repo"]]
    run_lists = group_runs(placed)
    runs = []
    for k, sids in enumerate(sorted(run_lists, key=lambda ss: min(sessions[s]["times"][0] for s in ss))):
        members = [sessions[s] for s in sids]
        rid = f"run{k + 1:03d}"
        for m in members:
            m["run"] = rid
        all_spans = activity_spans([t for m in members for t in m["times"]], gap_s)
        runs.append({
            "run": rid, "repo": members[0]["repo_name"], "sessions": sids,
            "worktrees": sorted({m["worktree_name"] for m in members}),
            "start": min(m["times"][0] for m in members), "end": max(m["times"][-1] for m in members),
            "active_minutes": round(sum(b - a for a, b in all_spans) / 60),
        })
    return placed, runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=SINCE_DEFAULT, help="first day counted (UTC), default %(default)s")
    ap.add_argument("--projects", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--run", default=date.today().isoformat(), help="results subdirectory name")
    ap.add_argument("--idle-gap", type=float, default=IDLE_GAP_S / 60, help="minutes of no events that end a span")
    ap.add_argument("--workers", type=int, default=4, help="scan processes (at most 4)")
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N transcripts (smoke runs)")
    ap.add_argument("--ticket-keys", default=None,
                    help="comma-separated ticket prefixes such as ABC,XYZ; default $SWARM_MEMORY_TICKET_KEYS, "
                         "else ticket_keys in the config.json beside the swarm_memory store")
    args = ap.parse_args(argv)
    global TICKET_KEYS
    TICKET_KEYS = load_ticket_keys(args.ticket_keys)
    if not TICKET_KEYS:
        print("replay: no ticket keys configured, so no prompt names a ticket", file=sys.stderr)

    t0 = time.time()
    since_ts = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp()
    gap_s = args.idle_gap * 60
    out_dir = os.path.join(HERE, "results", args.run)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Scan.
    tops, subs = discover(args.projects)
    if args.limit:
        tops = tops[:args.limit]
    jobs = [(p, since_ts, True) for p in tops]
    if args.limit:
        keys = {(os.path.dirname(p), os.path.basename(p)[:-6]) for p in tops}
        sub_paths = [sp for key, sps in subs.items() if key in keys for sp in sps]
    else:
        sub_paths = [sp for sps in subs.values() for sp in sps]
    sub_jobs = [(sp, since_ts, False) for sp in sorted(sub_paths)]
    workers = max(1, min(4, args.workers))
    if workers > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            scanned = pool.map(_scan_job, jobs + sub_jobs, chunksize=4)
    else:
        scanned = [_scan_job(j) for j in jobs + sub_jobs]
    by_path = {s["path"]: s for s in scanned}
    print(f"scanned {len(jobs)} sessions and {len(sub_jobs)} subagent transcripts in {time.time() - t0:.0f} s",
          file=sys.stderr)

    # 2. Sessions, repos, worktrees.
    merged = merge_scans(scanned)
    sessions, excluded = build_sessions(merged, gap_s)

    # 3. Runs.
    placed, runs = place_in_runs(sessions, gap_s)
    left_out = left_out_counts(sessions)
    run_by_id = {r["run"]: r for r in runs}
    gap_sensitivity = {}
    for g in (5, 15, 30):
        alt = [{**s, "spans": activity_spans(s["times"], g * 60)} for s in placed]
        rl = group_runs(alt)
        gap_sensitivity[g] = {"runs": len(rl), "sessions": sum(len(r) for r in rl)}
    first_pass = first_pass_runs(sessions)
    rl = group_runs([s for s in placed if s["worktree_name"] != "main checkout"])
    linked_only = {"runs": len(rl), "sessions": sum(len(r) for r in rl)}

    # 4. Repo file sets, relations.
    repos_in_play = sorted({s["repo"] for s in sessions.values()})
    files_by_repo = {}
    for repo in repos_in_play:
        fs = repo_file_set(repo, (datetime.fromtimestamp(since_ts, timezone.utc) - timedelta(days=31)).strftime("%Y-%m-%d"))
        for s in sessions.values():
            if s["repo"] == repo:
                fs |= {c["file"] for c in s["calls"] if c["file"] and c["tool"] in FILE_TOOLS and c["ok"]}
        files_by_repo[repo] = fs

    summary = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "since": args.since,
               "idle_gap_minutes": args.idle_gap, "windows_minutes": list(WINDOWS_MIN)}

    m3 = moment3(sessions, runs, files_by_repo)
    m4 = moment4(sessions, runs)
    m2 = moment2(sessions, files_by_repo, m4["sig_first_seen"])
    parity = error_steps_parity(tops, sessions, by_path)

    all_times = [t for s in sessions.values() for t in s["times"]]
    summary["corpus"] = {
        "session_transcripts": len(jobs), "subagent_transcripts": len(sub_jobs),
        "sessions_in_a_repo": len(sessions), "excluded": dict(excluded),
        "sessions_with_records_copied_into_another_transcript": sum(1 for s in sessions.values() if s["copied_into"]),
        "human_prompts": sum(len(s["prompts"]) for s in sessions.values()),
        "briefs": sum(len(s["briefs"]) for s in sessions.values()),
        "first_event": iso_day(min(all_times)) if all_times else None,
        "last_event": iso_day(max(all_times)) if all_times else None,
        "tool_calls": sum(len(s["calls"]) for s in sessions.values()),
        "repos": len(repos_in_play),
        "error_steps_parity": parity,
    }
    summary["runs"] = {
        "runs": len(runs), "sessions_in_runs": sum(len(r["sessions"]) for r in runs),
        "sessions_left_out_because_their_cwd_moved_between_worktrees": left_out["cwd moved between worktrees"],
        "sessions_left_out_because_their_cwd_moved_to_another_repo": left_out["cwd moved to another repo"],
        "repos_with_runs": len({r["repo"] for r in runs}),
        "worktrees_per_run_median": _median([len(r["worktrees"]) for r in runs]),
        "worktrees_per_run_max": max((len(r["worktrees"]) for r in runs), default=0),
        "sessions_per_run_max": max((len(r["sessions"]) for r in runs), default=0),
        "active_hours_total": round(sum(r["active_minutes"] for r in runs) / 60, 1),
        "idle_gap_sensitivity": gap_sensitivity,
        "without_main_checkout_sessions": linked_only,
        "first_pass_method_on_this_corpus": first_pass,
        "first_pass_reported": FIRST_PASS,
        "by_repo": dict(collections.Counter(r["repo"] for r in runs).most_common()),
    }
    summary["moment3"] = m3["summary"]
    summary["moment4"] = m4["summary"]
    summary["moment2"] = m2["summary"]
    examples = {"moment3": m3["examples"], "moment4": m4["examples"], "moment2": m2["examples"]}
    for r in runs:
        r["moment3_same_file_hits_60"] = m3["per_run"].get(r["run"], 0)
        r["moment3_same_or_related_hits_60"] = m3["per_run_any"].get(r["run"], 0)
        r["moment4_hits"] = m4["per_run"].get(r["run"], 0)
        r["start"], r["end"] = iso_minute(r["start"]), iso_minute(r["end"])
    summary["runs"]["runs_with_a_hit"] = {
        "moment 3 same file, 60 min": sum(1 for r in runs if r["moment3_same_file_hits_60"]),
        "moment 3 same or related file, 60 min": sum(1 for r in runs if r["moment3_same_or_related_hits_60"]),
        "moment 4, 240 min": sum(1 for r in runs if r["moment4_hits"]),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(os.path.join(out_dir, "runs.json"), "w") as fh:
        json.dump(runs, fh, indent=2)
    with open(os.path.join(out_dir, "examples.json"), "w") as fh:
        json.dump(examples, fh, indent=2)
    report = render_report(summary, runs, examples, run_by_id)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write(report)
    print(report)
    print(f"\nwrote {out_dir} in {time.time() - t0:.0f} s", file=sys.stderr)
    return 0


def first_pass_runs(sessions: dict) -> dict:
    """The 2026-11-16 first pass: worktree directories only, the encoded
    directory prefix as the repo, first-to-last timestamps capped at 12 h."""
    alt = []
    for s in sessions.values():
        pd = s["project_dir"]
        if "--claude-worktrees-" not in pd:
            continue
        start = s["times"][0]
        alt.append({"sid": s["sid"], "repo": pd.split("--claude-worktrees-")[0], "worktree": pd,
                    "spans": [(start, min(s["times"][-1], start + 12 * 3600))]})
    rl = group_runs(alt)
    return {"runs": len(rl), "sessions": sum(len(r) for r in rl),
            "repo_paths": len({alt_s["repo"] for alt_s in alt if any(alt_s["sid"] in r for r in rl)})}


def _tally_m3(events_by_run: dict, relations: dict, repo_of_run: dict, name_of_repo: dict):
    """Hit counts per repo for one definition of a touch. A tool call that
    touched several files counts once."""
    rows = collections.defaultdict(lambda: collections.defaultdict(int))
    notices = collections.defaultdict(set)
    per_run, per_run_any, pool = collections.Counter(), collections.Counter(), []
    top_files = collections.Counter()
    for rid, evs in events_by_run.items():
        repo = repo_of_run[rid]
        rname = name_of_repo[repo]
        row = rows[rname]
        row["calls"] += len({e["call"] for e in evs})
        row["edit calls"] += len({e["call"] for e in evs if e["edit"]})
        for w in WINDOWS_MIN:
            same = file_window_hits(evs, w * 60)
            same_calls = {h["event"]["call"] for h in same}
            row[f"same file|{w}"] += len(same_calls)
            row[f"edit after edit|{w}"] += len(edit_after_edit_calls(evs, w * 60))
            if w == 60:
                per_run[rid] += len(same_calls)
                for h in same:
                    notices[rname].add((h["event"]["sid"], h["other"]["sid"], h["event"]["file"]))
                    top_files[(rname, h["event"]["file"])] += 1
                    pool.append({**h, "run": rid})
            any_calls = set(same_calls)
            for rel_name, rel in relations[repo]["rels"].items():
                hit_calls = {h["event"]["call"] for h in file_window_hits(evs, w * 60, related=rel)}
                row[f"{rel_name}|{w}"] += len(hit_calls)
                any_calls |= hit_calls
            row[f"same or any related|{w}"] += len(any_calls)
            if w == 60:
                per_run_any[rid] += len(any_calls)
    for rname, keys in notices.items():
        rows[rname]["distinct notices same file 60"] = len(keys)
    totals = collections.defaultdict(int)
    for row in rows.values():
        for k, v in row.items():
            totals[k] += v
    top = [{"repo": r, "file": f, "hits_60": n} for (r, f), n in top_files.most_common(15)]
    return ({"by_repo": {k: dict(v) for k, v in rows.items()}, "total": dict(totals), "top_files_same_file_60": top},
            per_run, per_run_any, pool)


def moment3(sessions: dict, runs: list[dict], files_by_repo: dict) -> dict:
    """A touch is a file tool call on a repo file. Shell commands are not
    read: what a command touched cannot be told from its text."""
    file_events = collections.defaultdict(list)
    branch_of = collections.defaultdict(set)
    mix = collections.Counter()
    for s in sessions.values():
        rid = s.get("run")
        if not rid:
            continue
        repo = s["repo"]
        for k, c in enumerate(s["calls"]):
            mix["tool calls in runs"] += 1
            mix["Bash calls in runs"] += c["tool"] == "Bash"
            mix["file tool calls on a path outside the repo"] += c["tool"] in FILE_TOOLS and not c["file"]
            if not call_ran(c):
                continue
            ts = touch_ts(c)
            if ts is None:
                mix["calls with no result"] += 1
                continue
            base = {"ts": ts, "sid": s["sid"], "repo": repo, "call": f"{s['sid']}:{k}", "tool": c["tool"]}
            if c["tool"] in FILE_TOOLS and c["file"]:
                wt = worktree_of(c["path"], repo, s["worktree"]) or s["worktree"]
                file_events[rid].append({**base, "worktree": wt, "wt_name": worktree_label(wt, repo),
                                         "file": c["file"], "edit": c["tool"] in EDIT_TOOLS})
                branch_of[(repo, c["file"])].add(s["branch"])
    repo_of_run = {r["run"]: next(sessions[s]["repo"] for s in r["sessions"]) for r in runs}
    name_of_repo = {s["repo"]: s["repo_name"] for s in sessions.values()}

    # Relations, built over the files touched in runs.
    touched = collections.defaultdict(set)
    for rid, evs in file_events.items():
        touched[repo_of_run[rid]] |= {e["file"] for e in evs}
    relations = {}
    ontology_info = {}
    for repo, files in touched.items():
        allfiles = files_by_repo.get(repo, set()) | files
        py_idx = build_py_index(allfiles)
        packages = workspace_packages(repo, allfiles)
        wanted = {f: branch_of.get((repo, f), set()) for f in files if f.endswith(".py") or f.endswith(JS_SOURCE)}
        texts = read_repo_files(repo, wanted)
        imports = {}
        for f, versions in texts.items():
            imports[f] = set()
            for text in versions:
                imports[f] |= py_imports(f, text, py_idx, allfiles) if f.endswith(".py") else \
                    js_imports(f, text, allfiles, packages)
        neighbours = collections.defaultdict(set)
        for f, deps in imports.items():
            for d in deps:
                neighbours[f].add(d)
                neighbours[d].add(f)
        rels = {"same directory": same_directory_relation(files),
                "import neighbour": (lambda nb: (lambda f: nb.get(f, set())))(neighbours)}
        ont = None
        try:
            ont = ontology_rules(repo, files)
        except Exception as exc:  # an ontology that does not parse is reported, not fatal
            ontology_info[name_of_repo[repo]] = f"load failed: {type(exc).__name__}"
        if ont is not None:
            file_rules, source = ont
            by_rule = collections.defaultdict(set)
            for f, keys in file_rules.items():
                for key in keys:
                    by_rule[key].add(f)
            rel_rule = collections.defaultdict(set)
            for members in by_rule.values():
                for f in members:
                    rel_rule[f] |= members - {f}
            rels["same ontology rule"] = (lambda rr: (lambda f: rr.get(f, set())))(rel_rule)
            ontology_info[name_of_repo[repo]] = {
                "lib": source, "touched_files_linked_to_a_rule": sum(1 for f in files if file_rules.get(f)),
                "touched_files": len(files), "rules": len(by_rule)}
        relations[repo] = {"rels": rels, "parsed": len(texts), "wanted": len(wanted),
                           "with_imports": sum(1 for v in imports.values() if v)}

    file_tools, per_run, per_run_any, example_pool = _tally_m3(file_events, relations, repo_of_run, name_of_repo)

    example_pool.sort(key=lambda h: (not (h["event"]["edit"] and h["other"]["edit"]), h["event"]["ts"]))
    seen_files: set = set()
    example_pool = [h for h in example_pool
                    if not ((h["event"]["repo"], h["event"]["file"]) in seen_files
                            or seen_files.add((h["event"]["repo"], h["event"]["file"])))]
    examples = []
    for h in round_robin(example_pool, key=lambda h: h["event"]["repo"]):
        e, o = h["event"], h["other"]
        examples.append({
            "repo": name_of_repo[e["repo"]], "run": h["run"], "file": e["file"],
            "earlier": f'{o["tool"]} by session {o["sid"][:8]} in {o["wt_name"]}',
            "later": f'{e["tool"]} by session {e["sid"][:8]} in {e["wt_name"]}',
            "minutes_apart": round(h["gap_s"] / 60, 1), "at": iso_minute(e["ts"]),
        })
    rel_meta = {name_of_repo[r]: {k: v for k, v in info.items() if k != "rels"} for r, info in relations.items()}
    return {"summary": {"by_repo": file_tools["by_repo"], "total": file_tools["total"],
                        "top_files_same_file_60": file_tools["top_files_same_file_60"],
                        "tool_mix": dict(mix), "relations": rel_meta, "ontology": ontology_info},
            "examples": examples, "per_run": per_run, "per_run_any": per_run_any}


def moment4(sessions: dict, runs: list[dict]) -> dict:
    name_of_repo = {s["repo"]: s["repo_name"] for s in sessions.values()}
    failures_by_session = {}
    sig_first_seen = collections.defaultdict(dict)  # repo -> sig -> first ts
    counts = collections.defaultdict(lambda: collections.defaultdict(int))
    left_out = {"sessions whose cwd moved to another repo": 0, "their failures": 0}
    for s in sessions.values():
        fl = []
        calls = s["calls"]
        for c in calls:
            if c["ok"] is not False or not c["error"]:
                continue
            sig = failure_signature(c["error"])
            if sig is None:
                continue
            fts = c["rts"] or c["ts"]
            fl.append({"ts": fts, "sid": s["sid"], "worktree": s["worktree"], "wt_name": s["worktree_name"],
                       "sig": sig, "tool": c["tool"], "repo": s["repo"], "kind": failure_kind(c["error"])})
        if s["switched_repo"]:
            left_out["sessions whose cwd moved to another repo"] += 1
            left_out["their failures"] += len(fl)
            fl = []
        for f in fl:
            first = sig_first_seen[s["repo"]].get(f["sig"])
            if first is None or f["ts"] < first:
                sig_first_seen[s["repo"]][f["sig"]] = f["ts"]
        failures_by_session[s["sid"]] = fl

    # Context, comparable with MUD-476 F4: repeats of a signature first seen in another session of the repo.
    by_repo_all = collections.defaultdict(list)
    for s in sessions.values():
        by_repo_all[s["repo"]] += failures_by_session[s["sid"]]
    for repo, fl in by_repo_all.items():
        seen = collections.defaultdict(set)
        row = counts[name_of_repo[repo]]
        for f in sorted(fl, key=lambda f: f["ts"]):
            row["failures, all sessions"] += 1
            if seen[f["sig"]] - {f["sid"]}:
                row["repeats of another session's signature, all sessions"] += 1
            seen[f["sig"]].add(f["sid"])

    per_run = collections.Counter()
    example_pool = []
    for r in runs:
        fl = [f for sid in r["sessions"] for f in failures_by_session[sid]]
        rname = r["repo"]
        row = counts[rname]
        row["failures in runs"] += len(fl)
        for f in fl:
            row[f"failures in runs|{f['kind']}"] += 1
        row["hits any time in run"] += len(repeat_failure_hits(fl))
        for w in WINDOWS_MIN:
            row[f"hits|{w}"] += len(repeat_failure_hits(fl, window_s=w * 60))
        # The breakdown is over the widest window: an earlier failure days
        # before, in a run chained across days, is not "minutes earlier".
        hits = repeat_failure_hits(fl, window_s=max(WINDOWS_MIN) * 60)
        per_run[r["run"]] = len(hits)
        row["hits"] += len(hits)
        for h in hits:
            f = h["failure"]
            row[f"hits|kind|{f['kind']}"] += 1
            example_pool.append({**h, "run": r["run"]})
    totals = collections.defaultdict(int)
    for row in counts.values():
        for k, v in row.items():
            totals[k] += v
    sig_counts = collections.Counter(h["failure"]["sig"] for h in example_pool)
    example_pool.sort(key=lambda h: (h["failure"]["kind"] != "command or tool failure",
                                     -sig_counts[h["failure"]["sig"]], h["failure"]["ts"]))
    used: set = set()
    unique = []
    for h in example_pool:
        if h["failure"]["sig"] not in used:
            used.add(h["failure"]["sig"])
            unique.append(h)
    examples = []
    for h in round_robin(unique, key=lambda h: h["failure"]["repo"]):
        f, e = h["failure"], h["earlier"]
        examples.append({
            "repo": name_of_repo[f["repo"]], "run": h["run"], "signature": scrub(f["sig"]), "tool": f["tool"],
            "kind": f["kind"], "earlier": f'session {e["sid"][:8]} in {e["wt_name"]}',
            "later": f'session {f["sid"][:8]} in {f["wt_name"]}',
            "minutes_apart": round((f["ts"] - e["ts"]) / 60, 1),
            "hits_with_this_signature": sig_counts[f["sig"]], "at": iso_minute(f["ts"]),
        })
    top = collections.Counter((name_of_repo[h["failure"]["repo"]], scrub(h["failure"]["sig"]), h["failure"]["kind"])
                              for h in example_pool)
    return {"summary": {"by_repo": {k: dict(v) for k, v in counts.items()}, "total": dict(totals),
                        "top_hit_signatures": [{"repo": r, "signature": s, "kind": k, "hits": n}
                                               for (r, s, k), n in top.most_common(10)],
                        "left_out": left_out},
            "examples": examples, "per_run": per_run, "sig_first_seen": sig_first_seen}


def moment2(sessions: dict, files_by_repo: dict, sig_first_seen: dict) -> dict:
    rows = collections.defaultdict(lambda: collections.defaultdict(int))
    example_pool = []
    suffix_idx = {repo: build_suffix_index(files) for repo, files in files_by_repo.items()}
    populations = (("human prompts, all sessions", "prompts", False),
                   ("human prompts, sessions in runs", "prompts", True),
                   ("briefs from another session", "briefs", False))
    left_out = {"sessions whose cwd moved to another repo": 0, "their prompts": 0, "their briefs": 0}
    for s in sessions.values():
        if s["switched_repo"]:
            left_out["sessions whose cwd moved to another repo"] += 1
            left_out["their prompts"] += len(s["prompts"])
            left_out["their briefs"] += len(s["briefs"])
            continue
        repo, rname = s["repo"], s["repo_name"]
        files = files_by_repo.get(repo, set())
        for scope, field, runs_only in populations:
            if runs_only and not s.get("run"):
                continue
            row = rows[(rname, scope)]
            for p in s[field]:
                paths = prompt_paths(p["text"], files, cwd_prefix=s["cwd_prefix"], repo_root=repo,
                                     suffix_index=suffix_idx.get(repo), checkout_root=s["worktree"])
                tickets = prompt_tickets(p["text"])
                errors = prompt_error_lines(p["text"])
                known = [e for e in errors if (first := sig_first_seen.get(repo, {}).get(norm(e))) is not None
                         and first < p["ts"]]
                row["prompts"] += 1
                row["file path"] += bool(paths)
                row["error line"] += bool(errors)
                row["error line matching an earlier tool failure"] += bool(known)
                row["ticket key"] += bool(tickets)
                row["any of the three"] += bool(paths or errors or tickets)
                if scope == "human prompts, all sessions":
                    if paths:
                        example_pool.append({"kind": "file path", "value": paths[0], "repo": rname, "sid": s["sid"],
                                             "ts": p["ts"]})
                    if tickets:
                        example_pool.append({"kind": "ticket key", "value": tickets[0], "repo": rname,
                                             "sid": s["sid"], "ts": p["ts"]})
                    if errors:
                        example_pool.append({"kind": "error line", "value": scrub(norm(errors[0])), "repo": rname,
                                             "sid": s["sid"], "ts": p["ts"], "known": bool(known)})
    by_scope = collections.defaultdict(lambda: collections.defaultdict(int))
    for (rname, scope), row in rows.items():
        for k, v in row.items():
            by_scope[scope][k] += v
    example_pool.sort(key=lambda x: (x["kind"] != "error line" or not x.get("known"), x["ts"]))
    seen_values: set = set()
    example_pool = [x for x in example_pool if not (x["value"] in seen_values or seen_values.add(x["value"]))]
    picked = round_robin(example_pool, key=lambda x: (x["kind"], x["repo"]), k=6)
    examples = [{"repo": x["repo"], "session": x["sid"][:8], "kind": x["kind"], "value": x["value"],
                 "at": iso_minute(x["ts"]), **({"matches_earlier_failure": x["known"]} if "known" in x else {})}
                for x in picked]
    return {"summary": {"by_repo": {f"{r} ({scope})": dict(v) for (r, scope), v in sorted(rows.items())},
                        "total": {k: dict(v) for k, v in by_scope.items()}, "left_out": left_out},
            "examples": examples}


def error_steps_parity(tops: list[str], sessions: dict, by_path: dict, sample: int = 25) -> dict:
    """The scan's failure count against error_steps() itself, on a sample of transcripts."""
    picked = tops[:: max(1, len(tops) // sample)][:sample]
    match = 0
    mismatches = []
    for p in picked:
        sid = os.path.basename(p)[:-6]
        repo = sessions[sid]["repo"] if sid in sessions else "/"
        n = len(error_steps(p, repo, cap=10 ** 9))
        if n == by_path[p]["err_steps"]:
            match += 1
        else:
            mismatches.append({"session": sid[:8], "error_steps": n, "scan": by_path[p]["err_steps"]})
    return {"transcripts": len(picked), "matching": match, "mismatches": mismatches[:5]}


def render_report(summary: dict, runs: list[dict], examples: dict, run_by_id: dict) -> str:
    c, r = summary["corpus"], summary["runs"]
    lines = [f"# MUD-850 Phase 1 replay ({summary['generated']})", "",
             f"Sessions since {summary['since']}, idle gap {summary['idle_gap_minutes']:g} min, "
             f"windows {', '.join(str(w) for w in summary['windows_minutes'])} min.", "",
             "## Corpus", "",
             f"- {c['session_transcripts']} session transcripts and {c['subagent_transcripts']} subagent transcripts, "
             f"events {c['first_event']} to {c['last_event']}.",
             f"- {c['sessions_in_a_repo']} sessions ran in {c['repos']} git repos, {c['tool_calls']:,} tool calls, "
             f"{c['human_prompts']:,} human prompts, {c['briefs']:,} briefs. Excluded: {c['excluded']}.",
             f"- {c['sessions_with_records_copied_into_another_transcript']} sessions had records copied into a "
             f"resumed or forked transcript. Each record is counted once, for the session that wrote it.",
             f"- error_steps() parity: {c['error_steps_parity']['matching']} of "
             f"{c['error_steps_parity']['transcripts']} sampled transcripts give the same failure count.", "",
             "## Runs", "",
             f"- {r['runs']} runs, {r['sessions_in_runs']} sessions, {r['repos_with_runs']} repos. "
             f"Worktrees per run: median {r['worktrees_per_run_median']}, max {r['worktrees_per_run_max']}. "
             f"Active time {r['active_hours_total']} h.",
             f"- By repo: {r['by_repo']}.",
             f"- Left out of runs: {r['sessions_left_out_because_their_cwd_moved_between_worktrees']} sessions "
             f"whose cwd moved between worktrees of their repo, and "
             f"{r['sessions_left_out_because_their_cwd_moved_to_another_repo']} whose cwd moved to another repo.",
             f"- Idle gap sensitivity (runs, sessions): {r['idle_gap_sensitivity']}.",
             f"- Without main-checkout sessions: {r['without_main_checkout_sessions']}.",
             f"- Runs with at least one hit: {r['runs_with_a_hit']}.",
             f"- First-pass method on this corpus: {r['first_pass_method_on_this_corpus']}; "
             f"first pass reported {r['first_pass_reported']}.", ""]

    m3 = summary["moment3"]
    rel_names = ["same file", "same directory", "import neighbour", "same ontology rule", "same or any related"]
    lines += ["## Moment 3: another agent on the same or a related file", "",
              "Hits are file tool calls by agent B where another agent in the run touched the file (or a related "
              "file) within the window before. Denominator: file tool calls in runs.", "",
              "| repo | file tool calls | relation | 15 min | 60 min | 240 min |", "|---|---|---|---|---|---|"]
    for rname, row in sorted(m3["by_repo"].items(), key=lambda kv: -kv[1].get("calls", 0)) + \
            [("all repos", m3["total"])]:
        d = row.get("calls", 0)
        for rel in rel_names:
            if f"{rel}|15" not in row:
                continue
            lines.append(f"| {rname} | {d:,} | {rel} | " + " | ".join(
                f"{row.get(f'{rel}|{w}', 0):,} ({pct(row.get(f'{rel}|{w}', 0), d)})" for w in WINDOWS_MIN) + " |")
    lines += ["", "- Edit after edit, same file: " + ", ".join(
        f"{w} min {frac(m3['total'].get(f'edit after edit|{w}', 0), m3['total'].get('edit calls', 0))}"
        for w in WINDOWS_MIN) + " of edit calls in runs.",
              f"- Distinct notices (B session, A session, file) at 60 min, same file: "
              f"{m3['total'].get('distinct notices same file 60', 0):,}.",
              f"- Tool mix in runs: {m3['tool_mix']}.",
              f"- Import parse: {m3['relations']}.", f"- Ontology: {m3['ontology']}.", "", "Examples:", ""]
    for x in examples["moment3"]:
        lines.append(f"- {x['repo']} {x['run']}: `{x['file']}`, {x['earlier']}, then {x['later']} "
                     f"{x['minutes_apart']} min later ({x['at']} UTC).")

    m4 = summary["moment4"]
    kinds = ("command or tool failure", "hook block", "tool input rejected")
    lines += ["", "## Moment 4: an error another agent in the run already hit", "",
              "Hits are failures whose signature another worktree in the same run hit within the window before. "
              "Denominator: failures in runs. The breakdown columns are over the 240-minute hits.", "",
              "| repo | failures in runs | hits within 15 / 60 / 240 min | 240-min hit rate | hits by kind "
              "(failure / hook block / input rejected) | repeats of another session, all sessions |",
              "|---|---|---|---|---|---|"]
    for rname, row in sorted(m4["by_repo"].items(), key=lambda kv: -kv[1].get("failures in runs", 0)) + \
            [("all repos", m4["total"])]:
        d = row.get("failures in runs", 0)
        hits = row.get("hits", 0)
        lines.append(
            f"| {rname} | {d:,} | " + " / ".join(f"{row.get(f'hits|{w}', 0):,}" for w in WINDOWS_MIN) +
            f" | {pct(hits, d)} | " + " / ".join(f"{row.get(f'hits|kind|{k}', 0):,}" for k in kinds) +
            f" | {frac(row.get('repeats of another session' + chr(39) + 's signature, all sessions', 0), row.get('failures, all sessions', 0))} |")
    t = m4["total"]
    lines += ["", "- Failures in runs by kind: " + ", ".join(
        f"{k} {t.get(f'failures in runs|{k}', 0):,}" for k in kinds) + ".",
              f"- Hits any time in run, no window: {t.get('hits any time in run', 0):,}.",
              f"- Left out of every count and of the signature index: "
              f"{m4['left_out']['sessions whose cwd moved to another repo']:,} sessions whose cwd moved to another "
              f"repo, with {m4['left_out']['their failures']:,} failures.",
              f"- 240-min hits that are command or tool failures: {t.get('hits|kind|command or tool failure', 0):,}.",
              "", "Top signatures among hits:", ""]
    for x in m4["top_hit_signatures"]:
        lines.append(f"- {x['repo']}: `{x['signature']}` ({x['kind']}, {x['hits']} hits)")
    lines += ["", "Examples:", ""]
    for x in examples["moment4"]:
        lines.append(f"- {x['repo']} {x['run']}: `{x['signature']}` ({x['tool']}, {x['kind']}). {x['earlier']} first, "
                     f"{x['later']} {x['minutes_apart']} min later.")

    m2 = summary["moment2"]
    lines += ["", "## Moment 2: the prompt names a file, an error or a ticket", "",
              "Human prompts are what Michael typed or pasted. Briefs are the task text another Claude session "
              "wrote when it started a session, which is how side sessions and swarm workers get their prompt.", "",
              "| repo (scope) | prompts | file path | error line | error line matching an earlier failure | "
              "ticket key | any |", "|---|---|---|---|---|---|---|"]
    for rname, row in list(m2["by_repo"].items()) + [(f"all repos ({k})", v) for k, v in m2["total"].items()]:
        d = row.get("prompts", 0)
        lines.append(f"| {rname} | {d:,} | " + " | ".join(
            frac(row.get(k, 0), d) for k in ("file path", "error line", "error line matching an earlier tool failure",
                                             "ticket key", "any of the three")) + " |")
    lo2 = m2["left_out"]
    lines += ["", f"- Left out of every row: {lo2['sessions whose cwd moved to another repo']:,} sessions whose cwd "
                  f"moved to another repo, with {lo2['their prompts']:,} human prompts and {lo2['their briefs']:,} "
                  f"briefs.", "", "Examples:", ""]
    for x in examples["moment2"]:
        extra = f", matches an earlier tool failure: {x['matches_earlier_failure']}" if "matches_earlier_failure" in x else ""
        lines.append(f"- {x['repo']} session {x['session']}: {x['kind']} `{x['value']}`{extra} ({x['at']} UTC).")

    lines += ["", "## Runs with the most hits", "", "| run | repo | sessions | worktrees | start (UTC) | active min | "
              "M3 same file 60 min | M4 hits |", "|---|---|---|---|---|---|---|---|"]
    for run in sorted(runs, key=lambda x: -(x["moment3_same_file_hits_60"] + x["moment4_hits"]))[:12]:
        lines.append(f"| {run['run']} | {run['repo']} | {len(run['sessions'])} | {len(run['worktrees'])} | "
                     f"{run['start']} | {run['active_minutes']} | {run['moment3_same_file_hits_60']} | {run['moment4_hits']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())

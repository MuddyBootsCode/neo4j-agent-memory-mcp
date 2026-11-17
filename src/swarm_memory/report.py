"""Print what the Phase 2 hooks logged: sessions, activity, failures, would-deliver
rows and hook timings. Read-only on the store.

    python3 -m swarm_memory.report [--db PATH] [--minutes N] [--debug-log FILE ...]

``--debug-log`` reads Claude Code debug logs (``claude --debug-file``) for the
time each swarm_memory hook took as Claude Code measured it, process start
included, which is the time the hook adds to a tool call.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sqlite3
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_memory import store  # noqa: E402

_DEBUG_LINE = re.compile(r"Hooks: (\w+)(?::[^ ]*)? \[([^\]]*)\] finished with status (-?\d+) \((\d+)ms\)")


def pct(values, q: float):
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def debug_log_times(paths, marker: str = "swarm_memory") -> dict:
    """Per hook event, the durations Claude Code logged for hooks whose command names ``marker``."""
    out = collections.defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _DEBUG_LINE.search(line)
                if m and marker in m.group(2):
                    out[m.group(1)].append(int(m.group(4)))
    return out


def _short(sid) -> str:
    return (sid or "")[:8]


def _repo_name(key: str) -> str:
    return os.path.basename(os.path.dirname(key)) if os.path.basename(key) == ".git" else os.path.basename(key)


def _clock(ts) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""


def _table(title: str, header, rows) -> str:
    lines = [f"## {title}", ""]
    if not rows:
        return "\n".join(lines + ["(none)", ""])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v).replace("|", "/") for v in r) + " |")
    return "\n".join(lines + [""])


def build(db: str, minutes: float | None = None, debug_logs=()) -> str:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    since = time.time() - minutes * 60 if minutes else 0
    links = collections.defaultdict(lambda: collections.defaultdict(list))
    for rid, typ, key in con.execute("SELECT record_id, type, key FROM link"):
        links[rid][typ].append(key)
    rec = {r[0]: r for r in con.execute(
        "SELECT id, kind, ts, ts_end, session_id, agent_id, tool, action, status, tool_use_id, info FROM record"
        " WHERE ts >= ? OR kind = 'session' ORDER BY ts", (since,))}

    def files_of(rid):
        return ", ".join(k.split("::", 1)[1] for k in links[rid]["file"])

    sessions, activity, failures = [], [], []
    for rid, kind, ts, ts_end, sid, agent, tool, action, status, _tuid, info in rec.values():
        info = json.loads(info) if info else {}
        if kind == "session":
            wts = [os.path.basename(w) for w in links[rid]["worktree"]]
            here = ", ".join(_repo_name(k) for k in links[rid]["repo"]) or "none: cwd outside a git repo"
            sessions.append((_short(sid), here, ", ".join(wts),
                             ", ".join(_repo_name(k) for k in links[rid]["touched_repo"]),
                             info.get("branch"), ", ".join(links[rid]["ticket"]), _clock(ts), _clock(ts_end), status))
        elif kind == "activity":
            activity.append((rid, _clock(ts), _short(sid), _short(agent), tool, action, files_of(rid), info.get("program")))
        elif kind == "failure":
            after = round(ts_end - ts, 1) if status == "resolved" and ts_end else None
            failures.append((rid, _clock(ts), _short(sid), tool, info.get("program"), action, info.get("exit"),
                             info.get("sig"), info.get("head"), status, after))

    def describe(rid):
        r = rec.get(rid)
        if r is None:
            return f"#{rid}"
        if r[1] == "session":
            info = json.loads(r[10]) if r[10] else {}
            wt = ", ".join(os.path.basename(w) for w in links[rid]["worktree"])
            where = f"in {wt}" if wt else ("outside any repo, touched " + (", ".join(
                _repo_name(k) for k in links[rid]["touched_repo"]) or "none"))
            return (f"session {_short(r[4])} {where} (branch {info.get('branch')},"
                    f" ticket {', '.join(links[rid]['ticket']) or 'none'})")
        return f"{r[7]} of {files_of(rid) or '?'} by {_short(r[4])} at {_clock(r[2])}"

    deliver = []
    for ts, moment, sid, records, detail in con.execute(
            "SELECT ts, moment, session_id, records, detail FROM event WHERE kind = 'would_deliver' AND ts >= ?"
            " ORDER BY ts", (since,)):
        ids = json.loads(records) if records else []
        deliver.append((_clock(ts), moment, _short(sid), "; ".join(describe(i) for i in ids), detail))

    timing = collections.defaultdict(list)
    for hook, ms in con.execute("SELECT hook, ms FROM event WHERE kind = 'observed' AND ms IS NOT NULL AND ts >= ?",
                                (since,)):
        timing[hook].append(ms)
    errors = con.execute("SELECT detail, count(*) FROM event WHERE kind = 'hook_error' AND ts >= ? GROUP BY detail",
                         (since,)).fetchall()
    con.close()

    out = [f"# swarm_memory store {db}", ""]
    out.append(_table("Sessions", ["session", "repo (cwd)", "worktree", "touched repos", "branch", "ticket",
                                   "first seen", "last seen", "status"], sessions))
    out.append(_table("Activity", ["record", "time", "session", "subagent", "tool", "action", "files", "program"], activity))
    out.append(_table("Failures", ["record", "time", "session", "tool", "program", "kind", "exit", "signature",
                                   "error head", "status", "resolved after s"], failures))
    out.append(_table("Would deliver", ["time", "moment", "session", "would show", "detail"], deliver))
    out.append(_table("Hook time in the hook process (ms, from start of hook.py to commit)",
                      ["hook", "n", "p50", "p95", "max"],
                      [(h, len(v), pct(v, .5), pct(v, .95), max(v)) for h, v in sorted(timing.items())]))
    if debug_logs:
        dl = debug_log_times(debug_logs)
        allv = [x for v in dl.values() for x in v]
        rows = [(h, len(v), pct(v, .5), pct(v, .95), max(v)) for h, v in sorted(dl.items())]
        if allv:
            rows.append(("all", len(allv), pct(allv, .5), pct(allv, .95), max(allv)))
        out.append(_table("Hook time as Claude Code measured it (ms, process start included)",
                          ["hook", "n", "p50", "p95", "max"], rows))
    side = []
    for name in ("skips", "errors"):
        path = store.side_file(db, name)
        n = sum(1 for _ in open(path)) if os.path.exists(path) else 0
        side.append((f"{name} (side file)", n))
    side += [(f"hook_error {d}", n) for d, n in errors]
    out.append(_table("Errors and skips", ["what", "count"], side))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=None)
    ap.add_argument("--minutes", type=float, default=None, help="only the last N minutes")
    ap.add_argument("--debug-log", nargs="*", default=[])
    args = ap.parse_args(argv)
    print(build(store.db_path(args.db), args.minutes, args.debug_log))
    return 0


if __name__ == "__main__":
    sys.exit(main())

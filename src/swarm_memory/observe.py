"""Phase 2 handlers (MUD-850): write what a hook observed and log what Phase 3
would have delivered (D14). Nothing is returned to the model.

Records

* ``session``: one per session, made by the first hook the session fires
  (SessionStart, or any other event when the hooks were registered mid-session).
  Links: session, repo, worktree, and ticket from the branch name or, failing
  that, from the brief file or the first prompt. ``repo`` and ``worktree``
  come from the session's cwd only, so a session started outside any git repo
  has none. A repo whose files it touches from elsewhere is linked as
  ``touched_repo``: a lead running from ~/Projects still appears in that
  repo's roster without the repo becoming its own. ``info.branch`` holds the
  branch. ``ts_end`` is the last time any hook saw the session.
* ``activity``: one per successful tool call that touched repo files. A file
  tool is ``edit`` (Edit, Write, MultiEdit, NotebookEdit) or ``read`` (Read). A
  shell command that names repo files is ``mention``, because the command does
  not say whether it read or wrote them (D18). Links: session, repo, worktree,
  each file, and the program for a shell command.
* ``failure``: one per failed tool call, with the F4 signature, the program, the
  exit code and a redacted head of the error. It is resolved when a later call in
  the same session succeeds with the same command prefix, or with the same tool
  on the same file (replay rule). Interrupts are not failures.

Would deliver (one ``event`` row each, naming the records it would have shown)

* ``session_start``: other sessions seen in the last 60 minutes and not ended
  that run in this repo or touched its files. Logged again after resume and
  compaction, when the context is new.
* ``first_prompt``: at a session's first prompt, other sessions seen in the last
  60 minutes that hold a ticket the prompt names, or have edited a file it names.
* ``brief_file``: the same lookup over a swarm brief file (``SWARM_BRIEF.md``,
  ``EPIC_BRIEF.md``) the first time the session reads it, because a swarm
  worker's prompt only tells it to read that file.
* ``same_file_edit``: after an edit, another session's edit of the same file
  within 60 minutes, once per session, file and other session.

Never stored: command text, prompt text, tool output. The store keeps the
program, repo-relative paths, the signature and a short redacted error head.
"""

from __future__ import annotations

import json
import os
import time

from swarm_memory import rules, store

WINDOW_S = 60 * 60
ROSTER_LIMIT = 20
BRIEF_FILES = ("SWARM_BRIEF.md", "EPIC_BRIEF.md")
BRIEF_MAX_BYTES = 64 * 1024
HOOKS = ("SessionStart", "UserPromptSubmit", "PostToolUse", "PostToolUseFailure", "SessionEnd")
TOOL_HOOKS = ("PostToolUse", "PostToolUseFailure")


def run(raw: str, db_arg: str | None = None, t0: float | None = None, ts: float | None = None) -> None:
    """The hook body. Never raises. A busy store is a skip, counted in a side
    file. Any other exception is a ``hook_error`` event, or a line in a side file
    when the store itself is what failed. ``ts`` overrides the clock in tests."""
    t0 = time.perf_counter() if t0 is None else t0
    path = store.db_path(db_arg)
    ts = time.time() if ts is None else ts
    hook = session_id = None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload is not a JSON object")
        hook = payload.get("hook_event_name")
        session_id = payload.get("session_id")
        con = store.connect(path)
        try:
            handle(con, payload, ts, t0, store.ticket_keys(path))
        finally:
            con.close()
    except store.Busy:
        _side(path, "skips", ts, hook, session_id, "busy")
    except Exception as exc:
        _record_error(path, ts, hook, session_id, exc)


def _side(path, name, ts, hook, session_id, what) -> None:
    try:
        store.append_line(store.side_file(path, name), [f"{ts:.3f}", hook, (session_id or "")[:8], what])
    except Exception:
        pass


def _record_error(path, ts, hook, session_id, exc) -> None:
    detail = {"type": type(exc).__name__, "message": rules.scrub(str(exc))[:120]}
    try:
        con = store.connect(path)
        try:
            store.begin(con)
            store.log_event(con, "hook_error", ts, hook=hook, session_id=session_id if isinstance(session_id, str) else None,
                            detail=detail)
            con.execute("COMMIT")
        finally:
            con.close()
    except Exception:
        _side(path, "errors", ts, hook, session_id if isinstance(session_id, str) else None, detail["type"])


class Call:
    """The parts of a hook payload the handlers read."""

    def __init__(self, payload: dict, keys=()):
        self.payload = payload
        self.keys = tuple(keys)
        self.hook = payload.get("hook_event_name")
        self.sid = payload.get("session_id")
        cwd = payload.get("cwd")
        self.cwd = cwd if isinstance(cwd, str) and os.path.isabs(cwd) else ""
        agent = payload.get("agent_id")
        self.agent_id = agent if isinstance(agent, str) and agent else None
        tool = payload.get("tool_name")
        self.tool = tool if isinstance(tool, str) else None
        ti = payload.get("tool_input")
        self.tool_input = ti if isinstance(ti, dict) else {}
        tuid = payload.get("tool_use_id")
        self.tool_use_id = tuid if isinstance(tuid, str) and tuid else None
        self.dedupe = None


def handle(con, payload: dict, ts: float, t0: float, keys=()) -> None:
    call = Call(payload, keys)
    if call.hook not in HOOKS or not isinstance(call.sid, str) or not call.sid:
        return
    store.begin(con)
    try:
        call_key = f"call|{call.sid}|{call.tool_use_id}" if call.tool_use_id and call.hook in TOOL_HOOKS else None
        if call_key and con.execute("SELECT 1 FROM event WHERE dedupe = ?", (call_key,)).fetchone():
            # The same tool call delivered again: it changes nothing, the session included.
            detail = {"duplicate": True}
        else:
            call.dedupe = call.dedupe or call_key
            session = _ensure_session(con, call, ts)
            detail = _HANDLERS[call.hook](con, call, session, ts)
        store.log_event(con, "observed", ts, hook=call.hook, session_id=call.sid, agent_id=call.agent_id,
                        tool=call.tool, dedupe=call.dedupe, ms=(time.perf_counter() - t0) * 1000,
                        detail={k: v for k, v in detail.items() if v} or None)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def _ensure_session(con, call: Call, ts: float):
    loc = rules.locate(call.cwd) if call.cwd else None
    row = con.execute("SELECT id FROM record WHERE kind = 'session' AND session_id = ?", (call.sid,)).fetchone()
    if row is None:
        br = rules.branch(loc[3]) if loc else None
        ticket = rules.branch_ticket(br, call.keys)
        info = {"branch": br, "ticket_from": "branch" if ticket else None}
        agent_type = call.payload.get("agent_type") if call.hook == "SessionStart" else None
        if isinstance(agent_type, str) and agent_type:
            info["agent_type"] = agent_type
        rid = store.add_record(con, "session", ts, call.sid, ts_end=ts, action="started", status="active",
                               info={k: v for k, v in info.items() if v})
        store.add_links(con, rid, ts, [("session", call.sid), ("ticket", ticket)])
    else:
        rid = row[0]
        # Only a SessionStart (resume, clear, compact, fork) brings an ended
        # session back. A late event for an ended session leaves it ended.
        con.execute("UPDATE record SET ts_end = ?, status = 'active' WHERE id = ? AND (status != 'ended' OR ?)",
                    (ts, rid, call.hook == "SessionStart"))
    if loc:
        store.add_links(con, rid, ts, [("repo", loc[0]), ("worktree", os.path.realpath(loc[2]))])
    return rid, loc


# ---------------------------------------------------------------- SessionStart

def _on_session_start(con, call: Call, session, ts: float) -> dict:
    rid, loc = session
    source = call.payload.get("source")
    detail = {"source": source}
    if loc is None:
        return detail
    ticket = rules.branch_ticket(rules.branch(loc[3]), call.keys)
    if ticket:
        store.add_links(con, rid, ts, [("ticket", ticket)])
    rows = con.execute(
        "SELECT r.id FROM record r WHERE r.kind = 'session' AND r.ts_end >= ? AND r.session_id != ?"
        " AND r.status != 'ended'"
        " AND EXISTS (SELECT 1 FROM link l WHERE l.record_id = r.id AND l.type IN ('repo', 'touched_repo')"
        " AND l.key = ?)"
        " ORDER BY r.ts_end DESC LIMIT ?",
        (ts - WINDOW_S, call.sid, loc[0], ROSTER_LIMIT)).fetchall()
    if rows:
        store.log_event(con, "would_deliver", ts, hook=call.hook, session_id=call.sid, moment="session_start",
                        records=[r[0] for r in rows], detail={"source": source, "repo": loc[1]})
    detail["roster"] = len(rows)
    return detail


# ---------------------------------------------------------------- UserPromptSubmit

def _on_prompt(con, call: Call, session, ts: float) -> dict:
    rid, loc = session
    key = f"first_prompt|{call.sid}"
    if con.execute("SELECT 1 FROM event WHERE dedupe = ?", (key,)).fetchone():
        return {}
    call.dedupe = key
    prompt = call.payload.get("prompt")
    prompt = prompt if isinstance(prompt, str) else ""
    bases = [b for b in (call.cwd, loc[2] if loc else None) if b]
    return {"first": True, **_holders(con, call, rid, prompt, bases, ts, "first_prompt", "prompt")}


def _holders(con, call: Call, rid: int, text: str, bases: list, ts: float, moment: str, source: str) -> dict:
    """Moment 2 (D14): other sessions seen in the last 60 minutes that hold a
    ticket the text names, or have edited a repo file it names. A brief file
    is never a held file: each worker has its own. The text also supplies the
    session's ticket when it has none yet."""
    tickets = rules.prompt_tickets(text, call.keys)
    files = [fl for fl in (rules.file_location(p) for p in rules.named_files(text, bases))
             if fl and os.path.basename(fl[3]) not in BRIEF_FILES]
    if tickets and not con.execute("SELECT 1 FROM link WHERE record_id = ? AND type = 'ticket'", (rid,)).fetchone():
        store.add_links(con, rid, ts, [("ticket", tickets[0])])
        con.execute("UPDATE record SET info = json_set(coalesce(info, '{}'), '$.ticket_from', ?) WHERE id = ?",
                    (source, rid))
    shown: list[int] = []
    hit_tickets: list[str] = []
    hit_files: list[str] = []
    for t in tickets:
        rows = con.execute(
            "SELECT r.id FROM link l JOIN record r ON r.id = l.record_id"
            " WHERE l.type = 'ticket' AND l.key = ? AND r.kind = 'session' AND r.session_id != ?"
            " AND r.ts_end >= ? AND r.status != 'ended'",
            (t, call.sid, ts - WINDOW_S)).fetchall()
        if rows:
            hit_tickets.append(t)
            shown += [r[0] for r in rows]
    for repo_key, _name, _wt, rel in files:
        rows = con.execute(
            "SELECT a.id, a.session_id FROM link l JOIN record a ON a.id = l.record_id"
            " JOIN record s ON s.kind = 'session' AND s.session_id = a.session_id"
            " WHERE l.type = 'file' AND l.key = ? AND a.kind = 'activity' AND a.action = 'edit'"
            " AND a.session_id != ? AND s.ts_end >= ? AND s.status != 'ended' ORDER BY a.ts DESC",
            (rules.file_key(repo_key, rel), call.sid, ts - WINDOW_S)).fetchall()
        latest: dict = {}
        for aid, other in rows:
            latest.setdefault(other, aid)
        if latest:
            hit_files.append(rel)
            shown += list(latest.values())
    if shown:
        store.log_event(con, "would_deliver", ts, hook=call.hook, session_id=call.sid, agent_id=call.agent_id,
                        tool=call.tool, moment=moment, records=sorted(set(shown)),
                        detail={"tickets": hit_tickets, "files": hit_files})
    return {"tickets": len(tickets), "files": len(files), "held": len(set(shown))}


def _brief(con, call: Call, rid: int, desc: dict, ts: float) -> dict:
    """A swarm worker's prompt only says to read its brief file (ollama-swarm
    SWARM_BRIEF.md, epic-squad EPIC_BRIEF.md), so its brief is looked up the
    first time the session reads one."""
    if desc["action"] not in ("read", "mention"):
        return {}
    path = next((os.path.join(wt, rel) for _k, _n, wt, rel in desc["files"]
                 if os.path.basename(rel) in BRIEF_FILES), None)
    key = f"brief_file|{call.sid}"
    if path is None or con.execute("SELECT 1 FROM event WHERE dedupe = ?", (key,)).fetchone():
        return {}
    try:
        with open(path, "rb") as fh:
            text = fh.read(BRIEF_MAX_BYTES).decode("utf-8", "replace")
    except OSError:
        return {}
    store.log_event(con, "mark", ts, hook=call.hook, session_id=call.sid, dedupe=key)
    root = rules.locate(path)
    bases = [os.path.dirname(path)] + ([root[2]] if root else [])
    return {"brief": _holders(con, call, rid, text, bases, ts, "brief_file", "brief")}


# ---------------------------------------------------------------- tool calls

def _describe(call: Call) -> dict:
    """Files, action, program and resolution keys of a tool call."""
    tool = call.tool or ""
    ti = call.tool_input
    if tool in rules.FILE_TOOLS:
        path = ti.get("file_path") or ti.get("notebook_path")
        if isinstance(path, str) and path and not os.path.isabs(path) and call.cwd:
            path = os.path.join(call.cwd, path)
        fl = rules.file_location(path) if isinstance(path, str) and path else None
        return {"files": [fl] if fl else [], "action": "edit" if tool in rules.EDIT_TOOLS else "read",
                "program": None, "prefix_key": None, "file": rules.file_key(fl[0], fl[3]) if fl else None}
    if tool in rules.SHELL_TOOLS:
        cmd = ti.get("command") if isinstance(ti.get("command"), str) else ""
        files = [fl for fl in (rules.file_location(p) for p in rules.command_files(cmd, call.cwd)) if fl]
        return {"files": files, "action": "mention", "program": rules.command_program(cmd) or None,
                "prefix_key": rules.prefix_key(cmd), "file": None}
    return {"files": [], "action": None, "program": None, "prefix_key": None, "file": None}


def _links(call: Call, desc: dict) -> list:
    pairs = [("session", call.sid)]
    for repo_key, _name, worktree, rel in desc["files"]:
        pairs += [("repo", repo_key), ("worktree", worktree), ("file", rules.file_key(repo_key, rel))]
    if desc["program"]:
        pairs.append(("program", desc["program"]))
    return pairs


def _on_tool_success(con, call: Call, session, ts: float) -> dict:
    rid, loc = session
    desc = _describe(call)
    made = 0
    if desc["files"]:
        arid = store.add_record(con, "activity", ts, call.sid, agent_id=call.agent_id, tool=call.tool,
                                action=desc["action"], tool_use_id=call.tool_use_id,
                                info={"program": desc["program"]} if desc["program"] else None)
        if arid:
            made = 1
            store.add_links(con, arid, ts, _links(call, desc))
            here = loc[0] if loc else None
            store.add_links(con, rid, ts, [("touched_repo", f[0]) for f in desc["files"] if f[0] != here])
            if desc["action"] == "edit":
                _same_file_edit(con, call, desc["files"], ts)
    return {"activity": made, "files": len(desc["files"]), "resolved": _resolve(con, call, desc, ts),
            **_brief(con, call, rid, desc, ts)}


def _same_file_edit(con, call: Call, files, ts: float) -> None:
    for repo_key, name, _wt, rel in files:
        fk = rules.file_key(repo_key, rel)
        rows = con.execute(
            "SELECT r.id, r.session_id, r.ts FROM link l JOIN record r ON r.id = l.record_id"
            " WHERE l.type = 'file' AND l.key = ? AND l.ts >= ? AND r.kind = 'activity' AND r.action = 'edit'"
            " AND r.session_id != ? ORDER BY r.ts DESC",
            (fk, ts - WINDOW_S, call.sid)).fetchall()
        by_other: dict = {}
        for aid, other, ats in rows:
            by_other.setdefault(other, []).append((aid, ats))
        for other, hits in by_other.items():
            store.log_event(con, "would_deliver", ts, hook=call.hook, session_id=call.sid, agent_id=call.agent_id,
                            tool=call.tool, moment="same_file_edit", records=[a for a, _ in hits[:3]],
                            dedupe=f"same_file_edit|{call.sid}|{fk}|{other}",
                            detail={"repo": name, "file": rel, "age_s": round(ts - hits[0][1])})


def _resolve(con, call: Call, desc: dict, ts: float) -> int:
    """Resolve open failures of the same kind in this session. Like the
    replay's ``first_resolution``, a success counts only if it began after the
    failure came back: a concurrent call that started first is not the way
    past it. The start is the result time less ``duration_ms``."""
    response = call.payload.get("tool_response")
    response = response if isinstance(response, dict) else {}
    if (call.tool_input.get("run_in_background") or response.get("isAsync")
            or response.get("status") == "async_launched" or response.get("backgroundTaskId")):
        return 0  # a background launch answers before its work ends: no known outcome
    duration = call.payload.get("duration_ms")
    started = ts - duration / 1000.0 if isinstance(duration, (int, float)) and duration > 0 else ts
    rows = con.execute(
        "SELECT id, tool, info, ts FROM record WHERE kind = 'failure' AND status = 'open' AND session_id = ?",
        (call.sid,)).fetchall()
    resolved = 0
    for fid, tool, info, failed_at in rows:
        if tool != call.tool or failed_at > started:
            continue
        fi = json.loads(info) if info else {}
        if call.tool in rules.SHELL_TOOLS:
            same = fi.get("prefix_key") == desc["prefix_key"]
        elif fi.get("file") or desc["file"]:
            same = fi.get("file") == desc["file"]
        else:
            same = True
        if same:
            con.execute("UPDATE record SET status = 'resolved', ts_end = ?,"
                        " info = json_set(coalesce(info, '{}'), '$.resolved_by', ?) WHERE id = ?",
                        (ts, call.tool_use_id, fid))
            resolved += 1
    return resolved


def _on_tool_failure(con, call: Call, session, ts: float) -> dict:
    _rid, loc = session
    if call.payload.get("is_interrupt"):
        return {"interrupt": True}
    err = call.payload.get("error")
    err = err if isinstance(err, str) else ""
    desc = _describe(call)
    sig = rules.failure_signature(err)
    info = {"head": rules.error_head(err), "exit": rules.exit_code(err), "sig": sig, "program": desc["program"],
            "prefix_key": desc["prefix_key"], "file": desc["file"]}
    frid = store.add_record(con, "failure", ts, call.sid, agent_id=call.agent_id, tool=call.tool,
                            action=rules.failure_kind(err), status="open", tool_use_id=call.tool_use_id,
                            info={k: v for k, v in info.items() if v is not None})
    if frid:
        pairs = _links(call, desc) + [("signature", sig)]
        if not desc["files"] and loc:
            pairs += [("repo", loc[0]), ("worktree", os.path.realpath(loc[2]))]
        store.add_links(con, frid, ts, pairs)
    return {"failure": 1 if frid else 0, "signature": bool(sig)}


# ---------------------------------------------------------------- SessionEnd

def _on_session_end(con, call: Call, session, ts: float) -> dict:
    rid, _loc = session
    con.execute("UPDATE record SET status = 'ended', action = 'ended', ts_end = ? WHERE id = ?", (ts, rid))
    return {"reason": call.payload.get("reason")}


_HANDLERS = {
    "SessionStart": _on_session_start,
    "UserPromptSubmit": _on_prompt,
    "PostToolUse": _on_tool_success,
    "PostToolUseFailure": _on_tool_failure,
    "SessionEnd": _on_session_end,
}

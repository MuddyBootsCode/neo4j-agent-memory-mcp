"""The store: one SQLite file at user scope, shared by every repo and worktree.

Three tables (MUD-850 "Data store"):

* ``event``: append-only log. One ``observed`` row per hook call with the time
  the hook spent, one ``would_deliver`` row per notice Phase 3 would have shown,
  naming the records it would have shown, and one ``hook_error`` row per
  exception.
* ``record``: what was observed. ``session`` (one per session, ``ts_end`` is the
  last time a hook saw it), ``activity`` (a tool call that touched repo files)
  and ``failure`` (a failed tool call and its resolution).
* ``link``: record to the objects it is about, by type and key: ``repo`` (git
  common dir), ``worktree``, ``file`` (``<repo>::<repo-relative path>``),
  ``program``, ``signature``, ``ticket`` and ``session``, plus
  ``touched_repo`` from a session to a repo it touched from outside it. ``ts`` copies the
  record's time so a lookup by key can range on time. This table is the graph.

WAL mode lets many hook processes read while one writes. A hook takes the write
lock up front; if it is not free within ``BUSY_TIMEOUT_MS`` the hook skips and
counts the skip in a side file, so it never holds up a tool call.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

from swarm_memory import rules

ENV_DB = "SWARM_MEMORY_DB"
DEFAULT_DB = os.path.join(os.path.expanduser("~"), ".local", "share", "swarm-memory", "observe.sqlite3")
BUSY_TIMEOUT_MS = 100
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    hook TEXT,
    session_id TEXT,
    agent_id TEXT,
    tool TEXT,
    moment TEXT,
    records TEXT,
    dedupe TEXT,
    ms REAL,
    detail TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS event_dedupe ON event(dedupe) WHERE dedupe IS NOT NULL;
CREATE INDEX IF NOT EXISTS event_kind_ts ON event(kind, ts);

CREATE TABLE IF NOT EXISTS record (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    ts REAL NOT NULL,
    ts_end REAL,
    session_id TEXT NOT NULL,
    agent_id TEXT,
    tool TEXT,
    action TEXT,
    status TEXT,
    tool_use_id TEXT,
    info TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS record_one_session ON record(session_id) WHERE kind = 'session';
CREATE UNIQUE INDEX IF NOT EXISTS record_one_per_call ON record(tool_use_id, kind) WHERE tool_use_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS record_live_sessions ON record(ts_end) WHERE kind = 'session';
CREATE INDEX IF NOT EXISTS record_open_failures ON record(session_id) WHERE kind = 'failure' AND status = 'open';

CREATE TABLE IF NOT EXISTS link (
    record_id INTEGER NOT NULL REFERENCES record(id),
    type TEXT NOT NULL,
    key TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (record_id, type, key)
);
CREATE INDEX IF NOT EXISTS link_by_key ON link(type, key, ts);
"""


class Busy(Exception):
    """The write lock was not free within the busy timeout."""


def db_path(arg: str | None = None) -> str:
    return arg or os.environ.get(ENV_DB) or DEFAULT_DB


CONFIG_NAME = "config.json"


def config_path(db: str) -> str:
    """The local settings file beside the store, such as
    ``{"ticket_keys": ["ABC", "XYZ"]}``. Ticket prefixes live here, never in
    the repo."""
    return os.path.join(os.path.dirname(db) or ".", CONFIG_NAME)


def ticket_keys(db: str) -> tuple:
    """The configured ticket prefixes. With no file or no keys nothing names a
    ticket: sessions get no ticket link, and Moment 2 matches on files only."""
    try:
        with open(config_path(db), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return ()
    return rules.parse_ticket_keys(data.get("ticket_keys") if isinstance(data, dict) else None)


def connect(path: str) -> sqlite3.Connection:
    """Open the store, creating it (mode 0600) and its schema on first use."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, mode=0o700, exist_ok=True)
    if not os.path.exists(path):
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
    con = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    con.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    con.execute("PRAGMA synchronous=NORMAL")
    if con.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)
            con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        except sqlite3.OperationalError as exc:
            con.close()
            if _is_busy(exc):
                raise Busy() from exc
            raise
    return con


def _is_busy(exc: Exception) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def begin(con: sqlite3.Connection) -> None:
    """Take the write lock now, so reads and writes in one hook call see one state."""
    try:
        con.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        if _is_busy(exc):
            raise Busy() from exc
        raise


def _json(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def add_record(con, kind: str, ts: float, session_id: str, *, ts_end=None, agent_id=None, tool=None,
               action=None, status=None, tool_use_id=None, info=None) -> int | None:
    """Insert a record; None when this tool call already has one of this kind."""
    cur = con.execute(
        "INSERT OR IGNORE INTO record (kind, ts, ts_end, session_id, agent_id, tool, action, status, tool_use_id, info)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (kind, ts, ts_end, session_id, agent_id, tool, action, status, tool_use_id, _json(info)))
    return cur.lastrowid if cur.rowcount == 1 else None


def add_links(con, record_id: int, ts: float, pairs) -> None:
    con.executemany("INSERT OR IGNORE INTO link (record_id, type, key, ts) VALUES (?, ?, ?, ?)",
                    [(record_id, t, k, ts) for t, k in pairs if k])


def log_event(con, kind: str, ts: float, *, hook=None, session_id=None, agent_id=None, tool=None,
              moment=None, records=None, dedupe=None, ms=None, detail=None) -> bool:
    """Append an event; False when ``dedupe`` was already logged."""
    cur = con.execute(
        "INSERT OR IGNORE INTO event (ts, kind, hook, session_id, agent_id, tool, moment, records, dedupe, ms, detail)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, kind, hook, session_id, agent_id, tool, moment, _json(records), dedupe,
         None if ms is None else round(ms, 2), _json(detail)))
    return cur.rowcount == 1


def side_file(path: str, name: str) -> str:
    return f"{path}.{name}.log"


def append_line(path: str, fields) -> None:
    """Count a skip or an error that could not reach the store itself."""
    line = "\t".join(str(f) for f in fields) + "\n"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8", "replace"))
    finally:
        os.close(fd)


def now() -> float:
    return time.time()

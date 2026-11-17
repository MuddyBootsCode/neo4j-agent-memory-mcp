"""The store: one SQLite file at user scope, shared by every repo and worktree.

Three tables (MUD-850 "Data store"):

* ``event``: append-only log. One ``observed`` row per hook call with the time
  the hook spent, one ``would_deliver`` row per notice Phase 3 would have shown,
  naming the records it would have shown, and one ``hook_error`` row per
  exception.
* ``record``: what was observed. ``session`` (one per session, ``ts_end`` is the
  last time a hook saw it), ``activity`` (a file tool call on a repo file) and
  ``failure`` (a failed tool call).
* ``link``: record to the objects it is about, by type and key: ``repo`` (git
  common dir), ``worktree``, ``file`` (``<repo>::<repo-relative path>``),
  ``signature`` (the keyed hash), ``ticket`` and ``session``, plus
  ``touched_repo`` from a session to a repo it touched from outside it. ``ts``
  copies the record's time so a lookup by key can range on time. This table is
  the graph.

No text from a tool's output is stored. A failure keeps its signature only as
an HMAC under a random key kept beside the store (``<store>.key``, mode 0600),
so repeats still match and the hash cannot be checked against a guess. If the
key file goes missing while the store holds hashes, the next failure makes a
new key and logs a ``signature_key_new`` event: hashes from before it no longer
match those after it. A store that held text (schema version 1), values an
older rule let through (version 4), or anything read from shell command text
(versions 4 to 6), is migrated, then rewritten on a session hook with VACUUM
and a TRUNCATE checkpoint, so the replaced text is gone from the file's bytes as
well as its rows.

WAL mode lets many hook processes read while one writes. A hook takes the write
lock up front; if it is not free within ``BUSY_TIMEOUT_MS`` the hook skips and
counts the skip in a side file, so it never holds up a tool call.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time

from swarm_memory import rules

ENV_DB = "SWARM_MEMORY_DB"
DEFAULT_DB = os.path.join(os.path.expanduser("~"), ".local", "share", "swarm-memory", "observe.sqlite3")
BUSY_TIMEOUT_MS = 100
# 2: failures keep only a keyed signature hash; 3: the file holds no old text; 4: no recovery;
# 5: stored values meet the run 10 rules; 6: the file holds none of the values replaced at 5;
# 7: nothing read from shell command text; 8: the file holds none of the values removed at 7.
SCHEMA_VERSION = 8

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


KEY_BYTES = 32


def key_path(db: str) -> str:
    """The signature key beside the store: ``<store>.key``, mode 0600."""
    return f"{db}.key"


def signature_key(db: str) -> tuple[bytes, bool]:
    """The store's key for signature hashes, and whether this call made it.

    The key is made on first use from ``os.urandom`` and kept for the life
    of the store, so a signature always gives the same hash and repeats
    match. It is written to a temporary file and linked into place, so
    hooks that race to make it all end up with the one that won. A key file
    of the wrong size raises rather than being replaced."""
    path = key_path(db)
    try:
        with open(path, "rb") as fh:
            key = fh.read()
    except FileNotFoundError:
        key = None
    if key is not None:
        if len(key) != KEY_BYTES:
            raise ValueError(f"signature key {path} is {len(key)} bytes, not {KEY_BYTES}")
        return key, False
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, os.urandom(KEY_BYTES))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(tmp, path)
        made = True
    except FileExistsError:
        made = False
    finally:
        os.unlink(tmp)
    with open(path, "rb") as fh:
        return fh.read(), made


def signature_hash(key: bytes, signature: str) -> str:
    """The keyed hash a failure's signature is stored as: HMAC-SHA256, 32 hex
    characters. Without the key it cannot be checked against a guess."""
    return hmac.new(key, signature.encode("utf-8", "replace"), hashlib.sha256).hexdigest()[:32]


def _is_hash(key: str) -> bool:
    return len(key) == 32 and all(c in "0123456789abcdef" for c in key)


def migrate_to_hash_only(con: sqlite3.Connection, db: str) -> dict:
    """Schema version 1 to 2. Version 1 kept failure text: a redacted error
    head, and the signature in the record and as the signature link's key.
    Version 2 keeps only the signature's keyed hash, in both places.

    It runs in one write transaction and only while the store is still at
    version 1, so hooks that race to open the store migrate it once. A
    record or link already without text is left as it is. Returns counts."""
    con.execute("BEGIN IMMEDIATE")
    try:
        if con.execute("PRAGMA user_version").fetchone()[0] != 1:
            con.execute("ROLLBACK")
            return {}
        key, _made = signature_key(db)
        counts = {"failures": 0, "signatures_hashed": 0, "heads_dropped": 0, "links_hashed": 0}
        for rid, info in con.execute("SELECT id, info FROM record WHERE kind = 'failure'").fetchall():
            counts["failures"] += 1
            data = json.loads(info) if info else {}
            if "sig" not in data and "head" not in data:
                continue
            sig = data.pop("sig", None)
            counts["heads_dropped"] += data.pop("head", None) is not None
            if isinstance(sig, str) and sig:
                data["sig_hash"] = signature_hash(key, sig)
                counts["signatures_hashed"] += 1
            con.execute("UPDATE record SET info = ? WHERE id = ?", (_json(data or None), rid))
        for rid, text in con.execute("SELECT record_id, key FROM link WHERE type = 'signature'").fetchall():
            if not _is_hash(text):
                con.execute("UPDATE link SET key = ? WHERE record_id = ? AND type = 'signature' AND key = ?",
                            (signature_hash(key, text), rid, text))
                counts["links_hashed"] += 1
        con.execute(f"PRAGMA user_version={HASH_ONLY_VERSION}")
        con.execute("COMMIT")
        return counts
    except BaseException:
        con.execute("ROLLBACK")
        raise


HASH_ONLY_VERSION = 2  # failure text replaced by hashes in the rows; the file may still hold the old bytes
PURGED_VERSION = 3  # the file rewritten; failures still carry the fields recovery matching used
RECOVERY_DROPPED_VERSION = 4  # no recovery fields; values older rules let through may remain
RULES_APPLIED_VERSION = 5  # stored values meet the run 10 rules; the file may still hold the old bytes
RULES_PURGED_VERSION = 6  # the file rewritten after 5
SHELL_DROPPED_VERSION = 7  # nothing read from shell command text; the file may still hold the old bytes
# A version whose rows a migration rewrote, and the one its file reaches once rewritten too.
_PURGE_NEXT = {HASH_ONLY_VERSION: PURGED_VERSION, RULES_APPLIED_VERSION: RULES_PURGED_VERSION,
               SHELL_DROPPED_VERSION: SCHEMA_VERSION}
SHELL_TOOLS = ("Bash", "PowerShell")  # whose command text earlier versions read


def purge_freed_text(con: sqlite3.Connection) -> bool:
    """Version 2 to 3, 5 to 6 and 7 to 8: rewrite the file so the text a migration
    replaced is gone from its bytes, not only from its rows. SQLite leaves a
    replaced value in the page's free space and in older WAL frames. VACUUM
    writes a fresh copy of the database, and a TRUNCATE checkpoint moves it
    into the file and empties the WAL.

    It needs the write lock, and no other connection may still be reading an
    older snapshot. When either is busy it stops, leaves the version as it
    was, and a later session hook tries again. True when the file was rewritten."""
    target = _PURGE_NEXT.get(con.execute("PRAGMA user_version").fetchone()[0])
    if target is None:
        return False
    try:
        con.execute("VACUUM")
        if con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
            return False
        con.execute(f"PRAGMA user_version={target}")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return True
    except sqlite3.OperationalError as exc:
        if _is_busy(exc):
            return False
        raise


RECOVERY_KEYS = ("prefix_key", "file", "where", "resolved_by")


def drop_recovery(con: sqlite3.Connection) -> dict:
    """Version 3 to 4. Version 3 matched a failure to the later call that got
    past it: a failure kept a status, the call that resolved it and when, and
    the hashed command, path and directory it was matched on. Version 4 keeps
    none of that, and the index over open failures goes. It runs in one write
    transaction and only while the store is still at version 3. Returns counts."""
    con.execute("BEGIN IMMEDIATE")
    try:
        if con.execute("PRAGMA user_version").fetchone()[0] != PURGED_VERSION:
            con.execute("ROLLBACK")
            return {}
        counts = {"failures": 0, "changed": 0}
        rows = con.execute("SELECT id, info, status, ts_end FROM record WHERE kind = 'failure'").fetchall()
        for rid, info, status, ts_end in rows:
            counts["failures"] += 1
            data = json.loads(info) if info else {}
            kept = {k: v for k, v in data.items() if k not in RECOVERY_KEYS}
            if kept != data or status is not None or ts_end is not None:
                con.execute("UPDATE record SET info = ?, status = NULL, ts_end = NULL WHERE id = ?",
                            (_json(kept or None), rid))
                counts["changed"] += 1
        con.execute("DROP INDEX IF EXISTS record_open_failures")
        con.execute(f"PRAGMA user_version={RECOVERY_DROPPED_VERSION}")
        con.execute("COMMIT")
        return counts
    except BaseException:
        con.execute("ROLLBACK")
        raise


def apply_current_rules(con: sqlite3.Connection) -> dict:
    """Version 4 to 5: bring values stored under older rules to today's.

    * A program that is not one bare name leaves its record and its link. An
      older rule could store a wrapper's whole argument there
      (``env -S 'curl --user ...'``).
    * A failed file tool call's file link goes when no file is at that path in
      a worktree the record links. An older rule kept the path whether or not
      the file existed, so any text given as a path was stored.
    * A ``hook_error`` keeps the exception's type, not its message.

    A bare word an older rule took from after a wrapper option
    (``sudo -u alice id``) cannot be told from a program without the command,
    which was never stored, so it stays here; ``drop_shell_records()`` then
    removes every program. It runs in one write transaction and only while the
    store is still at version 4. Returns counts."""
    con.execute("BEGIN IMMEDIATE")
    try:
        if con.execute("PRAGMA user_version").fetchone()[0] != RECOVERY_DROPPED_VERSION:
            con.execute("ROLLBACK")
            return {}
        counts = {"programs": 0, "program_links": 0, "file_links": 0, "hook_errors": 0}
        rows = con.execute("SELECT id, info FROM record WHERE kind IN ('activity', 'failure') AND info IS NOT NULL")
        for rid, info in rows.fetchall():
            data = json.loads(info)
            if "program" in data and not rules.bare_name(data["program"]):
                del data["program"]
                con.execute("UPDATE record SET info = ? WHERE id = ?", (_json(data or None), rid))
                counts["programs"] += 1
        for rid, key in con.execute("SELECT record_id, key FROM link WHERE type = 'program'").fetchall():
            if not rules.bare_name(key):
                con.execute("DELETE FROM link WHERE record_id = ? AND type = 'program' AND key = ?", (rid, key))
                counts["program_links"] += 1
        tools = sorted(rules.FILE_TOOLS)
        failed = con.execute("SELECT id FROM record WHERE kind = 'failure' AND tool IN (%s)" % ",".join("?" * len(tools)),
                             tools).fetchall()
        for (rid,) in failed:
            roots = [k for (k,) in con.execute("SELECT key FROM link WHERE record_id = ? AND type = 'worktree'", (rid,))]
            for (key,) in con.execute("SELECT key FROM link WHERE record_id = ? AND type = 'file'", (rid,)).fetchall():
                rel = key.split("::", 1)[1] if "::" in key else ""
                if not (rel and any(os.path.isfile(os.path.join(root, rel)) for root in roots)):
                    con.execute("DELETE FROM link WHERE record_id = ? AND type = 'file' AND key = ?", (rid, key))
                    counts["file_links"] += 1
        for eid, detail in con.execute("SELECT id, detail FROM event WHERE kind = 'hook_error'").fetchall():
            data = json.loads(detail) if detail else {}
            kept = {"type": rules.bare_name(data.get("type"))}
            if kept != data:
                con.execute("UPDATE event SET detail = ? WHERE id = ?", (_json(kept if kept["type"] else None), eid))
                counts["hook_errors"] += 1
        con.execute(f"PRAGMA user_version={RULES_APPLIED_VERSION}")
        con.execute("COMMIT")
        return counts
    except BaseException:
        con.execute("ROLLBACK")
        raise


def drop_shell_records(con: sqlite3.Connection) -> dict:
    """Version 5 or 6 to 7: remove what earlier versions read from shell
    command text, since what a command touched cannot be told from its text.

    * Every stored program, in records and in links.
    * Activity records of shell commands (``mention``) and their links.
    * A failed shell command's file, repo and worktree links, which came from
      the paths its command named. It takes its session's repo and worktree
      instead, as a failure that names no file does.
    * ``touched_repo`` links, rebuilt from the file tool activity that is left.
    * The activity and file counts an ``observed`` row logged for a shell
      command.

    It runs in one write transaction and only while the store is at version 5
    or 6. ``purge_freed_text()`` then rewrites the file on a session hook.
    Returns counts."""
    con.execute("BEGIN IMMEDIATE")
    try:
        if con.execute("PRAGMA user_version").fetchone()[0] not in (RULES_APPLIED_VERSION, RULES_PURGED_VERSION):
            con.execute("ROLLBACK")
            return {}
        counts = {"mentions": 0, "mention_links": 0, "programs": 0, "program_links": 0, "shell_failures": 0,
                  "touched_repo_links_before": 0, "touched_repo_links_after": 0, "observed_rows": 0}
        mentions = con.execute("SELECT id FROM record WHERE kind = 'activity' AND (action = 'mention' OR tool IN (?, ?))",
                               SHELL_TOOLS).fetchall()
        for (rid,) in mentions:
            counts["mention_links"] += con.execute("DELETE FROM link WHERE record_id = ?", (rid,)).rowcount
            con.execute("DELETE FROM record WHERE id = ?", (rid,))
        counts["mentions"] = len(mentions)
        for rid, info in con.execute("SELECT id, info FROM record WHERE info IS NOT NULL").fetchall():
            data = json.loads(info)
            if "program" in data:
                del data["program"]
                con.execute("UPDATE record SET info = ? WHERE id = ?", (_json(data or None), rid))
                counts["programs"] += 1
        counts["program_links"] = con.execute("DELETE FROM link WHERE type = 'program'").rowcount
        failed = con.execute(
            "SELECT r.id, r.session_id, r.ts FROM record r WHERE r.kind = 'failure' AND r.tool IN (?, ?)"
            " AND EXISTS (SELECT 1 FROM link l WHERE l.record_id = r.id AND l.type = 'file')", SHELL_TOOLS).fetchall()
        for rid, sid, ts in failed:
            con.execute("DELETE FROM link WHERE record_id = ? AND type IN ('file', 'repo', 'worktree')", (rid,))
            con.execute(
                "INSERT OR IGNORE INTO link (record_id, type, key, ts) SELECT ?, l.type, l.key, ? FROM link l"
                " JOIN record s ON s.id = l.record_id WHERE s.kind = 'session' AND s.session_id = ?"
                " AND l.type IN ('repo', 'worktree')", (rid, ts, sid))
        counts["shell_failures"] = len(failed)
        counts["touched_repo_links_before"] = con.execute("DELETE FROM link WHERE type = 'touched_repo'").rowcount
        counts["touched_repo_links_after"] = con.execute(
            "INSERT OR IGNORE INTO link (record_id, type, key, ts)"
            " SELECT s.id, 'touched_repo', ar.key, min(a.ts) FROM record s"
            " JOIN record a ON a.kind = 'activity' AND a.session_id = s.session_id"
            " JOIN link ar ON ar.record_id = a.id AND ar.type = 'repo'"
            " WHERE s.kind = 'session' AND NOT EXISTS (SELECT 1 FROM link own WHERE own.record_id = s.id"
            " AND own.type = 'repo' AND own.key = ar.key) GROUP BY s.id, ar.key").rowcount
        observed = con.execute("SELECT id, detail FROM event WHERE kind = 'observed' AND tool IN (?, ?)"
                               " AND detail IS NOT NULL", SHELL_TOOLS).fetchall()
        for eid, detail in observed:
            data = json.loads(detail)
            kept = {k: v for k, v in data.items() if k not in ("activity", "files")}
            if kept != data:
                con.execute("UPDATE event SET detail = ? WHERE id = ?", (_json(kept or None), eid))
                counts["observed_rows"] += 1
        con.execute(f"PRAGMA user_version={SHELL_DROPPED_VERSION}")
        con.execute("COMMIT")
        return counts
    except BaseException:
        con.execute("ROLLBACK")
        raise


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
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if version in (0, 1, PURGED_VERSION, RECOVERY_DROPPED_VERSION, RULES_APPLIED_VERSION, RULES_PURGED_VERSION):
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)
            if version == 0:
                con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")  # a new store never held text
            elif version == 1:
                migrate_to_hash_only(con, path)  # to version 2; purge_freed_text() takes it to 3
            else:
                if version == PURGED_VERSION:
                    drop_recovery(con)  # to version 4
                if version in (PURGED_VERSION, RECOVERY_DROPPED_VERSION):
                    apply_current_rules(con)  # to version 5
                drop_shell_records(con)  # to version 7; purge_freed_text() takes it to 8
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

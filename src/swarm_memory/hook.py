#!/usr/bin/env python3
"""Phase 2 observe-and-log hook (MUD-850). One entry point for SessionStart,
UserPromptSubmit, PostToolUse, PostToolUseFailure and SessionEnd.

It reads the hook payload on stdin, writes records to the store, prints nothing
and always exits 0, so it can neither fail nor block a tool call, and the model
never sees it. Run it with ``python3 -I -S <path>/hook.py [--db PATH]``: ``-I``
and ``-S`` keep site-packages and environment variables out of a process that
starts on every tool call. The store is ``--db``, else ``$SWARM_MEMORY_DB``,
else ``~/.local/share/swarm-memory/observe.sqlite3``.
"""

import os
import sys
import time

T0 = time.perf_counter()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    db = argv[1] if len(argv) >= 2 and argv[0] == "--db" else None
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from swarm_memory import observe

        observe.run(sys.stdin.read(), db_arg=db, t0=T0)
    except BaseException as exc:  # noqa: BLE001 - observe.run counts its own errors; this is the package failing to load
        _count_outside_store(db, exc)
    return 0


def _count_outside_store(db, exc) -> None:
    """Count an error that happened before the store code could, in the side file."""
    try:
        path = db or os.environ.get("SWARM_MEMORY_DB") or os.path.join(
            os.path.expanduser("~"), ".local", "share", "swarm-memory", "observe.sqlite3")
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        fd = os.open(f"{path}.errors.log", os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(fd, f"{time.time():.3f}\thook.py\t\t{type(exc).__name__}\n".encode())
        finally:
            os.close(fd)
    except BaseException:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)

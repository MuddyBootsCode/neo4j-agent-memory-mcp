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
    except BaseException:  # noqa: BLE001 - observe.run counts what it can; the tool call must go on
        pass
    return 0


if __name__ == "__main__":
    main()
    sys.exit(0)

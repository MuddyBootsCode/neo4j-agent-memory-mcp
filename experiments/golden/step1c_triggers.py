"""Step 1c: write trigger sentences for every lesson in a pool (MUD-457).

Recall matches the prompt against the lesson, which is the fix, and the fix
only matches once you already know it. This generates the other side of
that match: two to four sentences per lesson describing the moment it
applies, in the words a developer would be using then. E2 scores retrieval
against them.

Offline and free — one local model call per lesson, no labels touched, the
scratch database untouched. Resumable: lessons already in triggers.json are
skipped, so an interrupted run continues where it stopped.

    GOLDEN_RUN=p5-e2 GOLDEN_POOL_FROM=results/p4-live/pool.json \\
    uv run --no-sync --project ../.. python -u step1c_triggers.py

Keyed by lesson id, which is the canonical text's hash and does not move
when the embedded string changes (MUD-456), so one generation serves every
E2 variant.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from lib import HERE, load_json, result_path, save_json

# Same basenames the context prefix uses, so a trigger sentence and the
# prefix name the same files.
MAX_FILES = 4
SAVE_EVERY = 10


def _files_line(files: list[str]) -> str:
    names = sorted({os.path.basename(p) for p in files if p})
    return ", ".join(names[:MAX_FILES])


async def main() -> None:
    src = os.environ.get("GOLDEN_POOL_FROM")
    if src:
        src = os.path.join(HERE, src) if not os.path.isabs(src) else src
        with open(src, encoding="utf-8") as fh:
            pool = json.load(fh)
    else:
        pool = load_json("pool.json")
    if not pool:
        raise SystemExit("no pool: set GOLDEN_POOL_FROM=results/<run>/pool.json")

    triggers: dict[str, list[str]] = load_json("triggers.json", {}) or {}
    todo = [it for it in pool if it["id"] not in triggers]
    print(f"{len(pool)} lessons, {len(triggers)} already generated, {len(todo)} to go")
    if not todo:
        return

    import agent_memory_mcp.baml_client.async_client as _async_client
    from agent_memory_mcp.providers import default_baml_options

    options = default_baml_options()
    t0 = time.time()
    failed = 0
    for n, it in enumerate(todo, 1):
        try:
            result = await _async_client.b.LessonTriggers(
                kind=it["kind"],
                lesson=it["text"],
                files=_files_line(it.get("files") or []),
                baml_options=options,
            )
            said = [s.strip() for s in (result.triggers or []) if s and s.strip()]
        except Exception as e:
            # A lesson with no triggers falls back to its own text at
            # scoring time, which is the baseline. Never a reason to stop.
            print(f"  [{it['id']}] failed: {e}")
            said = []
            failed += 1
        triggers[it["id"]] = said
        if n % SAVE_EVERY == 0 or n == len(todo):
            save_json("triggers.json", triggers)
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)} lessons, {rate:.1f}s each, "
                  f"{rate * (len(todo) - n) / 60:.0f} min left")

    save_json("triggers.json", triggers)
    counts = [len(v) for v in triggers.values()]
    empty = sum(1 for c in counts if c == 0)
    print(f"\ntriggers for {len(triggers)} lessons in {(time.time() - t0) / 60:.0f} min; "
          f"{sum(counts) / len(counts):.1f} per lesson, {empty} with none ({failed} calls failed)")
    print(f"written to {result_path('triggers.json')}")


if __name__ == "__main__":
    asyncio.run(main())

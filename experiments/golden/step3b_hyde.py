"""Step 3b: hypothetical lessons per query (MUD-459, P5 E4).

HyDE: the stored text is answer-shaped and a prompt is a question, so this
guesses what a past session might have written down that would help, and
step5 fuses each guess as its own vector leg. Never concatenated into the
prompt — the situation card (MUD-458) measured what more text in the query
string costs.

Offline and free: one call per query on the gate model, the same one the
recall gate uses, since this is what would run inside the hook's budget if
it ships. Resumable, keyed by query_id.

    GOLDEN_RUN=p5-e4 GOLDEN_QUERIES_FROM=results/p4-live/queries.json \\
    uv run --no-sync --project ../.. python -u step3b_hyde.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from lib import HERE, load_json, result_path, save_json

MAX_FILES = 4
SAVE_EVERY = 10


def _files_line(files: list[str]) -> str:
    names = sorted({os.path.basename(p) for p in files if p})
    return ", ".join(names[:MAX_FILES])


async def main() -> None:
    src = os.environ.get("GOLDEN_QUERIES_FROM")
    if src:
        src = os.path.join(HERE, src) if not os.path.isabs(src) else src
        with open(src, encoding="utf-8") as fh:
            queries = json.load(fh)
    else:
        queries = load_json("queries.json")
    if not queries:
        raise SystemExit("no queries: set GOLDEN_QUERIES_FROM=results/<run>/queries.json")

    rewrites: dict[str, list[str]] = load_json("rewrites.json", {}) or {}
    todo = [q for q in queries if str(q["query_id"]) not in rewrites]
    print(f"{len(queries)} queries, {len(rewrites)} already generated, {len(todo)} to go")
    if not todo:
        return

    import agent_memory_mcp.baml_client.async_client as _async_client
    from agent_memory_mcp.providers import gate_baml_options

    options = gate_baml_options()
    t0 = time.time()
    failed = 0
    for n, q in enumerate(todo, 1):
        try:
            result = await _async_client.b.HypotheticalLessons(
                prompt=q["prompt"],
                files=_files_line(q.get("files") or []),
                baml_options=options,
            )
            said = [s.strip() for s in (result.lessons or []) if s and s.strip()]
        except Exception as e:
            # A query with no guesses falls back to its prompt leg alone,
            # which is the baseline. Never a reason to stop.
            print(f"  [q{q['query_id']}] failed: {e}")
            said = []
            failed += 1
        rewrites[str(q["query_id"])] = said
        if n % SAVE_EVERY == 0 or n == len(todo):
            save_json("rewrites.json", rewrites)
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)} queries, {rate:.1f}s each")

    save_json("rewrites.json", rewrites)
    counts = [len(v) for v in rewrites.values()]
    print(f"\nrewrites for {len(rewrites)} queries in {(time.time() - t0) / 60:.0f} min; "
          f"{sum(counts) / len(counts):.1f} per query, {sum(1 for c in counts if c == 0)} with none "
          f"({failed} calls failed)")
    print(f"written to {result_path('rewrites.json')}")


if __name__ == "__main__":
    asyncio.run(main())

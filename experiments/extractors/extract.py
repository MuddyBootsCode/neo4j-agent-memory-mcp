"""Run one extractor variant over the shared source chunks.

    python extract.py --variant described --chunks 30 [--reset]

Each variant is real baml_src/extraction.baml text, mutated by a transform in
variants.py and executed on its own BamlRuntime — so the prompt and type
descriptions are rendered by BAML itself, not approximated. Production
baml_src/ is never modified and nothing is regenerated.

Every call appends a trace (rendered prompt, raw response, usage, timing) to
results/traces/<variant>.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baml_runtime as br
import source
from store import SubGraph
from variants import VARIANTS, baml_for


async def extract_one(variant, baml_text, chunk):
    parsed, trace = await br.extract(variant, baml_text, chunk["text"], chunk["id"])
    br.append_trace(variant, trace)
    return parsed, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    br.load_env()
    prod = br.production_baml()
    baml_text = baml_for(a.variant, prod)

    items = source.chunks(a.chunks)
    print(f"{a.variant}: {len(items)} chunks — {VARIANTS[a.variant]['blurb']}")

    with SubGraph(a.variant) as g:
        g.ensure_schema()
        if a.reset:
            print(f"  reset: deleted {g.reset()} nodes")

        t0 = time.perf_counter()
        ok = err = 0
        totals = [0, 0, 0]

        async def run_all():
            nonlocal ok, err, totals
            sem = asyncio.Semaphore(a.workers)

            async def one(chunk):
                async with sem:
                    return chunk, await extract_one(a.variant, baml_text, chunk)

            for fut in asyncio.as_completed([one(c) for c in items]):
                try:
                    chunk, (result, _trace) = await fut
                except Exception as e:
                    err += 1
                    print(f"  FAILED {type(e).__name__}: {str(e)[:120]}")
                    continue
                counts = g.write(result, chunk["id"])
                totals = [t + c for t, c in zip(totals, counts)]
                ok += 1
                if ok % 5 == 0:
                    print(f"  {ok}/{len(items)} chunks", flush=True)

        asyncio.run(run_all())

        stats, types = g.stats()
        print(
            f"\n{a.variant}: {ok} ok, {err} failed in {time.perf_counter()-t0:.0f}s"
        )
        print(f"  emitted: {totals[0]} entities, {totals[1]} relations, {totals[2]} prefs")
        print(f"  stored:  {stats}")
        print(f"  types:   {types[:8]}")
        print(f"  traces:  {br.trace_path(a.variant)}")


if __name__ == "__main__":
    main()

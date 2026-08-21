"""Run the local judge over the selected queries.

    python run.py                 # all selected queries, resumable
    python run.py --limit 1       # smoke test

One Ollama call per query, judging its whole candidate set — not one call per
pair, which would be 10x the calls for the same information.

Sequential by default: Ollama serves one model instance, so concurrent requests
queue behind each other and only make the per-call latency in the trace a lie.

Resumable: a query already present in results/verdicts.json is skipped, so a
run interrupted after twenty queries costs twenty calls, not thirty.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from runtime import CONFIGS, append_trace, judge  # noqa: E402
from selection import candidates_text  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def verdicts_path(config, corpus="production"):
    tail = "" if corpus == "production" else f".{corpus}"
    return os.path.join(RESULTS, f"verdicts.{config}{tail}.json")


def load_verdicts(config, corpus="production"):
    path = verdicts_path(config, corpus)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_verdicts(v, config, corpus="production"):
    with open(verdicts_path(config, corpus), "w") as fh:
        json.dump(v, fh, indent=1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true", help="re-judge queries already done")
    ap.add_argument("--config", default="think-reason", choices=list(CONFIGS))
    ap.add_argument("--corpus", default="production")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    with open(os.path.join(RESULTS, f"selected.{args.corpus}.json")) as fh:
        selected = json.load(fh)

    done = {} if args.redo else load_verdicts(args.config, args.corpus)
    todo = [q for q in selected if str(q["query_id"]) not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"corpus {args.corpus} · config {cfg['name']} — {cfg['blurb']}")
    print(f"{len(todo)} to judge, {len(done)} cached")
    started = time.time()

    for n, q in enumerate(todo, 1):
        t0 = time.time()
        parsed, trace = await judge(
            q["prompt"], candidates_text(q), q["query_id"], cfg)
        append_trace(trace, f"{cfg['name']}.{args.corpus}")

        verdicts = (parsed or {}).get("verdicts") or []
        by_id = {}
        for v in verdicts:
            try:
                by_id[int(v["id"])] = v
            except (KeyError, TypeError, ValueError):
                continue

        valid = {c["id"] for c in q["candidates"]}
        # A model that invents an id has not judged a candidate, and a model
        # that skips one has not either. Both are recorded rather than patched
        # over, because schema compliance is the first gate on a local model.
        done[str(q["query_id"])] = {
            "query_id": q["query_id"],
            "ok": trace.get("ok", False),
            "json_ok": trace.get("json_ok"),
            "error": trace.get("error"),
            "duration_ms": trace.get("duration_ms"),
            "attempts": trace.get("attempts"),
            "input_tokens": trace.get("input_tokens"),
            "output_tokens": trace.get("output_tokens"),
            "thinking": trace.get("thinking"),
            "raw_response": trace.get("raw_response"),
            "verdicts": {str(k): v for k, v in by_id.items() if k in valid},
            "hallucinated_ids": sorted(set(by_id) - valid),
            "missing_ids": sorted(valid - set(by_id)),
        }
        save_verdicts(done, args.config, args.corpus)

        rec = done[str(q["query_id"])]
        flag = "ok " if rec["ok"] else "FAIL"
        print(
            f"[{n}/{len(todo)}] q{q['query_id']:>3} {flag} "
            f"{time.time() - t0:6.1f}s  "
            f"verdicts={len(rec['verdicts'])}/{len(q['candidates'])} "
            f"miss={len(rec['missing_ids'])} halluc={len(rec['hallucinated_ids'])} "
            f"out_tok={rec['output_tokens']}"
            + (f"  {rec['error'][:80]}" if rec["error"] else "")
        )

    print(f"total {time.time() - started:.0f}s -> {verdicts_path(args.config, args.corpus)}")


if __name__ == "__main__":
    asyncio.run(main())

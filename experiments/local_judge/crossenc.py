"""Cross-encoder rerankers as the gate, on the same candidates.

    python crossenc.py

A bi-encoder embedded each memory once, at store time, and compares two
independently-made vectors at query time. A cross-encoder puts the prompt and
one memory through the model together, so it can compare their tokens — which
is the only reason it can tell GRA-602 from GRA-577, and the same reason the
LLM judge could.

Scoring is cheap enough to run the whole 200-query set, which buys an honest
threshold: a cross-encoder emits an uncalibrated score, so the cutoff is swept
on the `calibrate` split and reported on `report`. Tuning it on the reported
queries would be exactly the mistake MUD-386 exists to fix.

Writes results/crossenc.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from score import RELEVANT, prf  # noqa: E402
from selection import build  # noqa: E402

RESULTS = os.path.join(HERE, "results")

MODELS = [
    "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
]

# The 30 queries the LLM configs ran on — the first 30 of the report split.
# Recomputed rather than read back, so the two sets cannot drift apart.
def llm_query_ids(corpus):
    return {q["query_id"] for q in build(30, "report", corpus)}


def rows_for(queries, scores):
    """One row per (query, candidate) with the cross-encoder score attached."""
    out = []
    for q in queries:
        for c in q["candidates"]:
            out.append({
                "query_id": q["query_id"],
                "split": q["split"],
                "id": c["id"],
                "sim": c["sim"],
                "kind": c["kind"],
                "claude": c["claude"],
                "ce": scores[(q["query_id"], c["id"])],
            })
    return out


def by_query(rows):
    out = {}
    for r in rows:
        out.setdefault(r["query_id"], []).append(r)
    return out


def top_k(rows, k):
    """Best k per query by cross-encoder score.

    The shape a reranker is actually built for. Its scores are not calibrated
    across queries — median 0.001 with a p90 of 0.03 — so a single global
    cutoff cannot use ordering signal that AUC says is present. Ranking within
    a query can.
    """
    picked = []
    for group in by_query(rows).values():
        picked.extend(sorted(group, key=lambda r: -r["ce"])[:k])
    return picked


def relative(rows, frac):
    """Keep candidates within `frac` of the best score for their own query."""
    picked = []
    for group in by_query(rows).values():
        top = max(r["ce"] for r in group)
        if top <= 0:
            continue
        picked.extend([r for r in group if r["ce"] >= frac * top])
    return picked


def sweep_relative(rows, steps=100):
    best = None
    for i in range(1, steps + 1):
        frac = i / steps
        m = prf(relative(rows, frac), rows)
        if best is None or m["f1"] > best[1]["f1"]:
            best = (frac, m)
    return best


def sweep(rows, lo, hi, steps=200):
    """Threshold that maximises F1 on whatever rows are handed in."""
    best = None
    for i in range(steps + 1):
        t = lo + (hi - lo) * i / steps
        picked = [r for r in rows if r["ce"] >= t]
        m = prf(picked, rows)
        if best is None or m["f1"] > best[1]["f1"]:
            best = (t, m)
    return best


def evaluate(model_name, queries, llm_ids):
    from sentence_transformers import CrossEncoder

    t0 = time.time()
    model = CrossEncoder(model_name, device="mps")
    load_s = time.time() - t0

    pairs, keys = [], []
    for q in queries:
        for c in q["candidates"]:
            pairs.append((q["prompt"], c["text"]))
            keys.append((q["query_id"], c["id"]))

    t0 = time.time()
    raw = model.predict(pairs, batch_size=32, show_progress_bar=False)
    score_s = time.time() - t0
    scores = {k: float(v) for k, v in zip(keys, raw)}

    # Per-query latency measured on its own: a batch of 2,000 pairs amortises
    # overhead that a live hook scoring 10 pairs would pay in full.
    single = queries[0]
    single_pairs = [(single["prompt"], c["text"]) for c in single["candidates"]]
    model.predict(single_pairs, batch_size=32, show_progress_bar=False)  # warm
    t0 = time.time()
    for _ in range(5):
        model.predict(single_pairs, batch_size=32, show_progress_bar=False)
    per_query_ms = (time.time() - t0) / 5 * 1000

    rows = rows_for(queries, scores)
    lo, hi = min(r["ce"] for r in rows), max(r["ce"] for r in rows)
    calibrate = [r for r in rows if r["split"] == "calibrate"]
    report = [r for r in rows if r["split"] == "report"]
    llm_set = [r for r in rows if r["query_id"] in llm_ids]

    threshold, cal_metrics = sweep(calibrate, lo, hi)
    frac, cal_rel = sweep_relative(calibrate)

    # Cosine top-k on identical rows, so top-k is compared against the same
    # shape rather than against the global cutoff it beats trivially.
    def cos_top_k(group_rows, k):
        picked = []
        for group in by_query(group_rows).values():
            picked.extend(sorted(group, key=lambda r: -r["sim"])[:k])
        return picked

    gates = {}
    for k in (1, 2, 3, 4, 5):
        gates[f"top_{k}"] = {
            "report": prf(top_k(report, k), report),
            "llm_30": prf(top_k(llm_set, k), llm_set),
        }
        gates[f"cosine_top_{k}"] = {
            "report": prf(cos_top_k(report, k), report),
            "llm_30": prf(cos_top_k(llm_set, k), llm_set),
        }
    gates["relative"] = {
        "frac": round(frac, 3),
        "report": prf(relative(report, frac), report),
        "llm_30": prf(relative(llm_set, frac), llm_set),
    }

    return {
        "gates": gates,
        "calibrate_relative": cal_rel,
        "model": model_name,
        "threshold": round(threshold, 4),
        "score_range": [round(lo, 4), round(hi, 4)],
        "load_s": round(load_s, 1),
        "batch_s_2000_pairs": round(score_s, 2),
        "per_query_ms": round(per_query_ms, 1),
        "calibrate": cal_metrics,
        "report": prf([r for r in report if r["ce"] >= threshold], report),
        "llm_30": prf([r for r in llm_set if r["ce"] >= threshold], llm_set),
        "scores": {f"{k[0]}|{k[1]}": round(v, 5) for k, v in scores.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--corpus", default="production")
    args = ap.parse_args()

    queries = build(args.queries, split=None, corpus=args.corpus)
    llm_ids = llm_query_ids(args.corpus)
    pairs = sum(len(q["candidates"]) for q in queries)
    print(f"corpus {args.corpus} — {len(queries)} queries, {pairs} pairs, "
          f"{len(llm_ids)} of them shared with the LLM runs")

    out = []
    for name in args.models:
        print(f"\n--- {name}")
        try:
            r = evaluate(name, queries, llm_ids)
        except Exception as exc:
            print(f"    FAILED {type(exc).__name__}: {str(exc)[:200]}")
            continue
        out.append(r)
        print(f"    load {r['load_s']}s  batch {r['batch_s_2000_pairs']}s  "
              f"per query {r['per_query_ms']}ms")
        print(f"    threshold {r['threshold']} swept on calibrate "
              f"(range {r['score_range'][0]} to {r['score_range'][1]})")
        for split in ("calibrate", "report", "llm_30"):
            m = r[split]
            print(f"    global    {split:<10} inj {m['per_query']:.2f}  "
                  f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")
        print(f"    {'gate':<16}{'inj':>6}{'P':>8}{'R':>8}{'F1':>8}   (report split)")
        for name, gm in r["gates"].items():
            m = gm["report"]
            print(f"    {name:<16}{m['per_query']:>6.2f}{m['precision']:>8.3f}"
                  f"{m['recall']:>8.3f}{m['f1']:>8.3f}")

    tail = "" if args.corpus == "production" else f".{args.corpus}"
    path = os.path.join(RESULTS, f"crossenc{tail}.json")
    with open(path, "w") as fh:
        json.dump({"relevant_at": RELEVANT, "corpus": args.corpus,
                   "models": out}, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

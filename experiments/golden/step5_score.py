"""Step 5: score retrieval configs against the labels, with true recall.

Configs, all importing production code rather than copying it:

- cosine20  — the hybrid query with threshold 0 and no anchor boost, top 20.
              Its recall is the retriever ceiling: what any reranker or gate
              downstream can reach at all.
- D         — production ranking (threshold 0.45, anchor boost 0.15), top
              GATE_DEPTH, ungated. What the gate sees.
- E         — D screened by the production ScreenRecalledMemories gate
              (BAML, routed wherever NAM_LLM_PROVIDER points; ollama here).
              What the hook injects.

Metrics per config: precision (relevant / injected), recall (relevant
injected / relevant in the whole pool for that query), coverage (queries
with >= 1 relevant injected), items per query, and P@5. Recall is over the
full labeled pool, not the retrieved top-k.

Latency per stage (embed, vector query, gate) is measured here directly,
p50/p95 over queries.
"""

from __future__ import annotations

import asyncio
import os
import time

from lib import GOLDEN_DB, lesson_id, lesson_text, load_json, save_json
from mem import open_client

CAP = 5


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


async def _retrieve(client, q: dict, *, threshold: float, boost: float, limit: int) -> tuple[list[dict], dict]:
    from agent_memory_mcp.mcp._coding_tools import (
        CODING_MEMORY_INDEX, HYBRID_CANDIDATES, _HYBRID_QUERY, _embed, _render_memory,
    )

    timing = {}
    t0 = time.perf_counter()
    embedding = await _embed(client, q["prompt"])
    timing["embed_ms"] = (time.perf_counter() - t0) * 1000
    if embedding is None:
        return [], timing
    t0 = time.perf_counter()
    rows = await client.graph.execute_read(_HYBRID_QUERY, {
        "index": CODING_MEMORY_INDEX, "embedding": embedding,
        "candidates": max(HYBRID_CANDIDATES, limit), "threshold": threshold,
        "anchor_boost": boost, "limit": limit, "repo": q["repo"],
        "files": q["files"], "task_key": None,
    })
    timing["vector_ms"] = (time.perf_counter() - t0) * 1000
    items = []
    for row in rows:
        m = _render_memory(row)
        props = {k: v for k, v in (row.get("props") or {}).items() if k != "embedding"}
        m["id"] = lesson_id(q["repo"], m["kind"], lesson_text(m["kind"], props))
        items.append(m)
    return items, timing


async def main() -> None:
    from agent_memory_mcp.mcp._coding_tools import ANCHOR_BOOST, GATE_DEPTH, HYBRID_THRESHOLD, screen_memories

    queries = load_json("queries.json")
    pool = load_json("pool.json")
    labels = load_json("labels.json")
    if not (queries and pool and labels):
        raise SystemExit("run steps 1-4 first")
    pool_by_repo: dict[str, set[str]] = {}
    for it in pool:
        pool_by_repo.setdefault(it["repo"], set()).add(it["id"])

    def rel(qid: int, lid: str) -> bool | None:
        return labels.get(f"{qid}:{lid}")

    db = os.environ.get("GOLDEN_REUSE_DB") or GOLDEN_DB
    configs = ["cosine20", "D", "E"]
    agg = {c: {"injected": 0, "relevant": 0, "unlabeled": 0, "covered": 0, "top5_injected": 0, "top5_relevant": 0, "recall_num": 0} for c in configs}
    relevant_total = 0
    timings: dict[str, list[float]] = {"embed_ms": [], "vector_ms": [], "gate_ms": []}
    per_query = []

    async with open_client(db) as client:
        # Warm the lazy embedder so the first query's embed time is not the model load.
        await _retrieve(client, {**queries[0], "files": []}, threshold=0.0, boost=0.0, limit=1)
        for q in queries:
            qid = q["query_id"]
            relevant_ids = {lid for lid in pool_by_repo.get(q["repo"], set()) if rel(qid, lid)}
            relevant_total += len(relevant_ids)

            cos, t1 = await _retrieve(client, q, threshold=0.0, boost=0.0, limit=20)
            d, t2 = await _retrieve(client, q, threshold=HYBRID_THRESHOLD, boost=ANCHOR_BOOST, limit=GATE_DEPTH)
            timings["embed_ms"].append(t2["embed_ms"])
            timings["vector_ms"].append(t2.get("vector_ms", 0.0))
            t0 = time.perf_counter()
            e = await screen_memories(q["prompt"], list(d)) if d else []
            timings["gate_ms"].append((time.perf_counter() - t0) * 1000)

            row = {"query_id": qid, "anchorable": q["anchorable"], "short": q["short"], "relevant_in_pool": len(relevant_ids)}
            for name, items in (("cosine20", cos), ("D", d), ("E", e)):
                got = [it["id"] for it in items]
                hits = [lid for lid in got if rel(qid, lid)]
                a = agg[name]
                a["injected"] += len(got)
                a["relevant"] += len(hits)
                a["unlabeled"] += sum(1 for lid in got if rel(qid, lid) is None)
                a["covered"] += 1 if hits else 0
                a["recall_num"] += len(set(hits) & relevant_ids)
                a["top5_injected"] += len(got[:CAP])
                a["top5_relevant"] += sum(1 for lid in got[:CAP] if rel(qid, lid))
                row[name] = {"ids": got, "hits": len(hits)}
            per_query.append(row)
            print(f"q{qid:<3} pool_rel={len(relevant_ids):<2} cos20={row['cosine20']['hits']}/{len(cos)} D={row['D']['hits']}/{len(d)} E={row['E']['hits']}/{len(e)}  gate {timings['gate_ms'][-1]:.0f}ms")

    n = len(queries)
    table = {}
    for name, a in agg.items():
        table[name] = {
            "precision": a["relevant"] / a["injected"] if a["injected"] else None,
            "recall": a["recall_num"] / relevant_total if relevant_total else None,
            "coverage": a["covered"] / n,
            "items_per_query": a["injected"] / n,
            "p_at_5": a["top5_relevant"] / a["top5_injected"] if a["top5_injected"] else None,
            "unlabeled_injected": a["unlabeled"],
            **{k: a[k] for k in ("injected", "relevant")},
        }
    latency = {k: {"p50_ms": _pct(v, 0.5), "p95_ms": _pct(v, 0.95)} for k, v in timings.items()}
    summary = {"queries": n, "pool": len(pool), "relevant_pairs": relevant_total,
               "relevant_per_query": relevant_total / n, "configs": table, "latency": latency,
               "gate_depth": GATE_DEPTH, "threshold": HYBRID_THRESHOLD, "anchor_boost": ANCHOR_BOOST,
               "embedding_model": os.environ.get("NAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")}
    save_json("scores.json", {"summary": summary, "per_query": per_query})

    print(f"\n{n} queries, {len(pool)} lessons, {relevant_total} relevant pairs ({relevant_total / n:.2f}/query)")
    print(f"{'config':<10}{'precision':>10}{'recall':>8}{'P@5':>7}{'coverage':>10}{'items/q':>9}")
    for name, t in table.items():
        f = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
        print(f"{name:<10}{f(t['precision']):>10}{f(t['recall']):>8}{f(t['p_at_5']):>7}{f(t['coverage']):>10}{t['items_per_query']:>9.2f}")
    print("latency p50/p95 ms: " + ", ".join(f"{k} {v['p50_ms']:.0f}/{v['p95_ms']:.0f}" for k, v in latency.items()))


if __name__ == "__main__":
    asyncio.run(main())

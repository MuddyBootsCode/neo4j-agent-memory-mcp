"""Step 7: recall@5, and the two precision levers, over one graded pool.

The MUD-401 sweep measured precision only -- it grades what a config
returned and never looks at what it did not. So "29% precision" says the
injected context is 29% useful; it says nothing about how many relevant
lessons were missed. This measures that.

Method: retrieve the top 20 for a sample of queries instead of the top 5,
grade all of them, then read three things off the same pool.

- **recall@5** -- of the relevant lessons the top 20 contained, what share
  made the top 5. High means ranking is finding what exists and the fix for
  precision is returning fewer items; low means real memories are being
  buried below the cut and ranking itself needs work.
- **config F** -- the same items re-sorted with ANCHOR_BOOST removed. The
  D run found anchored items less relevant than unanchored ones (24% vs
  33%), which would make the boost a net negative. Re-ranking the same
  graded pool tests that for free.
- **config E** -- the config-C judge gate applied to D's top 5. Precision
  is scored from the independent grading pass, never from the gate's own
  verdicts, so this is not the gate marking its own homework.

Caveat this cannot escape: it only sees the top 20. A lesson that never
ranks that high is invisible here, so recall@5 is an optimistic bound --
it is recall within reach of the retriever, not recall over the corpus.
Grading all 178 lessons per query would cost ~3,500 judge calls.

Writes results/probe.json. Does not touch the step 4-6 artifacts.
"""

from __future__ import annotations

import asyncio
import os

from judge import call as judge_call
from lib import REPO_NAME, SWEEP_DB, load_json, sample_evenly, save_json
from mem import open_client
from step4_configs import GATE_SYSTEM, GATE_USER_TMPL

PROBE_QUERIES = int(os.environ.get("SWEEP_PROBE_QUERIES", "20"))
DEPTH = int(os.environ.get("SWEEP_PROBE_DEPTH", "20"))
CAP = 5
# Graded in two calls rather than one: a 20-item rubric in a single prompt
# strains the local judge's per-item discipline, which is the failure mode
# the grading prompt is written to avoid.
GRADE_CHUNK = 10

GRADE_SYSTEM = (
    "You audit whether a stored engineering memory is genuinely relevant to "
    "a developer's question. Answer only with the requested JSON."
)

GRADE_USER_TMPL = """\
A developer asked the QUERY below. For each numbered CANDIDATE, judge \
independently whether it is SPECIFICALLY on point for that query -- would it \
actually help answer it? Shared vocabulary or the same general subject area \
is NOT enough; say false unless the candidate bears on the question asked.

QUERY:
{query}

CANDIDATES:
{items}

Return JSON: {{"verdicts": [{{"id": <int>, "relevant": true|false}}, ...]}} \
with exactly one entry per candidate id above, including ones you judge \
irrelevant.
"""


async def _retrieve(client, prompt: str, files: list[str]) -> list[dict]:
    """D's query at probe depth, carrying raw score and anchored flag."""
    from agent_memory_mcp.mcp._coding_tools import (
        ANCHOR_BOOST,
        CODING_MEMORY_INDEX,
        HYBRID_CANDIDATES,
        HYBRID_THRESHOLD,
        _HYBRID_QUERY,
        _embed,
        _render_memory,
    )

    embedding = await _embed(client, prompt)
    if embedding is None:
        return []
    rows = await client.graph.execute_read(
        _HYBRID_QUERY,
        {
            "index": CODING_MEMORY_INDEX,
            "embedding": embedding,
            "candidates": max(HYBRID_CANDIDATES, DEPTH),
            "threshold": HYBRID_THRESHOLD,
            "anchor_boost": ANCHOR_BOOST,
            "limit": DEPTH,
            "repo": REPO_NAME,
            "files": files,
            "task_key": None,
        },
    )
    items = []
    for rank, row in enumerate(rows):
        item = _render_memory(row)
        item["id"] = rank  # rank in D's current ordering
        items.append(item)
    return items


async def _grade(prompt: str, items: list[dict]) -> dict[int, bool | None]:
    """Independent relevance verdicts, chunked. Missing ids stay None."""
    verdicts: dict[int, bool | None] = {it["id"]: None for it in items}
    for start in range(0, len(items), GRADE_CHUNK):
        chunk = items[start : start + GRADE_CHUNK]
        items_text = "\n".join(
            f"{it['id']}. [{it['kind']}] {it['text']}" for it in chunk
        )
        parsed, err = await judge_call(
            GRADE_SYSTEM,
            GRADE_USER_TMPL.format(query=prompt, items=items_text),
            max_tokens=2000,
            timeout=180,
        )
        if parsed is None:
            print(f"    grade chunk failed: {err}")
            continue
        for v in parsed.get("verdicts", []):
            if isinstance(v, dict) and v.get("id") in verdicts:
                verdicts[v["id"]] = bool(v.get("relevant"))
    return verdicts


async def _gate(prompt: str, items: list[dict]) -> set[int]:
    """Config C's keep/drop gate, applied to D's top CAP."""
    if not items:
        return set()
    items_text = "\n".join(
        f"{i}. [{it['kind']}] {it['text']}" for i, it in enumerate(items)
    )
    parsed, err = await judge_call(
        GATE_SYSTEM,
        GATE_USER_TMPL.format(query=prompt, items=items_text),
        max_tokens=1200,
        timeout=180,
    )
    if parsed is None:
        print(f"    gate failed: {err}")
        return set()
    return {
        items[v["id"]]["id"]
        for v in parsed.get("verdicts", [])
        if isinstance(v, dict)
        and v.get("keep") is True
        and isinstance(v.get("id"), int)
        and 0 <= v["id"] < len(items)
    }


def _precision(pairs: list[bool | None]) -> tuple[int, int]:
    graded = [p for p in pairs if p is not None]
    return sum(1 for p in graded if p), len(graded)


async def main() -> None:
    queries = load_json("queries.json")
    if not queries:
        raise SystemExit("run step3_queries.py first")
    sample = sample_evenly(queries, min(PROBE_QUERIES, len(queries)))
    print(f"probing {len(sample)} of {len(queries)} queries at depth {DEPTH}\n")

    records = []
    async with open_client(SWEEP_DB) as client:
        for q in sample:
            print(f"--- q{q['query_id']}: {q['prompt'][:64]!r}")
            items = await _retrieve(client, q["prompt"], q["files"])
            if not items:
                print("    no candidates")
                continue
            verdicts = await _grade(q["prompt"], items)
            kept = await _gate(q["prompt"], items[:CAP])

            # F: same pool, boost removed -- re-sort by raw similarity.
            by_score = sorted(items, key=lambda it: it["score"], reverse=True)

            rel_total = sum(1 for v in verdicts.values() if v)
            rel_top5 = sum(1 for it in items[:CAP] if verdicts[it["id"]])
            print(
                f"    relevant: {rel_top5} in top{CAP}, {rel_total} in top{len(items)}"
                f"  | gate kept {len(kept)}"
            )
            records.append({
                "query_id": q["query_id"],
                "prompt": q["prompt"],
                "anchorable": q.get("anchorable"),
                "depth": len(items),
                "d_order": [it["id"] for it in items],
                "f_order": [it["id"] for it in by_score],
                "gate_kept": sorted(kept),
                "verdicts": {str(k): v for k, v in verdicts.items()},
                "items": [
                    {
                        "id": it["id"], "kind": it["kind"], "score": it["score"],
                        "anchored": it.get("anchored"), "text": it["text"][:300],
                    }
                    for it in items
                ],
            })

    # ---- aggregate -------------------------------------------------------
    rel_top5 = rel_all = 0
    d_hits: list[bool | None] = []
    f_hits: list[bool | None] = []
    e_hits: list[bool | None] = []
    ranks: list[int] = []
    for r in records:
        v = {int(k): val for k, val in r["verdicts"].items()}
        rel_all += sum(1 for val in v.values() if val)
        rel_top5 += sum(1 for i in r["d_order"][:CAP] if v.get(i))
        d_hits += [v.get(i) for i in r["d_order"][:CAP]]
        f_hits += [v.get(i) for i in r["f_order"][:CAP]]
        e_hits += [v.get(i) for i in r["d_order"][:CAP] if i in set(r["gate_kept"])]
        ranks += [pos for pos, i in enumerate(r["d_order"]) if v.get(i)]

    recall_at_5 = rel_top5 / rel_all if rel_all else None
    summary = {
        "queries": len(records),
        "depth": DEPTH,
        "relevant_in_top5": rel_top5,
        "relevant_in_depth": rel_all,
        "recall_at_5": recall_at_5,
        "rank_histogram": {
            str(k): ranks.count(k) for k in sorted(set(ranks))
        },
        "precision": {},
    }
    for name, hits in (("D (shipped)", d_hits), ("F (no boost)", f_hits), ("E (D+gate)", e_hits)):
        rel, graded = _precision(hits)
        summary["precision"][name] = {
            "relevant": rel, "graded": graded,
            "precision": (rel / graded) if graded else None,
            "items_per_query": graded / len(records) if records else None,
        }

    save_json("probe.json", {"summary": summary, "per_query": records})

    print("\n================ RECALL ================")
    print(f"relevant found in top {DEPTH}: {rel_all}")
    print(f"relevant inside top {CAP}:    {rel_top5}")
    print(f"recall@{CAP}: {recall_at_5:.0%}" if recall_at_5 is not None else "recall@5: n/a")
    print("\nwhere the relevant ones ranked:")
    for k, n in summary["rank_histogram"].items():
        print(f"  rank {int(k) + 1:2d}: {'#' * n} ({n})")
    print("\n================ PRECISION LEVERS ================")
    print(f"{'config':<16}{'precision':<12}{'items/query':<14}{'relevant/graded'}")
    for name, s in summary["precision"].items():
        p = f"{s['precision']:.0%}" if s["precision"] is not None else "n/a"
        print(f"{name:<16}{p:<12}{s['items_per_query']:<14.2f}{s['relevant']}/{s['graded']}")


if __name__ == "__main__":
    asyncio.run(main())

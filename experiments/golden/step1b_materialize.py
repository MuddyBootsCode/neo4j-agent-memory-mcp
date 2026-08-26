"""Step 1b: rebuild a scratch database from a committed pool.json, no LLM.

Retrieval experiments (MUD-406: embedder, BM25, fusion, gate variants) need
the same lessons under a different index, and the labels keyed by lesson id
stay valid as long as the lesson text is unchanged. This writes every pool
lesson back as a :CodingMemory node with its props, ABOUT edges and a
session, embedding it with whatever NAM_EMBEDDING_MODEL names, and creates
both indexes.

    GOLDEN_RUN=p3-bge GOLDEN_POOL_FROM=results/p1-capture/pool.json \\
    NAM_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 NAM_EMBEDDING_DIMENSIONS=768 \\
    uv run --no-sync --project ../.. python step1b_materialize.py

Copies pool.json, queries.json and labels.json from the source run into the
new run directory so steps 5 and 6 work unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time

from lib import GOLDEN_DB, HERE, drop_database, lesson_text, result_path, save_json
from mem import LOCAL_EMBEDDING_CONFIG, open_client
from step1_corpus import _create_database


# Lifecycle counters are written by the write path, not carried in $props,
# so anchored_memory_write resets them to the first-sighting defaults. A
# pool exported after MUD-405 carries what the real run accumulated; the
# outcome prior (MUD-407) reads exactly these, so a rebuild that dropped
# them would measure a ranker with no history to read.
_COUNTERS = ("evidence_count", "served_count", "helpful", "harmful", "outcome_weight")


def _counters_from(item: dict) -> dict:
    props = item.get("props") or {}
    return {k: props[k] for k in _COUNTERS if props.get(k) is not None}


def _props_from(item: dict) -> dict:
    """Node props for a pool item. Newer pools carry ``props`` verbatim;
    older ones are parsed back from the embedding text."""
    if item.get("props"):
        return {k: v for k, v in item["props"].items() if k not in _COUNTERS}
    kind, text = item["kind"], item["text"]
    symptom = None
    if kind != "Decision" and " | " in text:
        symptom, text = text.split(" | ", 1)
    props: dict = {"confidence": item.get("confidence") or 0.7}
    if kind == "Decision":
        head, _, reason = text.partition(" — ")
        props.update({"text": head, "reason": item.get("reason") or reason})
    elif kind == "Gotcha":
        props["text"] = text
    else:
        attempt, _, why = text.partition(" — failed: ")
        props.update({"attempt": attempt, "why_failed": why})
    if symptom:
        props["symptom"] = symptom
    return props


async def main() -> None:
    src = os.environ.get("GOLDEN_POOL_FROM")
    if not src:
        raise SystemExit("set GOLDEN_POOL_FROM=results/<run>/pool.json")
    src = os.path.join(HERE, src) if not os.path.isabs(src) else src
    src_dir = os.path.dirname(src)
    with open(src, encoding="utf-8") as fh:
        pool = json.load(fh)
    for name in ("queries.json", "labels.json", "session_split.json"):
        if os.path.exists(os.path.join(src_dir, name)):
            shutil.copy(os.path.join(src_dir, name), result_path(name))

    from agent_memory_mcp.capture.cypher import anchored_memory_write, session_upsert
    from agent_memory_mcp.mcp._coding_tools import _embed, ensure_coding_memory_index

    print(f"recreating {GOLDEN_DB} with embedder {LOCAL_EMBEDDING_CONFIG}...")
    await drop_database(GOLDEN_DB)
    await _create_database(GOLDEN_DB)

    t0 = time.time()
    mismatched = 0
    restored = 0
    async with open_client(GOLDEN_DB) as client:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sessions = {it["session"] for it in pool}
        for sid in sessions:
            repo = next(it["repo"] for it in pool if it["session"] == sid)
            q, p = session_upsert("golden-corpus-agent", sid, repo, "main", None, ts)
            await client.graph.execute_write(q, p)
        first = True
        for it in pool:
            props = _props_from(it)
            text = lesson_text(it["kind"], props)
            if text != it["text"]:
                mismatched += 1
            vector = await _embed(client, text)
            if first:
                ok = await ensure_coding_memory_index(client)
                print(f"indexes ensured: {ok}")
                first = False
            q, p = anchored_memory_write(it["kind"], props, it["session"], it["repo"], it["files"], None, ts, embedding=vector)
            rows = await client.graph.execute_write(q, p)
            counters = _counters_from(it)
            if counters and rows:
                restored += 1
                await client.graph.execute_write(
                    "MATCH (m) WHERE elementId(m) = $eid SET m += $counters",
                    {"eid": rows[0]["eid"], "counters": counters},
                )
    save_json("pool.json", pool)
    save_json("corpus_stats.json", {"materialized_from": src, "lessons": len(pool),
                                    "embedding": LOCAL_EMBEDDING_CONFIG, "text_mismatches": mismatched,
                                    "counters_restored": restored})
    print(f"materialized {len(pool)} lessons in {time.time() - t0:.0f}s; "
          f"{mismatched} whose rebuilt text differs from the pool (labels for those are approximate); "
          f"{restored} with lifecycle counters restored")


if __name__ == "__main__":
    asyncio.run(main())

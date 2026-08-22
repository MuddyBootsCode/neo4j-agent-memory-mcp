"""Step 1: split sessions per repo and build the lesson pool in GOLDEN_DB.

Runs production capture (extract_coding_memory + anchored_memory_write) over
the earlier 60% of each repo's sessions, newest MAX_CORPUS_SESSIONS per repo.
The later 40% are reserved for queries, so no query session leaks into the
pool. Messages are not stored: the golden set measures the lesson plane.

Reuse: GOLDEN_REUSE_DB=<name> skips extraction and points the run at an
existing scratch database (e.g. the MUD-401 `sweepcorpus`), and
GOLDEN_REUSE_SPLIT=<path> copies that build's session split so the query
sessions stay disjoint from it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time

from lib import (
    GOLDEN_DB,
    MAX_CORPUS_SESSIONS,
    REPO_ROOTS,
    drop_database,
    list_sessions,
    repo_name,
    result_path,
    save_json,
    split_sessions,
    touched_files,
)
from mem import open_client

MAX_TRANSCRIPT_CHARS = 80_000


async def _create_database(name: str) -> None:
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "graphmemory")),
    )
    try:
        async with driver.session(database="system") as session:
            await session.run(f"CREATE DATABASE `{name}` IF NOT EXISTS")
        await asyncio.sleep(2)
    finally:
        await driver.close()


def _split() -> dict:
    reuse = os.environ.get("GOLDEN_REUSE_SPLIT")
    if reuse:
        with open(reuse, encoding="utf-8") as fh:
            split = json.load(fh)
        # Older splits carry no repo; every session in them belongs to REPO_ROOTS[0].
        for key in ("corpus_sessions", "query_sessions"):
            for s in split[key]:
                s.setdefault("repo", repo_name(REPO_ROOTS[0]))
                s.setdefault("repo_root", REPO_ROOTS[0])
        shutil.copy(reuse, result_path("session_split.source.json"))
        return split

    corpus, query = [], []
    for root in REPO_ROOTS:
        sessions = list_sessions(root)
        c, q = split_sessions(sessions)
        c = c[-MAX_CORPUS_SESSIONS:]
        print(f"{repo_name(root)}: {len(sessions)} sessions -> corpus {len(c)} (newest of {len(sessions) - len(q)}), query {len(q)}")
        corpus += c
        query += q
    return {"corpus_sessions": corpus, "query_sessions": query}


async def main() -> None:
    split = _split()
    save_json("session_split.json", split)

    reuse_db = os.environ.get("GOLDEN_REUSE_DB")
    if reuse_db:
        print(f"reusing existing database {reuse_db!r}; no extraction run")
        save_json("corpus_stats.json", {"reused_db": reuse_db})
        return

    from agent_memory_mcp.capture.cypher import anchored_memory_write, session_upsert
    from agent_memory_mcp.extraction.coding import extract_coding_memory
    from agent_memory_mcp.hook.capture_hook import extract_transcript_text
    from agent_memory_mcp.mcp._coding_tools import _embed, ensure_coding_memory_index, memory_embedding_text

    print(f"recreating {GOLDEN_DB}...")
    await drop_database(GOLDEN_DB)
    await _create_database(GOLDEN_DB)

    stats = {"sessions_processed": 0, "sessions_failed": 0, "by_kind": {}, "dropped_unanchored": 0,
             "anchor_rates": [], "per_session": [], "errors": []}
    async with open_client(GOLDEN_DB) as client:
        for s in split["corpus_sessions"]:
            t0 = time.time()
            session_id = f"golden-{s['session']}"
            repo = s["repo"]
            print(f"=== {repo} / {s['session']}")
            try:
                transcript = extract_transcript_text(s["path"], max_chars=MAX_TRANSCRIPT_CHARS)
                files = touched_files(s["path"], s["repo_root"])
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                q, p = session_upsert("golden-corpus-agent", session_id, repo, "main", None, ts)
                await client.graph.execute_write(q, p)

                extracted = await extract_coding_memory(transcript, branch="main", task=None, files=files)
                stats["dropped_unanchored"] += extracted["dropped_unanchored"]
                if extracted["anchor_rate"] is not None:
                    stats["anchor_rates"].append(extracted["anchor_rate"])

                items = []
                for d in extracted["decisions"]:
                    items.append(("Decision", {"text": d["text"], "reason": d["reason"], "confidence": d["confidence"]}, d["anchor_files"]))
                for g in extracted["gotchas"]:
                    items.append(("Gotcha", {"text": g["text"], "confidence": g["confidence"]}, g["anchor_files"]))
                for de in extracted["dead_ends"]:
                    items.append(("DeadEnd", {"attempt": de["attempt"], "why_failed": de["why_failed"], "confidence": de["confidence"]}, de["anchor_files"]))

                vectors = [await _embed(client, memory_embedding_text(k, pr)) for k, pr, _ in items]
                if any(v is not None for v in vectors):
                    await ensure_coding_memory_index(client)
                for (kind, props, paths), vector in zip(items, vectors):
                    q, p = anchored_memory_write(kind, props, session_id, repo, paths, None, ts, embedding=vector)
                    await client.graph.execute_write(q, p)
                    stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1

                stats["sessions_processed"] += 1
                stats["per_session"].append({
                    "session": s["session"], "repo": repo, "transcript_chars": len(transcript),
                    "touched_files": len(files), "lessons": len(items),
                    "anchor_rate": extracted["anchor_rate"], "dropped": extracted["dropped_unanchored"],
                    "elapsed_s": round(time.time() - t0, 1),
                })
                print(f"  lessons={len(items)} anchor_rate={extracted['anchor_rate']} dropped={extracted['dropped_unanchored']} ({time.time() - t0:.0f}s)")
            except Exception as exc:  # noqa: BLE001
                stats["sessions_failed"] += 1
                stats["errors"].append({"session": s["session"], "error": str(exc)[:500]})
                print(f"  FAILED: {exc}")

    save_json("corpus_stats.json", stats)
    print(f"\ncorpus: {stats['sessions_processed']} sessions, by_kind={stats['by_kind']}, dropped={stats['dropped_unanchored']}")


if __name__ == "__main__":
    asyncio.run(main())

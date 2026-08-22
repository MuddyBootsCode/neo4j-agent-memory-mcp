"""Step 1: split sessions per repo and build the lesson pool in GOLDEN_DB.

Runs the production capture pipeline (``capture_transcript``: windows,
extraction, error-step candidates, curation, anchored writes) over the
earlier 60% of each repo's sessions, newest MAX_CORPUS_SESSIONS per repo,
using the same transcript rendering and file/error collection the SessionEnd
hook uses. The later 40% are reserved for queries, so no query session
leaks into the pool.

Reuse: GOLDEN_REUSE_DB=<name> skips extraction and points the run at an
existing scratch database, and GOLDEN_REUSE_SPLIT=<path> copies that
build's session split so the query sessions stay disjoint from it.
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
)
from mem import open_client


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

    from agent_memory_mcp.hook.capture_hook import (
        error_steps,
        extract_transcript_text,
        transcript_touched_files,
    )
    from agent_memory_mcp.mcp._coding_tools import capture_transcript

    print(f"recreating {GOLDEN_DB}...")
    await drop_database(GOLDEN_DB)
    await _create_database(GOLDEN_DB)

    stats = {"sessions_processed": 0, "sessions_failed": 0, "by_kind": {}, "dropped_unanchored": 0,
             "curated": {}, "anchor_rates": [], "windows": 0, "per_session": [], "errors": []}
    async with open_client(GOLDEN_DB) as client:
        for s in split["corpus_sessions"]:
            t0 = time.time()
            session_id = f"golden-{s['session']}"
            repo = s["repo"]
            print(f"=== {repo} / {s['session']}")
            try:
                transcript = extract_transcript_text(s["path"])
                files = transcript_touched_files(s["path"], s["repo_root"])
                steps = error_steps(s["path"], s["repo_root"])
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                result = await capture_transcript(
                    client, transcript=transcript, agent_id="golden-corpus-agent",
                    session_id=session_id, repo=repo, branch="main", task_key=None,
                    files=files, error_steps=steps, ts=ts,
                )
                for kind, n in result["by_kind"].items():
                    stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + n
                for key, n in result["curated"].items():
                    stats["curated"][key] = stats["curated"].get(key, 0) + n
                stats["dropped_unanchored"] += result["dropped_unanchored"]
                stats["windows"] += result["windows"]
                if result["anchor_rate"] is not None:
                    stats["anchor_rates"].append(result["anchor_rate"])
                stats["sessions_processed"] += 1
                stats["per_session"].append({
                    "session": s["session"], "repo": repo, "transcript_chars": len(transcript),
                    "touched_files": len(files), "error_steps": len(steps),
                    "windows": result["windows"], "stored": result["stored"], "by_kind": result["by_kind"],
                    "curated": result["curated"], "anchor_rate": result["anchor_rate"],
                    "dropped": result["dropped_unanchored"], "elapsed_s": round(time.time() - t0, 1),
                })
                print(f"  chars={len(transcript)} files={len(files)} errors={len(steps)} windows={result['windows']} "
                      f"stored={result['stored']} curated={result['curated']} anchor_rate={result['anchor_rate']} "
                      f"({time.time() - t0:.0f}s)")
            except Exception as exc:  # noqa: BLE001
                stats["sessions_failed"] += 1
                stats["errors"].append({"session": s["session"], "error": str(exc)[:500]})
                print(f"  FAILED: {exc}")
            save_json("corpus_stats.json", stats)

    zero = sum(1 for p in stats["per_session"] if p["stored"] == 0)
    stats["zero_lesson_sessions"] = zero
    save_json("corpus_stats.json", stats)
    print(f"\ncorpus: {stats['sessions_processed']} sessions, by_kind={stats['by_kind']}, "
          f"curated={stats['curated']}, dropped={stats['dropped_unanchored']}, zero-lesson sessions={zero}")


if __name__ == "__main__":
    asyncio.run(main())

"""Step 2: export the lesson pool from the scratch database to pool.json.

The committed pool is the evidence for a run; the database is disposable
(D3). Ids are content hashes (lib.lesson_id) so a rebuild that reproduces a
lesson keeps its labels.
"""

from __future__ import annotations

import asyncio
import os

from lib import GOLDEN_DB, lesson_id, lesson_text, save_json
from mem import open_client

_POOL_QUERY = """
    MATCH (m:CodingMemory)-[:MADE_IN]->(s:CodingSession)
    OPTIONAL MATCH (m)-[:ABOUT]->(f:CodeFile)
    WITH m, s, [p IN collect(DISTINCT f.path) WHERE p IS NOT NULL] AS files
    RETURN labels(m) AS labels, properties(m) AS props, s.repo AS repo,
           s.id AS session, files, m.embedding IS NOT NULL AS embedded
"""


async def main() -> None:
    db = os.environ.get("GOLDEN_REUSE_DB") or GOLDEN_DB
    pool: dict[str, dict] = {}
    async with open_client(db) as client:
        rows = await client.graph.execute_read(_POOL_QUERY, {})
    dupes = 0
    for row in rows:
        kind = next(label for label in row["labels"] if label in ("Decision", "Gotcha", "DeadEnd"))
        props = {k: v for k, v in row["props"].items() if k != "embedding"}
        text = lesson_text(kind, props)
        lid = lesson_id(row["repo"], kind, text)
        if lid in pool:
            dupes += 1
            continue
        pool[lid] = {
            "id": lid, "repo": row["repo"], "kind": kind, "text": text,
            "reason": props.get("reason"), "files": sorted(row["files"]),
            "session": row["session"], "confidence": props.get("confidence"),
            "embedded": bool(row["embedded"]),
        }
    ordered = [pool[k] for k in sorted(pool)]
    save_json("pool.json", ordered)
    by_repo: dict[str, int] = {}
    for item in ordered:
        by_repo[item["repo"]] = by_repo.get(item["repo"], 0) + 1
    print(f"pool: {len(ordered)} lessons from {db} ({dupes} exact duplicates collapsed); by repo {by_repo}")


if __name__ == "__main__":
    asyncio.run(main())

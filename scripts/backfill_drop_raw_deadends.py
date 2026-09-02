"""Delete the DeadEnd nodes that are raw tool output, not lessons.

MUD-404 turned errored tool steps into zero-LLM DeadEnd candidates for the
curator to judge. Two paths wrote them unjudged: the curator's fail-open
kept every candidate when the model call raised or covered too few, and the
hook treated any tool result matching an error keyword as a failure, so a
successful ``cat`` of a file mentioning "Traceback" qualified. By 2026-09-02
the store held 4,751 such nodes (help screens, runbook excerpts, permission
prompts) against ~2,000 real lessons; they were 54% of what recall served
(MUD-407, finding F5). Both write paths are fixed; this removes what they
wrote.

A raw error step is the only DeadEnd written at ``confidence = 0.5``: the
extractor clamps its own to 0.7 or higher (``extraction/coding.py``). A node
a session rated helpful is kept regardless.

Before deleting, every doomed node is exported (props minus embedding) so
the run is reversible by hand, and any lesson one of them expired through a
SUPERSEDES edge gets its ``expired_at`` and ``superseded_by`` cleared.

    uv run python scripts/backfill_drop_raw_deadends.py            # dry run
    uv run python scripts/backfill_drop_raw_deadends.py --apply --export out.jsonl

Reads NEO4J_* from the environment, exactly as the server does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

RAW_CONFIDENCE = 0.5
BATCH = 500

_MATCH = """
    MATCH (m:DeadEnd)
    WHERE m.confidence = $conf AND coalesce(m.helpful, 0) = 0
"""

_COUNT = _MATCH + " RETURN count(m) AS n"

_EXPORT = _MATCH + """
    RETURN elementId(m) AS eid,
           [k IN keys(m) WHERE k <> 'embedding' | [k, toString(m[k])]] AS props
"""

_SUPERSEDED = _MATCH + """
    MATCH (m)-[:SUPERSEDES]->(o)
    RETURN elementId(o) AS eid, labels(o)[0] AS kind,
           left(coalesce(o.symptom, o.text, o.attempt, ''), 100) AS text
"""

_RESTORE = _MATCH + """
    MATCH (m)-[:SUPERSEDES]->(o)
    SET o.expired_at = null, o.superseded_by = null
    RETURN count(o) AS n
"""

_DELETE_BATCH = _MATCH + """
    WITH m LIMIT $batch
    DETACH DELETE m
    RETURN count(*) AS n
"""


def _serialise(value):
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--apply", action="store_true", help="delete; without it, report only")
    parser.add_argument(
        "--export",
        default=None,
        help="JSONL path for the deleted nodes (default: raw-deadends-<utc>.jsonl in cwd)",
    )
    args = parser.parse_args()

    from pydantic import SecretStr

    from neo4j_agent_memory import MemoryClient, MemorySettings
    from neo4j_agent_memory.config.settings import Neo4jConfig

    settings = MemorySettings(
        neo4j=Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=SecretStr(os.environ.get("NEO4J_PASSWORD", "graphmemory")),
            database=args.database,
        ),
    )
    client = MemoryClient(settings)
    await client.connect()
    params = {"conf": RAW_CONFIDENCE}

    try:
        total = (await client.graph.execute_read(_COUNT, params))[0]["n"]
        superseded = await client.graph.execute_read(_SUPERSEDED, params)
        print(f"{total} raw DeadEnd node(s) at confidence {RAW_CONFIDENCE} in {args.database!r}")
        print(f"{len(superseded)} lesson(s) they expired via SUPERSEDES will be restored")
        for row in superseded[:10]:
            print(f"  restore [{row['kind']}] {row['text']}")
        if not total:
            return 0

        if not args.apply:
            sample = await client.graph.execute_read(_EXPORT + " LIMIT 8", params)
            for row in sample:
                props = dict(row["props"])
                print(f"  drop  {props.get('attempt', '')[:70]!r} — {props.get('why_failed', '')[:60]!r}")
            print("dry run — nothing deleted (pass --apply)")
            return 0

        export = args.export or f"raw-deadends-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
        rows = await client.graph.execute_read(_EXPORT, params)
        with open(export, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps({"eid": row["eid"], "props": {k: _serialise(v) for k, v in row["props"]}}) + "\n")
        print(f"exported {len(rows)} node(s) to {export}")

        restored = (await client.graph.execute_write(_RESTORE, params))[0]["n"]
        print(f"restored {restored} superseded lesson(s)")

        deleted = 0
        while True:
            rows = await client.graph.execute_write(_DELETE_BATCH, {**params, "batch": BATCH})
            n = rows[0]["n"] if rows else 0
            if not n:
                break
            deleted += n
            print(f"  deleted {deleted}/{total}", end="\r", flush=True)
        print(f"deleted {deleted} raw DeadEnd node(s)")
        remaining = (await client.graph.execute_read(_COUNT, params))[0]["n"]
        print(f"remaining at confidence {RAW_CONFIDENCE}: {remaining}")
        return 0 if remaining == 0 else 1
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

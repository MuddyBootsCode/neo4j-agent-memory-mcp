"""Recount ``evidence_count`` as sessions-families, not sessions.

``reassert_write`` added one unit of evidence per session that restated a
lesson. Subagent sessions are ``parent:spawn``, so a swarm of two hundred
subagents restating one lesson pushed it to evidence 60–114 in days, and
the guardrail label (``evidence_count >= 3``) covered 45% of the store
(MUD-435 F3). The write path now counts once per family (the parent and
its spawns; see ``capture.cypher.session_family``). This recounts what is
already stored from the ``MADE_IN`` and ``REASSERTED_IN`` edges, which are
kept per session and are the record.

    uv run python scripts/backfill_evidence_families.py            # dry run
    uv run python scripts/backfill_evidence_families.py --apply --export out.jsonl

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

BATCH = 1000

# The family of a session id, in Cypher: the same rule as session_family().
_FAMILY = """
    CASE WHEN s.id CONTAINS ':' AND NOT s.id STARTS WITH ':'
         THEN split(s.id, ':')[0] ELSE s.id END
"""

_RECOUNT = f"""
    MATCH (m) WHERE (m:Gotcha OR m:Decision OR m:DeadEnd)
    OPTIONAL MATCH (m)-[:MADE_IN|REASSERTED_IN]->(s:CodingSession)
    WITH m, size(collect(DISTINCT {_FAMILY})) AS families
    WITH m, coalesce(m.evidence_count, 1) AS old,
         CASE WHEN families < 1 THEN 1 ELSE families END AS new
"""

_SUMMARY = _RECOUNT + """
    RETURN count(m) AS lessons,
           sum(CASE WHEN old <> new THEN 1 ELSE 0 END) AS changed,
           sum(CASE WHEN old >= 3 THEN 1 ELSE 0 END) AS guardrails_old,
           sum(CASE WHEN new >= 3 THEN 1 ELSE 0 END) AS guardrails_new,
           max(old) AS max_old, max(new) AS max_new
"""

_EXPORT = _RECOUNT + """
    WHERE old <> new
    RETURN elementId(m) AS eid, labels(m)[0] AS kind, old, new,
           left(coalesce(m.text, m.attempt, ''), 120) AS text
    ORDER BY old - new DESC
"""

_APPLY = _RECOUNT + """
    WHERE old <> new
    WITH m, new LIMIT $batch
    SET m.evidence_count = new
    RETURN count(m) AS n
"""


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--apply", action="store_true", help="write; without it, report only")
    parser.add_argument(
        "--export",
        default=None,
        help="JSONL path for the old/new counts (default: evidence-families-<utc>.jsonl in cwd)",
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
    try:
        row = (await client.graph.execute_read(_SUMMARY, {}))[0]
        print(
            f"{row['lessons']} lessons in {args.database!r}: {row['changed']} change; "
            f"guardrails (evidence >= 3) {row['guardrails_old']} -> {row['guardrails_new']}; "
            f"max evidence {row['max_old']} -> {row['max_new']}"
        )
        if not row["changed"]:
            return 0
        rows = await client.graph.execute_read(_EXPORT, {})
        for r in rows[:8]:
            print(f"  [{r['kind']}] {r['old']:>4} -> {r['new']:<3} {r['text'][:80]!r}")
        if not args.apply:
            print("dry run — nothing written (pass --apply)")
            return 0

        export = args.export or f"evidence-families-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
        with open(export, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"eid": r["eid"], "kind": r["kind"], "old": r["old"], "new": r["new"]}) + "\n")
        print(f"exported {len(rows)} old/new pair(s) to {export}")

        written = 0
        while True:
            out = await client.graph.execute_write(_APPLY, {"batch": BATCH})
            n = out[0]["n"] if out else 0
            if not n:
                break
            written += n
            print(f"  updated {written}/{row['changed']}", end="\r", flush=True)
        print(f"updated {written} lesson(s)")
        after = (await client.graph.execute_read(_SUMMARY, {}))[0]
        print(f"remaining to change: {after['changed']}; guardrails now {after['guardrails_new']}")
        return 0 if after["changed"] == 0 else 1
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

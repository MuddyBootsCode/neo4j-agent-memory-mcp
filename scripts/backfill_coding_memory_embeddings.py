"""Backfill the :CodingMemory label and embeddings onto existing lessons.

MUD-401 changed coding_recall to rank Decision/Gotcha/DeadEnd nodes by
similarity to the prompt. That reads through a vector index on the shared
:CodingMemory label, which only nodes written after the change carry — every
lesson captured before it is invisible to the new query and reachable only by
the anchor-first fallback.

This walks the existing nodes, embeds each one with the same text
capture_session_memory would use, and adds the label. Idempotent: nodes that
already have an embedding are skipped unless --force is given.

    uv run python scripts/backfill_coding_memory_embeddings.py --dry-run
    uv run python scripts/backfill_coding_memory_embeddings.py

Reads NEO4J_* and the embedding provider vars from the environment, exactly
as the server does, so it embeds with whatever the server would use. Point it
at one database with --database.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from agent_memory_mcp.capture.cypher import RECALL_KINDS, SHARED_RECALL_LABEL  # noqa: E402
from agent_memory_mcp.mcp._bootstrap import bootstrap_upstream_patches  # noqa: E402
from agent_memory_mcp.mcp._coding_tools import (  # noqa: E402
    _embed,
    ensure_coding_memory_index,
    memory_embedding_text,
)

# elementId() is stable for the life of the node, which is all this needs --
# it never outlives the run.
_SELECT = """
    MATCH (m)
    WHERE (__KINDS__)
      AND ($force OR m.embedding IS NULL)
    RETURN elementId(m) AS eid, labels(m) AS labels, properties(m) AS props
    ORDER BY m.created_at
"""

_UPDATE = f"""
    MATCH (m) WHERE elementId(m) = $eid
    SET m:{SHARED_RECALL_LABEL}, m.embedding = $embedding
"""

# Label-only update for nodes whose text is empty: they can never match a
# vector query, but the label keeps the graph's shape uniform.
_LABEL_ONLY = f"""
    MATCH (m) WHERE elementId(m) = $eid
    SET m:{SHARED_RECALL_LABEL}
"""


def _select_query() -> str:
    disjunction = " OR ".join(f"m:{kind}" for kind in sorted(RECALL_KINDS))
    return _SELECT.replace("__KINDS__", disjunction)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change and write nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed nodes that already have an embedding",
    )
    args = parser.parse_args()

    bootstrap_upstream_patches()

    from pydantic import SecretStr

    from agent_memory_mcp.providers import embedding_kwargs_from_env
    from neo4j_agent_memory import MemoryClient, MemorySettings
    from neo4j_agent_memory.config.settings import EmbeddingConfig, Neo4jConfig

    # Same env-var contract as the server, so the backfill embeds with
    # whatever the server would -- mixing providers would mix dimensions and
    # every vector query would fail.
    settings = MemorySettings(
        neo4j=Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=SecretStr(os.environ.get("NEO4J_PASSWORD", "graphmemory")),
            database=args.database,
        ),
        embedding=EmbeddingConfig(**embedding_kwargs_from_env()),
    )
    client = MemoryClient(settings)
    await client.connect()

    try:
        rows = await client.graph.execute_read(_select_query(), {"force": args.force})
        print(f"{len(rows)} lesson node(s) to backfill in database {args.database!r}")
        if not rows:
            return 0

        if args.dry_run:
            for row in rows[:10]:
                kind = next(
                    (label for label in row["labels"] if label in RECALL_KINDS), "?"
                )
                text = memory_embedding_text(kind, row["props"])
                print(f"  [{kind}] {text[:90]}")
            if len(rows) > 10:
                print(f"  ... and {len(rows) - 10} more")
            print("dry run — nothing written")
            return 0

        if not await ensure_coding_memory_index(client):
            print(
                "ERROR: could not create the vector index — check that an "
                "embedder is configured. Nothing was written.",
                file=sys.stderr,
            )
            return 1

        embedded = labelled = failed = 0
        for row in rows:
            kind = next(
                (label for label in row["labels"] if label in RECALL_KINDS), None
            )
            if kind is None:  # pragma: no cover - the query filters these out
                continue
            text = memory_embedding_text(kind, row["props"])
            vector = await _embed(client, text) if text else None
            if vector is None:
                await client.graph.execute_write(_LABEL_ONLY, {"eid": row["eid"]})
                labelled += 1
                if text:
                    failed += 1
                continue
            await client.graph.execute_write(
                _UPDATE, {"eid": row["eid"], "embedding": vector}
            )
            embedded += 1

        print(f"embedded: {embedded}")
        print(f"labelled without an embedding: {labelled}")
        if failed:
            print(f"  of which {failed} had text but the embedder failed — rerun to retry")
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

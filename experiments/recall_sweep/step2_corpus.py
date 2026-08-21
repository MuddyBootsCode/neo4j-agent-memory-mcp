"""Step 2: build the sweepcorpus database from the corpus sessions.

For each corpus session:
  1. render the transcript (extract_transcript_text, 80k tail cap — the
     capture convention)
  2. reconstruct touched files from Edit/Write tool_use entries
  3. run extract_coding_memory (BAML, routed to Ollama via NAM_LLM_PROVIDER)
  4. write session_upsert + anchored_memory_write per kept item into
     sweepcorpus, via client.graph (the MemoryClient's own Neo4j session,
     already pointed at sweepcorpus)
  5. store the session's individual user/assistant turns as Messages with
     local embeddings, and persist extracted preferences, so config A
     (cosine-only search) has a corpus to search

Never touches the default `neo4j` database — every write in this file goes
through a MemoryClient/driver explicitly constructed against SWEEP_DB.

Idempotent-ish: DROPs and recreates sweepcorpus first (per the task's "drop
and recreate at start for a clean run" instruction), so reruns start clean.
"""

from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("NAM_LLM_PROVIDER", "ollama")

from lib import (  # noqa: E402
    REPO_NAME,
    SWEEP_DB,
    load_json,
    save_json,
    touched_files,
)
from mem import open_client  # noqa: E402

MAX_TRANSCRIPT_CHARS = 80_000
MAX_MESSAGE_CHARS = 8_000


async def _recreate_database() -> None:
    from neo4j import AsyncGraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "graphmemory")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session(database="system") as session:
            await session.run(f"DROP DATABASE `{SWEEP_DB}` IF EXISTS")
        await asyncio.sleep(2)
        async with driver.session(database="system") as session:
            await session.run(f"CREATE DATABASE `{SWEEP_DB}` IF NOT EXISTS")
        await asyncio.sleep(2)
        async with driver.session(database="system") as session:
            result = await session.run(
                f"SHOW DATABASE `{SWEEP_DB}` YIELD currentStatus"
            )
            record = await result.single()
            status = record["currentStatus"] if record else None
        print(f"sweepcorpus database status: {status}")
    finally:
        await driver.close()


def _session_turns(transcript_path: str) -> list[tuple[str, str]]:
    """(role, text) for every real user/assistant turn in the FULL session.

    Unlike the extraction transcript (tail-capped at 80k for the LLM call),
    message storage uses the whole session so config A has a full corpus to
    search — capped per-message to keep embeddings cheap on outlier blobs.
    """
    import json as _json

    turns: list[tuple[str, str]] = []
    with open(transcript_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            try:
                record = _json.loads(raw)
            except Exception:
                continue
            if not isinstance(record, dict) or record.get("isMeta"):
                continue
            role = record.get("type")
            if role not in ("user", "assistant"):
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            text = None
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                    text = None
                else:
                    parts = [
                        b.get("text", "").strip()
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    parts = [p for p in parts if p]
                    text = " ".join(parts) if parts else None
            if text:
                turns.append((role, text[:MAX_MESSAGE_CHARS]))
    return turns


async def main() -> None:
    from agent_memory_mcp.capture.cypher import anchored_memory_write, session_upsert
    from agent_memory_mcp.mcp._coding_tools import (
        _embed,
        ensure_coding_memory_index,
        memory_embedding_text,
    )
    from agent_memory_mcp.extraction.coding import extract_coding_memory
    from agent_memory_mcp.extraction.unified import persist_preferences
    from agent_memory_mcp.hook.capture_hook import extract_transcript_text

    split = load_json("session_split.json")
    if split is None:
        raise SystemExit("run step1_split.py first")
    corpus_sessions = split["corpus_sessions"]

    print(f"recreating {SWEEP_DB} database...")
    await _recreate_database()

    stats = {
        "sessions_processed": 0,
        "sessions_failed": 0,
        "by_kind": {"Decision": 0, "Gotcha": 0, "DeadEnd": 0, "CodingPreference": 0},
        "anchor_rates": [],
        "dropped_unanchored_total": 0,
        "messages_stored": 0,
        "messages_failed": 0,
        "preferences_stored": 0,
        "errors": [],
        "per_session": [],
    }

    async with open_client(SWEEP_DB) as client:
        for s in corpus_sessions:
            session_id = f"sweep-{s['session']}"
            t0 = time.time()
            print(f"\n=== corpus session {s['session']} ===")
            try:
                transcript_text = extract_transcript_text(
                    s["path"], max_chars=MAX_TRANSCRIPT_CHARS
                )
                files = touched_files(s["path"])
                print(f"  transcript chars: {len(transcript_text)}  touched files: {len(files)}")

                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                # Session node always written, even if extraction yields nothing.
                q, p = session_upsert(
                    "sweep-corpus-agent", session_id, REPO_NAME, "main", None, ts
                )
                await client.graph.execute_write(q, p)

                extracted = await extract_coding_memory(
                    transcript_text, branch="main", task=None, files=files
                )
                stats["dropped_unanchored_total"] += extracted["dropped_unanchored"]
                if extracted["anchor_rate"] is not None:
                    stats["anchor_rates"].append(extracted["anchor_rate"])

                items: list[tuple[str, dict, list[str], str | None]] = []
                for d in extracted["decisions"]:
                    items.append((
                        "Decision",
                        {"text": d["text"], "reason": d["reason"], "confidence": d["confidence"]},
                        d["anchor_files"], None,
                    ))
                for g in extracted["gotchas"]:
                    items.append((
                        "Gotcha",
                        {"text": g["text"], "confidence": g["confidence"]},
                        g["anchor_files"], None,
                    ))
                for de in extracted["dead_ends"]:
                    items.append((
                        "DeadEnd",
                        {"attempt": de["attempt"], "why_failed": de["why_failed"], "confidence": de["confidence"]},
                        de["anchor_files"], None,
                    ))

                # Embed every lesson exactly as capture_session_memory does,
                # so config D measures production's real read against
                # production's real write -- including the order. The
                # embedder is created lazily, so ensuring the index before
                # the first embed finds no dimensions and silently skips.
                vectors = []
                for kind, props, _paths, _task in items:
                    vector = await _embed(
                        client, memory_embedding_text(kind, props)
                    )
                    if vector is not None:
                        stats["embedded"] = stats.get("embedded", 0) + 1
                    vectors.append(vector)
                if any(v is not None for v in vectors):
                    await ensure_coding_memory_index(client)

                for (kind, props, anchor_paths, task_key), vector in zip(items, vectors):
                    q, p = anchored_memory_write(
                        kind, props, session_id, REPO_NAME, anchor_paths, task_key,
                        ts, embedding=vector,
                    )
                    await client.graph.execute_write(q, p)
                    stats["by_kind"][kind] += 1
                stats["by_kind"]["CodingPreference"] += len(extracted["preferences"])

                prefs_stored = await persist_preferences(client, extracted["preferences"])
                stats["preferences_stored"] += prefs_stored

                # Messages: whole-session turns, individually embedded.
                turns = _session_turns(s["path"])
                msg_ok = 0
                msg_fail = 0
                for role, text in turns:
                    try:
                        await client.short_term.add_message(
                            session_id=session_id,
                            role=role,
                            content=text,
                            extract_entities=False,
                            generate_embedding=True,
                        )
                        msg_ok += 1
                    except Exception as exc:  # noqa: BLE001
                        msg_fail += 1
                        stats["errors"].append(
                            {"session": s["session"], "stage": "add_message", "error": str(exc)[:300]}
                        )
                stats["messages_stored"] += msg_ok
                stats["messages_failed"] += msg_fail

                stats["sessions_processed"] += 1
                elapsed = time.time() - t0
                print(
                    f"  decisions={len(extracted['decisions'])} gotchas={len(extracted['gotchas'])} "
                    f"dead_ends={len(extracted['dead_ends'])} prefs={len(extracted['preferences'])} "
                    f"anchor_rate={extracted['anchor_rate']} dropped={extracted['dropped_unanchored']} "
                    f"messages={msg_ok}/{msg_ok + msg_fail}  ({elapsed:.1f}s)"
                )
                stats["per_session"].append({
                    "session": s["session"],
                    "session_id": session_id,
                    "transcript_chars": len(transcript_text),
                    "touched_files": files,
                    "decisions": len(extracted["decisions"]),
                    "gotchas": len(extracted["gotchas"]),
                    "dead_ends": len(extracted["dead_ends"]),
                    "preferences": len(extracted["preferences"]),
                    "anchor_rate": extracted["anchor_rate"],
                    "dropped_unanchored": extracted["dropped_unanchored"],
                    "messages_stored": msg_ok,
                    "messages_failed": msg_fail,
                    "elapsed_s": elapsed,
                })
            except Exception as exc:  # noqa: BLE001
                stats["sessions_failed"] += 1
                stats["errors"].append(
                    {"session": s["session"], "stage": "extract_coding_memory", "error": str(exc)[:500]}
                )
                print(f"  FAILED: {exc}")

    save_json("corpus_stats.json", stats)
    print("\n=== corpus build complete ===")
    print(f"sessions processed: {stats['sessions_processed']}  failed: {stats['sessions_failed']}")
    print(f"by_kind: {stats['by_kind']}")
    print(f"messages stored: {stats['messages_stored']}  failed: {stats['messages_failed']}")
    print(f"preferences stored: {stats['preferences_stored']}")
    print(f"anchor_rates: {stats['anchor_rates']}")


if __name__ == "__main__":
    asyncio.run(main())

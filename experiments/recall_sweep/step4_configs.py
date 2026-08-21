"""Step 4: for each query, produce the three injection candidate sets.

  A. cosine-only  — embedding search over sweepcorpus messages + preferences
                     (mirrors memory_search's vector path), threshold 0.5,
                     merged and sorted by similarity, capped at 5.
  B. anchor-first  — the production _MEMORIES_QUERY traversal from the
                     query's reconstructed files (repo=REPO_NAME, task_key
                     always None here — the sweep has no WorkTask context),
                     capped at 5.
  C. anchor-first + judge gate — every config-B item screened by qwen-judge
                     with a terse keep/drop prompt; kept items only.

Every candidate gets a query-scoped id (A0..A4, B0..B4) so the grading step
(step5) can grade the union of A ∪ B once per query and every config just
references those ids — C's ids are always a subset of B's.
"""

from __future__ import annotations

import asyncio

from judge import call as judge_call
from lib import REPO_NAME, SWEEP_DB, load_json, save_json
from mem import open_client

COSINE_THRESHOLD = 0.5
CAP = 5

GATE_SYSTEM = (
    "You screen candidate memories before they are injected into a coding "
    "assistant's context. Answer only with the requested JSON."
)

GATE_USER_TMPL = """\
For the QUERY below, decide for each numbered ITEM: is this item useful \
context for answering the query? Answer strictly per item — do not let one \
item's answer bleed into another's.

QUERY:
{query}

ITEMS:
{items}

Return JSON: {{"verdicts": [{{"id": <int>, "keep": true|false}}, ...]}} with \
exactly one entry per item id listed above, including items you reject.
"""


def _item_text(kind: str, props: dict) -> str:
    if kind == "DeadEnd":
        return f"{props.get('attempt', '')} — failed: {props.get('why_failed', '')}"
    return props.get("text") or props.get("preference") or str(props)


async def _config_a(client, prompt: str) -> list[dict]:
    messages = await client.short_term.search_messages(
        query=prompt, limit=10, threshold=COSINE_THRESHOLD
    )
    prefs = await client.long_term.search_preferences(
        query=prompt, limit=10, threshold=COSINE_THRESHOLD
    )
    pool = []
    for m in messages:
        sim = (m.metadata or {}).get("similarity")
        pool.append({
            "kind": "Message",
            "text": m.content,
            "role": m.role.value if hasattr(m.role, "value") else str(m.role),
            "similarity": sim,
        })
    for p in prefs:
        sim = (p.metadata or {}).get("similarity")
        pool.append({
            "kind": "Preference",
            "text": p.preference,
            "category": p.category,
            "similarity": sim,
        })
    pool.sort(key=lambda x: (x["similarity"] if x["similarity"] is not None else -1), reverse=True)
    top = pool[:CAP]
    for i, item in enumerate(top):
        item["id"] = f"A{i}"
    return top


# The production _MEMORIES_QUERY from mcp/_coding_tools.py, imported directly
# rather than copied, so the sweep measures the real query shape.
async def _config_b(client, files: list[str]) -> list[dict]:
    from agent_memory_mcp.mcp._coding_tools import _MEMORIES_QUERY, _render_memory

    rows = await client.graph.execute_read(
        _MEMORIES_QUERY, {"repo": REPO_NAME, "files": files, "task_key": None}
    )
    rendered = [_render_memory(row) for row in rows][:CAP]
    for i, item in enumerate(rendered):
        item["id"] = f"B{i}"
    return rendered


async def _config_c(prompt: str, b_items: list[dict]) -> tuple[list[str], dict]:
    """Gate config B's items. Returns (kept_ids, trace)."""
    if not b_items:
        return [], {"skipped": True}
    items_text = "\n".join(
        f"{i}. [{it['kind']}] {it['text']}" for i, it in enumerate(b_items)
    )
    user = GATE_USER_TMPL.format(query=prompt, items=items_text)
    parsed, err = await judge_call(GATE_SYSTEM, user, max_tokens=1200, timeout=180)
    trace = {"error": err, "ok": err is None and parsed is not None}
    if parsed is None:
        return [], trace
    verdicts = {v.get("id"): v.get("keep") for v in parsed.get("verdicts", []) if isinstance(v, dict)}
    trace["verdicts"] = verdicts
    kept = [b_items[i]["id"] for i in range(len(b_items)) if verdicts.get(i) is True]
    return kept, trace


async def main() -> None:
    queries = load_json("queries.json")
    if not queries:
        raise SystemExit("run step3_queries.py first (no queries found)")

    out = []
    errors = []
    async with open_client(SWEEP_DB) as client:
        for q in queries:
            qid = q["query_id"]
            print(f"\n--- query {qid}: {q['prompt'][:70]!r} ---")
            try:
                a_items = await _config_a(client, q["prompt"])
            except Exception as exc:  # noqa: BLE001
                a_items = []
                errors.append({"query_id": qid, "stage": "config_a", "error": str(exc)[:300]})
            try:
                b_items = await _config_b(client, q["files"])
            except Exception as exc:  # noqa: BLE001
                b_items = []
                errors.append({"query_id": qid, "stage": "config_b", "error": str(exc)[:300]})

            c_ids, gate_trace = await _config_c(q["prompt"], b_items)
            if not gate_trace.get("ok") and b_items:
                errors.append({"query_id": qid, "stage": "config_c_gate", "error": gate_trace.get("error")})

            print(
                f"  A={len(a_items)} B={len(b_items)} C(kept from B)={len(c_ids)}"
                + ("" if gate_trace.get('ok', True) else f"  GATE FAIL: {gate_trace.get('error')}")
            )

            out.append({
                "query_id": qid,
                "session": q["session"],
                "prompt": q["prompt"],
                "files": q["files"],
                "config_a": a_items,
                "config_b": b_items,
                "config_c_ids": c_ids,
                "gate_trace": gate_trace,
            })

    save_json("candidates.json", out)
    save_json("config_errors.json", errors)
    print(f"\nwrote candidates.json for {len(out)} queries; {len(errors)} errors logged")


if __name__ == "__main__":
    asyncio.run(main())

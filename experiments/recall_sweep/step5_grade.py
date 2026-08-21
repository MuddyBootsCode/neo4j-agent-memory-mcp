"""Step 5: grade every (query, item) pair injected by ANY config.

Grading uses the SAME judge model (qwen-judge) as config C's gate, but a
DIFFERENTLY-WORDED prompt — a separate framing ("audit", skeptical of
keyword overlap) rather than the gate's "would this help answer" framing.
This does not remove the shared-model bias (documented as the sweep's
biggest caveat: see README.md), only the shared-wording bias.

Grades once per query for the union of config_a ∪ config_b (config_c's ids
are always a subset of config_b's, so no separate grading pass is needed for
C — its precision reads off the same grades as B's).
"""

from __future__ import annotations

import asyncio

from judge import call as judge_call
from lib import load_json, save_json

GRADE_SYSTEM = (
    "You are auditing a memory-retrieval system for relevance, independent of "
    "whatever system selected these candidates. Be skeptical: shared "
    "vocabulary or topic is not enough — the candidate must genuinely help "
    "answer the specific query. Answer only with the requested JSON."
)

GRADE_USER_TMPL = """\
QUERY (what the user actually asked):
{query}

For each numbered CANDIDATE record below, judge independently: if this \
record were shown to an assistant answering the query above, would it \
change or improve the answer? Do not credit a record just because it shares \
words with the query — judge whether it is SPECIFICALLY on point.

CANDIDATES:
{items}

Return JSON: {{"grades": [{{"id": <int>, "relevant": true|false, "reason": \
"<one short sentence>"}}, ...]}} with exactly one entry per candidate id, \
including ones you judge irrelevant.
"""


def _combined_items(record: dict) -> list[dict]:
    """Union of config_a and config_b items for one query, each with its id."""
    combined = list(record.get("config_a") or [])
    combined += list(record.get("config_b") or [])
    return combined


async def main() -> None:
    candidates = load_json("candidates.json")
    if not candidates:
        raise SystemExit("run step4_configs.py first (no candidates found)")

    grades = {}
    errors = []

    for record in candidates:
        qid = record["query_id"]
        items = _combined_items(record)
        if not items:
            grades[str(qid)] = {}
            print(f"query {qid}: no injected items — nothing to grade")
            continue

        items_text = "\n".join(
            f"{i}. [{it['kind']}] {it['text']}" for i, it in enumerate(items)
        )
        user = GRADE_USER_TMPL.format(query=record["prompt"], items=items_text)
        parsed, err = await judge_call(GRADE_SYSTEM, user, max_tokens=2000, timeout=180)

        if parsed is None:
            errors.append({"query_id": qid, "error": err})
            grades[str(qid)] = {"_error": err}
            print(f"query {qid}: GRADE FAIL: {err}")
            continue

        by_idx = {}
        for g in parsed.get("grades", []):
            if isinstance(g, dict) and isinstance(g.get("id"), int):
                by_idx[g["id"]] = g

        by_item_id = {}
        for i, it in enumerate(items):
            g = by_idx.get(i)
            if g is None:
                by_item_id[it["id"]] = {"relevant": None, "reason": "missing from judge output"}
            else:
                by_item_id[it["id"]] = {
                    "relevant": g.get("relevant"),
                    "reason": g.get("reason"),
                }
        grades[str(qid)] = by_item_id
        n_relevant = sum(1 for v in by_item_id.values() if v["relevant"] is True)
        print(f"query {qid}: graded {len(by_item_id)} items, {n_relevant} relevant")

    save_json("grades.json", grades)
    save_json("grade_errors.json", errors)
    print(f"\nwrote grades.json for {len(grades)} queries; {len(errors)} grading errors")


if __name__ == "__main__":
    asyncio.run(main())

"""Step 6: aggregate metrics, write results.json and README.md."""

from __future__ import annotations

import json
import os

from lib import HERE, load_json, save_json

CONFIGS = ["A", "B", "C"]


def _config_items(record: dict, config: str) -> list[dict]:
    if config == "A":
        return list(record.get("config_a") or [])
    if config == "B":
        return list(record.get("config_b") or [])
    if config == "C":
        by_id = {it["id"]: it for it in (record.get("config_b") or [])}
        return [by_id[i] for i in (record.get("config_c_ids") or []) if i in by_id]
    raise ValueError(config)


def _grade_for(grades_for_query: dict, item_id: str):
    if not grades_for_query or "_error" in grades_for_query:
        return None  # whole-query grading failure — excluded
    return grades_for_query.get(item_id)


def _tally(candidates: list[dict], grades: dict) -> tuple[dict, list[dict]]:
    """Per-config metrics and per-query rows over the given candidate records."""
    n_queries = len(candidates)
    per_query_records = []
    metrics = {
        c: {"injected": 0, "relevant": 0, "excluded": 0, "queries_with_injection": 0,
            "queries_with_relevant": 0, "zero_injection_queries": 0}
        for c in CONFIGS
    }

    for record in candidates:
        qid = record["query_id"]
        g = grades.get(str(qid), {})
        query_row = {"query_id": qid, "prompt": record["prompt"], "files": record["files"],
                     "configs": {}}

        for c in CONFIGS:
            items = _config_items(record, c)
            m = metrics[c]
            m["injected"] += len(items)
            if len(items) == 0:
                m["zero_injection_queries"] += 1
            else:
                m["queries_with_injection"] += 1

            relevant_count = 0
            excluded_count = 0
            item_rows = []
            for it in items:
                grade = _grade_for(g, it["id"])
                if grade is None or grade.get("relevant") is None:
                    excluded_count += 1
                    verdict = None
                else:
                    verdict = bool(grade["relevant"])
                    if verdict:
                        relevant_count += 1
                item_rows.append({
                    "id": it["id"], "kind": it.get("kind"), "text": it.get("text"),
                    "similarity": it.get("similarity"),
                    "relevant": verdict,
                    "reason": (grade or {}).get("reason") if grade else None,
                })
            m["relevant"] += relevant_count
            m["excluded"] += excluded_count
            if relevant_count > 0:
                m["queries_with_relevant"] += 1

            query_row["configs"][c] = {
                "injected_count": len(items),
                "relevant_count": relevant_count,
                "excluded_count": excluded_count,
                "items": item_rows,
            }
        per_query_records.append(query_row)

    summary = {}
    for c in CONFIGS:
        m = metrics[c]
        gradeable = m["injected"] - m["excluded"]
        precision = (m["relevant"] / gradeable) if gradeable > 0 else None
        summary[c] = {
            "n_queries": n_queries,
            "injected_total": m["injected"],
            "relevant_total": m["relevant"],
            "excluded_total": m["excluded"],
            "precision": precision,
            "items_per_query": m["injected"] / n_queries if n_queries else None,
            "coverage": m["queries_with_relevant"] / n_queries if n_queries else None,
            "zero_injection_rate": m["zero_injection_queries"] / n_queries if n_queries else None,
        }
    return summary, per_query_records


def _table(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print(f"{'config':<8}{'queries':<9}{'precision':<12}{'items/query':<14}{'coverage':<12}{'zero-inj rate':<14}")
    for c in CONFIGS:
        s = summary[c]
        prec = f"{s['precision']:.0%}" if s["precision"] is not None else "n/a"
        cov = f"{s['coverage']:.0%}" if s["coverage"] is not None else "n/a"
        zir = f"{s['zero_injection_rate']:.0%}" if s["zero_injection_rate"] is not None else "n/a"
        ipq = f"{s['items_per_query']:.2f}" if s["items_per_query"] is not None else "n/a"
        print(f"{c:<8}{s['n_queries']:<9}{prec:<12}{ipq:<14}{cov:<12}{zir:<14}")


def main() -> None:
    candidates = load_json("candidates.json") or []
    grades = load_json("grades.json") or {}
    corpus_stats = load_json("corpus_stats.json") or {}
    config_errors = load_json("config_errors.json") or []
    grade_errors = load_json("grade_errors.json") or []
    queries = load_json("queries.json") or []
    split = load_json("session_split.json") or {}

    n_queries = len(candidates)
    summary, per_query_records = _tally(candidates, grades)

    # Stratified runs deliberately over-sample queries that config B could
    # answer at all, so a single average over the whole set is not an estimate
    # of anything. Break the strata out; report each on its own terms.
    anchorable = {q["query_id"]: q.get("anchorable") for q in queries}
    strata = {}
    if any(v is not None for v in anchorable.values()):
        for name, want in (("anchorable", True), ("unanchorable", False)):
            subset = [r for r in candidates if anchorable.get(r["query_id"]) is want]
            if subset:
                strata[name], _ = _tally(subset, grades)
        for r in per_query_records:
            r["anchorable"] = anchorable.get(r["query_id"])

    results = {
        "meta": {
            "n_corpus_sessions": len(split.get("corpus_sessions", [])),
            "n_query_sessions": len(split.get("query_sessions", [])),
            "n_total_sessions": split.get("total_sessions"),
            "n_queries": n_queries,
        },
        "corpus_stats": corpus_stats,
        "summary": summary,
        "strata": strata,
        "per_query": per_query_records,
        "config_errors": config_errors,
        "grade_errors": grade_errors,
    }
    # The deliverable results.json lives at experiments/recall_sweep/results.json
    # (top level, per the task) — results/ underneath holds intermediate,
    # per-step artifacts (session_split, corpus_stats, candidates, grades, ...).
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    # ---- console tables ----
    _table("ALL QUERIES" + (" (stratified sample -- see strata below)" if strata else ""), summary)
    for name, sub in strata.items():
        _table(name.upper() + " QUERIES", sub)

    print(f"\nwrote results.json ({n_queries} queries, {len(candidates)} candidate records)")


if __name__ == "__main__":
    main()

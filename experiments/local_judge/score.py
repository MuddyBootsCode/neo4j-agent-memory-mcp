"""Score the local judge against the Claude labels already on disk.

    python score.py

Two questions, kept separate:

1. Agreement — does the local model grade the way Claude graded? Reported as
   exact 3-way match, binary match, and Cohen's kappa, because on a set that is
   67% zeros raw agreement is flattered by the majority class alone.

2. Gatekeeping — if the model decided what to inject instead of the similarity
   cutoff, would the injection be better? Precision, recall and F1 for each
   gate over the same candidates, scored on the Claude grades.

Recall here is bounded by the candidate pool: the denominator is the relevant
records inside each query's top 10, not everything relevant in the corpus. A
gate cannot inject what retrieval never surfaced, so this measures the gate,
not the retriever.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SHIPPING_THRESHOLD = 0.50
TUNED_THRESHOLD = 0.38   # MUD-386
RELEVANT = 1             # Claude grade >= 1 counts as relevant


def load(config, corpus="production"):
    tail = "" if corpus == "production" else f".{corpus}"
    with open(os.path.join(RESULTS, f"selected.{corpus}.json")) as fh:
        selected = json.load(fh)
    with open(os.path.join(RESULTS, f"verdicts.{config}{tail}.json")) as fh:
        verdicts = json.load(fh)
    return selected, verdicts


def pairs(selected, verdicts):
    """Flatten to one row per (query, candidate) with both judges' opinions."""
    rows = []
    for q in selected:
        rec = verdicts.get(str(q["query_id"]))
        if rec is None:
            continue
        for c in q["candidates"]:
            v = (rec.get("verdicts") or {}).get(str(c["id"]))
            rows.append({
                "query_id": q["query_id"],
                "prompt": q["prompt"],
                "project": q["project"],
                "id": c["id"],
                "rank": c["rank"],
                "sim": c["sim"],
                "kind": c["kind"],
                "text": c["text"],
                "claude": c["claude"],
                "qwen": _grade(v),
                "keep": bool(v.get("keep")) if isinstance(v, dict) else None,
                "reason": (v or {}).get("reason") if isinstance(v, dict) else None,
                "judged": isinstance(v, dict),
            })
    return rows


def _grade(v):
    if not isinstance(v, dict):
        return None
    try:
        g = int(v.get("grade"))
    except (TypeError, ValueError):
        return None
    return g if g in (0, 1, 2) else None


def prf(selected_rows, all_rows):
    """Precision/recall/F1 of a gate's selection against the Claude grades."""
    tp = sum(1 for r in selected_rows if (r["claude"] or 0) >= RELEVANT)
    fp = len(selected_rows) - tp
    relevant = sum(1 for r in all_rows if (r["claude"] or 0) >= RELEVANT)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / relevant if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "injected": len(selected_rows),
        "per_query": len(selected_rows) / len({r["query_id"] for r in all_rows}),
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def kappa(rows, key_a="claude", key_b="qwen"):
    """Cohen's kappa over the 3-way grades."""
    usable = [r for r in rows if r[key_a] is not None and r[key_b] is not None]
    if not usable:
        return None
    n = len(usable)
    observed = sum(1 for r in usable if r[key_a] == r[key_b]) / n
    expected = 0.0
    for g in (0, 1, 2):
        pa = sum(1 for r in usable if r[key_a] == g) / n
        pb = sum(1 for r in usable if r[key_b] == g) / n
        expected += pa * pb
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else None


def confusion(rows):
    m = {a: {b: 0 for b in (0, 1, 2)} for a in (0, 1, 2)}
    for r in rows:
        if r["claude"] in m and r["qwen"] in m[r["claude"]]:
            m[r["claude"]][r["qwen"]] += 1
    return m


def build(selected, verdicts):
    rows = pairs(selected, verdicts)
    judged = [r for r in rows if r["judged"]]
    n_queries = len({r["query_id"] for r in rows})

    gates = {
        "threshold_0.50_shipping": prf([r for r in rows if r["sim"] >= SHIPPING_THRESHOLD], rows),
        "threshold_0.38_tuned": prf([r for r in rows if r["sim"] >= TUNED_THRESHOLD], rows),
        "qwen_keep": prf([r for r in judged if r["keep"]], rows),
        "qwen_grade_ge_1": prf([r for r in judged if (r["qwen"] or 0) >= 1], rows),
        "qwen_grade_eq_2": prf([r for r in judged if r["qwen"] == 2], rows),
    }

    exact = sum(1 for r in judged if r["claude"] == r["qwen"])
    binary = sum(
        1 for r in judged
        if ((r["claude"] or 0) >= RELEVANT) == ((r["qwen"] or 0) >= RELEVANT)
    )

    calls = [verdicts[k] for k in verdicts]
    ok = [c for c in calls if c.get("ok")]
    durations = sorted(c["duration_ms"] for c in ok if c.get("duration_ms"))

    return {
        "queries": n_queries,
        "candidates": len(rows),
        "judged": len(judged),
        "claude_relevant": sum(1 for r in rows if (r["claude"] or 0) >= RELEVANT),
        "claude_grades": {g: sum(1 for r in rows if r["claude"] == g) for g in (0, 1, 2)},
        "qwen_grades": {g: sum(1 for r in judged if r["qwen"] == g) for g in (0, 1, 2)},
        "agreement": {
            "exact": round(exact / len(judged), 3) if judged else None,
            "binary": round(binary / len(judged), 3) if judged else None,
            "kappa": kappa(judged),
        },
        "confusion": confusion(judged),
        "gates": gates,
        "calls": {
            "total": len(calls),
            "ok": len(ok),
            "json_ok": sum(1 for c in calls if c.get("json_ok")),
            "retries": sum((c.get("attempts") or 1) - 1 for c in calls),
            "missing_ids": sum(len(c.get("missing_ids") or []) for c in calls),
            "hallucinated_ids": sum(len(c.get("hallucinated_ids") or []) for c in calls),
            "median_ms": durations[len(durations) // 2] if durations else None,
            "total_s": round(sum(durations) / 1000, 1) if durations else None,
            "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
            "thinking_chars": sum(len(c.get("thinking") or "") for c in calls),
        },
        "rows": rows,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="think-reason")
    ap.add_argument("--corpus", default="production")
    args = ap.parse_args()

    selected, verdicts = load(args.config, args.corpus)
    report = build(selected, verdicts)
    report["config"] = args.config
    report["corpus"] = args.corpus
    tail = "" if args.corpus == "production" else f".{args.corpus}"
    out = os.path.join(RESULTS, f"score.{args.config}{tail}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)

    print(f"[{args.config}] {report['queries']} queries, {report['candidates']} candidates, "
          f"{report['claude_relevant']} relevant by Claude")
    a = report["agreement"]
    print(f"agreement  exact={a['exact']}  binary={a['binary']}  kappa={a['kappa']}")
    print(f"{'gate':<26}{'inject':>7}{'prec':>7}{'recall':>8}{'F1':>7}")
    for name, g in report["gates"].items():
        print(f"{name:<26}{g['per_query']:>7.2f}{g['precision']:>7.3f}"
              f"{g['recall']:>8.3f}{g['f1']:>7.3f}")
    c = report["calls"]
    print(f"calls ok={c['ok']}/{c['total']} json_ok={c['json_ok']} "
          f"retries={c['retries']} missing={c['missing_ids']} "
          f"halluc={c['hallucinated_ids']} median={c['median_ms']}ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

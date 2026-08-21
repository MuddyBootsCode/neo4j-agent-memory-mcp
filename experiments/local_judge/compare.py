"""Compare the three judging configs on identical candidates.

    python compare.py

Isolates two costs that a single run would confound: the thinking block, and
the per-candidate reason field. Dropping thinking is the obvious latency win.
Dropping the reason is the subtle one — writing a justification before
committing to a grade is chain-of-thought inside the answer, so it may be
carrying accuracy that the thinking block was not.

Writes results/compare.json.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from runtime import CONFIGS  # noqa: E402
from score import build, load  # noqa: E402

RESULTS = os.path.join(HERE, "results")
GATE = "qwen_grade_ge_1"


def main():
    out = {}
    for name, cfg in CONFIGS.items():
        try:
            selected, verdicts = load(name)
        except FileNotFoundError:
            continue
        if not verdicts:
            continue
        report = build(selected, verdicts)
        report["config"] = name
        report["blurb"] = cfg["blurb"]
        with open(os.path.join(RESULTS, f"score.{name}.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        out[name] = report

    rows = []
    for name, r in out.items():
        c, g, a = r["calls"], r["gates"], r["agreement"]
        rows.append({
            "config": name,
            "blurb": r["blurb"],
            "queries": r["queries"],
            "median_s": round((c["median_ms"] or 0) / 1000, 1),
            "output_tokens_per_call": round(c["output_tokens"] / max(c["total"], 1)),
            "json_ok": f"{c['json_ok']}/{c['total']}",
            "missing_ids": c["missing_ids"],
            "hallucinated_ids": c["hallucinated_ids"],
            "kappa": a["kappa"],
            "binary_agreement": a["binary"],
            "precision": g[GATE]["precision"],
            "recall": g[GATE]["recall"],
            "f1": g[GATE]["f1"],
            "inject_per_prompt": round(g[GATE]["per_query"], 2),
        })

    with open(os.path.join(RESULTS, "compare.json"), "w") as fh:
        json.dump({"gate": GATE, "rows": rows}, fh, indent=1)

    head = f"{'config':<14}{'med s':>7}{'out tok':>9}{'json':>8}{'kappa':>7}{'prec':>7}{'recall':>8}{'F1':>7}{'inject':>8}"
    print(f"gate: {GATE}")
    print(head)
    for r in rows:
        print(f"{r['config']:<14}{r['median_s']:>7.1f}{r['output_tokens_per_call']:>9}"
              f"{r['json_ok']:>8}{r['kappa']:>7}{r['precision']:>7.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>7.3f}{r['inject_per_prompt']:>8.2f}")
    print(f"wrote {os.path.join(RESULTS, 'compare.json')}")


if __name__ == "__main__":
    main()

"""Score retrieval for each variant's subgraph against the same query set.

    python evaluate.py --queries 80

Stages, cached to results/ so any one can be rerun alone:
  retrieve  embed each variant's records + the queries, keep top-k
  pool      union the candidates across variants — every variant's hits get
            judged, so none is penalised for surfacing something another missed
  judge     one LLM call per query labels its whole candidate set
  score     nDCG, threshold sweep, paired bootstrap on the held-out split

The threshold sweep exists because the recall hook filters by absolute
similarity rather than taking top-k, so rank metrics alone describe something
the production code does not do.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")

import queries as queries_mod
from store import SubGraph, all_variants_present
from variants import VARIANTS, embed_text

POOL_K = 10
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

JUDGE_SYSTEM = """You label retrieval relevance for an AI assistant's memory system.

The system injects stored memories into a coding assistant's context before it
answers a user's prompt. Decide whether each memory would genuinely help the
assistant respond better to that prompt.

2 = directly informs the answer: a preference the answer should follow, a fact
    the prompt asks about, a constraint that changes the response.
1 = same topic, useful background, but the answer is substantially the same
    without it.
0 = unrelated, or so generic it adds nothing. Most memories are 0 for most
    prompts. Merely sharing vocabulary with the prompt is 0 — a record about a
    different PR, ticket, or migration number than the one asked about is 0.

Be strict. Injected noise wastes context and misleads the model.
Label every candidate, including the 0s."""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate": {"type": "integer"},
                    "label": {"type": "integer", "enum": [0, 1, 2]},
                },
                "required": ["candidate", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def _path(n):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, n)


def _load(n, d=None):
    p = _path(n)
    return json.load(open(p)) if os.path.exists(p) else d


def _save(n, o):
    json.dump(o, open(_path(n), "w"), indent=1)


def _normed(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)


def _anthropic_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = os.path.join(os.path.dirname(os.path.dirname(HERE)), ".env")
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")


def variant_records(name):
    """Retrievable records for a variant, as the retriever would see them."""
    with SubGraph(name) as g:
        ents, prefs = g.records()
    out = [{"kind": "entity", "text": embed_text(name, e)} for e in ents]
    out += [
        {
            "kind": "preference",
            "text": f"[{p['category']}] {p['preference']}"
            + (f" ({p['context']})" if p.get("context") else ""),
        }
        for p in prefs
    ]
    return out


def stage_retrieve(names, n_queries):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(6)
    qs = _load("queries.json")
    if not qs:
        qs = queries_mod.sample(n_queries)
        _save("queries.json", qs)
    qtexts = [q["prompt"] for q in qs]

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    qv = _normed(model.encode(qtexts, batch_size=32, show_progress_bar=False))

    runs = _load("runs.json", {})
    corpora = _load("corpora.json", {})
    for name in names:
        recs = variant_records(name)
        corpora[name] = recs
        dv = _normed(
            model.encode([r["text"] for r in recs], batch_size=32, show_progress_bar=False)
        )
        sims = qv @ dv.T
        runs[name] = [
            [
                {"doc": int(j), "sim": round(float(sims[i][j]), 4)}
                for j in np.argsort(-sims[i])[:POOL_K]
            ]
            for i in range(len(qtexts))
        ]
        print(f"  {name}: {len(recs)} records indexed")
    _save("runs.json", runs)
    _save("corpora.json", corpora)
    return runs


def stage_judge(names, model, workers):
    """Judge per (query, variant) — candidate texts differ between variants."""
    _anthropic_key()
    import anthropic

    qs = _load("queries.json")
    corpora = _load("corpora.json")
    runs = _load("runs.json")
    labels = _load("labels.json", {})

    todo = [
        (name, qi)
        for name in names
        for qi in range(len(qs))
        if any(f"{name}|{qi}|{c['doc']}" not in labels for c in runs[name][qi])
    ]
    print(f"  {len(todo)} (variant, query) pairs to judge")
    client = anthropic.Anthropic()

    def judge(name, qi):
        cands = [c["doc"] for c in runs[name][qi]]
        listing = "\n".join(
            f"[{n}] ({corpora[name][d]['kind']}) {corpora[name][d]['text'][:300]}"
            for n, d in enumerate(cands)
        )
        r = client.messages.create(
            model=model,
            max_tokens=3000,
            system=JUDGE_SYSTEM,
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content":
                       f"User prompt:\n{qs[qi]['prompt']}\n\n"
                       f"Candidate memories:\n{listing}\n\nLabel every candidate."}],
        )
        data = json.loads("".join(b.text for b in r.content if b.type == "text"))
        out = {f"{name}|{qi}|{d}": 0 for d in cands}
        for item in data.get("labels", []):
            n = item.get("candidate")
            if isinstance(n, int) and 0 <= n < len(cands):
                out[f"{name}|{qi}|{cands[n]}"] = int(item.get("label", 0))
        return out

    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge, n, q): (n, q) for n, q in todo}
        for fut in cf.as_completed(futs):
            try:
                labels.update(fut.result())
            except Exception as e:
                print(f"  {futs[fut]} FAILED {type(e).__name__}: {str(e)[:100]}")
            done += 1
            if done % 25 == 0 or done == len(todo):
                _save("labels.json", labels)
                print(f"  judged {done}/{len(todo)}", flush=True)
    _save("labels.json", labels)
    dist = {}
    for v in labels.values():
        dist[v] = dist.get(v, 0) + 1
    print(f"  label distribution: {dict(sorted(dist.items()))}")


def _ndcg(top, name, qi, labels, k=5):
    gains = [labels.get(f"{name}|{qi}|{c['doc']}", 0) for c in top[:k]]
    ideal = sorted((v for key, v in labels.items()
                    if key.startswith(f"{name}|{qi}|")), reverse=True)[:k]
    def dcg(g):
        return sum(x / np.log2(i + 2) for i, x in enumerate(g))
    return dcg(gains) / dcg(ideal) if dcg(ideal) > 0 else None


def _bootstrap(a, b, n=10000, seed=3):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    diffs = [x - y for x, y in pairs]
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        for _ in range(n)
    )
    return (round(float(sum(diffs) / len(diffs)), 4),
            [round(float(means[int(n * .025)]), 4), round(float(means[int(n * .975)]), 4)])


def stage_score(names, split):
    qs = _load("queries.json")
    runs = _load("runs.json")
    labels = _load("labels.json")
    if not labels:
        sys.exit("no labels — run the judge stage")

    idx = [i for i, q in enumerate(qs) if not split or q.get("split") == split]
    print(f"\nscoring {len(idx)} queries" + (f" (split={split})" if split else ""))

    print("\n=== nDCG@5 ===")
    nd = {}
    for name in names:
        s = [_ndcg(runs[name][qi], name, qi, labels) for qi in idx]
        nd[name] = s
        v = [x for x in s if x is not None]
        print(f"  {name:12} {statistics.mean(v):.3f}  (n={len(v)})")

    if "baseline" in names:
        print("\n=== paired bootstrap vs baseline ===")
        for name in names:
            if name == "baseline":
                continue
            r = _bootstrap(nd[name], nd["baseline"])
            if r:
                sig = "significant" if r[1][0] > 0 else "not significant"
                print(f"  {name:12} {r[0]:+.3f}  95% CI {r[1]}  {sig}")

    print("\n=== threshold sweep ===")
    def f1(p, r):
        return 2 * p * r / (p + r) if p and r else 0.0
    for name in names:
        rows = []
        for t in [x / 100 for x in range(20, 82, 4)]:
            P, R, INJ, NZ = [], [], [], []
            for qi in idx:
                rel = {int(k.split("|")[2]) for k, v in labels.items()
                       if k.startswith(f"{name}|{qi}|") and v == 2}
                kept = [c for c in runs[name][qi] if c["sim"] >= t]
                INJ.append(len(kept))
                hit = sum(1 for c in kept if c["doc"] in rel)
                if kept:
                    P.append(hit / len(kept))
                if rel:
                    R.append(hit / len(rel))
                else:
                    NZ.append(len(kept))
            p = statistics.mean(P) if P else 0
            r = statistics.mean(R) if R else 0
            rows.append((t, statistics.mean(INJ), p, r, f1(p, r),
                         statistics.mean(NZ) if NZ else 0))
        best = max(rows, key=lambda x: x[4])
        print(f"\n  {name}  — best F1 {best[4]:.3f} at threshold {best[0]:.2f}")
        print(f"    {'thr':>5} {'inject':>7} {'prec':>6} {'recall':>7} {'F1':>6} {'noise':>6}")
        for row in rows:
            mark = "  <=" if row[0] == best[0] else ""
            print(f"    {row[0]:>5.2f} {row[1]:>7.2f} {row[2]:>6.3f} "
                  f"{row[3]:>7.3f} {row[4]:>6.3f} {row[5]:>6.2f}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["retrieve", "judge", "score", "all"])
    ap.add_argument("--variants", default=None)
    ap.add_argument("--queries", type=int, default=80)
    ap.add_argument("--judge-model", default="claude-opus-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="report", choices=["calibrate", "report", "all"])
    a = ap.parse_args()

    names = ([v for v in a.variants.split(",") if v] if a.variants
             else [v for v in all_variants_present() if v in VARIANTS])
    if not names:
        sys.exit("no variant subgraphs found — run extract.py first")
    print(f"variants: {names}")

    if a.stage in ("retrieve", "all"):
        stage_retrieve(names, a.queries)
    if a.stage in ("judge", "all"):
        stage_judge(names, a.judge_model, a.workers)
    if a.stage in ("score", "all"):
        stage_score(names, None if a.split == "all" else a.split)


if __name__ == "__main__":
    main()

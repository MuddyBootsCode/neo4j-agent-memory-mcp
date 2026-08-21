"""Assemble the judging set from the extractor experiment's cached results.

No model, no embedding, no Neo4j — everything needed is already on disk from
the variant sweep: the sampled prompts, the per-variant record corpora, the
top-10 retrieval runs, and 2,000 Claude-judged production pairs.

    python select.py --queries 30

Writes results/selected.json.
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(os.path.dirname(HERE), "extractors", "results")
RESULTS = os.path.join(HERE, "results")

DEFAULT_CORPUS = "production"


def _load(name):
    with open(os.path.join(SWEEP, name)) as fh:
        return json.load(fh)


def build(n, split="report", corpus=DEFAULT_CORPUS):
    """The judging set for one corpus.

    Corpus selection matters more than it looks: `production` entities carry a
    bare name, `described` entities carry a self-contained sentence. Retrieval
    was run separately over each, so the candidates differ too — this is not
    the same 300 records with more text attached.
    """
    queries = _load("queries.json")
    records = _load("corpora.json")[corpus]
    runs = _load("runs.json")[corpus]
    labels = _load("labels.json")

    picked = []
    for i, q in enumerate(queries):
        if len(picked) >= n:
            break
        if split and q.get("split") != split:
            continue
        candidates = []
        for rank, hit in enumerate(runs[i]):
            doc = hit["doc"]
            record = records[doc]
            candidates.append({
                "id": doc,
                "rank": rank,
                "sim": hit["sim"],
                "kind": record["kind"],
                "text": record["text"],
                # The grade Claude gave this exact pair during the sweep.
                # Absent means the pair was never judged; kept as None rather
                # than defaulted to 0, which would invent agreement.
                "claude": labels.get(f"{corpus}|{i}|{doc}"),
            })
        picked.append({
            "query_id": i,
            "corpus": corpus,
            "prompt": q["prompt"],
            "project": q["project"],
            "split": q["split"],
            "candidates": candidates,
        })
    return picked


def candidates_text(query):
    """The candidate block as the model sees it.

    Kind is included and similarity is not: the score is what we are testing
    the model against, so showing it would let the model anchor on the very
    ranking we want an independent opinion of.
    """
    return "\n".join(
        f"[{c['id']}] ({c['kind']}) {c['text']}" for c in query["candidates"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--split", default="report")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    args = ap.parse_args()

    picked = build(args.queries, args.split, args.corpus)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"selected.{args.corpus}.json")
    with open(out, "w") as fh:
        json.dump(picked, fh, indent=1)

    pairs = sum(len(q["candidates"]) for q in picked)
    judged = sum(1 for q in picked for c in q["candidates"] if c["claude"] is not None)
    print(f"{len(picked)} queries, {pairs} candidates, {judged} with Claude labels")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

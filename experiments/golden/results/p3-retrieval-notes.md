# P3 retrieval runs — 2026-08-22

All runs score the `p1-capture` pool (225 lessons) and its Opus 5 labels
(513 relevant pairs) against the same 100 queries, rematerialized into a
scratch DB with `step1b_materialize.py` under each configuration. Code
`2624a24` for p3-bm25, `0d4c423` (adds the 6 s gate cap) for p3-bge and
p3-ctx. No relabeling: lesson text is unchanged, so labels carry over.

| run | embedder | legs | query | cosine20 R | D P / R / P@5 | E P / R / items | gate p50 / p95 |
|---|---|---|---|---|---|---|---|
| p1-capture (reference) | MiniLM-384 | vector + anchor boost | prompt | 39% | 14% / 28% / 19% | 27% / 18% / 3.3 | 3.6 s / 35 s |
| p3-bm25 | MiniLM-384 | vector + BM25, RRF | prompt | 39% | 14% / 27% / 19% | 27% / 17% / 3.2 | 3.2 s / 32 s |
| p3-bge | bge-base-768 | vector + BM25, RRF | prompt | 35% | 13% / 25% / 17% | 26% / 16% / 3.2 | 3.2 s / 3.5 s |
| p3-ctx | MiniLM-384 | vector + BM25, RRF | prompt + 2 prior turns | 27% | 11% / 22% / 14% | 15% / 15% / 5.3 | 3.5 s / 6.0 s |

Absolute relevant lessons reached, cosine20 / D / E: p1 201 / 142 / 90,
p3-bm25 198 / 138 / 86, p3-bge 181 / 129 / 82, p3-ctx 137 / 112 / 78.

## Findings

F1. **BM25 fusion is neutral on this workload.** Precision, recall and P@5
are identical to the vector-only reference within one or two items. The
prompts are mostly short conversational follow-ups with no path, env var
or error string for BM25 to match; the leg costs ~50 ms p50. Kept: it
serves recall when no embedder is available and it will matter on prompts
that paste an error, which this query set under-represents.

F2. **bge-base is not better than MiniLM here; it is slightly worse.**
Cosine top-20 recall 35% vs 39%, D 25% vs 27%. The +0.094 nDCG@5 from the
extractor sweep was measured on entity/preference records with descriptions
and a different query distribution; it does not transfer to lesson text
against these prompts. Embedding 4x slower (39 ms vs 8 ms p50). Decision
D1 is reversed: MiniLM stays the default, bge stays available by config
(`NAM_EMBEDDING_MODEL`, `NAM_EMBEDDING_DIMENSIONS`, backfill
`--recreate-index`) for a workload that measures differently.

F3. **Query expansion with prior turns hurts.** Prepending the previous two
human prompts and the last assistant text drops cosine top-20 recall from
39% to 27% and gate precision from 27% to 15%. Two effects: the longer
query dilutes the embedding, and the labels were made against the bare
prompt plus file context, so the expanded query asks a different question
from the one that was labeled. Not shipped.

F4. **The gate cap works.** With `NAM_RECALL_GATE_TIMEOUT=6`, p95 gate
latency fell from 31-35 s to 3.5-6 s and the five calls that exceeded it
fell open inside the budget instead of stalling for 900 s. This is the one
retrieval change with an unambiguous effect.

F5. **Nothing on the retrieval side has moved the per-prompt numbers.**
Anchor boost (baseline), BM25, a stronger embedder, and query expansion
all land within noise of vector-only MiniLM. The ceiling at depth 20 is
35-39% of relevant lessons and the gate turns 14% precision into 27% at a
third of the items. The remaining levers are not in the ranker: the gate's
depth and cost (screening 20-40 instead of 10), ranking signals that do not
depend on the prompt text (evidence count, served/helpful counters — P2),
and the label set itself, which is generous on short prompts.

## Caveats

One pool, one repo, 100 queries, one labeler. Differences under ~3 points
or ~10 relevant items are noise at this size. The p1 reference and p3-bm25
differ only in the BM25 leg; p3-bge and p3-ctx also carry the gate cap.

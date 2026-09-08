# p5-e1: contextual prefix on the embedded lesson text (MUD-456, 2026-09-08)

Anthropic's contextual retrieval applied to lessons: prepend where a
lesson came from before embedding it. Four variants over the p4-live pool
and queries, ablated up front rather than only on a win. Baseline is
`p5-repro` (the prerequisite's no-op check), prior off, `cosine20,D`.

```bash
cd experiments/golden
for v in 1 files repo kind; do
  GOLDEN_RUN=p5-e1-$v GOLDEN_POOL_FROM=results/p4-live/pool.json GOLDEN_DB=goldenp5 \
    NAM_EMBED_CONTEXT_PREFIX=$v uv run --no-sync --project ../.. python -u step1b_materialize.py
  GOLDEN_RUN=p5-e1-$v GOLDEN_DB=goldenp5 GOLDEN_CONFIGS=cosine20,D \
    NAM_RECALL_OUTCOME_WEIGHT=0 NAM_RECALL_EVIDENCE_WEIGHT=0 \
    uv run --no-sync --project ../.. python -u step5_score.py
done
```

| variant | prefix | cosine20 relevant | cosine20 recall | Δ | D precision | D recall | D P@5 |
|---|---|---|---|---|---|---|---|
| baseline | none | 310 | 30.0% | — | 20.7% | 20.0% | 25.8% |
| full | repo · kind · files | 349 | 33.8% | +3.8 | 21.9% | 21.2% | 28.2% |
| **files** | **files** | **370** | **35.8%** | **+5.8** | **21.5%** | **20.8%** | **24.8%** |
| repo | repo | 306 | 29.6% | −0.4 | 21.3% | 20.6% | 25.0% |
| kind | kind | 314 | 30.4% | +0.4 | 21.9% | 21.2% | 25.8% |

**The filenames are doing all the work, and the other two parts get in
their way.** Files alone beats the full prefix by 2 points. Repo is a
constant inside a repo-scoped search — the same token on every lesson,
pure dilution, and it measures as such. Kind is three values over 287
lessons and moves nothing.

**Below the bar.** +5.8 points against the ≥10 the P5 plan asks for. It
is the largest single move any variant has produced since P1, and it
costs nothing at query time, but on its own it does not clear.

## Where the lift lands, which is not where it was expected

cosine20 recall by what file context the query has:

| variant | no files at all (40) | files, none in the pool (9) | shares a pool file (51) |
|---|---|---|---|
| baseline | 32.1% | 35.9% | 28.0% |
| full | 41.4% | 35.9% | 28.0% |
| files | **47.1%** | 38.5% | 27.4% |

The prefix helps most where the session edited nothing, and slightly
hurts where the session edited a file the lesson is anchored to. So this
is not the anchor signal arriving by another route: filenames in the
lesson text work as topic words that match the prompt's vocabulary, and
they pay off exactly in the cell where anchoring has nothing to offer.
Where the anchor already applies, the filenames are shared by every
nearby lesson and only dilute.

That splits the file signal in two, on disjoint halves of the query set,
and it is a prediction for E6 (MUD-461): an anchor leg can only fire on
the 51 queries that share a file — the cell the prefix does not improve.

Not a labelling artifact: 186 of 287 pool lessons carry anchors (65%) and
they hold 651 of 1,033 relevant labels (63%), so relevance is not
over-represented on anchored lessons to begin with.

## Carried forward

`NAM_EMBED_CONTEXT_PREFIX=files`, not `=1`. Default stays off.

`pool.json`, `queries.json`, `labels.json` and `session_split.json` under
each run directory are byte-identical copies of p4-live's (step1b copies
them from `GOLDEN_POOL_FROM`); five P5 runs would have added ~10 MB of
duplicates, so only what each run produced is committed.

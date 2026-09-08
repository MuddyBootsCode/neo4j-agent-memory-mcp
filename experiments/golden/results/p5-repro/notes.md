# p5-repro (MUD-456, 2026-09-08)

The no-op check for the `lesson_id` prerequisite. `memory_embedding_text`
is now the canonical lesson text — what `lesson_id` hashes and what keys
duplicate detection — and `memory_embedding_input` is what reaches the
embedder. With `NAM_EMBED_CONTEXT_PREFIX` unset the two are the same
string, so p4-live has to reproduce exactly, and it does: every summary
metric equal to the decimal and every per-query id list identical.

```bash
cd experiments/golden
GOLDEN_RUN=p5-repro GOLDEN_POOL_FROM=results/p4-live/pool.json GOLDEN_DB=goldenp5 \
  uv run --no-sync --project ../.. python -u step1b_materialize.py
GOLDEN_RUN=p5-repro GOLDEN_DB=goldenp5 GOLDEN_CONFIGS=cosine20,D \
  NAM_RECALL_OUTCOME_WEIGHT=0 NAM_RECALL_EVIDENCE_WEIGHT=0 \
  uv run --no-sync --project ../.. python -u step5_score.py
```

| config | precision | recall | P@5 | coverage |
|---|---|---|---|---|
| cosine20 | 15.5% | 30.0% | 24.4% | 81% |
| D | 20.7% | 20.0% | 25.8% | 71% |

Identical to `results/p4-live/scores-prior-off.json`: 310 relevant of
2,000 injected at top-20, 207 of 1,000 at D, 1,033 relevant pairs over
287 lessons.

**The P5 baseline is cosine20 recall 30.0%, not the 27% MUD-455 was
written against.** 27% (338 of 1,263 pairs) is the withdrawn pre-fix run
described in `../p4-live/notes.md` — 330 lessons, the 43 leaked
query-family lessons still in the pool, counters unrestored. The ≥10
point bar for E1–E6 is therefore 40%.

`pool.json`, `queries.json`, `labels.json` and `session_split.json` are
byte-identical copies of p4-live's (step1b copies them from
`GOLDEN_POOL_FROM`), so only what this run produced is committed here.

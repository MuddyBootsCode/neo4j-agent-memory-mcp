# p4-outcome (MUD-407)

Re-score of the P2 pool with the outcome prior in the ranker.

| config | precision | recall | P@5 | coverage | items/query |
|---|---|---|---|---|---|
| cosine20 | 9.9% | 38.6% | 15.6% | 57% | 20.0 |
| D | 13.8% | 26.9% | 18.6% | 48% | 10.0 |
| E | 23.1% | 17.0% | 28.0% | 43% | 3.8 |

Against p2-regress: D is identical to three decimals, which is the point —
the ungated ranking did not move. E differs by 0.8 points on precision and
0.2 items per query, which is the local gate's own run-to-run variance over
the same candidate list in the same order, not the prior.

**The prior was not exercised by this run.** `step1b` reports
`counters_restored: 0`: the committed pool predates the props export, so
every lesson materialized with `evidence_count = 1` and no
`outcome_weight`, and the prior evaluated to the same constant (1.02) for
all 225 lessons. A constant multiplier cannot reorder anything, so this run
shows no regression and no effect. It is not evidence that the prior helps.

`step2_pool` now carries the counters in `props` and `step1b` restores them
after the write, so the next pool exported from a store with real history
will measure the prior for the first time. That run is what should set
`NAM_RECALL_OUTCOME_WEIGHT` / `NAM_RECALL_EVIDENCE_WEIGHT`; the shipped
0.2 / 0.1 are a starting point, not a measured choice.

Run: `GOLDEN_RUN=p4-outcome GOLDEN_POOL_FROM=results/p2-regress/pool.json`
→ `step1b_materialize.py`, `step5_score.py`, `step6_teardown.py`.

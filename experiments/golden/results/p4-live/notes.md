# p4-live and p4-live-b (MUD-435 S6, 2026-09-07)

First golden run on a pool exported from the live store with real
lifecycle counters, after five days of the MUD-407 outcome loop. Pool:
gradgraph-auth-platform, 330 unexpired lessons, 200 with counters
restored, 7 with `outcome_weight` (14 helpful / 1 harmful verdicts store
wide). Relabel was the full 700 Opus calls, $26.40: lesson ids hash the
extraction text, and none of the p2 pool's ids survive because that pool
came from a separate extraction run. Queries are p2's 100.

The same materialized pool was scored with the prior off
(`NAM_RECALL_OUTCOME_WEIGHT=0 NAM_RECALL_EVIDENCE_WEIGHT=0`,
`scores-prior-off.json`) and on (shipped 0.2 / 0.1, `scores.json`).
`p4-live-b` is the same pool re-exported after the A4 evidence backfill
(max evidence 114 → 4, ids identical, labels reused, 3 top-up calls),
scored with the prior on.

| run | D precision | D recall | D P@5 | E precision | E recall | E items/q | gate p50 |
|---|---|---|---|---|---|---|---|
| prior off | 22.3% | 17.7% | 27.8% | 36.7% | 2.9% | 0.98 | 3.3 s |
| prior on, pre-backfill evidence | 21.7% | 17.2% | 27.6% | 23.9% | 6.9% | 3.64 | 5.0 s |
| prior on, post-backfill evidence (b) | 22.3% | 17.7% | 27.8% | 25.1% | 6.2% | 3.11 | 5.0 s |

**D is the measurement.** With post-backfill evidence the prior reorders
nothing measurable: identical to prior off. On the inflated pre-backfill
evidence it cost 0.6 points of precision, i.e. the evidence term was
rewarding swarm reassertion, which is what A4 removed. The outcome term
has 7 lessons to act on and no visible effect at this sample size.

**E is gate noise, not the prior.** Production captures ran during all
three passes; the local gate's p50 was 3.3–5.0 s against a 6 s cap, and
E's items per query swung 0.98 → 3.64 on timeouts alone. Do not read E
across these passes.

Against p2-regress (D 13.8% / 26.9%): a different pool, not comparable.
The higher precision reflects the A1 raw-DeadEnd purge; the lower recall
reflects 12.6 relevant lessons per query in this pool versus p2's fewer.

MUD-407 done-when: counters populate from real sessions, and the ranker
reads them without a precision regression. Met, on a small sample. The
shipped 0.2 / 0.1 remain unmeasured as a choice; there is no signal to
tune them on until helpful/harmful verdicts reach the low hundreds.

# p4-live (MUD-435 S6, 2026-09-07)

First golden run on a pool exported from the live store with real
lifecycle counters, after five days of the MUD-407 outcome loop.

Pool: gradgraph-auth-platform lessons from the live store, unexpired,
exported after the A4 evidence backfill (max evidence 4), with the 43
lessons from query-session families dropped by step1b's holdout filter
(a prompt replayed against knowledge extracted from its own session is
not recall). 287 lessons, 284 with counters restored, 5 with
`outcome_weight`, 3 at evidence ≥ 3. Queries are p2's 100. Labels: the
full relabel cost 700 Opus calls, $26.40, on an earlier export of the
same lessons (lesson ids hash the extraction text, so none of p2's ids
carry over from its separate extraction run); this run reused them, 0
calls.

An earlier pair of runs on this pool was withdrawn: step1b never
restored counters for lessons without file anchors (the anchored write
returned no eid through an UNWIND over an empty list; 122 of 330
lessons, 5 of the 7 weighted), and the pool still held the 43 leaked
lessons. Both fixed on the MUD-435 branch (PR #26); these are the
numbers under the fixed harness.

The same materialized pool was scored with the prior off
(`NAM_RECALL_OUTCOME_WEIGHT=0 NAM_RECALL_EVIDENCE_WEIGHT=0`,
`scores-prior-off.json`) and on (shipped 0.2 / 0.1, `scores.json`).

| config | prior off | prior on |
|---|---|---|
| D precision | 20.7% | 20.8% |
| D recall | 20.0% | 20.1% |
| D P@5 | 25.8% | 25.6% |
| D relevant hits / 1,000 | 207 | 208 |
| E precision | 21.2% | 23.7% |
| E items / query | 5.79 | 5.11 |
| gate p50 | 6.0 s | 5.8 s |

**D is the measurement: the prior is neutral.** One relevant hit in a
thousand, with 5 weighted lessons and 3 at evidence ≥ 3 to act on. No
regression, no effect at this sample size.

**E is gate noise.** Production captures ran during both passes and the
local gate's p50 sat at the 6 s cap, so most E calls fell open to
ungated; E's numbers move with timeouts, not with the prior.

Against p2-regress (D 13.8% / 26.9%): a different pool, not comparable.

MUD-407 done-when (counters populate from real sessions; ranker reads
them without a precision regression): met, on a small sample. The
shipped 0.2 / 0.1 remain unmeasured as a choice until helpful/harmful
verdicts reach the low hundreds (15 store-wide today, about 3 a day).

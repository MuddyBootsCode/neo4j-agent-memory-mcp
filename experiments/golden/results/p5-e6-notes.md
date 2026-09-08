# p5-e6: files as a fused RRF leg instead of a boost (MUD-461, 2026-09-08)

The anchor was a +0.15 score boost that measured at zero effect (MUD-403).
This makes it a leg: the lessons anchored to the files being edited,
ranked by cosine among themselves, fused by RRF with the vector and BM25
legs, with at most `NAM_RECALL_ANCHOR_SLOTS` (2) of the returned slots
going to lessons *only* that leg found. Behind `NAM_RECALL_ANCHOR_LEG=1`,
default off.

Scored on the p4-live pool and labels, prior off, `GOLDEN_CONFIGS=D`.
cosine20 is unchanged by construction — the leg is a fusion input, not an
index. Crossed with E1's filename prefix, since both use the file signal.

```bash
cd experiments/golden
run () {  # name, prefix, leg
  GOLDEN_RUN=$1 GOLDEN_POOL_FROM=results/p4-live/pool.json GOLDEN_DB=goldenp5 \
    NAM_EMBED_CONTEXT_PREFIX="$2" uv run --no-sync --project ../.. python -u step1b_materialize.py
  GOLDEN_RUN=$1 GOLDEN_DB=goldenp5 GOLDEN_CONFIGS=D NAM_RECALL_ANCHOR_LEG=$3 \
    NAM_RECALL_OUTCOME_WEIGHT=0 NAM_RECALL_EVIDENCE_WEIGHT=0 \
    uv run --no-sync --project ../.. python -u step5_score.py
}
run p5-e6-base "" 0; run p5-e6-leg "" 1
run p5-e6-files files 0; run p5-e6-files-leg files 1
```

| index | anchor leg | D precision | D recall | D P@5 | coverage | relevant hits |
|---|---|---|---|---|---|---|
| bare | off | 20.7% | 20.0% | 25.8% | 71% | 207 |
| bare | **on** | 22.2% | 21.5% | 29.6% | 76% | 222 |
| files prefix | off | 21.5% | 20.8% | 24.8% | 73% | 215 |
| files prefix | **on** | **23.4%** | **22.7%** | 27.2% | 75% | **234** |

**Keep it.** MUD-461's rule was: keep only if D recall on the queries it
can fire on rises without a precision drop. Both rose, and the whole-set
numbers are the best D has measured in this project — against the shipped
p4-live D of 20.8% precision / 20.1% recall.

## The control holds exactly

D recall split by whether the query shares an edited file with any pool
lesson, which is the only place the leg can act:

| index | anchor leg | can fire (51) | cannot fire (49) |
|---|---|---|---|
| bare | off | 20.2% | 19.8% |
| bare | on | 22.8% | 19.8% |
| files prefix | off | 18.8% | 23.3% |
| files prefix | on | 22.1% | 23.3% |

The cannot-fire half is identical to the decimal with the leg on and off,
in both indexes. Nothing leaked. On the firing half the leg lifts recall
20.2% → 22.8%, precision 22.7% → 25.7%, and coverage 73% → 82%.

**The two halves are additive, which is the finding.** E1's prefix lifts
the queries with no file context and does nothing for the anchored ones
(it actually costs them, 20.2% → 18.8% here); the leg lifts the anchored
ones and provably does nothing for the rest. Together: 20.0% → 22.7% D
recall, 20.7% → 23.4% precision, +27 relevant hits per 1,000 injected.
One file signal, two disjoint query populations, and the boost that was
supposed to serve both served neither.

## Codex review, and why the numbers did not move

The adversarial review found the cap could discard a lesson the vector or
BM25 leg had independently retrieved: the same lesson is several nodes
with distinct eids, `dedupe_fused` may keep the anchor leg's copy of one
(a better outcome prior can outrank a two-leg twin), and the surviving
row's ranks then read anchor-only. Reproduced, fixed — the cap now spares
any lesson whose canonical text appeared in a non-anchor leg — and
covered by a regression.

Every number above is unchanged after the fix, to the decimal. The golden
pool cannot trigger it: `step2_pool` collapses duplicates by canonical
text, so step1b writes exactly one node per lesson and there is no twin to
split across legs. The defect is real in the live store, where the same
lesson captured in several sessions is several nodes, and that is where it
would have bitten.

The leg runs also carry `cosine20` now, and it comes back identical to the
same index without the leg (30.0% bare, 35.8% with the filename prefix) —
the "unchanged by construction" claim, measured.

Same artifact policy as p5-e1: `pool.json`, `queries.json`, `labels.json`
and `session_split.json` in each run directory are byte-identical copies
of p4-live's, so only what each run produced is committed.

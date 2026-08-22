# P1 capture run — 2026-08-22

Code: `0197000` (MUD-404 capture pipeline). Same 40 corpus sessions and
the same 100 queries as `baseline/`; the pool was rebuilt from scratch with
the new pipeline and relabeled (Claude Opus 5, 22,500 pairs, 513 relevant,
$17.33, zero missing verdicts). Retrieval code unchanged from baseline.

## What the capture change did to the pool

| | baseline | p1-capture |
|---|---|---|
| lessons (Decision / Gotcha / DeadEnd) | 178 (66 / 83 / 29) | 225 (92 / 121 / 12) |
| relevant (query, lesson) pairs | 421 (4.2 / query) | 513 (5.1 / query) |
| sessions with zero lessons | 13 of 40 at anchor rate 0 | 6 of 40 |
| lessons carrying a symptom | 0 | 133 (59%) |
| near-duplicate pairs (cosine ≥ 0.85, MiniLM) | 4 | 4 |
| session-local identifiers in text (regex) | 17 (10%) | 41 (18%) |
| avg lesson chars (embedded text) | 133 | 292 |
| capture time per session | one 80k call | 106 s avg, 78 windows over 40 sessions |

Curator over 1,174 candidates: WRITE 282, ALREADY_KNOWN 200, NOT_DURABLE
625, UNSUPPORTED 65, uncovered 2. Most ALREADY_KNOWN verdicts were repeats
across windows of one session; cross-session duplicates stayed at 4 pairs.
Raw error-step candidates were almost all rejected (DeadEnd 29 → 12), which
is the intended behaviour: they are offered, not written.

## Retrieval against the new pool (same code as baseline)

| config | precision | recall | P@5 | coverage | items/q | relevant retrieved |
|---|---|---|---|---|---|---|
| cosine20 | 10% (9%) | 39% (44%) | 16% (16%) | 57% (57%) | 20.0 | 201 (186) |
| D ungated | 14% (13%) | 28% (30%) | 19% (16%) | 51% (50%) | 10.0 | 142 (127) |
| E shipped gate | 27% (29%) | 18% (23%) | 32% (31%) | 43% (48%) | 3.3 | 90 (95) |

Baseline in parentheses. Latency p50 / p95 ms: embed 8 / 13, vector 14 /
19, gate 3,621 / 35,036 (one query hit BAML's 900 s timeout and fell open).

## Reading

F1. Capture did what it was meant to: more sessions produce lessons, 22%
more relevant material exists, and lessons now carry the symptom the next
prompt will contain. The absolute number of relevant lessons the retriever
reaches rose 8% (cosine top-20) and 12% (D top-10).

F2. Per-prompt rates did not move, and the gate kept slightly fewer (90 vs
95). The ranker is the floor: 61% of relevant lessons are outside cosine's
top 20, and P@5 is unchanged. Longer embedded text with symptoms did not
help MiniLM; this is the P3 work (bge-base, BM25 on symptom/path, fused
ranking), which this run now measures cleanly against.

F3. Gate latency is worse than the baseline on this pool (p95 35 s) and
one call timed out at 900 s. The per-prompt gate on a loaded local Ollama
is not inside any budget; P3 must replace or bound it.

F4. Session-local identifiers rose from 10% to 18% by the regex, mostly
because symptoms quote error text that contains numbers and names. That is
the point of the field; the prompt rule against SHAs and counts applies to
`text`, and the regex cannot tell the two apart. Spot-read of 9 random
lessons: symptoms are real error lines, texts are reusable instructions.

## Caveats

Same as baseline: one repo, one user, one model's labels, generous on
short prompts. Pool content differs between runs, so the recall denominators
are not the same set; compare the absolute counts and the rates together.

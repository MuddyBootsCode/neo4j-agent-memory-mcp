# p5-e4: HyDE rewrites fused as their own legs (MUD-459, 2026-09-08)

The stored text is answer-shaped and a prompt is a question. HyDE guesses
what a past session might have written down — a decision, a gotcha, a dead
end — and each guess is embedded and fused as **its own vector leg** beside
the prompt's, never concatenated into the query. That distinction is the
whole experiment: E3 measured what more text in the query string costs
(30.0% → 23.2% cosine20 recall), and a leg cannot do that damage, because a
bad guess costs a leg's worth of rank and leaves the prompt leg untouched.

`HypotheticalLessons` (BAML), `step3b_hyde.py` generates offline keyed by
query_id, `GOLDEN_HYDE` points step5 at the file, and
`retrieve_candidates(extra_embeddings=...)` is the shippable form.

| config | cos20 recall | cos20 P@5 | D precision | D recall | D P@5 | D coverage | relevant / 1,000 |
|---|---|---|---|---|---|---|---|
| baseline | 30.0% | 24.4% | 20.7% | 20.0% | 25.8% | 71% | 207 |
| E3 card (query concatenated) | 23.2% | 16.4% | 16.5% | 16.0% | 22.8% | 66% | 165 |
| E4 hyde alone | 32.9% | 27.8% | 23.0% | 22.3% | 30.4% | 76% | 230 |
| E2 + E6 | 34.7% | 30.4% | 24.7% | 23.9% | 31.4% | 79% | 247 |
| **E2 + E4 + E6** | **38.4%** | **33.0%** | **27.2%** | **26.3%** | **35.8%** | **80%** | **272** |

**The best result P5 has produced, and it clears the revised bar four
times over.** Against the baseline: D precision +6.5, D recall +6.3, D P@5
+10.0, coverage +9, and 272 relevant lessons per 1,000 injected against
207 — a third more. The ceiling moves 8.4 points, which even the retired
≥10 cosine20 bar nearly accepts.

**Fused, not concatenated, is the whole story.** Same query-side
information, the two shapes measured on the same harness: as a card inside
the query string, −6.8 points of ceiling; as legs beside it, +2.9 alone
and +3.7 on top of the index-side work.

## Rewrite quality decides it, and the gate model cannot supply it

The first generation ran on `qwen3-gate` (4B, constrained JSON), which is
what would run inside the hook. It echoed the prompt template back as
"lessons" on **54 of 100 queries**. Regenerated on `qwen-judge` (35B):
zero leakage, 3.0 rewrites per query, 2.1 s each. Both scored on the same
index:

| rewrites | cos20 recall | cos20 P@5 | D precision | D recall |
|---|---|---|---|---|
| qwen-judge (35B) | 38.4% | 33.0% | 27.2% | 26.3% |
| none | 34.7% | 30.4% | 24.7% | 23.9% |
| qwen3-gate (4B, 54% leaked) | 31.3% | 25.2% | 23.0% | 22.3% |

**Bad rewrites are worse than no rewrites** — the leaky set costs 3.4
points of ceiling against not doing it at all. So this ships only with a
model that can write them, which is MUD-435's R2 hosted-gate
recommendation restated with a number attached: 2.1 s per prompt on the
local 35B against the hook's 8 s budget, and only while the capture lane
is idle.

Latency: four vector legs instead of one takes the query's p50 from ~50 ms
to ~120 ms, which is noise next to the generation call.

## Provenance

The Codex review found that rewrites were keyed by query_id alone, on both
sides: `step3b_hyde` resumed on the id, and step5 attached the guesses by
the id, so a query regenerated under its own number would silently inherit
the guesses written for the prompt it replaced — an invalid comparison
that reports nothing. Each entry now carries the fingerprint of the query
it was generated from; step3b regenerates what no longer matches and step5
refuses to score a mismatch. Both committed files were migrated to that
shape against the queries they were actually generated from, and the run
above reproduces to the decimal afterwards.

Same artifact policy as the other P5 runs; `rewrites.json` and
`rewrites-gate.json` live once, under `p5-e4/`.

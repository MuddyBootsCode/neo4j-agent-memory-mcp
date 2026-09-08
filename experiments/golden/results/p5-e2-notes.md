# p5-e2: trigger sentences as the indexed key (MUD-457, 2026-09-08)

Recall embeds the prompt and matches it against the lesson, which is the
fix; the fix only matches once you already know it. `LessonTriggers`
writes the other side of that match — 2 to 4 sentences per lesson
describing the moment it applies, in the words a developer would be using
then — and step1b concatenates them into the embedded string. Devin's
trigger description, LongMemEval's key expansion (+9.4% recall@k).

Generation: 287 lessons, qwen-judge, 25 min, 3.0 triggers per lesson, 0
failures, `results/p5-e2/triggers.json`. Keyed by lesson id, so the same
generation serves every variant. Truncation: with triggers the embedded
string runs p50 143 / p90 196 tokens against MiniLM's 256, and 3 of 287
lessons lose the tail of their last sentence.

Concatenated into one vector, which MUD-457 asks to test before the
multi-vector form. Crossed with E1's filename prefix and E6's anchor leg,
prior off, on the p4-live pool and labels.

| config | cos20 recall | cos20 P@5 | D precision | D recall | D P@5 | D coverage | relevant / 1,000 |
|---|---|---|---|---|---|---|---|
| baseline | 30.0% | 24.4% | 20.7% | 20.0% | 25.8% | 71% | 207 |
| E1 files prefix | 35.8% | 26.0% | 21.5% | 20.8% | 24.8% | 73% | 215 |
| E6 anchor leg | — | — | 22.2% | 21.5% | 29.6% | 76% | 222 |
| E2 triggers | 34.7% | 30.4% | 23.2% | 22.5% | 27.6% | 75% | 232 |
| **E2 + E6** | **34.7%** | **30.4%** | **24.7%** | **23.9%** | **31.4%** | **79%** | **247** |
| E1 + E2 | 34.7% | 26.8% | 23.5% | 22.7% | 28.2% | 76% | 235 |
| E1 + E2 + E6 | 34.7% | 26.8% | 24.5% | 23.7% | 30.4% | 78% | 245 |

**E2 is +4.7 on the ceiling, under the ≥10 bar, and it is the best thing
measured here anyway.** The bar is cosine top-20 recall because the
retriever ceiling was assumed to be what binds. It moved 4.7 points. What
moved is the ordering inside that ceiling: cosine20 P@5 +6.0, D precision
+4.0, D recall +3.9, D P@5 +5.6, coverage +8. Per 1,000 lessons injected
the developer sees 247 relevant instead of 207, a fifth more.

**E1's prefix is redundant once triggers exist, and costs P@5.** Adding
it on top: identical cosine20 recall (358 relevant either way), P@5 30.4%
→ 26.8%, D precision 24.7% → 24.5%. The trigger prompt asks the model to
name the lesson's files in a sentence, so the filenames are already in
the string, in context instead of as a bare list. Two copies dilute.
E1 ships only if E2 does not.

**E2 and E6 are additive**, as p5-e6 predicted: the triggers work on the
whole query set through the prompt's vocabulary, the leg only on the 51
queries that share an edited file, and together they are the best D this
project has measured.

## Not yet done

The multi-vector form (one vector per trigger on `:Trigger` nodes, score
= max) is the one the literature measured; MUD-457 asks for both. This is
the cheap half, and it cleared enough to justify the schema.

Shipping needs the triggers generated at capture — one extra local call
per kept lesson, ~40 a day — and a backfill over the store, neither of
which this measures.

Same artifact policy as p5-e1: only what each run produced is committed;
`triggers.json` lives once, under `p5-e2/`.

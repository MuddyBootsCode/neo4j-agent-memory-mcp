# Golden baseline — 2026-08-22

Code: `cdaae17` (production recall as merged in PR #14). Pool: the MUD-401
`sweepcorpus` build, 40 `gradgraph-auth-platform` sessions, 178 lessons
(Decision 66, Gotcha 83, DeadEnd 29), embedded with `all-MiniLM-L6-v2`.
Queries: 100 real prompts from the 38 later sessions, 39 anchorable / 61
not, 59 short (< 15 words) / 41 long. Labels: Claude Opus 5, effort medium,
17,800 pairs, 421 relevant (2.4%, 4.2 per query). Cost $13.61 over 405 calls,
397/400 first-pass calls served from cache. Gate: `qwen-judge` via Ollama.

## Results

| config | precision | recall (full pool) | P@5 | coverage | items/query |
|---|---|---|---|---|---|
| cosine20 (retriever ceiling) | 9% | 44% | 16% | 57% | 20.0 |
| D — shipped ranking, ungated top 10 | 13% | 30% | 16% | 50% | 10.0 |
| E — D + gate (what the hook injects) | 29% | 23% | 31% | 48% | 3.3 |

By stratum (precision / recall):

| | D | E |
|---|---|---|
| anchorable (39) | 15% / 32% | 38% / 26% |
| not anchorable (61) | 11% / 29% | 23% / 20% |
| short (59) | 11% / 33% | 25% / 26% |
| long (41) | 15% / 27% | 35% / 20% |

Latency (p50 / p95 ms, measured in-process): embed 7 / 11, vector 13 / 15,
gate 3,482 / 16,473.

## Findings

F1. The self-graded 69% was the judge agreeing with itself. Against
independent labels the shipped path is 29% precision at 3.3 items per
prompt, and it drops recall from 30% to 23%.

F2. Retrieval is the floor. 56% of relevant lessons are not in cosine's top
20 at all; no gate or reranker downstream can reach them. Ranking inside the
top 20 is also weak: P@5 is 16% for both cosine and D, so the anchor boost
adds nothing measurable.

F3. The gate's latency is not inside the hook budget. p50 3.5 s, p95 16.5 s
against `NAM_HOOK_TIMEOUT=8`. On the slow tail the hook times out and falls
back silently, which looks like an empty recall rather than a slow one.

F4. Short prompts are most of the workload (59/100 after a 4-word floor) and
score worst. A prompt like "Keep the lineage doc" carries no retrievable
content; anything useful has to come from the session's files and task.

## Hand spot-check (20 queries, every 5th)

Reviewed the Opus-relevant lessons for q0, q5, …, q95. Agreement at the
query level: 13/17 queries with positives looked right (q50, q55, q60, q85
are good examples: deployment-order, tool-description refresh, stacked-PR
merge state). Generous cases: q20 ("Not fine. We need to tune this") got six
cip4/metro lessons that are topical, not on point; q65 ("It's not a benign
behavior, it's just gone") got two KeyError gotchas; q90 ("Keep the lineage
doc") got migration-mount lessons. Roughly 30% of positive pairs on short
follow-up prompts are inferred from the file context rather than the prompt.
Three queries are the identical "Read ./EPIC_BRIEF.md … Follow it exactly."
(q25, q80, q95) and correctly have no positives. No false negatives were
checkable in this pass. Treat recall numbers as relative between runs, not
absolute.

## Caveats

- One repo, one user, one extraction run. Lesson text is what the MUD-401
  extractor produced; lessons it never extracted are invisible to every
  config.
- Labels are one model's judgement at medium effort.
- Gate latency was measured with back-to-back calls on a loaded Ollama; the
  p50 is higher than the 1.5 s recorded in MUD-402.

# Recall-precision sweep: coding-memory pivot

Measures whether anchor-first recall beats plain embedding search, and
whether a judge gate helps — over real Claude Code session transcripts as
the corpus.

Two runs so far. MUD-395 could not answer the question (3-transcript corpus,
zero file overlap, config B returned nothing). MUD-401 reran it on a corpus
with real file continuity and got an answer.

## Headline (MUD-401, gradgraph-auth-platform + worktrees)

Anchorable queries — the stratum chosen to give anchor-first its best case:

| config | precision | items/query | coverage | zero-injection |
|---|---|---|---|---|
| A. cosine-only | **52%** (78/150) | 5.00 | 97% | 0% |
| B. anchor-first | **12%** (17/147) | 4.90 | 33% | 0% |
| C. anchor-first + judge gate | **24%** (15/63) | 2.10 | 33% | 17% |

**Anchor-first is four times worse than cosine-only on precision, on the
queries selected to favour it.** The judge gate doubles B's precision by
discarding 57% of its items, and still lands less than half of A.

Config B is production's `_MEMORIES_QUERY`, imported directly rather than
copied — this is what `coding_recall` returns today.

Unanchorable queries (the other half of the sample):

| config | precision | items/query | coverage | zero-injection |
|---|---|---|---|---|
| A. cosine-only | 50% (73/147) | 5.00 | 87% | 0% |
| B. anchor-first | n/a — 0 candidates | 0.00 | 0% | 100% |
| C. anchor-first + judge | n/a — inherits B | 0.00 | 0% | 100% |

A holds 50% precision where B returns nothing at all. Cosine-only degrades
gracefully; anchor-first degrades to silence.

## What this says about the design

1. **The anchor is the wrong retrieval key.** Sharing a file with a past
   session says the two are *near* each other, not that the past lesson
   answers the present question. 88% of what B returned was judged
   off-point.
2. **The prompt is the signal, and `coding_recall` throws it away.** Its
   own docstring says `prompt` is "accepted for future relevance ranking
   but unused in v1". Config A does nothing but use it, and wins fourfold.
3. **Anchor-first cannot fire most of the time.** Only 30 of 146 candidate
   prompts (21%) had a touched file the corpus had also seen. Most prompts
   arrive before anything has been edited in that session, so there is
   nothing to anchor to yet. Raising precision would not fix coverage.
4. **A judge gate is a real improvement over a threshold**, doubling
   precision at half the volume — but it is a filter, not a retriever. It
   cannot rescue candidates the anchor never surfaced.

The cheapest change consistent with all four: rank by prompt similarity and
use the anchor as a boost, not as the gate.

## Corpus

`gradgraph-auth-platform`, 95 sessions (main checkout plus every linked
worktree's project directory — worktrees share one repo identity per
`git_sweep.repo_name`). Chronological 60/40 split, corpus capped at the 40
**newest** corpus-side sessions, leaving 38 query sessions.

- 40 sessions processed, 0 failed.
- Extracted by kind: Gotcha 83, Decision 67, CodingPreference 67, DeadEnd 28.
- `dropped_unanchored`: 89 (27% of everything extracted).
- 13 of 40 sessions scored `anchor_rate` 0.0 — they edited no files, so every
  lesson they produced was discarded by design.
- 2,581 messages stored with local embeddings, 0 failures.

## Query set

60 queries from 146 candidate prompts (real human-typed turns, >= 8 words),
stratified: all 30 anchorable, plus 30 sampled evenly from the rest. The
stratum is recorded per query and the report breaks the two out — averaging
a deliberately biased sample estimates nothing.

## Caveats

- **Only the file anchor was exercised.** The corpus is built with
  `task_key=None`, so no `WorkTask` nodes exist in `sweepcorpus`. Config B's
  query has two anchor paths, `ABOUT->CodeFile` and `CONCERNS->WorkTask`;
  this run tests the first. Production has both, so B's real hit rate is
  somewhat higher than measured here — though the live graph anchors tasks
  by `repo/branch`, which on `main` collapses every session into one bucket.
- **Grader and gate share a model.** Config C's keep/drop gate and the
  independent grading pass both run on `qwen-judge`. Differently-worded
  prompts remove wording bias, not model bias, so C's 24% is upward-biased
  relative to a truly independent judge. Unlike the MUD-395 run, the gate
  actually fired here (63 items kept from 147), so this bias is live.
- **Stratified sample.** The "ALL QUERIES" table over-samples anchorable
  queries 30/60 against a true rate of 21%. Read the strata, not the total.
- **Corpus from one project**, extracted by a local 35B model with
  384-dimensional local embeddings. No claim of generalization to other
  codebases, working styles, or extraction models.
- **Judge compliance was good**: 3 of 447 graded item-slots (0.7%) came back
  without a verdict and were excluded from precision's denominator. Zero
  config errors, zero grading errors.

## Prior run (MUD-395)

3 transcripts total, split 2 corpus / 1 query, zero file overlap between
them. Config A scored 52% precision (29/56); configs B and C returned zero
candidates on all 12 queries and could not be scored. The run showed only
that cosine-only degrades gracefully at tiny corpus sizes. Its `results.json`
is in git history at the MUD-401 baseline commit.

Notably, A's precision barely moved between the runs — 52% on 56 graded
pairs then, 51% on 297 now (151/297 across both strata).

## Method

1. **Split** (`step1_split.py`). Chronological 60/40 over all sessions for the
   corpus repo. `SWEEP_REPO_ROOT` selects the repo, `SWEEP_INCLUDE_WORKTREES=1`
   folds in worktree project dirs, `SWEEP_MAX_CORPUS_SESSIONS` caps the corpus
   side (newest kept).
2. **Corpus build** (`step2_corpus.py`). Per session: render the transcript
   tail (80k cap, the capture convention), reconstruct touched files from
   `Edit`/`Write` `tool_use` entries, run `extract_coding_memory` on
   `qwen-judge`, write kept items into the isolated `sweepcorpus` database.
   Also stores every real turn as a `Message` with a local embedding so
   config A has something to search.
3. **Queries** (`step3_queries.py`). Real human prompts only; `SWEEP_STRATIFY=1`
   keeps every anchorable query and fills the rest of the budget evenly.
4. **Configs** (`step4_configs.py`), each capped at 5 items.
   - **A** — `search_messages` + `search_preferences`, threshold 0.5, merged
     by similarity.
   - **B** — production's `_MEMORIES_QUERY`, imported from
     `src/agent_memory_mcp/mcp/_coding_tools.py`.
   - **C** — every config-B item screened by `qwen-judge`.
5. **Grading** (`step5_grade.py`). Every injected pair graded by `qwen-judge`
   under an audit framing worded independently of the gate, explicitly
   warning against crediting shared vocabulary over genuine relevance.
6. **Report** (`step6_report.py`). Aggregates `results.json`, prints the
   overall table plus one per stratum.

## Reproduce

```bash
cd experiments/recall_sweep
export SWEEP_REPO_ROOT=/path/to/corpus-repo SWEEP_INCLUDE_WORKTREES=1 \
       SWEEP_STRATIFY=1 SWEEP_MAX_QUERIES=60 \
       NAM_LLM_PROVIDER=ollama NAM_EMBEDDING_PROVIDER=sentence_transformers \
       BAML_LOG=warn
for s in step1_split step2_corpus step3_queries step4_configs step5_grade step6_report; do
    uv run --no-sync --project ../.. python -u $s.py || break
done
```

Needs Neo4j and Ollama running. `sweepcorpus` is recreated at the start of
step 2; nothing is left in the live `neo4j` database.

## Artifacts

- `results.json` — per-query records: prompt, stratum, reconstructed files,
  every injected item per config with its judge verdict and one-line reason
- `results/*.json` — per-step intermediates (`session_split`, `corpus_stats`,
  `queries`, `candidates`, `grades`, `config_errors`, `grade_errors`)

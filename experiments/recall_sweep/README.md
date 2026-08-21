# Recall-precision sweep: coding-memory pivot

Measures whether anchor-first recall beats plain embedding search, and
whether a judge gate helps — over real Claude Code session transcripts as
the corpus.

Two runs so far. MUD-395 could not answer the question (3-transcript corpus,
zero file overlap, config B returned nothing). MUD-401 reran it on a corpus
with real file continuity and got an answer.

## Headline (MUD-401, gradgraph-auth-platform + worktrees)

Four configs. **B is what production shipped; D is what it does now.**

Anchorable queries — the stratum chosen to give anchor-first its best case:

| config | precision | items/query | coverage | zero-injection |
|---|---|---|---|---|
| A. cosine over messages | **51%** (77/150) | 5.00 | 93% | 0% |
| B. anchor-first (v1) | **13%** (19/147) | 4.90 | 27% | 0% |
| C. anchor-first + judge gate | 25% (18/73) | 2.43 | 27% | 7% |
| D. hybrid: similarity + anchor boost | **25%** (38/150) | 5.00 | 67% | 0% |

Unanchorable queries — the other half of the sample:

| config | precision | items/query | coverage | zero-injection |
|---|---|---|---|---|
| A. cosine over messages | 41% (61/150) | 5.00 | 77% | 0% |
| B / C | n/a — 0 candidates | 0.00 | 0% | 100% |
| D. hybrid | **33%** (49/149) | 5.00 | 63% | 0% |

**The fix works.** Against the v1 read it replaces, on identical data graded
in the same pass by the same judge: precision roughly doubles (13% -> 25%),
coverage goes from 27% to 67%, and the 100% zero-injection rate on
unanchorable queries goes to 0% at 33% precision. Anchor-first answered half
the queries; the hybrid answers all of them.

**It does not reach config A**, and that comparison is not like-for-like: A
searches 2,581 messages and 68 preferences, D searches 178 lessons. A is
fourteen times the corpus. A also returns raw conversation turns rather than
distilled lessons, so it is a different product, not a drop-in replacement --
but the gap is large enough to be worth a follow-up: the recall hook
currently reaches the message plane only when coding_recall declares
fallback, and on this evidence it should reach it every time.

## What this says about the design

1. **The anchor is the wrong retrieval key.** Sharing a file with a past
   session says the two are *near* each other, not that the past lesson
   answers the present question. 87% of what B returned was judged
   off-point.
2. **The prompt is the signal, and v1 `coding_recall` threw it away.** Its
   docstring said `prompt` was "accepted for future relevance ranking but
   unused in v1". Using it doubles precision (config D) and using it over a
   larger corpus quadruples it (config A).
3. **Anchor-first cannot fire most of the time.** Only 30 of 146 candidate
   prompts (21%) had a touched file the corpus had also seen. Most prompts
   arrive before anything has been edited in that session, so there is
   nothing to anchor to yet. Raising precision would not fix coverage.
4. **A judge gate is a real improvement over a threshold**, doubling
   precision at half the volume — but it is a filter, not a retriever. It
   cannot rescue candidates the anchor never surfaced.

The cheapest change consistent with all four -- shipped as config D and now
production's read: rank by prompt similarity and use the anchor as a boost,
not as the gate.

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
- **Run-to-run variance is real.** The first MUD-401 run scored A at 52%/50%
  across the two strata; this one, after a corpus rebuild, scores 51%/41%.
  Extraction runs on a local model and grading is a fresh LLM pass, so treat
  single-run gaps under ~10 points as noise. The B-vs-D comparison is not
  exposed to this: both were retrieved from the same corpus and graded in the
  same pass by the same judge.
- **Corpus from one project**, extracted by a local 35B model with
  384-dimensional local embeddings. No claim of generalization to other
  codebases, working styles, or extraction models.
- **Judge compliance was good**: 1 of 596 graded item-slots (0.7%) came back
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
   - **D** — production's `_HYBRID_QUERY` with its real constants
     (`HYBRID_THRESHOLD`, `ANCHOR_BOOST`, `HYBRID_CANDIDATES`), also
     imported rather than copied.
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

# Golden recall set (MUD-403)

A fixed measurement for coding-memory recall. Every phase of the memory plan
(MUD-404 to MUD-407) is scored against it before it merges.

What it fixes about the earlier sweeps: the labeler is not the pipeline's
model, recall is over the whole lesson pool rather than the retrieved top-k,
prompts include the short ones the hook actually sees, and every artifact is
committed under `results/<run>/`.

## Pipeline

| step | writes | what |
|---|---|---|
| `step1_corpus.py` | `session_split.json`, `corpus_stats.json` | 60/40 chronological split per repo; production capture over the earlier sessions into the scratch DB. `GOLDEN_REUSE_DB` skips extraction. |
| `step1b_materialize.py` | `pool.json`, copies of queries/labels | Rebuilds the scratch DB from a committed `pool.json` under the current embedder, no LLM. Retrieval experiments reuse the labels. `GOLDEN_POOL_FROM`. |
| `step2_pool.py` | `pool.json` | Lessons exported from the DB. Id = sha1(repo, kind, embedding text), so a rebuild that reproduces a lesson keeps its labels. |
| `step3_queries.py` | `queries.json` | 100 real human prompts from the query sessions, 25 per cell of anchorable × short, each with the files edited before it. |
| `step4_label.py` | `labels.json`, `label_usage.json` | Claude Opus 5 labels every (query, lesson) pair in the same repo. Rubric + 50 sorted lessons cached in `system`, query in `messages`, structured output. Chunk-major so one cache prefix stays hot. Stops if the second call on a chunk shows no cache read. Resumable. |
| `step5_score.py` | `scores.json` | cosine20 (retriever ceiling), D (production ranking, ungated), E (D + production gate). Precision, recall over the full pool, P@5, coverage, items/query; embed/vector/gate latency p50/p95. |
| `step6_teardown.py` | | Drops the scratch DB. Evidence is the results directory, never the store. |

```bash
cd experiments/golden
GOLDEN_RUN=baseline GOLDEN_REPOS=~/Projects/gradgraph-auth-platform ./run.sh
```

Env: `GOLDEN_RUN` (results subdir), `GOLDEN_REPOS` (comma-separated checkouts),
`GOLDEN_DB`, `GOLDEN_REUSE_DB` / `GOLDEN_REUSE_SPLIT`, `GOLDEN_MAX_QUERIES`,
`GOLDEN_LABEL_MODEL`, `GOLDEN_LABEL_CHUNK`, `GOLDEN_KEEP_DB`. `.env` supplies
`ANTHROPIC_API_KEY` for the labeler; `NAM_LLM_PROVIDER=ollama` keeps every
BAML call local.

## Re-scoring a change

Retrieval changes (MUD-406): `GOLDEN_POOL_FROM=results/<run>/pool.json
step1b_materialize.py`, then `step5_score.py` (`NAM_EMBEDDING_MODEL`,
`NAM_EMBEDDING_DIMENSIONS`, `GOLDEN_QUERY_CONTEXT` to prepend prior turns),
then `step6_teardown.py`, under a new `GOLDEN_RUN` name. Capture changes (MUD-404): rebuild the pool with `step1`,
relabel with `step4` (only lessons whose text changed cost new calls), then
score. Compare `scores.json` across runs.

## Caveats

- One user's transcripts, one repo per run so far.
- Labels are one model's judgement, not ground truth. Spot-check recorded in the run's `notes.md`.
- `step5` measures recall within the labeled pool; lessons the extractor never produced are invisible to every config.

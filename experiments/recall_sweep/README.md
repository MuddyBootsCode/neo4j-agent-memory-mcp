# Recall-precision sweep: coding-memory pivot (MUD-395)

Measures whether anchor-first recall beats plain embedding search, and
whether a judge gate helps — over this project's own Claude Code session
transcripts as the corpus.

## Headline

| config | precision | items/query | coverage | zero-injection rate |
|---|---|---|---|---|
| A. cosine-only | **52%** (29/56 gradeable; 4 excluded) | 5.00 | 100% (12/12) | 0% |
| B. anchor-first | **n/a** — 0 candidates on every query | 0.00 | 0% | 100% (12/12) |
| C. anchor-first + judge gate | **n/a** — inherits B's 0 candidates | 0.00 | 0% | 100% (12/12) |

At this corpus size, cosine-only is the only config that returns anything.
Anchor-first never fires because the 2 corpus sessions and the 1 query
session touched disjoint files — see caveat 1.

## Biggest caveat: sample size and file disjointness

This project has exactly **3** Claude Code session transcripts total, not
the 40 the design budgets for. Chronological 60/40 split gives 2 corpus
sessions and 1 query session (12 usable query prompts). Per the task's
explicit fallback ("run the sweep anyway... flag the sample size
prominently"), that's what ran — nothing here is padded or fabricated.

The two corpus sessions touched:

- `src/agent_memory_mcp/hook/recall_hook.py`, `tests/test_recall_hook.py`,
  `pyproject.toml`, `README.md` (session `a3998677`, MUD-394-era recall-hook
  work)
- *(no files — a planning/discussion session with no Edit/Write calls)*
  (session `dc34fb22`)

The query session touched:

- `tests/integration/conftest.py`, `src/agent_memory_mcp/providers.py`,
  `src/agent_memory_mcp/capture/git_sweep.py`, `tests/test_git_sweep.py`,
  `docker-compose.yml`, `src/agent_memory_mcp/mcp/server.py` (MUD-395
  coding-memory-pivot work)

Zero overlap. Config B's anchor-first query (`_MEMORIES_QUERY`, matched via
`(m)-[:ABOUT]->(f:CodeFile)` on `f.path IN $files`) has nothing to anchor to
for any of the 12 queries, so it returns empty every time; config C (which
only filters B's output) inherits that. This is a genuine property of
anchor-first at small corpus sizes — it depends on file-level continuity
across sessions, and with 2 sessions covering unrelated tickets there is
none — not a bug in the query or the sweep. It does mean this run cannot
answer "does anchor-first beat cosine-only" from data: it can only show
that cosine-only degrades gracefully to non-trivial results (52% precision)
while anchor-first degrades to nothing, in a corpus this thin. Re-running
once more coding sessions accumulate (or seeding the corpus from a
longer-lived project) is the natural next step.

## Method

1. **Corpus/query split.** Chronological, 60/40, over the 3 session
   transcripts under
   `~/.claude/projects/-Users-muddybootscode-Projects-neo4j-agent-memory-mcp/*.jsonl`
   (the `memory/` subdirectory excluded, per instructions).
2. **Corpus build** (`step2_corpus.py`). For each corpus session: rendered
   the transcript tail (`extract_transcript_text`, 80k-char cap — the
   capture convention), reconstructed touched files from `Edit`/`Write`
   `tool_use` entries, ran `extract_coding_memory(transcript, branch="main",
   task=None, files=...)` routed to `qwen-judge` via
   `NAM_LLM_PROVIDER=ollama`, and wrote kept items (`session_upsert` then
   `anchored_memory_write` per item) into the isolated `sweepcorpus`
   database via the connected `MemoryClient`'s own graph session. Also
   stored every real user/assistant turn in the *whole* session (not just
   the 80k tail) as a `Message` with a local `sentence-transformers`
   embedding, and persisted extracted `CodingPreference`s through
   `persist_preferences`, so config A has a corpus to search.
   - **Touched-file path convention.** `tool_input.file_path` is absolute;
     reconstructing a repo-relative `CodeFile.path` from it needs to match
     what production's `edited_files()` would produce (`git status
     --porcelain -C <cwd>`, paths relative to *cwd*). When a session ran
     inside a linked git worktree (`.worktrees/<name>/...`), the path is
     normalized relative to that worktree, not the main checkout — this
     mattered for the query session, which ran inside
     `.worktrees/coding-memory-pivot/`.
3. **Query set** (`step3_queries.py`). From the one query session: real
   human-typed prompts, filtering `isMeta` entries, tool-result turns,
   injected synthetic content (`<task-notification>`, `<system-reminder>`,
   `<command-message>`, etc.), and slash commands, kept at >= 8 words. All
   12 candidate prompts found were kept (well under the 30 cap). File
   context per query: files touched in that session strictly *before* the
   prompt's line index — straightforward here since the transcript is
   already chronological, so this is the "ordering is easy" branch, not the
   whole-session fallback.
4. **Configs** (`step4_configs.py`), each capped at 5 items:
   - **A. cosine-only** — `client.short_term.search_messages` +
     `client.long_term.search_preferences` against `sweepcorpus`, threshold
     0.5, merged and sorted by similarity.
   - **B. anchor-first** — the production `_MEMORIES_QUERY` from
     `src/agent_memory_mcp/mcp/_coding_tools.py`, imported directly (not
     copied), run with `{repo: "neo4j-agent-memory-mcp", files: <query's
     reconstructed files>, task_key: None}`.
   - **C. anchor-first + judge gate** — every config-B item screened by
     `qwen-judge` with a terse keep/drop prompt ("is this item useful
     context for answering the query").
5. **Grading** (`step5_grade.py`). Every `(query, item)` pair injected by A
   or B is graded by `qwen-judge` with a prompt worded independently of the
   gate — an "audit" framing that explicitly warns against crediting shared
   vocabulary over genuine relevance. C's items are a subset of B's, so they
   reuse B's grades rather than triggering a third pass.
6. **Report** (`step6_report.py`). Aggregates `results.json` and this table.

## Corpus stats

- 2 corpus sessions processed, 0 failed.
- Extracted by kind: Decision 2, Gotcha 6, DeadEnd 3, CodingPreference 6.
- `dropped_unanchored`: 16 total (1 from session `a3998677`, 15 from
  `dc34fb22` — that session touched no files at all via Edit/Write, so
  every extracted Decision/Gotcha/DeadEnd there had nothing to anchor to
  and was dropped by design).
- `anchor_rate`: 0.92 (`a3998677`), 0.0 (`dc34fb22`).
- 227 messages stored with local embeddings, 0 failures.
- 6 preferences persisted.

Full per-session detail: `results/corpus_stats.json`.

## Other caveats

- **Grader/gate share a model.** Config C's keep/drop gate and the
  independent grading pass both run on `qwen-judge`. Differently-worded
  prompts remove *wording* bias but not *model* bias — the same model's
  judgment underlies both, so C's precision would be upward-biased relative
  to a truly independent judge. This run could not actually exercise that
  bias (0 gate calls fired — B was always empty), so it is a live caveat
  for the *next* run with file overlap, not a number that needed flagging
  here.
- **Reconstructed file context.** Touched-files come from `Edit`/`Write`
  `tool_use` entries in the transcript, not from git. A session that only
  reads files (no edits) shows 0 touched files even if it discussed those
  files at length — this happened for corpus session `dc34fb22` and for 7
  of the 12 queries (the early, still-discussing part of the query
  session).
- **Corpus from one project.** Everything is dogfooded from this repo's own
  Claude Code sessions. No claim of generalization to other codebases,
  working styles, or corpus sizes.
- **Judge compliance.** 4 of 60 graded item-slots (6.7%) came back with no
  verdict for that item id from `qwen-judge` and were excluded from
  precision's denominator — not counted as either relevant or irrelevant.
  Zero grading calls failed outright (0 entries in `grade_errors.json`) and
  zero config calls failed (0 entries in `config_errors.json`).

## Artifacts

- `experiments/recall_sweep/README.md` — this file
- `experiments/recall_sweep/results.json` — raw per-query records: prompt,
  reconstructed files, and every injected item per config with its judge
  verdict and one-line reason
- `experiments/recall_sweep/results/*.json` — intermediate per-step
  artifacts (`session_split.json`, `corpus_stats.json`, `queries.json`,
  `candidates.json`, `grades.json`, `config_errors.json`,
  `grade_errors.json`)
- `experiments/recall_sweep/*.py` — rerunnable pipeline: `step1_split.py` →
  `step2_corpus.py` → `step3_queries.py` → `step4_configs.py` →
  `step5_grade.py` → `step6_report.py`; shared helpers in `lib.py`, `mem.py`,
  `judge.py`

`sweepcorpus` is dropped at the end of the run — nothing from this sweep is
left in the live `neo4j` database.

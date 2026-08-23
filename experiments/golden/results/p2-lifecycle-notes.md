# P2 lifecycle runs — 2026-08-22

Code `14ff3a3` (MUD-405). Two checks: a retrieval regression on the P1 pool
and labels (the recall path now filters expired lessons and records what it
served), and a capture smoke over real sessions to see the accumulation
paths fire.

## Retrieval regression (`p2-regress/`, MiniLM + BM25 fused, P1 pool and labels)

| config | precision | recall | P@5 | coverage | items/q |
|---|---|---|---|---|---|
| cosine20 | 10% | 39% | 16% | 57% | 20.0 |
| D | 14% | 27% | 19% | 48% | 10.0 |
| E | 24% | 17% | 28% | 42% | 3.6 |

Identical to p3-bm25 for cosine20 and D; E within noise (27% → 24% on 360
vs 318 injected, the gate's own run-to-run variance). Gate p50 / p95 2.7 s /
6.0 s under the cap. No regression.

## Capture smoke (`p2-smoke/`, 5 newest `gradgraph-auth-platform` sessions, scratch DB)

Curator over 235 candidates: WRITE 53, ALREADY_KNOWN 23, SUPERSEDES 2,
NOT_DURABLE 96, UNSUPPORTED 61. Stored 49 lessons (Decision 17, Gotcha 27,
DeadEnd 5); preferences went to the upstream store (CodingPreference 0).

Lifecycle counts after the run: 13 of 49 lessons reasserted at least once
(`evidence_count` up to 5), 17 `REASSERTED_IN` edges, 2 lessons expired
with a `SUPERSEDES` edge from their replacement, 0 zero-lesson sessions.
Example at evidence 4: "Chose to use `scripts.lib.tool_catalog` at runtime
for the public tool list instead of the frozen `ALL_TOOLS` constant …".

Constraints verified on the live database after the container rebuild:
`code_agent_id`, `coding_session_id`, `work_task_key`, `change_sha`,
`code_file_repo_path` (composite), all UNIQUENESS.

## What is not measured here

- `RESOLVED_BY` needs a served lesson and a later commit in the same
  session; the smoke has no recall calls, so the count is 0 by
  construction. The builder and tool path are unit-tested.
- Ranking does not read `evidence_count` / `served_count` / `helpful` yet;
  that is the remaining MUD-406 item, now that the signals exist.
- The "new, old" sample query in the log printed no rows despite 2
  `SUPERSEDES` edges; the edge count is the evidence, the sample query's
  output is unexplained and noted here rather than chased.

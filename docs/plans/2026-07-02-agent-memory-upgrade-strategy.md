# Agent Memory — Upgrade Strategy & Reasoned Assessment

**Date:** 2026-07-02
**Companion to:** [`docs/reviews/2026-07-02-agent-memory-deep-dive.md`](../reviews/2026-07-02-agent-memory-deep-dive.md) (the detailed, file:line findings)
**Purpose:** (1) a consolidated, prioritized recommendation list; (2) a multi-agent implementation strategy for executing it; (3) a candid opinion on whether this is a good memory system and what I would change.

---

## Part 1 — Is this a good memory system? (Reasoned opinion)

Short version: **the design instincts are good and above the median for this class of project, but it is not yet a trustworthy memory system, because several of its headline primitives are verifiably broken end-to-end and the test strategy is structurally unable to notice.** The gap is not vision — it's the seam between the code and the database.

### What's genuinely good

- **The ambition is right.** A memory system for agents *should* do entity/relationship extraction, temporal fact evolution, and relevance-ranked retrieval. This project attempts all three rather than being a glorified vector store, and it has a real ontology (POLE+O plus per-vertical schemas).
- **It measures quality.** The golden extraction dataset with precision/recall gates, the tiered round-trip retrieval test with a negative-query floor, and the 25-case routing eval are things most memory projects never build. Whoever wrote these understood that "the LLM extracted something" is not the same as "the extraction is correct."
- **The operational posture is mature.** Graceful-degradation ladders, Docker lifecycle discipline, single-flight routing cache, structured logging. This reads like someone who has run software in production.

### Why I don't yet trust it as *memory*

Memory has one job: **store a fact and give it back correctly later.** On that axis, the current implementation has verified failures in its two most differentiated features:

1. **Temporal facts don't round-trip.** Facts are written with ISO-string `valid_from`/`valid_until` but queried with integer epoch-millis comparisons, and `STRING <= INTEGER` is `null` in Cypher — so `temporal_query` silently excludes exactly the facts that have temporal data. The "what was true on date X" feature returns the wrong answer, and `memory_search` even labels future-dated facts as "expired." The flagship capability is, today, anti-functional.
2. **Cross-database references are dead on arrival.** The `HAS_REFERENCE` edge that cross-DB resolution matches on is never created anywhere, so every vertical write leaks an orphan node and `entity_lookup`'s cross-references are always empty. The multi-DB architecture's connective tissue doesn't exist.

Neither bug is exotic; both are invisible because **every unit test mocks the database.** A memory system whose tests never store-then-read against a real graph is testing its own mocks, not its memory. That's the root problem, and it's why I'd withhold "production-ready" regardless of how polished the surrounding code is.

### The deeper architectural question I'd raise

Beyond the bugs, I'd challenge one design decision: **separate Neo4j databases per vertical, with an LLM routing every read and write.** It's the source of the most complexity and the most breakage in the codebase:

- It creates the cross-DB proxy problem (currently broken) that wouldn't exist in a single graph.
- It puts a nondeterministic LLM call on the hot path of *every* uncached store and search — latency, cost, and a failure mode where a Bedrock hiccup misroutes a write into the wrong database *permanently*.
- Neo4j has no cross-database joins, so fan-out + merge + rerank is all application-level glue that has to be re-derived and re-tested.

The same domain separation is achievable inside **one database** with node labels/namespaces and metadata filters — no cross-DB proxies, no routing LLM required for correctness (routing becomes an optional relevance optimization, not a correctness dependency), and entity dedup works globally instead of being defeated by per-DB UUIDs. I'm not certain the multi-DB split is wrong — if the real driver is hard tenant isolation or per-vertical backup/retention, it earns its keep — but **as currently justified (domain-specific ontologies), it's paying a steep complexity tax for something labels would buy cheaply.** I'd want that decision re-litigated explicitly before building more on top of it.

### The other structural liability: the overlay

The package shadows the upstream `neo4j-agent-memory` import namespace, monkeypatches its privates, and `deploy.sh` copies files over site-packages in prod. It works today, but it's the least testable and most upgrade-fragile part of the system, and it's unpinned against upstream. This is technical debt that compounds: every future fix has to reason about which copy of the code is running.

### Bottom line

Good bones, real quality instincts, genuinely broken primitives, and a test strategy that hides the breakage. **It's a strong prototype that is one focused correctness-and-testing pass away from being a good memory system** — and possibly one architectural simplification (collapse the verticals into one graph) away from being a *simpler* good memory system. The recommendations below are ordered to get it there.

---

## Part 2 — Complete Recommendation List

Severity: **C**=critical, **H**=high, **M**=medium, **L**=low. Status as of this document.

### Correctness — Critical

| ID | Recommendation | Subsystem | Status |
|----|----------------|-----------|--------|
| R1 | Unify temporal property representation (choose epoch millis), convert at the `add_fact` call site, migrate existing data, delete the "string means expired" hack in `memory_search` | temporal | **Open** |
| R2 | Fix or remove cross-DB ProxyRef: create the `HAS_REFERENCE` edge keyed on entity identity + round-trip test, or delete the feature and stop leaking orphan nodes | proxy/registry | **Open** |
| R3 | Enforce `graph_query` read-only at the DB layer (READ-access-mode transaction) | mcp tools | **Done** (commit `6b45b21`) |
| R4 | Rotate the retired `claude-sonnet-4-20250514` Bedrock pin to a current model; re-baseline golden thresholds; make BAML extraction failure non-fatal to `memory_store` (currently it aborts *after* the message node is committed → partial write) | extraction/baml | **Open** |

### Correctness — High

| ID | Recommendation | Subsystem | Status |
|----|----------------|-----------|--------|
| R5 | Make fact create + supersede a single atomic Cypher write (fixes the concurrent-store mutual-invalidation race and the crash-between-writes duplicate) | temporal | Open |
| R6 | Gate supersession on the new fact being current/open-ended; consult `is_current_state` so a historical fact ("used to work at Acme") doesn't invalidate the current one | temporal | Open |
| R7 | Add an object-equality check so re-affirming an identical fact refreshes rather than supersedes (stop destroying `valid_from` history) | temporal | Open |
| R8 | Set `expired_at` on all supersession paths (not just the contradiction path) so `knowledge_state` transaction-time queries are correct | temporal | Open |
| R9 | Allowlist-validate `entity_type` in `entity_lookup` (label injection sink) | mcp tools | Open |
| R10 | Add authentication to the HTTP transport; stop documenting `0.0.0.0` bind without an auth gate | server | Open |
| R11 | Resolve/cancel the pending routing future on leader cancellation (coalesced waiters currently hang forever on `CancelledError`) | routing | Open |
| R12 | On router failure, fan out to all DBs for *search* and surface a degradation flag; make *storage* routing failure an explicit error rather than a silent write to the general DB | routing | Open |
| R13 | Guard the reranker: if zero results are kept, return the originals (fail-open on bad structured output, not just on exceptions) | routing | Open |

### Correctness — Medium

| ID | Recommendation | Subsystem | Status |
|----|----------------|-----------|--------|
| R14 | Normalize subject/predicate case (and trim) before supersession/contradiction matching | temporal | Open |
| R15 | Add `idx >= 0` bounds check before invalidating a contradiction candidate (negative index invalidates the wrong fact) | temporal | Open |
| R16 | Fix `Z`-suffix ISO parsing for Python 3.10, or raise the floor to 3.11 in `pyproject.toml` | temporal | Open |
| R17 | Preserve distinct relation types between the same node pair (current `MERGE` on `:RELATED_TO` collapses `ASSIGNED_TO` + `BLOCKED_BY` into one edge) | extraction | Open |
| R18 | Make the name-based relation fallback type-aware (currently links an arbitrary same-named entity via `LIMIT 1`) | extraction | Open |
| R19 | Enforce `limit` after cross-DB merge (fan-out returns up to N_dbs × limit) | routing/merge | Open |
| R20 | Fix cache-key normalization: preserve the part separator and don't truncate storage keys to 128 chars (collisions misroute persistent writes) | routing | Open |
| R21 | Harden extraction prompts against injection: role-separate/fence stored content in `extraction.baml` and `reasoning.baml` (vertical ontologies already do this) | extraction/baml | Open |
| R22 | Fix silent `database=` fallbacks in `temporal_query`/`fact_evolution`/`knowledge_state`/`memory_search` — either honor the requested vertical or tell the caller it wasn't found | mcp tools | Open |
| R23 | Surface partial fan-out failures: add an `errors` field instead of listing a down DB in `databases_searched` as if it succeeded | routing/merge | Open |
| R24 | Fix `entity_lookup` searching only `target_dbs[0]` after paying for routing (either search all targets or skip routing) | mcp tools | Open |

### Structural / Fragility

| ID | Recommendation | Subsystem | Status |
|----|----------------|-----------|--------|
| R25 | **Re-litigate the multi-DB-vertical decision** (see Part 1). Either document the hard-isolation justification, or collapse to a single graph with label namespaces + metadata filters, eliminating the proxy problem and routing-as-correctness-dependency | architecture | Open (decision) |
| R26 | Replace the same-namespace overlay + deploy-time site-packages copy with a renamed package (or vendored fork); pin the upstream `neo4j-agent-memory` version | packaging | Open |
| R27 | Deduplicate the embedder patch (two drift-prone copies); make monkeypatches apply on all server construction paths or fail loudly | server | Open |
| R28 | Reduce per-store LLM/embedding calls (up to 3 LLM + 2 embed per fact): reuse the fact's stored embedding in contradiction search; consider batching | temporal/extraction | Open |
| R29 | Externalize hardcoded thresholds (routing 0.3, ambiguity 0.6, rerank 0.4) so code and prompt can't drift and they're tunable without regenerating the BAML client | routing | Open |
| R30 | Make `migration.py` real (invoked at startup, per-vertical, batched, format-consistent with R1) or delete it | temporal | Open |
| R31 | Validate `Resilient` fallback provider keys at startup; preflight `AWS_REGION` | extraction/baml | Open |

### Testing / CI

| ID | Recommendation | Subsystem | Status |
|----|----------------|-----------|--------|
| R32 | **Add a Neo4j service to CI**, split `-m integration` from unit runs, make Bedrock-dependent tests skip (not error) without creds, and take the live LLM eval off the PR merge gate | ci/tests | Open |
| R33 | Add end-to-end `memory_store` → `memory_search` tests through the actual MCP tool functions against a real Neo4j | tests | Open |
| R34 | Add the temporal store→query round-trip test (the regression gate for R1) | tests | Open |
| R35 | Add concurrency tests: parallel same-subject stores (R5), parallel same-entity `add_message` (dedup race) | tests | Open |
| R36 | Add reranker success-path, ProxyRef round-trip (R2), and failure-injection (Bedrock down, one DB down) tests | tests | Open |
| R37 | Delete/repair tests that can't fail: `test_health.py` (no `/health` route exists), the tautological `test_baml_variants` winner assertion, the `TestFactSearch` self-tests | tests | Open |

---

## Part 3 — Multi-Agent Implementation Strategy

The work above is ~30 items across five subsystems. Executing it well in parallel is mostly a **dependency-and-conflict-management** problem, not a raw-throughput problem. Two facts dominate the plan:

- **`_tools.py` (1,600+ lines) is a merge-conflict hotspot.** Many fixes touch it. Uncoordinated parallel edits will conflict badly.
- **Almost every correctness fix needs a real-Neo4j integration test to prove it.** So the test harness is a hard prerequisite, not a parallel nice-to-have.

### Guiding principles

1. **Integration-test-first.** No correctness fix merges without a real-Neo4j test that fails before and passes after. This is the whole remedy for the "mocks hide the bug" problem.
2. **Worktree isolation for parallel edits.** Agents that touch overlapping files run in separate git worktrees; a dedicated **integrator** rebases and resolves, rather than agents pushing to a shared branch.
3. **One coherent subsystem per agent.** Group by file-locality and conceptual coupling (all temporal-lifecycle fixes = one agent), not by ticket. This minimizes cross-agent conflict and keeps each agent's context focused.
4. **Adversarially verify.** Each correctness claim is checked by a second agent prompted to *refute* it (reproduce the bug on the old code, confirm the fix, probe for a regression) before it's considered done.
5. **Decision gates block dependent work.** R1 (temporal representation) and R25 (multi-DB decision) are choices that many other items depend on. Resolve them first; don't let agents guess.

### Wave 0 — Foundations (mostly serial; unblocks everything)

These gate the rest and should land before the parallel waves.

| Task | Agent role | Why it's first | Verification gate |
|------|-----------|----------------|-------------------|
| **CI + integration harness** (R32) | Test-infra | Every later fix needs a real-Neo4j test bed and a CI that runs it | CI green with a Neo4j service; integration tests run, not skip |
| **Decision: temporal representation** (R1 design) | Architect | R5–R8, R30, R34 all depend on the chosen type | Written decision (epoch millis) + migration approach |
| **Decision: multi-DB vs single graph** (R25) | Architect | Determines whether R2 is "fix proxy" or "delete proxy"; reshapes routing work | Written decision with justification |
| **Packaging redesign** (R26/R27) | Platform | Riskiest structural change; doing it early means every later fix lands in the clean layout, not the overlay | Renamed/pinned package; full suite green; deploy dry-run |

> Sequencing note: Wave 0 is where a human decision-maker is most needed. The two "Decision" tasks are the ones an agent should *propose* (with options + recommendation) but a human should *ratify*, because they're irreversible-ish and shape everything downstream. Use `AskUserQuestion` at these gates.

### Wave 1 — Parallel correctness fixes (worktree-isolated)

Once Wave 0 lands, these run concurrently. Grouped to minimize `_tools.py` collisions — each group owns a bounded region or file set. Each group ships with its integration test (R33–R36).

| Group | Items | Primary files | Runs parallel with |
|-------|-------|---------------|--------------------|
| **A. Temporal lifecycle** | R1(impl), R5, R6, R7, R8, R14, R15, R16, R30 | `temporal/*.py`, `_tools.py` fact branch | B, C, D, E |
| **B. Proxy / cross-DB** | R2 (per R25 decision) | `_proxy.py`, `_merge.py`, `_tools.py` entity_lookup | A, C, D, E |
| **C. Routing / rerank** | R11, R12, R13, R19, R20, R23, R24, R29 | `routing/router.py`, `_merge.py`, `_tools.py` search | A, B, D, E |
| **D. Extraction** | R4, R17, R18, R21, R28, R31 | `extraction/*.py`, `baml_src/*.baml` | A, B, C, E |
| **E. Security surface** | R9, R10, R22 | `_tools.py` entity_lookup, `server.py` transport | A, B, C, D |

**Conflict management:** groups A, B, C, E all touch `_tools.py`. Options, in order of preference:
1. **Refactor first (recommended):** a short Wave-0.5 task splits `_tools.py` into per-tool modules (`_tools_memory.py`, `_tools_temporal.py`, `_tools_graph.py`, …) so each Wave-1 group owns a file. One-time cost, permanent conflict relief, and it improves the codebase.
2. **If not refactoring:** each group edits a designated function region only; the integrator merges in a fixed order (A→B→C→E), rebasing each on the prior. Slower, more integrator toil.

### Wave 2 — Integration & verification

| Task | Agent role | Depends on |
|------|-----------|-----------|
| Cross-cutting integration tests (R33, R35) — E2E store→search, concurrency | Test | A, C, D merged |
| Failure-injection tests (R36) — Bedrock down, one DB down | Test | C, D, E merged |
| Adversarial verification pass — each Wave-1 fix re-checked by a refuter agent | Verifier | all of Wave 1 |
| Test cleanup (R37) — delete/repair tests that can't fail | Test | anytime after Wave 0 |

### Wave 3 — Polish

Documentation refresh (the README claims features that R1/R2 make false today — update once fixed), threshold externalization follow-through (R29), and any deferred `L`-severity items.

### Orchestration shape

```
Wave 0 (serial, human-gated):  CI harness → [R1 decision, R25 decision] → packaging redesign → _tools.py split
                                     │
Wave 1 (parallel, worktrees):        ├── Agent A: temporal lifecycle
                                     ├── Agent B: proxy / cross-DB
                                     ├── Agent C: routing / rerank
                                     ├── Agent D: extraction
                                     └── Agent E: security surface
                                     │   (each ships its own integration test; integrator rebases)
Wave 2 (parallel after merge):       ├── cross-cutting integration + concurrency tests
                                     ├── failure-injection tests
                                     └── adversarial verify (refuter per fix)
Wave 3:                              docs + threshold externalization + L-severity tail
```

**Why this shape:** the two irreversible decisions and the shared-file refactor are front-loaded so the expensive parallel wave never has to guess or fight merge conflicts; every correctness agent carries its own proof (a failing-then-passing real-DB test); and an independent verifier tries to break each fix before it's trusted — the same discipline that would have caught the original bugs.

### Practical guardrails

- **Log what's dropped.** If any agent bounds scope (skips a case, samples), it must say so — silent truncation is how the current gaps formed.
- **No fix without a red test first.** The test proving the bug must fail on `main` before the fix; otherwise the fix is unverified.
- **Human ratifies the two decisions and the packaging change.** Everything else can run autonomously; those three are where a wrong autonomous call is expensive to unwind.

---

## Appendix — Suggested first three PRs (if executed incrementally rather than as a fleet)

1. **CI + integration harness (R32)** — unblocks proving everything else. Smallest change with the highest leverage.
2. **Temporal representation unification + round-trip test (R1 + R34)** — fixes the single most-broken headline feature, with the regression gate that keeps it fixed.
3. **Proxy decision + fix/delete (R25 → R2)** — resolves the other dead primitive and forces the architectural conversation early, before more is built on the multi-DB assumption.

R3 (read-only `graph_query`) is already merged as the proof-of-concept for this workflow: DB-layer fix + unit tests for the previously-untested guard + an integration test that reproduces the bypass on a live server.

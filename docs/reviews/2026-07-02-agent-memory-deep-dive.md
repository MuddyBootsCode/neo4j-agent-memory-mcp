# Agent Memory Implementation — Deep-Dive Review

**Date:** 2026-07-02
**Scope:** Full review of the agent memory implementation: MCP server layer, BAML extraction pipeline, temporal fact management, routing/reranking/cross-database search, and the test suite. Five parallel subsystem reviews plus independent verification of every critical finding. Unit suite run locally: **186 passed in 1.76s**.

---

## Executive Summary

The architecture is genuinely good: the vertical/registry design, the routing cache with single-flight stampede prevention, the graceful-degradation ladders, and the extraction eval harness are all above the bar for a project at this stage. But there are **four show-stopper findings**, each independently verified against the code:

1. **Point-in-time temporal queries are broken by a type mismatch** — facts store `valid_from`/`valid_until` as ISO strings, queries compare against integer epoch millis. In Neo4j `STRING <= INTEGER` is `null`, so `temporal_query` silently excludes every fact that actually has temporal data. The headline feature works only for facts with *no* temporal info.
2. **The cross-database ProxyRef feature is dead on arrival** — `create_proxy_reference` writes a bare node; the `HAS_REFERENCE` edge that `resolve_proxy_references` matches on is never created anywhere (`grep` confirms one read, zero writes). Every vertical store leaks an orphan node; `cross_references` is always empty.
3. **The "read-only" `graph_query` tool can write to the database** — upstream `execute_read` is a bare `session.run()` in a default (write-capable) session; the only guard is a keyword regex that deliberately allows `CALL apoc.*`. Compounding it, `entity_lookup` interpolates the caller-controlled `entity_type` directly into Cypher (`f" AND e:{entity_type}"`) — a write-capable injection sink. And the HTTP transport has **no authentication** while the README documents binding `0.0.0.0` on a public IP.
4. **The pinned Bedrock model is past retirement** — `clients.baml` pins `claude-sonnet-4-20250514`, retired June 15, 2026 (before today). When Bedrock stops serving it, every extraction fails — and because BAML failures propagate out of `add_message` *after* the message node is committed, `memory_store` returns an error for a write that partially succeeded.

A fifth structural issue underlies much of the fragility: the **same-namespace overlay** on the unpinned upstream `neo4j-agent-memory` package, with `deploy.sh` copying overlay files over site-packages in production. It is the root cause of the recursion guards, the duplicated embedder patch, and the least testable behavior in the repo.

The test suite is the mirror image of the code: excellent eval harnesses (golden extraction dataset, retrieval quality tiers, routing eval) that **never run in CI**, and a fast all-mock unit layer that structurally cannot catch any of the four show-stoppers above. CI has no Neo4j service and no AWS credentials; the integration tests skip silently or hard-fail there.

---

## 1. What Works

These are strengths worth keeping and building on:

### Architecture & operations
- **Lifespan phasing with graceful degradation** (`mcp/server.py:52-239`): Docker → general client (with retry) → vertical DB creation → per-vertical clients → router/reranker, each phase independently guarded. If Neo4j is unavailable the server still boots and tools return structured errors instead of the process dying.
- **Docker manager discipline** (`mcp/_docker.py`): TCP-probe before touching Docker, exponential backoff, only stops containers it started (`_we_started` guard), timeouts on every subprocess call.
- **Per-DB failure isolation in fan-out** (`mcp/_registry.py:82-94`): parallel `asyncio.gather` with per-DB exceptions converted to error dicts; one vertical being down does not fail the whole search.
- **Structured tool logging** (`mcp/_logging.py`): start/end/error events with timing and param truncation.

### Routing & caching
- **Two-tier TTL caches with single-flight coalescing** (`routing/router.py:82-138, 207-219`): concurrent identical queries share one LLM call; failures are deliberately not cached; structured `routing_decision:` log lines with latency and cache status.
- **Sensible reranker economics** (`router.py:310-312, 348-350`): skips the LLM for ≤3 results, fails open on exception.
- **Verticals registry as single source of truth** (`verticals.py:29-75`) with a documented add-a-vertical recipe.
- **Disambiguation flow** for ambiguous queries with an explicit `database=` escape hatch.

### Extraction
- **Type-safe extraction via BAML enums** — entity types can never be arbitrary strings from the primary extractor; confidence clamped to [0,1] at every conversion site.
- **Relation hallucination filtering** (`extraction/baml_extractor.py:105-116`): drops relations whose endpoints weren't extracted.
- **No Cypher injection through vertical persistence** (`extraction/vertical_extractor.py:142, 187-196`): SET clauses built only from hardcoded keys; LLM-produced relation types stored as properties on a fixed edge type; labels go through the upstream sanitizer.
- **Genuinely async throughout** — zero sync BAML client usage in `src/`; no event-loop blocking.
- **Measured extraction quality**: the prompt variant in production won an empirical A/B comparison, and golden-set thresholds pin regressions (entity recall ≥60%, precision ≥40%).

### Temporal design (the *design*, not the data layer — see §2.1)
- Correct half-open interval semantics (`valid_from <= pit`, `valid_until > pit`).
- Idempotent supersession writes guarded by `WHERE f.valid_until IS NULL`.
- Graceful degradation ladder in contradiction detection (no embedder → SPO fallback; BAML failure → SPO fallback).
- Conservative contradiction prompt ("when in doubt, do NOT mark as contradicted").
- Env kill-switches (`NAM_TEMPORAL_EXTRACTION`, `NAM_CONTRADICTION_DETECTION`).
- Temporal indexes created at init for all verticals, including composite `(subject, predicate)`.

### Testing (the good parts)
- **Golden extraction dataset**: 20 hand-labeled conversations across 10 categories with real precision/recall gates (`tests/integration/test_extraction_golden.py`).
- **Round-trip retrieval quality tiers** (`tests/integration/test_round_trip.py`): easy/medium/hard/negative queries with a ≥70% aggregate hit-rate gate and negative-query assertions.
- **Graph structural invariants** (`tests/integration/test_graph_integrity.py`): no orphaned entities, no duplicate `(name, type)` pairs, all messages embedded.
- **Routing eval harness**: 25 live scenarios, per-category breakdown, CSV/JSON artifacts, threshold-gated exit code.
- **Overlay drift tripwire** (`tests/test_overlay_comparison.py`): fails loudly if upstream adds a tool the overlay doesn't cover.

---

## 2. Critical Findings (fix first)

### 2.1 Temporal property types are incoherent — point-in-time queries drop exactly the facts that have temporal data — **CRITICAL**

Upstream `add_fact` serializes `valid_from.isoformat()` / `valid_until.isoformat()` → **string** properties on Fact nodes (verified: `.venv/.../memory/long_term.py:655-656`). `temporal_fact_query` binds `$pit` as **integer** epoch millis (`temporal/lifecycle.py:166-172`). In Neo4j, `STRING <= INTEGER` evaluates to `null`, so the `WHERE` clause excludes the row.

**Three representations coexist** in production data: `NULL` (no temporal info), ISO string (add_fact path), and int epoch millis (supersession `lifecycle.py:48`, contradiction invalidation `contradiction.py:216`, migration `migration.py:28`).

Consequences:
- `temporal_query` returns facts with `valid_from = NULL` and migration-backfilled facts only. A fact stored with "since March" extraction never appears at any point in time.
- `memory_search` contains a knowing hack around this (`mcp/_tools.py:286-287`): any string-typed `valid_until` is treated as **expired** — so a fact with an explicit *future* expiry ("contract valid until 2027") is labeled expired, sorted last, and dropped when `include_expired=False`.
- Mixed formats leak to consumers: `str(row["valid_until"])` yields `"1709827200000"` for some facts and `"2026-03-01T00:00:00"` for others in the same response.

**Fix:** pick one representation (epoch millis), convert at the `add_fact` call site in `_tools.py` (or normalize on write), write a data migration, and delete the string-means-expired hack.

### 2.2 Cross-database ProxyRef feature is dead code that leaks orphan nodes — **CRITICAL (functional)**

`create_proxy_reference` (`mcp/_proxy.py:35-53`) CREATEs a bare `:ProxyRef` node. `resolve_proxy_references` (`_proxy.py:77-84`) matches `(e:Entity {id})-[:HAS_REFERENCE]->(p:ProxyRef)`. **`HAS_REFERENCE` is never created anywhere in the codebase** (verified by grep: one read, zero writes). Additionally, proxies are keyed on the vertical *message/fact id* while resolution is driven by an *Entity id* — even with the edge, the linkage targets the wrong node type.

Result: every vertical `memory_store` leaks an orphan node into the general DB; `entity_lookup`'s `cross_references` is always empty; `merge_entity_results` (`_merge.py:50-76`) is dead code with no callers. `_proxy.py` has zero tests — one round-trip test would have caught this.

**Fix:** either wire the feature properly (create the edge, key on entity identity, add a round-trip test) or delete it and stop creating orphan nodes.

### 2.3 "Read-only" tools can write; write-capable Cypher injection; unauthenticated HTTP — **CRITICAL (security)**

Three compounding issues:

1. **`execute_read` does not enforce read-only.** The comment at `_tools.py:76-77` ("The database itself will reject writes when executed via execute_read()") is false — upstream `execute_read` is a bare `session.run()` in a default write-capable session (verified: `.venv/.../graph/client.py:104-107`). The regex in `_is_read_only_query` is the *only* guard.
2. **The regex is bypassable.** `WRITE_PATTERNS` (`_tools.py:78-90`) deliberately allows `CALL <procedure>` (the intent, per the comment at `_tools.py:74-77`, is to permit read procedures like `db.index.vector.queryNodes` and `apoc.meta.data`). But it only blocks `CALL {` subqueries — any *write* APOC procedure passes: `apoc.create.node(...)`, `apoc.merge.*`, `apoc.refactor.*`, `apoc.periodic.iterate(...)`, `apoc.trigger.*`, and notably `apoc.cypher.doIt(...)` (the write-mode sibling of `apoc.cypher.runFirstColumn`). APOC is enabled in the compose file. Enumerating these in the denylist is a losing game — the set changes across APOC versions and any string-built procedure name or comment/casing trick evades a regex over a language the server itself parses.
3. **`entity_lookup` Cypher injection.** `_tools.py:700-701`: `type_filter = f" AND e:{entity_type}"` — a caller/LLM-controlled string interpolated into the query. `Person WITH e MATCH (n) DETACH DELETE n //` is a valid payload, and because of (1) it mutates the graph.

Plus: **the HTTP transport has no authentication** (`server.py:401-402`), and the README documents `--host 0.0.0.0` and a public IP endpoint. Anyone who can reach the port gets full tool access, including the injection sinks above.

**Verified:** upstream `execute_read` is byte-for-byte identical to `execute_write` — both call `session.run()` on `_get_session()`, which opens `driver.session(database=...)` with **no `default_access_mode`** (defaults to WRITE). There is zero DB-level read enforcement (`.venv/.../graph/client.py:84-127`).

**Fix — enforcement options, ranked (a denylist regex cannot catch all cases):**

1. **Read-only account / true READ transaction — best, and a genuine catch-all.** Prod runs Neo4j Enterprise (RBAC available), so either give the `graph_query` path a dedicated user with a read-only role, or open the session in read access mode (`driver.session(default_access_mode=neo4j.READ_ACCESS)` / the managed `session.execute_read(...)` transaction function). Key property: a READ-mode transaction rejects not just write *clauses* but any procedure whose declared mode is WRITE/SCHEMA — so `apoc.cypher.doIt` and every other write APOC procedure fails with *"Cannot perform write operation in read access mode"* **without maintaining any list**. This is why it is strictly better than regex or EXPLAIN. (The read-only *account* specifically needs Enterprise — Community has no custom RBAC — which prod has.) Because the overlay already monkeypatches this upstream client, the cleanest place to add a real read-mode `execute_read` is in that patch rather than upstream.
2. **EXPLAIN-based validation — a real improvement over regex, with one caveat.** `EXPLAIN <query>` never executes (only PROFILE does) and returns a plan the *server* parsed, so you can walk the operator tree for write operators (`CreateNode`, `CreateRelationship`, `SetProperty`, `Delete*`, `Merge`) — this robustly kills the parsing-error class the regex can't. **Caveat:** EXPLAIN does not see inside APOC — `CALL apoc.cypher.doIt(...)` shows only an opaque `ProcedureCall` operator with no write operator in the plan. To close that you'd cross-reference each `ProcedureCall` against `SHOW PROCEDURES` and reject any with `mode` of WRITE/SCHEMA/DBMS — at which point you've rebuilt what option 1 gives for free. Use EXPLAIN only if option 1 is unavailable.
3. **Keeping the current regex — only as defense-in-depth.** If it stays, at minimum add `apoc.cypher.doIt`, `apoc.cypher.runWrite`, `apoc.create.*`, `apoc.merge.*`, `apoc.refactor.*`, `apoc.periodic.*`, `apoc.trigger.*`, and treat the list as permanently incomplete. Ideally it becomes a cheap first-pass reject in *front* of a read transaction, not the sole gate.

Separately: validate `entity_type` against the known type allowlist (it can't be parameterized as a label, so it must be whitelisted), and put an auth layer (or at least a bearer token + bind guidance) in front of the HTTP transport. Recommended posture: read-only account **and** READ access mode (belt and suspenders), with the regex demoted to a fast-fail nicety.

**Status (2026-07-02):** Option 1 (READ access mode) implemented for `graph_query` — `_execute_read_only` in `mcp/_tools.py` now runs the query in a managed read transaction, and the regex is demoted to an explicit first-pass check. Unit tests cover the keyword pre-check (previously untested), the read-transaction wiring, and the APOC-write gap; an integration test (`tests/integration/test_graph_query_readonly.py`) proves a live server rejects both a plain `CREATE` and a parameter-smuggled `apoc.cypher.doIt` write. **Still open:** provisioning a read-only Neo4j account (deployment/RBAC), the `entity_type` allowlist in `entity_lookup`, and HTTP transport auth.

### 2.4 Retired model pin + extraction failure semantics = production outage waiting — **CRITICAL (operational)**

- `clients.baml:21` pins `us.anthropic.claude-sonnet-4-20250514-v1:0` — deprecated with retirement **June 15, 2026**, which has passed. When Bedrock stops serving it, every extraction/routing/reranking call fails.
- When BAML extraction fails, the exception propagates out of upstream `add_message` **after the Message node is committed** (no try/except around `_extract_and_link_entities`; verified upstream `memory/short_term.py:530-548`). The MCP client receives `{"error": ...}` for a write that partially succeeded — retrying clients store duplicates; non-retrying clients believe the memory was lost.
- The documented `client_name="Anthropic"` (`baml_extractor.py:27`, `reasoning_extractor.py:19`) **does not exist** in `clients.baml` — setting `NAM_EXTRACTION__BAML_CLIENT=Anthropic` makes every extract call fail at dispatch. Tests set this value but never call `extract()`, so CI passes.

**Fix:** rotate the model pin (and re-baseline golden thresholds), wrap the `add_message` extraction path so a BAML outage degrades to "message stored, entities skipped," and either add the Anthropic client or remove it from the docs/defaults.

---

## 3. High-Severity Logic Bugs

### Temporal
- **Historical facts supersede current ones** (`_tools.py:574-586`): supersession runs unconditionally, ignoring `parsed_valid_until` and the extraction's `is_current_state`. Storing "Alice *used to* work at Acme" invalidates the true current fact "Alice WORKS_AT Globex."
- **Concurrent-store race mutually invalidates both facts** — `add_fact` and `supersede_matching_facts` are separate transactions with no locking. Interleaved stores of the same subject+predicate leave a supersession *cycle* and zero active facts. A crash between the two calls leaves duplicate active facts. The fix for both is a single atomic Cypher write (CREATE + supersede in one statement).
- **Re-affirming an identical fact destroys its history** (`lifecycle.py:41-57`): no object check, so storing the same fact twice supersedes the original and resets `valid_from` to now — "known since March" becomes "known since today."
- **`expired_at` is set only on the contradiction path** (`contradiction.py:216` vs `lifecycle.py:48-49`), so `knowledge_state` (which filters on `expired_at`) shows SPO-superseded facts as currently believed forever.
- **Negative LLM index invalidates the wrong fact** (`contradiction.py:208-210`): no `idx >= 0` check; `-1` resolves to the last candidate.
- **Case-sensitive subject/predicate matching** defeats supersession for `alice`/`Alice` or `works_at`/`WORKS_AT` — no normalization anywhere.
- **`Z`-suffix ISO parsing fails on Python 3.10** (`lifecycle.py:17`) while the BAML prompt instructs the LLM to emit `Z`-suffixed dates and `pyproject.toml` declares `>=3.10`.

### Routing & search
- **Coalesced waiters hang forever on leader cancellation** (`router.py:130, 164-175`): `except Exception` doesn't catch `CancelledError`; the pending future is popped but never resolved, so waiters await it indefinitely. Fix: cancel/resolve the future in `finally`.
- **Cache key collisions can misroute writes** (`router.py:22-37`): normalization strips the `|` part separator and truncates at 128 chars — two stored memories sharing a 128-char normalized prefix silently reuse each other's *storage* routing decision. Misrouted writes are persistent.
- **Routing failure silently hides all vertical data**: on LLM error the fallback is general-DB-only (`router.py:166-173`) with no degradation flag — during a Bedrock outage, search returns "no results" for everything in the verticals, and new writes land permanently in the general DB. A better search fallback is fan-out-to-all; storage failures should be surfaced.
- **Reranker can silently discard all results** (`router.py:330-340`): filtering trusts the LLM's `keep` flags and exact ID matching with no "if zero kept, return originals" floor. Hallucinated or mangled IDs = valid results vanish. It fails open on exceptions but fails closed on bad structured output.
- **`limit` not enforced after merge**: fan-out returns up to N_dbs × limit per memory type, bloating responses and reranker prompts.
- **`entity_lookup` pays for routing then searches only `target_dbs[0]`** (`_tools.py:686`), contradicting its own docstring.
- **Silent `database=` fallbacks**: several tools (memory_search without registry, `temporal_query`, `fact_evolution`, `knowledge_state`) silently fall back to the general DB when the requested vertical isn't registered — general-DB results labeled as if the request was honored.
- **Partial fan-out failures are invisible**: an errored DB is skipped in the merge but still listed in `databases_searched`, with no `errors` field.

### Extraction
- **Distinct relation types between the same node pair collapse into one edge** (`vertical_extractor.py:187-196`): `MERGE` on `:RELATED_TO` with `relation_type` set only `ON CREATE` — `ASSIGNED_TO` and `BLOCKED_BY` between the same entities lose the second type.
- **Name-based relation fallback links arbitrary same-named entities** (`vertical_extractor.py:205-229`): lowercase name match with `LIMIT 1`, type not part of the match — wrong edges get MERGEd permanently.
- **Prompt injection → memory poisoning** (`extraction.baml:37-61`, `reasoning.baml`): stored content is interpolated into the same prompt block as the instructions with no role separation or fencing (the vertical ontologies do use `_.role("user")`, the core extractor doesn't). Injected content can fabricate schema-valid entities/facts persisted as trusted memory, and `SynthesizeExplanation` re-interpolates previously stored trace steps — a second-order injection channel. Bounded by the typed output schema, but worth hardening.
- **Double extraction per vertical-routed message**: POLE+O extraction inside `add_message` plus vertical extraction on the same content — two full LLM calls, two overlapping entity sets, no cross-linking.

---

## 4. Structural Fragility: the Overlay Architecture

This is the biggest medium-term risk even though nothing is "broken" today:

- The repo's package **shadows the same `neo4j_agent_memory` import namespace** as the upstream dependency, via `__path__` manipulation and exec-the-base-init tricks (`src/neo4j_agent_memory/__init__.py`, `extraction/__init__.py`). The base package's `__init__` executes twice under a synthetic module name — duplicate module-level state and latent `isinstance` hazards.
- The upstream dependency is **unpinned** (`pyproject.toml:7`: `neo4j-agent-memory[mcp,openai]`, no version constraint) while the code reaches into upstream privates (`client.graph._driver`, `client.short_term._embedder`, `client._extractor`) and monkeypatches `_create_embedder` and `extraction.factory.create_extractor`. Any upstream minor bump can silently break the patch targets — the patch works only because upstream uses a *deferred* import; if that import is ever hoisted, the patch silently stops applying.
- **`deploy/deploy.sh` copies overlay files over site-packages in production**, overwriting the upstream `__init__.py` (which contains the ~1,000-line `MemoryClient`) with the 71-line overlay shim. Correctness on the prod box then depends on `sys.path` ordering. This is the root cause of the recursion guards and is untestable in the dev layout.
- **The monkeypatches apply only on the lifespan path** (`server.py:73, 119`): `Neo4jMemoryMCPServer` and `create_mcp_server(settings=None)` install no lifespan, so Bedrock embedding silently doesn't work on those construction paths. The embedder patch also exists in two drift-prone copies (`server.py:77-116` vs `_embedder_patch.py:12-51`).

**Recommendation:** rename the overlay package (e.g. `neo4j_agent_memory_mcp`) and import from upstream explicitly, or vendor/fork upstream, or upstream the Bedrock/BAML support. Pin the upstream version either way. This one change eliminates the recursion guards, the deploy-time file copying, and an entire class of silent-breakage risk.

Other fragility worth noting:
- `NAM_VERTICALS` is half-dynamic: arbitrary verticals get databases and clients, but the BAML routing enum is frozen at meetings/projects/research — custom verticals are unroutable; removing a vertical strands its data silently.
- Up to **3 LLM calls + 2 embedding calls per stored fact** (routing, temporal extraction, embedding, a redundant re-embedding in contradiction candidate search, contradiction detection), all sequential, defaults ON. Consider reusing the fact's stored embedding and batching.
- Thresholds scattered across code and prompts (routing 0.3 in code, ambiguity 0.6 prompt-only, rerank 0.4 prompt-only) — can't tune without regenerating the BAML client, and code/prompt already disagree (`>=` vs `>`).
- `temporal/migration.py` is dead code: never invoked at startup, backfills only `valid_from` as int (diverging from add_fact's strings), single-DB, unbatched.
- `Resilient` fallback chain requires OpenAI/Gemini keys that nothing validates at startup — in a Bedrock-only deployment it just adds two more failures' worth of latency.
- BAML drift: `pyproject.toml` allows `baml-py>=0.70.0` while the generator pins 0.219.0; prod regenerates the client at deploy time, so what runs is not what was reviewed; the "git diff empty after generate" check from the plan isn't enforced in CI.

---

## 5. Testing: What Needs to Be Tested Further

### The structural problem
The unit layer (186 tests, 1.76s) mocks everything below the tool functions — `graph.execute_read/execute_write` return whatever the test author imagines, so **no test executes any Cypher against a real Neo4j**. That is precisely why the temporal type bug (§2.1), the ProxyRef bug (§2.2), and the relation-collapse bug all shipped: they live in the seam between Python and the database that mocks cannot see.

Meanwhile CI (`.github/workflows/test.yml`) runs `pytest tests/ --ignore=tests/test_routing_eval.py` with **no Neo4j service and no AWS credentials**:
- Neo4j-backed integration tests (`test_round_trip`, `test_cross_session`, `test_graph_integrity`, `test_smoke`) silently **skip** — the best tests in the repo never run in CI.
- `test_extraction_golden.py` / `test_baml_variants.py` don't depend on the Neo4j fixture and call Bedrock with a hardcoded `AWS_PROFILE` — in CI they hard-fail rather than skip.
- The routing-eval job runs a **live LLM eval on every PR** with a 0.80 threshold — a nondeterministic merge gate — and evaluates **OpenAI** while production routes with **Bedrock**, so the measured 92% may not describe production at all.
- The checked-in `routing_eval_results.json` is stale and test-fitted: one expectation was changed to match model output after the recorded run, and the BAML doc comments still contradict the harness expectations.

### Tests that can't fail (fix or delete)
- `tests/test_health.py`: asserts `server is not None`; there is **no `/health` route anywhere in the codebase** despite the docstring.
- `test_baml_variants.py::test_compare_variants`: asserts `max(all_results) >= baseline` where the max includes baseline — tautology.
- `test_multi_db.py::TestFactSearch` (4 tests) and `test_graph_augment_false_skips_traversal`: re-implement the production logic locally and assert on their own code.
- `test_docker_manager.py::test_returns_none_when_not_found`: asserts `result is None or isinstance(result, Path)` — satisfied by anything.
- `TestExtractTemporalContext` depends on BAML being *unavailable* — a developer with live creds gets a real LLM call and a possible failure.

### Prioritized list of tests to add

1. **Fix CI first**: add a Neo4j service container, split `-m integration` from unit runs, make Bedrock-dependent tests skip (not error) without creds, and take the live eval out of the PR merge gate (run on `[eval]` commits or nightly). Highest-leverage change available.
2. **Temporal store→query round trip against real Neo4j**: store a fact with `valid_from`, query it via `temporal_query`. This fails today (§2.1) — it's the regression test for the fix.
3. **End-to-end `memory_store` → `memory_search` through the MCP tool functions** against the `testharness` DB. The 1,618-line tool layer has never touched a real database.
4. **`graph_query` hardening tests**: APOC write bypass, `CALL {` variants, comment/unicode obfuscation, plus `entity_lookup` `entity_type` injection payloads — as end-to-end rejection tests once the fixes land.
5. **ProxyRef round trip**: create → resolve, asserting a non-empty `cross_references` (currently impossible).
6. **Concurrency**: two simultaneous `memory_store` calls with the same subject+predicate (supersession race, §3); N concurrent `add_message` calls mentioning the same entity (MERGE dedup race on the product's core claim).
7. **Reranker success path**: mocked `RerankOutput` with mixed keep/drop, hallucinated IDs, and the all-dropped case (should floor to originals once fixed).
8. **Supersession semantics**: historical fact (`is_current_state=false` / closed `valid_until`) must *not* supersede the current fact; same-object re-store must not destroy history.
9. **Server lifespan test**: run the 5-phase lifespan with mocked client/docker; assert embedder patch, factory patch, registry population, index creation, and `close_all` — replacing the self-simulating `test_server_baml_patch.py`.
10. **Failure injection at the tool layer**: BAML down mid-`memory_store` (assert "message stored, extraction skipped" once §2.4 is fixed); one vertical DB down during real fan-out with an `errors` field in the response.

---

## 6. Prioritized Action Plan

**P0 — correctness/security (this week):**
1. Unify temporal property representation (epoch millis), convert at write, migrate existing data, delete the string-means-expired hack (§2.1).
2. Fix or remove ProxyRef: create the `HAS_REFERENCE` edge keyed on entity identity, or delete the feature (§2.2).
3. Enforce read-only `graph_query` at the DB layer — a read-only Enterprise account and/or READ-access-mode transactions (which reject write APOC procedures like `apoc.cypher.doIt` for free, unlike the regex or EXPLAIN); allowlist-validate `entity_type` in `entity_lookup`; add auth to the HTTP transport (§2.3).
4. Rotate the retired Sonnet 4 Bedrock pin; re-baseline golden thresholds; make extraction failure non-fatal to `memory_store` (§2.4).

**P1 — data integrity (next):**
5. Make create+supersede one atomic Cypher write; gate supersession on the new fact being current/open-ended; add object check for identical-fact re-stores; set `expired_at` on all supersession paths; normalize subject/predicate case.
6. Guard the reranker (zero-kept floor) and surface routing degradation (fan-out-to-all on router failure for search; explicit error for storage).
7. Resolve/cancel pending routing futures on leader cancellation; fix the cache-key separator/truncation for the storage path.

**P2 — structure (planned work):**
8. Replace the same-namespace overlay + deploy-time site-packages copying with a renamed package or vendored fork; pin the upstream version.
9. Fix CI (Neo4j service, marker split, cred-less skips, eval off the merge gate) and add the top-5 tests from §5.
10. Deduplicate the embedder patch; make monkeypatches apply on all construction paths or fail loudly.

---

*Methodology: five parallel subsystem reviews (MCP server layer, extraction pipeline, temporal facts, routing/search, test suite), each reading its full subsystem plus relevant upstream code in the installed dependency. All critical findings (§2.1–2.4) were independently re-verified against the source before inclusion. Unit suite executed locally: 186 passed.*

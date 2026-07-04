# Remaining Work — Multi-Agent Orchestration Plan

**Date:** 2026-07-02
**Companions:** [deep-dive review](../reviews/2026-07-02-agent-memory-deep-dive.md) · [upgrade strategy](2026-07-02-agent-memory-upgrade-strategy.md)
**Scope:** the work still open *after* the single-graph refactor (R25/R1/R2/R3/R4/R9/R17/R18/R28 already landed). This document is the executable orchestration plan — waves, agent roles, dependencies, conflict management, verification gates, and a runnable `Workflow` skeleton.

---

## Context: where the codebase is now

The multi-database vertical architecture is gone. Memory lives in one Neo4j graph; extraction is a single unified `ExtractMemory` pass; the LLM router is off the hot path. Unit suite: 112 passing. What remains is correctness hardening, security posture, packaging, and CI — no more architectural forks. That makes the remaining work **highly parallelizable**: the items cluster into subsystem-local groups with few cross-dependencies.

### The remaining items

| ID | Item | Primary files |
|----|------|---------------|
| R5 | Make fact create + supersede one atomic Cypher write (fixes concurrent-store mutual invalidation and crash-between-writes) | `temporal/lifecycle.py`, `_tools.py` fact branch |
| R6 | Gate supersession on the new fact being current/open-ended (a historical fact must not supersede the current one); consult `is_current_state` | `_tools.py` fact branch, `temporal/lifecycle.py` |
| R7 | Object-equality check — re-affirming an identical fact refreshes, not supersedes (stop destroying `valid_from`) | `temporal/lifecycle.py` |
| R8 | Set `expired_at` on **all** supersession paths (not just contradiction), so `knowledge_state` is correct | `temporal/lifecycle.py`, `temporal/contradiction.py` |
| R14 | Normalize subject/predicate case + trim before matching | `temporal/lifecycle.py`, `temporal/contradiction.py` |
| R15 | `idx >= 0` bounds check before invalidating a contradiction candidate | `temporal/contradiction.py` |
| R16 | Fix `Z`-suffix ISO parse on Python 3.10, or raise the floor to 3.11 | `temporal/lifecycle.py`, `pyproject.toml` |
| R10 | Authenticate the HTTP transport; stop documenting bare `0.0.0.0` | `mcp/server.py`, README |
| RBAC | Provision + document a read-only Neo4j account for the query path (belt to the READ-transaction suspenders already in place) | `mcp/server.py`, config, deploy docs |
| R21 | Fence stored content in extraction/reasoning prompts (role separation is partial) | `baml_src/extraction.baml`, `baml_src/reasoning.baml` |
| R31 | Validate `Resilient` provider keys / `AWS_REGION` at startup | `mcp/server.py` |
| R26 | Replace the same-namespace overlay + deploy-time site-packages copy with a renamed package (or vendored fork); pin the upstream version | `src/neo4j_agent_memory/__init__.py`, `extraction/__init__.py`, `deploy/deploy.sh`, `pyproject.toml` |
| R27 | Deduplicate the embedder patch; make it apply on all server construction paths or fail loudly | `mcp/server.py`, `mcp/_embedder_patch.py` |
| R32 | Add a Neo4j service to CI; run integration tests; make Bedrock-dependent tests skip (not error) without creds | `.github/workflows/test.yml`, `tests/integration/conftest.py` |
| CI‑wf | Remove the dead `routing-eval` CI job (its test was deleted) — **needs `workflow` scope** | `.github/workflows/test.yml` |

---

## Principles

1. **Integration-test-first.** No correctness fix merges without a real-Neo4j test that fails before and passes after. The E2E harness already exists (`tests/integration/test_memory_e2e.py`); extend it.
2. **Worktree isolation for parallel edits.** Agents that touch overlapping files run in separate git worktrees; a dedicated integrator rebases and resolves.
3. **One coherent subsystem per agent.** Group by file-locality so intra-group conflicts are near zero.
4. **Adversarially verify.** Each correctness claim is re-checked by a second agent prompted to *refute* it (reproduce the bug on the old code, confirm the fix, probe for a regression) before it's trusted.
5. **Human gates two things:** the `workflow`-scope CI edit (CI‑wf) and the packaging rename (R26) — the former this agent can't push, the latter is irreversible-ish and touches everything.

---

## Work groups (parallel units)

Grouped so each group owns a bounded file set. The two `_tools.py` touchers (T and S) edit **different regions** (fact branch vs `graph_query`/transport), so they merge cleanly in order T→S.

| Group | Items | Owns | Ships |
|-------|-------|------|-------|
| **T. Temporal correctness** | R5, R6, R7, R8, R14, R15, R16 | `temporal/*.py`, `_tools.py` fact branch, `pyproject.toml` python floor | atomic supersede + semantics tests in `test_memory_e2e.py` (concurrency, historical-fact, re-affirm, `expired_at`, case) |
| **S. Security surface** | R10, RBAC | `mcp/server.py` transport, README deploy | HTTP-auth unit test (401 without token, 200 with); read-only-account config + docs |
| **X. Extraction hardening** | R21, R31 | `baml_src/*.baml`, `mcp/server.py` startup preflight | prompt-injection extraction test (adversarial content yields no fabricated fact); startup-preflight unit test |
| **C. CI & test harness** | R32 (+ CI‑wf handoff) | `.github/workflows/test.yml`, `tests/integration/conftest.py` | green CI with a Neo4j service; Bedrock tests skip cleanly without creds |
| **P. Packaging redesign** | R26, R27 | overlay `__init__` files, `deploy/deploy.sh`, `pyproject.toml`, `mcp/server.py` embedder patch | renamed/pinned package; full suite green; deploy dry-run |

---

## Wave plan

```
Wave 0 (serial, human-gated):  C — CI Neo4j-service harness  ──►  proves every later fix
                                    │  (CI‑wf edit handed to a human with workflow scope)
                                    ▼
Wave 1 (parallel, worktrees):   ├── Agent T: temporal correctness   (+ integration tests)
                                ├── Agent S: HTTP auth + read-only account
                                └── Agent X: prompt fencing + startup preflight
                                    │  integrator merges T → S → X, rebasing each
                                    ▼
Wave 2 (parallel):              ├── concurrency tests (parallel same-S+P store; parallel same-entity add_message)
                                └── adversarial verify: one refuter per T/S/X fix
                                    ▼
Wave 3 (serial, isolated):      P — overlay packaging rename + embedder-patch dedup
                                    │  full-suite gate + deploy dry-run; human ratifies the rename
                                    ▼
                                Done → squash-merge PR per group, or one integration PR
```

**Why this order.** CI-with-Neo4j (Wave 0) is the leverage point — it's what makes every correctness fix provable, and it unblocks nothing else so it goes first. The functional fixes (Wave 1) are subsystem-local and run concurrently. Packaging (Wave 3) is deferred to last so the fixes land on the current layout; the rename then becomes a mechanical change with a full-suite gate rather than a moving target under everyone else.

### Conflict management

- `temporal/*.py` and the `_tools.py` fact branch are wholly owned by **Agent T** — no intra-group races.
- **Agent S** edits `_tools.py` only in the `graph_query`/transport region and `server.py` transport setup — disjoint from T's fact branch. Integrator merges T first, then rebases S.
- **Agent X** edits `baml_src/` and `server.py` startup — the `server.py` overlap with S is the one hotspot; assign S the transport block and X the startup-preflight block, and have the integrator apply S→X in that file.
- **Agent P** (Wave 3) touches everything by rename; it runs alone after Waves 1–2 land, so there is nothing to conflict with.

### Verification gates (per group)

- **T:** `pytest tests/integration/test_memory_e2e.py -m integration` green, including new concurrency and semantics cases; adversarial refuter cannot reproduce the concurrent-store double-invalidation.
- **S:** unit test asserts the HTTP app returns 401 without a token and 200 with; README no longer documents an unauthenticated `0.0.0.0` bind.
- **X:** adversarial-content extraction test asserts no fabricated entity/fact escapes the typed schema; startup with a missing `Resilient` key logs a clear preflight error.
- **P:** full unit + integration suite green on the renamed package; `deploy/deploy.sh --dry-run` (or a staging run) shows no site-packages shadowing.
- **All:** no fix without a red test first (the test must fail on the pre-fix code).

---

## Runnable `Workflow` skeleton

This maps the plan onto the `Workflow` tool. It is a *pipeline*: each group flows find→fix→verify independently, with the packaging phase gated behind the rest. Author it as a script and run via the Workflow tool (or run each phase as its own invocation to stay in the loop between waves).

```js
export const meta = {
  name: 'agent-memory-remaining-work',
  description: 'Land the post-refactor open items (temporal, security, hardening) with adversarial verification',
  phases: [
    { title: 'CI harness' },
    { title: 'Fix' },
    { title: 'Verify' },
    { title: 'Packaging' },
  ],
}

// Wave 0 — CI harness must land (and be confirmed green) before fixes rely on it.
phase('CI harness')
await agent(
  'Add a Neo4j service container to .github/workflows/test.yml, split unit vs ' +
  '`-m integration`, and make Bedrock-dependent integration tests skip (not error) ' +
  'without AWS creds. Do NOT touch application code. Note: the workflow file edit ' +
  'may require human push (workflow scope) — output the exact YAML.',
  { label: 'ci-harness', phase: 'CI harness', isolation: 'worktree' }
)

// Wave 1+2 — each group: fix (worktree) → adversarial verify. Pipeline, no barrier.
const GROUPS = [
  { key: 'temporal',   prompt: 'Implement R5,R6,R7,R8,R14,R15,R16 in temporal/*.py and the _tools.py fact branch. ' +
      'Make create+supersede one atomic Cypher statement; gate supersession on the new fact being current/open-ended; ' +
      'add an object-equality refresh path; set expired_at on all supersede paths; normalize subject/predicate case; ' +
      'bounds-check the contradiction index; fix Z-suffix parsing or raise the python floor. ' +
      'Add failing-first integration tests to tests/integration/test_memory_e2e.py.' },
  { key: 'security',   prompt: 'Implement R10 + read-only account: add bearer-token auth to the HTTP transport in mcp/server.py ' +
      '(401 without, 200 with), stop documenting an unauthenticated 0.0.0.0 bind, and add config + docs for a read-only ' +
      'Neo4j account used by the query path. Add a unit test for the auth gate. Edit only the transport region of _tools.py/server.py.' },
  { key: 'hardening',  prompt: 'Implement R21 + R31: fence stored content in baml_src/extraction.baml and reasoning.baml so injected ' +
      'instructions cannot fabricate records, and add a startup preflight in mcp/server.py that validates Resilient provider keys / AWS_REGION. ' +
      'Add an adversarial-content extraction test.' },
]

const VERDICT = { type: 'object', properties: {
  real: { type: 'boolean' }, reproduced_before: { type: 'boolean' }, notes: { type: 'string' },
}, required: ['real', 'notes'], additionalProperties: false }

const results = await pipeline(
  GROUPS,
  g => agent(g.prompt, { label: `fix:${g.key}`, phase: 'Fix', isolation: 'worktree' }),
  (fixOutput, g) => parallel(['correctness', 'regression', 'repro-old-bug'].map(lens => () =>
    agent(
      `Adversarially verify the ${g.key} changes via the ${lens} lens. Check out the pre-fix code, confirm the bug ` +
      `reproduced there, confirm the fix resolves it, and probe for a regression. Return your verdict.\n\n${fixOutput}`,
      { label: `verify:${g.key}:${lens}`, phase: 'Verify', schema: VERDICT }
    )
  )).then(vs => ({ group: g.key, verdicts: vs.filter(Boolean) }))
)

const shaky = results.flat().filter(r => r.verdicts.filter(v => v.real).length < 2)
if (shaky.length) log(`NEEDS REWORK before packaging: ${shaky.map(r => r.group).join(', ')}`)

// Wave 3 — packaging rename runs alone, only after the above is clean.
phase('Packaging')
await agent(
  'Implement R26+R27: rename the overlay package to stop shadowing the upstream import namespace ' +
  '(or vendor/fork it), pin the upstream neo4j-agent-memory version in pyproject.toml, remove the ' +
  'deploy.sh site-packages copy, and deduplicate the embedder patch so it applies on all server ' +
  'construction paths or fails loudly. Gate on the FULL unit+integration suite green. This is ' +
  'irreversible-ish — stop and surface a summary for human ratification before finalizing.',
  { label: 'packaging', phase: 'Packaging', isolation: 'worktree', effort: 'high' }
)
```

**Notes on running it.** Prefer running Waves 0, 1–2, and 3 as **separate** `Workflow` invocations so a human reviews between them (especially before Packaging). The `isolation: 'worktree'` flags matter here — T, S, and X mutate overlapping files and must not share a tree. The adversarial-verify stage is the safeguard that would have caught the original concurrency and type bugs; keep the "≥2 of 3 lenses confirm real" gate.

---

## Guardrails

- **No fix without a red test first** — the test proving the bug must fail on the current code before the fix.
- **Log what's dropped** — if any agent bounds scope (skips a case, samples), it says so; silent truncation is how the original gaps formed.
- **Human ratifies two calls** — the `workflow`-scope CI edit and the packaging rename. Everything else runs autonomously.
- **Concurrency is the highest-value new coverage** — the temporal race (R5) and the entity-dedup MERGE race are the two failure modes most likely to bite in production; they get dedicated real-Neo4j tests.

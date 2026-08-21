# Extractor experiments

Compares memory-extraction schemas by running each one over the same source
text, writing its output to its own subgraph, and scoring retrieval against the
same query set.

Motivation: recall quality is bounded by what extraction stores. Entities are
currently written with a name and no description, so the text embedded for an
entity is a bare string like `PR #136`. Measured over 3,251 judged pairs,
entities were rated relevant 4.8% of the time against 18.5% for preferences,
while making up 89% of the corpus.

## Self-contained by design

- **Imports nothing from `src/`.** Changing production cannot change results
  here, and a broken experiment cannot break the server.
- **No BAML codegen, no container rebuild.** Variants call the Anthropic API
  directly with structured outputs. A variant is a prompt plus a JSON schema, so
  adding one is editing `variants.py`.
- **Own subgraph per variant.** Every node carries the `:Exp` label and a
  `variant` property; nothing touches production `:Entity` / `:Preference`
  nodes. Variants coexist, so two runs can be compared without a reset.

The tradeoff: this measures the *schema and prompt*, not BAML itself. BAML is a
typed wrapper over the same structured call. Whichever variant wins gets ported
into `baml_src/extraction.baml` and verified there.

## Run

```bash
export ANTHROPIC_API_KEY=...          # or rely on .env at the repo root

# 1. Extract. Same source chunks for every variant, so the comparison is paired.
uv run --with anthropic --with neo4j python experiments/extractors/extract.py \
    --variant baseline --chunks 30
uv run --with anthropic --with neo4j python experiments/extractors/extract.py \
    --variant described --chunks 30
uv run --with anthropic --with neo4j python experiments/extractors/extract.py \
    --variant workflow --chunks 30

# 2. Inspect what each one stored.
uv run --with neo4j python experiments/extractors/show.py

# 3. Score retrieval. Needs the local embedder.
uv run --extra local --with neo4j --with anthropic \
    python experiments/extractors/evaluate.py --queries 80
```

`--reset` on `extract.py` clears just that variant's subgraph before writing.

## Variants

| name | entity fields | types | embedded text |
|------|---------------|-------|---------------|
| `baseline` | name, type, subtype | production enum | name only |
| `described` | + required description | production enum | name + description |
| `workflow` | + required description | + EPIC, PULL_REQUEST, ISSUE, LESSON | name + description |

`baseline` reproduces today's behaviour and is the control. `described` isolates
the effect of one field. `workflow` adds the engineering types on top, so the
two effects stay separable.

# Local judge

Puts a local model in the seat the similarity cutoff currently occupies, and
reads back why it kept what it kept.

Recall ships with no reasoning in it: `recall_hook.py` injects whatever
`memory_search` returns above `DEFAULT_THRESHOLD`. This measures whether a
model deciding instead would inject better, and makes the decision readable.

## Free by construction

Everything needed is already on disk from the extractor variant sweep — the
sampled prompts, the record corpora, the top-10 retrieval runs, and 2,000
Claude-judged production pairs. No re-extraction, no re-embedding, no Neo4j.
The local judge is scored directly against labels that already exist, over
byte-identical candidates, with the rubric copied verbatim from
`extractors/evaluate.py` so a disagreement is a model difference and not a
prompt artifact.

## Run

```bash
ollama create qwen-judge -f Modelfile     # once — see "The model" below
uv run python experiments/local_judge/selection.py --queries 30
BAML_LOG=warn uv run python experiments/local_judge/run.py --config terse-only
uv run python experiments/local_judge/compare.py
uv run python experiments/local_judge/render.py
```

### Configs

`--config` on `run.py` selects the shape:

| config | thinking | reason field | median call |
|--------|----------|--------------|-------------|
| `think-reason` | on | yes | 52.3s |
| `terse-reason` | off | yes | 9.0s |
| `terse-only` | off | no | 4.2s |

`compare.py` scores all three against the same candidates and writes
`results/compare.json`. The fastest config is also the most accurate — thinking
made the model generous, promoting 65 records off zero that Claude called
worthless. See MUD-394.

`reasoning_effort: "none"` is the only flag that actually stops Ollama
generating thinking tokens. `think: false` and
`chat_template_kwargs.enable_thinking` strip reasoning from the response while
the model generates it anyway: 126 and 197 completion tokens for a one-word
answer, against 2 with `reasoning_effort`.

`run.py` is resumable: a query already in `results/verdicts.json` is skipped,
so an interrupted run costs what it already spent and no more. `--redo`
forces a re-judge, `--limit N` smoke-tests.

Roughly 55s per call on an M5 Max, one call per prompt. No API spend.

## The model

`qwen-judge` is a derived tag, not the upstream build:

```
FROM rafw007/Qwen3.6-35B-A3B-mlx-claude-coder-abliterated:latest
SYSTEM "You are a precise assistant. Follow the user's instructions exactly."
```

Upstream ships a Polish-language red-team operator persona as its `SYSTEM`
prompt, which biases a relevance-judging task and surfaces in the reasoning
text. Note that `SYSTEM ""` does not clear it — Ollama treats the empty value
as a no-op and the inherited prompt survives, so it has to be replaced with a
real string.

`max_tokens` in the BAML client is the **output** budget, not the context
window. The model declares `num_ctx 262144` itself. The budget matters because
thinking is on and thinking tokens count against it.

## Reading the reasoning

Ollama returns a thinking model's reasoning in `reasoning_content`, which the
OpenAI schema does not have, so BAML's `raw_llm_response` — `message.content` —
drops it silently. The tell is the token count: ~5k output tokens against a
~1.5k-token answer, with an empty thinking field. `runtime.py` recovers it from
the Collector's untouched HTTP response body.

## Files

| file | does |
|------|------|
| `selection.py` | assembles the judging set from the sweep's cached JSON |
| `judge_source.py` | the `JudgeRecall` BAML source, injected into the file map |
| `runtime.py` | BAML runtime routed to Ollama; Collector capture |
| `run.py` | drives the calls, resumable, records compliance failures |
| `score.py` | agreement and gate comparison against the Claude labels |
| `compare.py` | scores every config that has been run, side by side |
| `render.py` | builds the self-contained HTML page |

The one production change this depends on is the `QwenCoder` client in
`baml_src/clients.baml` and `ollama_registry()` in `providers.py`, both opt-in
behind `NAM_LLM_PROVIDER=ollama`.

## Reading the numbers

Recall is bounded by the candidate pool: the denominator is the relevant
records inside each query's top 10, not everything relevant in the corpus. A
gate cannot inject what retrieval never surfaced, so this measures the gate,
not the retriever.

Agreement is reported as exact 3-way match, binary match, and Cohen's kappa.
The kappa is the one to read — the label set is about two-thirds zeros, so raw
agreement is flattered by the majority class alone.

Tracked in MUD-394.

"""Step 4: label every (query, lesson) pair in the same repo with Claude Opus 5.

Cost shape: the rubric plus one sorted chunk of CHUNK lessons sits in
`system` behind a cache breakpoint; only the query changes per call. Calls
run chunk-major so one prefix stays hot across all queries. Resumable:
labels.json is keyed "query_id:lesson_id" and existing keys are skipped.

The labeler is deliberately not the pipeline's model. Every prior number was
qwen-judge grading qwen-judge; this is the independent reference.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from lib import LABEL_MODEL, PRICE, load_json, result_path, save_json

CHUNK = int(os.environ.get("GOLDEN_LABEL_CHUNK", "50"))
EFFORT = os.environ.get("GOLDEN_LABEL_EFFORT", "medium")
CONCURRENCY = int(os.environ.get("GOLDEN_LABEL_CONCURRENCY", "4"))

RUBRIC = """\
You are building a relevance benchmark for a coding assistant's memory. The \
assistant stores short lessons learned in earlier work sessions on a \
repository: decisions (what was chosen and why), gotchas (constraints that \
cost time to discover), and dead ends (attempts that failed and why).

You will be shown a developer's prompt from a later session, with the files \
that session had edited before the prompt, and a numbered list of stored \
lessons. For each lesson decide whether injecting it into the assistant's \
context at that moment would materially help: it answers part of the prompt, \
prevents a mistake the prompt is about to make, or changes what the assistant \
should do. Shared vocabulary, the same subsystem, or the same file is not \
enough on its own. A lesson about a file the prompt concerns is relevant only \
if the prompt touches the behaviour the lesson describes.

Be strict. Most lessons are irrelevant to most prompts. Return a verdict for \
every lesson id, including the irrelevant ones.

STORED LESSONS (repository: {repo}):
{lessons}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "relevant": {"type": "boolean"}},
                "required": ["id", "relevant"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def _lesson_line(item: dict) -> str:
    files = f" (files: {', '.join(item['files'][:4])})" if item["files"] else ""
    reason = f" Reason: {item['reason']}" if item.get("reason") else ""
    return f"{item['id']}. [{item['kind']}] {item['text']}{reason}{files}"


def _cost(usage) -> float:
    p = PRICE.get(LABEL_MODEL)
    if not p:
        return 0.0
    return (
        usage.input_tokens * p["input"]
        + usage.output_tokens * p["output"]
        + (usage.cache_creation_input_tokens or 0) * p["cache_write"]
        + (usage.cache_read_input_tokens or 0) * p["cache_read"]
    ) / 1_000_000


async def _label_one(client, system, q: dict, ids: list[str]) -> tuple[dict | None, object, float]:
    files = ", ".join(q["files"][:10]) or "(none)"
    user = (
        f"Files edited in this session before the prompt: {files}\n\n"
        f"DEVELOPER PROMPT:\n{q['prompt']}\n\n"
        f"Return one verdict per lesson id ({len(ids)} ids)."
    )
    t0 = time.time()
    response = await client.messages.create(
        model=LABEL_MODEL,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if response.stop_reason == "refusal":
        print(f"  q{q['query_id']}: refusal ({response.stop_details})", file=sys.stderr)
        return None, response.usage, time.time() - t0
    text = next(b.text for b in response.content if b.type == "text")
    verdicts = {v["id"]: bool(v["relevant"]) for v in json.loads(text)["verdicts"]}
    return verdicts, response.usage, time.time() - t0


async def main() -> None:
    import anthropic

    queries = load_json("queries.json")
    pool = load_json("pool.json")
    if not queries or not pool:
        raise SystemExit("run steps 1-3 first")
    labels: dict[str, bool] = load_json("labels.json", {})
    usage_log: list[dict] = load_json("label_usage.json", [])

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    by_repo: dict[str, list[dict]] = {}
    for item in pool:
        by_repo.setdefault(item["repo"], []).append(item)

    state = {"cost": sum(u["cost_usd"] for u in usage_log), "calls": 0}

    async def run(q: dict, ci: int, repo: str, system, ids: list[str]) -> int:
        async with sem:
            verdicts, u, elapsed = await _label_one(client, system, q, ids)
        if verdicts is None:
            return 0
        missing = [lid for lid in ids if lid not in verdicts]
        for lid in ids:
            if lid in verdicts:
                labels[f"{q['query_id']}:{lid}"] = verdicts[lid]
        cost = _cost(u)
        state["cost"] += cost
        state["calls"] += 1
        usage_log.append({
            "query_id": q["query_id"], "repo": repo, "chunk": ci, "model": LABEL_MODEL, "effort": EFFORT,
            "input": u.input_tokens, "output": u.output_tokens,
            "cache_write": u.cache_creation_input_tokens, "cache_read": u.cache_read_input_tokens,
            "cost_usd": round(cost, 5), "elapsed_s": round(elapsed, 1),
            "missing": len(missing), "relevant": sum(verdicts.values()),
        })
        if state["calls"] % 10 == 0:
            save_json("labels.json", labels)
            save_json("label_usage.json", usage_log)
        print(
            f"  q{q['query_id']:<3} chunk {ci}: {sum(verdicts.values()):>2} relevant"
            f"{' missing=' + str(len(missing)) if missing else ''}"
            f"  cache_read={u.cache_read_input_tokens} out={u.output_tokens}"
            f"  ${cost:.4f}  ({elapsed:.1f}s)  total ${state['cost']:.2f}"
        )
        return u.cache_read_input_tokens or 0

    for repo, items in by_repo.items():
        items.sort(key=lambda it: it["id"])
        repo_queries = [q for q in queries if q["repo"] == repo]
        chunks = [items[i : i + CHUNK] for i in range(0, len(items), CHUNK)]
        print(f"{repo}: {len(items)} lessons in {len(chunks)} chunks x {len(repo_queries)} queries")
        for ci, chunk in enumerate(chunks):
            ids = [it["id"] for it in chunk]
            system = [{
                "type": "text",
                "text": RUBRIC.format(repo=repo, lessons="\n".join(_lesson_line(it) for it in chunk)),
                "cache_control": {"type": "ephemeral"},
            }]
            todo = [q for q in repo_queries if any(f"{q['query_id']}:{lid}" not in labels for lid in ids)]
            if not todo:
                continue
            # First call alone so the cache entry exists before the fan-out;
            # the second call must read it or the cost assumption is wrong.
            await run(todo[0], ci, repo, system, ids)
            if len(todo) > 1:
                cache_read = await run(todo[1], ci, repo, system, ids)
                if cache_read == 0:
                    raise SystemExit(
                        "cache_read_input_tokens is 0 on the second call for this chunk: "
                        "a silent cache invalidator is at work; stopping before the cost runs"
                    )
            await asyncio.gather(*(run(q, ci, repo, system, ids) for q in todo[2:]))

    save_json("labels.json", labels)
    save_json("label_usage.json", usage_log)
    relevant = sum(1 for v in labels.values() if v)
    print(f"\nlabels: {len(labels)} pairs, {relevant} relevant ({relevant / max(len(labels), 1):.1%}); "
          f"{state['calls']} calls this run; total cost ${state['cost']:.2f} -> {result_path('labels.json')}")


if __name__ == "__main__":
    asyncio.run(main())

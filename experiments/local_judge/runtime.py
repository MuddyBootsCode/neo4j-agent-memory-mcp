"""BAML runtime for the local judge, routed to Ollama via ClientRegistry.

Mirrors experiments/extractors/baml_runtime.py: build a real BamlRuntime from
the production file map plus one extra file, and read results out of the
Collector rather than BAML's generated Pydantic types, which are frozen at
codegen and know nothing about JudgeRecall.

Reading through the Collector is also the point of the exercise — it is what
makes the model's reasoning, the rendered prompt, the retries and the token
counts observable instead of inferred.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from judge_source import JUDGE_BAML, JUDGE_KEY  # noqa: E402

RESULTS = os.path.join(HERE, "results")

# The three configs isolate two separate costs. Thinking is the obvious one —
# 87% of generated characters in the first run. The reason field is the
# subtle one: writing a justification before committing to a grade is
# chain-of-thought inside the answer, so dropping it may cost accuracy on its
# own. Measuring them together would confound the two.
CONFIGS = {
    "think-reason": {
        "name": "think-reason",
        "function": "JudgeRecall",
        "options": {},
        "blurb": "thinking on, one-sentence reason per candidate",
    },
    "terse-reason": {
        "name": "terse-reason",
        "function": "JudgeRecall",
        "options": {"reasoning_effort": "none"},
        "blurb": "thinking off, reason kept",
    },
    "terse-only": {
        "name": "terse-only",
        "function": "JudgeRecallTerse",
        "options": {"reasoning_effort": "none"},
        "blurb": "thinking off, no reason — the production shape",
    },
}

_runtime = None
_lock = threading.Lock()

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
OPEN_THINK_RE = re.compile(r"<think>(.*)", re.S)


def split_thinking(raw):
    """Separate a thinking block from the answer.

    Qwen3.6 emits reasoning inline in the content. It has to come off before
    JSON extraction: the thinking routinely contains braces, so scanning for
    the first '{' over the whole string picks up a fragment of the model's
    scratch work instead of its answer.
    """
    text = raw or ""
    blocks = THINK_RE.findall(text)
    if blocks:
        return "\n".join(b.strip() for b in blocks), THINK_RE.sub("", text).strip()
    # Budget exhausted mid-thought: an unclosed <think> and no answer at all.
    m = OPEN_THINK_RE.search(text)
    if m:
        return m.group(1).strip(), ""
    return "", text.strip()


def parse_raw(raw):
    """JSON out of a raw model response, tolerating fences and thinking."""
    _, text = split_thinking(raw)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise json.JSONDecodeError("no JSON object found", text[:80], 0)
        text = text[start:end + 1]
    return json.loads(text)


def reasoning_from_body(body_text):
    """`reasoning_content` out of a raw Ollama chat-completions body."""
    if not body_text:
        return ""
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError:
        return ""
    for choice in body.get("choices") or []:
        msg = choice.get("message") or {}
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _registry(extra_options=None):
    """The local client, with per-config option overrides.

    Built here rather than through providers.ollama_registry() only because a
    config needs to add request fields to it. The base options still come from
    production, so what the experiment runs is what the server would run.

    reasoning_effort="none" is the one flag that actually stops a thinking
    model from thinking on Ollama. `think: false` and
    `chat_template_kwargs.enable_thinking` both strip reasoning from the
    RESPONSE while the model generates it anyway — 126 and 197 completion
    tokens for a one-word answer, against 2 with reasoning_effort. Either would
    have looked like a latency win and measured nothing.
    """
    os.environ.setdefault("NAM_LLM_PROVIDER", "ollama")
    from baml_py import ClientRegistry

    from agent_memory_mcp.providers import (
        OLLAMA_CLIENT_NAME,
        ollama_client_options,
    )

    options = ollama_client_options()
    options.update(extra_options or {})

    registry = ClientRegistry()
    registry.add_llm_client(
        name=OLLAMA_CLIENT_NAME,
        provider="openai-generic",
        options=options,
        retry_policy="StandardRetry",
    )
    registry.set_primary(OLLAMA_CLIENT_NAME)
    return registry


def runtime():
    global _runtime
    with _lock:
        if _runtime is None:
            from baml_py import BamlRuntime

            from agent_memory_mcp.baml_client.inlinedbaml import get_baml_files

            files = dict(get_baml_files())
            files[JUDGE_KEY] = JUDGE_BAML
            _runtime = BamlRuntime.from_files("baml_src", files, os.environ.copy())
        return _runtime


async def judge(user_prompt, candidates_text, query_id, config=None):
    """One judging call. Returns (parsed_or_None, trace).

    Never raises on a model failure: a call that times out or returns
    unparseable JSON is a result about the local model, not an accident, and
    the run should keep going and report it.
    """
    from baml_py import Collector

    cfg = config or CONFIGS["think-reason"]
    rt = runtime()
    collector = Collector(name="judge")

    err = None
    try:
        result = await rt.call_function(
            cfg["function"],
            {"user_prompt": user_prompt, "candidates": candidates_text},
            rt.create_context_manager(),
            None, _registry(cfg.get("options")), [collector],
            os.environ.copy(), {"query": str(query_id)}, None, None,
        )
        # A failed call RETURNS a failed FunctionResult rather than raising.
        if not result.is_ok():
            err = f"call not ok: {str(result)[:300]}"
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:400]

    log = collector.last
    trace = {"query": query_id, "config": cfg["name"], "error": err}
    parsed = None

    if log is not None:
        call = log.selected_call or (log.calls[-1] if log.calls else None)
        raw = log.raw_llm_response
        # Ollama returns a thinking model's reasoning in `reasoning_content`,
        # a field the OpenAI schema does not have, so BAML's raw_llm_response
        # — which is `message.content` — drops it. The token counts give it
        # away: ~5k output tokens for a ~1.5k-token answer. The only place it
        # survives is the untouched HTTP response body.
        http_body = None
        if call is not None and call.http_response is not None:
            try:
                http_body = call.http_response.body.text()
            except Exception:
                http_body = None
        thinking, answer = split_thinking(raw)
        if not thinking:
            thinking = reasoning_from_body(http_body)
        trace.update(
            attempts=len(log.calls),
            duration_ms=log.timing.duration_ms if log.timing else None,
            input_tokens=log.usage.input_tokens if log.usage else None,
            output_tokens=log.usage.output_tokens if log.usage else None,
            client=getattr(call, "client_name", None),
            prompt=(call.http_request.body.text()
                    if call is not None and call.http_request else None),
            raw_response=raw,
            thinking=thinking,
            answer=answer,
        )
        if raw:
            try:
                parsed = parse_raw(raw)
                trace["json_ok"] = True
            except json.JSONDecodeError as exc:
                trace["json_ok"] = False
                trace["parse_error"] = str(exc)[:200]
        else:
            trace["json_ok"] = False
            trace["parse_error"] = "empty response"

    trace["ok"] = err is None and parsed is not None
    return parsed, trace


def append_trace(trace, config="think-reason"):
    os.makedirs(os.path.join(RESULTS, "traces"), exist_ok=True)
    with open(os.path.join(RESULTS, "traces", f"judge.{config}.jsonl"), "a") as fh:
        fh.write(json.dumps(trace) + "\n")

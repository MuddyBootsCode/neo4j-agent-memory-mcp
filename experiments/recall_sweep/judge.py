"""Direct calls to the local qwen-judge model via its OpenAI-compatible endpoint.

Not routed through BAML: this experiment needs two differently-worded prompts
(gate vs. grade) and a plain HTTP call is simpler than standing up a second
BAML runtime for it. reasoning_effort="none" is passed as an extra body field
(the only thing that actually stops Qwen3.6 emitting thinking tokens — see
experiments/local_judge/README.md) and Qwen3.6 still occasionally emits an
inline <think> block regardless, so responses are parsed tolerantly the same
way experiments/local_judge/runtime.py does.
"""

from __future__ import annotations

import json
import re

from lib import OLLAMA_BASE_URL, OLLAMA_MODEL

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
OPEN_THINK_RE = re.compile(r"<think>(.*)", re.S)

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _client


def strip_thinking(raw: str) -> str:
    text = raw or ""
    if THINK_RE.search(text):
        return THINK_RE.sub("", text).strip()
    m = OPEN_THINK_RE.search(text)
    if m:
        return ""  # unclosed <think>: budget exhausted mid-thought, no answer
    return text.strip()


def parse_json(raw: str) -> dict:
    text = strip_thinking(raw)
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise json.JSONDecodeError("no JSON object found", text[:200], 0)
        text = text[start:end + 1]
    return json.loads(text)


async def call(
    system: str,
    user: str,
    *,
    max_tokens: int = 4000,
    timeout: float = 180.0,
    retries: int = 1,
) -> tuple[dict | None, str | None]:
    """One judge call. Returns (parsed_dict_or_None, error_or_None).

    Never raises — a timeout, connection error, or unparseable response is a
    result about the model, logged and excluded by callers, not a crash.
    """
    client = _get_client()
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body={"reasoning_effort": "none"},
            )
            raw = resp.choices[0].message.content or ""
            try:
                return parse_json(raw), None
            except json.JSONDecodeError as exc:
                last_err = f"json_decode_error: {exc}: raw[:300]={raw[:300]!r}"
        except Exception as exc:  # noqa: BLE001 - a model/transport failure is data
            last_err = f"{type(exc).__name__}: {exc}"
    return None, last_err

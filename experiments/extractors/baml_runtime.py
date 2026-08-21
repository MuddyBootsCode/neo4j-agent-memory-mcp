"""Run ExtractMemory through a real BAML runtime built from mutated .baml source.

`baml_client/globals.py` builds the production singleton with
`BamlRuntime.from_files("baml_src", get_baml_files(), env)`. A variant is the
same call over a mutated copy of that file map, so the prompt and the type
descriptions are rendered by the real BAML runtime rather than approximated.

Results are read from the Collector's `raw_llm_response` rather than BAML's
`cast_to`, because the generated Pydantic types are frozen at codegen time and
would drop any field a variant adds. That also means no `@@dynamic` marker and
no regeneration are needed, and production baml_src/ is never modified.
"""

from __future__ import annotations

import json
import os
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

EXTRACTION_KEY = "extraction.baml"

_runtimes: dict[str, tuple] = {}
_lock = threading.Lock()


def load_env():
    """Populate ANTHROPIC_API_KEY from the repo .env if not already set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    path = os.path.join(REPO, ".env")
    if os.path.exists(path):
        for line in open(path):
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")


def parse_raw(raw):
    """JSON out of a raw model response.

    Models fence their output as ```json ... ``` often enough that a bare
    json.loads silently drops 2-7% of chunks — a quiet corpus loss that biases
    whichever variant fences more. BAML's own parser tolerates the fence; this
    matches it.
    """
    text = (raw or "").strip()
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


def production_baml() -> str:
    """The unmodified extraction.baml source, as BAML itself sees it."""
    from agent_memory_mcp.baml_client.inlinedbaml import get_baml_files

    return get_baml_files()[EXTRACTION_KEY]


def _build(variant, baml_text):
    from baml_py import BamlRuntime

    from agent_memory_mcp.baml_client.inlinedbaml import get_baml_files

    files = dict(get_baml_files())
    files[EXTRACTION_KEY] = baml_text
    rt = BamlRuntime.from_files("baml_src", files, os.environ.copy())
    return rt, None


def runtime_for(variant, baml_text):
    """One cached runtime per variant — construction parses the whole map."""
    with _lock:
        if variant not in _runtimes:
            _runtimes[variant] = _build(variant, baml_text)
        return _runtimes[variant]


def _client_registry():
    """Route to the Anthropic API when a key is set, matching production."""
    from agent_memory_mcp.providers import anthropic_registry

    return anthropic_registry()


async def render_prompt(variant, baml_text, text):
    """The prompt BAML would send, without spending a completion.

    Used by the verification step to prove the control is byte-identical to
    production and that each variant differs on exactly one axis.
    """
    rt, ctx = runtime_for(variant, baml_text)
    # build_request takes (fn, args, ctx, tb, cr, env_vars, is_stream) — no
    # collectors/tags/abort, unlike call_function.
    req = await rt.build_request(
        "ExtractMemory", {"text": text}, rt.create_context_manager(),
        None, _client_registry(), os.environ.copy(), False,
    )
    return req.body.text()


async def extract(variant, baml_text, text, chunk_id):
    """Run one extraction. Returns (parsed_dict, trace_record).

    Raises on transport failure or unparseable output; callers decide whether a
    chunk failure is fatal. A parse failure is itself a finding — a variant whose
    schema the model cannot satisfy is a worse variant.
    """
    from baml_py import Collector

    rt, ctx = runtime_for(variant, baml_text)
    collector = Collector(name=variant)

    err = None
    try:
        result = await rt.call_function(
            "ExtractMemory", {"text": text}, rt.create_context_manager(),
            None, _client_registry(), [collector],
            os.environ.copy(), {"variant": variant}, None, None,
        )
        # A failed call RETURNS a failed FunctionResult rather than raising.
        # Without this check a provider outage becomes an empty extraction that
        # counts as a success and silently deflates the variant.
        if not result.is_ok():
            err = f"call not ok: {str(result)[:300]}"
    except Exception as exc:  # keep the trace even when the call fails
        err = f"{type(exc).__name__}: {exc}"[:400]

    log = collector.last
    trace = {"variant": variant, "chunk": chunk_id, "error": err}
    parsed = {"entities": [], "relations": [], "preferences": []}

    if log is not None:
        call = log.selected_call or (log.calls[-1] if log.calls else None)
        trace.update(
            attempts=len(log.calls),
            duration_ms=log.timing.duration_ms if log.timing else None,
            input_tokens=log.usage.input_tokens if log.usage else None,
            output_tokens=log.usage.output_tokens if log.usage else None,
            client=getattr(call, "client_name", None),
            prompt=(call.http_request.body.text()
                    if call is not None and call.http_request else None),
            raw_response=log.raw_llm_response,
        )
        raw = log.raw_llm_response
        if raw:
            try:
                parsed = parse_raw(raw)
                trace["json_ok"] = True
            except json.JSONDecodeError as exc:
                trace["json_ok"] = False
                trace["parse_error"] = str(exc)[:200]
            # Would BAML's own coercion have accepted this? Recorded, never
            # relied on: the generated types are frozen at codegen, so a variant
            # that adds a field is EXPECTED to fail here while still being a
            # perfectly good extraction. Informative for `production`, which
            # must pass.
            try:
                from agent_memory_mcp.baml_client import stream_types, types

                rt.parse_llm_response(
                    "ExtractMemory", raw, types, types, stream_types, False,
                    rt.create_context_manager(), None, _client_registry(),
                    os.environ.copy(),
                )
                trace["baml_parse_ok"] = True
            except Exception as exc:
                trace["baml_parse_ok"] = False
                trace["baml_parse_error"] = f"{type(exc).__name__}: {exc}"[:200]

    trace["ok"] = err is None
    if err:
        raise RuntimeError(err)
    for key in ("entities", "relations", "preferences"):
        parsed.setdefault(key, [])
    return parsed, trace


def trace_path(variant):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "traces")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{variant}.jsonl")


def append_trace(variant, trace):
    with open(trace_path(variant), "a") as fh:
        fh.write(json.dumps(trace) + "\n")

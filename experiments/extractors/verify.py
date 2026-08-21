"""Prove the control is production and each variant moves exactly one axis.

Renders each variant's prompt through BAML without spending a completion
(`build_request`), then diffs. If the `production` variant is not byte-identical
to what the unmodified client would send, every downstream number is void.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baml_runtime as br
from variants import VARIANTS, baml_for

SAMPLE = "Michael: PR #136 under epic GRA-564 closes GRA-588. I prefer stacked PRs."


def prompt_text(body):
    """Prompt text out of a rendered request body.

    Compared instead of the raw body because the JSON serializer does not fix
    key order ("model" vs "max_tokens" first), which is not a difference in what
    the model is asked.
    """
    payload = json.loads(body)
    out = []
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            out.append(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block["text"])
    return "\n".join(out)


async def main():
    br.load_env()
    prod_text = br.production_baml()

    # The reference: the prompt the shipped client renders for this input.
    from agent_memory_mcp.baml_client.async_client import b
    from agent_memory_mcp.providers import anthropic_registry

    # Same registry production uses when ANTHROPIC_API_KEY is set; without it
    # ExtractMemory falls to its declared Bedrock client and needs AWS_REGION.
    ref = prompt_text((await b.request.ExtractMemory(
        text=SAMPLE, baml_options={"client_registry": anthropic_registry()}
    )).body.text())

    prompts = {}
    for name in VARIANTS:
        prompts[name] = prompt_text(
            await br.render_prompt(name, baml_for(name, prod_text), SAMPLE)
        )

    ok = True
    print("1. CONTROL FIDELITY")
    if prompts["production"] == ref:
        print(f"   PASS  production is byte-identical to the shipped client "
              f"({len(ref)} chars)")
    else:
        ok = False
        print("   FAIL  production differs from the shipped client:")
        for line in list(difflib.unified_diff(
            ref.splitlines(), prompts["production"].splitlines(),
            "shipped", "variant", lineterm="", n=1))[:20]:
            print("        " + line)

    print("\n2. VARIANT ISOLATION (vs production)")
    base = prompts["production"]
    for name, spec in VARIANTS.items():
        if name == "production":
            continue
        added = sum(1 for ln in difflib.unified_diff(
            base.splitlines(), prompts[name].splitlines(), lineterm="", n=0)
            if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in difflib.unified_diff(
            base.splitlines(), prompts[name].splitlines(), lineterm="", n=0)
            if ln.startswith("-") and not ln.startswith("---"))
        if added == 0 and removed == 0:
            ok = False
            flag = "FAIL  transform was a no-op"
        else:
            flag = "ok"
        print(f"   {name:18} {len(prompts[name]):>5} chars  "
              f"+{added:<3} -{removed:<3} axis={spec['axis']:<28} {flag}")

    print("\n3. AXIS SANITY")
    checks = [
        ("no-type-descr", "DO NOT extract unnamed roles", False),
        ("sharp-type-descr", "if the name would not fit on a label", True),
        ("prompt-examples", "Worked examples", True),
        ("described", "self-contained sentence", True),
        ("workflow", "PULL_REQUEST", True),
        ("production", "DO NOT extract unnamed roles", True),
    ]
    for name, needle, want in checks:
        got = needle in prompts[name]
        status = "ok" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"   {name:18} {'contains' if want else 'omits':>8} "
              f"{needle!r:45} {status}")

    print("\nVERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

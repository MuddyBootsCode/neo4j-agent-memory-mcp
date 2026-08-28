"""Keep the recall-gate model resident in Ollama (MUD-407).

The recall gate (``ScreenRecalledMemories``) has a hard deadline
(``NAM_RECALL_GATE_TIMEOUT``, default 6s). Ollama unloads models between
uses, and a cold reload of even a small model can eat most of that budget
on a contended box. Ollama's OpenAI-compatible ``/v1`` endpoint does not
accept ``keep_alive``, but the native ``POST /api/generate`` with
``{"model": ..., "keep_alive": -1, "prompt": ""}`` loads a model and pins
it resident.

This module runs that native call on server startup and then on a
background interval, so the gate model is always warm when a recall
arrives. It is strictly fail-open: any failure is logged at debug and
swallowed — a broken warm-up must never take the server down, and the gate
itself already fails open to ungated recall.

The warm-up only runs when NAM_LLM_PROVIDER=ollama and
NAM_OLLAMA_GATE_MODEL names a model different from NAM_OLLAMA_MODEL:
pinning the 36B main model resident would evict everything else on the
box, and with no dedicated gate model there is nothing to keep warm.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

WARMUP_INTERVAL_S = 600.0
_WARMUP_REQUEST_TIMEOUT_S = 30.0


def warmup_enabled() -> bool:
    """True only when a dedicated gate model is configured on Ollama."""
    from agent_memory_mcp import providers

    if not providers.ollama_enabled():
        return False
    gate = providers.ollama_gate_model()
    return gate != providers.ollama_main_model()


def native_ollama_base_url() -> str:
    """Ollama's native API base, derived from NAM_OLLAMA_URL.

    NAM_OLLAMA_URL points at the OpenAI-compatible ``/v1`` endpoint; the
    native API (which accepts ``keep_alive``) lives at the same host with
    the ``/v1`` suffix stripped.
    """
    import os

    from agent_memory_mcp import providers

    base = os.environ.get("NAM_OLLAMA_URL", providers.DEFAULT_OLLAMA_URL)
    base = base.strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _post_keep_alive(base_url: str, model: str) -> None:
    """Blocking native generate call that loads ``model`` and pins it."""
    body = json.dumps(
        {"model": model, "keep_alive": -1, "prompt": ""}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_WARMUP_REQUEST_TIMEOUT_S):
        pass


async def warm_gate_model_once() -> bool:
    """One warm-up attempt. Fail-open: returns False on any error."""
    from agent_memory_mcp import providers

    try:
        base_url = native_ollama_base_url()
        model = providers.ollama_gate_model()
        await asyncio.to_thread(_post_keep_alive, base_url, model)
        logger.debug("gate model %s pinned resident via %s", model, base_url)
        return True
    except Exception:
        logger.debug("gate model warm-up failed (fail-open)", exc_info=True)
        return False


async def gate_warmup_loop(interval_s: float = WARMUP_INTERVAL_S) -> None:
    """Warm immediately, then re-pin on an interval forever."""
    while True:
        await warm_gate_model_once()
        await asyncio.sleep(interval_s)


def start_gate_warmup() -> asyncio.Task | None:
    """Start the warm-up loop as a background task, or None when not enabled.

    Fail-open like everything else here: an error starting the task is
    logged at debug and swallowed.
    """
    try:
        if not warmup_enabled():
            return None
        from agent_memory_mcp import providers

        task = asyncio.get_running_loop().create_task(
            gate_warmup_loop(), name="nam-gate-warmup"
        )
        logger.info(
            "gate warm-up started: pinning %s resident every %.0fs",
            providers.ollama_gate_model(),
            WARMUP_INTERVAL_S,
        )
        return task
    except Exception:
        logger.debug("could not start gate warm-up (fail-open)", exc_info=True)
        return None


def stop_gate_warmup(task: asyncio.Task | None) -> None:
    """Cancel a task returned by ``start_gate_warmup``. None is a no-op."""
    if task is not None:
        task.cancel()

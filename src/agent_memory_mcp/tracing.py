"""Opik tracing for the memory-pipeline judges (MUD-427).

One trace per judge call — ``recall-gate``, ``curator``, ``served-rater``,
``extraction`` — carrying the exact strings the model saw, the full verdict
list (incl. the new reason fields), and metadata (model tag, session/repo/
task where available, elapsed ms, kept/of counts).

Configuration comes ONLY from the Opik SDK's own env vars: ``OPIK_API_KEY``,
``OPIK_WORKSPACE``, ``OPIK_PROJECT_NAME``. With no key set, every function
here is a silent no-op and the ``opik`` package is never imported. Any
exception inside tracing is caught and logged at debug — tracing must never
break capture or recall (the same fail-open contract as the judges
themselves, see ``rate_served_lessons``).

Delivery: the Opik client batches trace creates in a background thread
(2s flush interval) and registers an atexit flush, so the long-lived MCP
server needs no per-call flush. The capture pipeline still calls
:func:`flush` once at session end — a latency-free point — so capture-side
traces survive an unclean shutdown.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Transcript inputs are truncated to this many trailing chars in the trace;
# the model still gets the full text.
TRANSCRIPT_TRACE_CHARS = 4000

_client: Any = None
_client_failed = False
_client_lock = threading.Lock()


def enabled() -> bool:
    """True when an Opik API key is configured."""
    return bool(os.environ.get("OPIK_API_KEY", "").strip())


def _get_client() -> Any:
    """The shared Opik client, or None when tracing is off or broken.

    A client that failed to construct stays off for the life of the process:
    retrying a misconfigured SDK on every judge call would pay its failure
    cost inside the recall hook.
    """
    global _client, _client_failed
    if not enabled() or _client_failed:
        return None
    if _client is None:
        with _client_lock:
            if _client is None and not _client_failed:
                try:
                    from opik import Opik

                    _client = Opik(_show_misconfiguration_message=False)
                except Exception:
                    logger.debug("opik client init failed; tracing disabled", exc_info=True)
                    _client_failed = True
    return _client


def truncate_transcript(text: str) -> str:
    """The tail of a transcript for trace input; the model sees the full text."""
    return (text or "")[-TRANSCRIPT_TRACE_CHARS:]


def model_tag(*, gate: bool = False) -> str:
    """The model tag a judge call routes to, mirroring provider selection."""
    try:
        from agent_memory_mcp import providers

        if providers.ollama_enabled():
            return providers.ollama_gate_model() if gate else providers.ollama_main_model()
        if providers.anthropic_enabled():
            return os.environ.get("NAM_ANTHROPIC_MODEL", providers.DEFAULT_ANTHROPIC_MODEL)
        return "bedrock"
    except Exception:
        return "unknown"


def emit_trace(
    name: str,
    *,
    input: dict[str, Any],
    output: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log one judge call as an Opik trace. Silent no-op without a key;
    swallows every failure — a trace is never worth a broken recall."""
    client = _get_client()
    if client is None:
        return
    try:
        client.trace(
            name=name,
            input=input,
            output=output,
            metadata={k: v for k, v in (metadata or {}).items() if v is not None},
        )
    except Exception:
        logger.debug("opik trace emit failed", exc_info=True)


def flush(timeout: int = 5) -> None:
    """Best-effort bounded flush of batched traces. No-op without a client."""
    if _client is None:
        return
    try:
        _client.flush(timeout=timeout)
    except Exception:
        logger.debug("opik flush failed", exc_info=True)

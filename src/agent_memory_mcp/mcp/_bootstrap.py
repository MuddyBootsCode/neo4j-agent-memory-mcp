"""Single fail-loud bootstrap for the upstream monkeypatches (R27).

This package extends the pinned upstream ``neo4j-agent-memory`` library in
exactly two places:

1. **Embedder factory** — ``MemoryClient._create_embedder`` gains a Bedrock
   branch (:mod:`agent_memory_mcp.mcp._embedder_patch`).
2. **Extractor factory** — ``extraction.factory.create_extractor`` gains an
   env-gated BAML override for direct-library callers
   (:mod:`agent_memory_mcp.mcp._extractor_patch`).

Historically the embedder patch was duplicated inline in the server lifespan
and only applied on that one construction path, so ``Neo4jMemoryMCPServer``
and ``create_mcp_server(settings=None)`` silently ran without Bedrock
support. :func:`bootstrap_upstream_patches` is now the ONE place patches are
applied, and every server construction path goes through it:

- ``create_mcp_server(...)`` (both with and without settings)
- ``Neo4jMemoryMCPServer.__init__`` (pre-connected client wrapper)
- the integration-test client fixtures

The individual patch functions validate their upstream targets and raise
``PatchTargetError`` when the pinned upstream changed shape; this module
wraps any failure in :class:`BootstrapError` so a broken patch aborts server
startup loudly instead of degrading into a silent no-op.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_bootstrapped = False


class BootstrapError(RuntimeError):
    """A required upstream monkeypatch could not be applied."""


def bootstrap_upstream_patches(*, force: bool = False) -> None:
    """Apply all upstream monkeypatches exactly once, or fail loudly.

    Safe to call from every server construction path: the first call applies
    the patches, later calls return immediately. The underlying patch
    functions are themselves idempotent, so even a forced re-run never
    double-wraps a factory.

    Args:
        force: re-run the (idempotent) patch functions even if a previous
            call already succeeded. Intended for tests.

    Raises:
        BootstrapError: if any patch cannot be applied — e.g. the pinned
            upstream ``neo4j-agent-memory`` no longer exposes the expected
            ``MemoryClient._create_embedder`` / ``create_extractor``
            targets. Callers must not catch-and-continue: a server without
            these patches silently loses Bedrock embeddings and BAML
            extraction.
    """
    global _bootstrapped

    with _lock:
        if _bootstrapped and not force:
            return

        from agent_memory_mcp.mcp._embedder_patch import patch_embedder_factory
        from agent_memory_mcp.mcp._extractor_patch import patch_extractor_factory

        try:
            patch_embedder_factory()
            patch_extractor_factory()
        except Exception as exc:
            raise BootstrapError(
                "Failed to apply upstream monkeypatches (Bedrock embedder / "
                "BAML extractor factory). The server cannot start without "
                f"them: {exc}"
            ) from exc

        _bootstrapped = True
        logger.info("Upstream patches bootstrapped (embedder + extractor factory)")

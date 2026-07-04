"""Tests for the fail-loud upstream patch bootstrap (R27).

The bootstrap must:
- apply BOTH patches (Bedrock embedder + BAML extractor factory),
- apply each exactly once (idempotent, never double-wraps),
- raise loudly when an upstream patch target is missing or changed shape,
- run on every server construction path.
"""

from unittest.mock import MagicMock

import pytest

import agent_memory_mcp.mcp._bootstrap as bootstrap_mod
from agent_memory_mcp.mcp._bootstrap import BootstrapError, bootstrap_upstream_patches
from agent_memory_mcp.mcp._embedder_patch import PatchTargetError


@pytest.fixture
def fresh_bootstrap(monkeypatch):
    """Reset the module-level 'already bootstrapped' flag for the test."""
    monkeypatch.setattr(bootstrap_mod, "_bootstrapped", False)
    yield


def test_bootstrap_applies_both_patches(fresh_bootstrap):
    import neo4j_agent_memory.extraction.factory as factory_mod
    from neo4j_agent_memory import MemoryClient

    bootstrap_upstream_patches()

    assert getattr(MemoryClient._create_embedder, "_nam_bedrock_patched", False), (
        "embedder patch not applied to MemoryClient._create_embedder"
    )
    assert getattr(factory_mod.create_extractor, "_nam_baml_patched", False), (
        "extractor factory patch not applied to create_extractor"
    )


def test_bootstrap_applies_exactly_once(fresh_bootstrap):
    import neo4j_agent_memory.extraction.factory as factory_mod
    from neo4j_agent_memory import MemoryClient

    bootstrap_upstream_patches()
    embedder_after_first = MemoryClient._create_embedder
    factory_after_first = factory_mod.create_extractor

    # Second call is a no-op fast path...
    bootstrap_upstream_patches()
    assert MemoryClient._create_embedder is embedder_after_first
    assert factory_mod.create_extractor is factory_after_first

    # ...and even a forced re-run never double-wraps.
    bootstrap_upstream_patches(force=True)
    assert MemoryClient._create_embedder is embedder_after_first
    assert factory_mod.create_extractor is factory_after_first


def test_bootstrap_raises_when_embedder_target_missing(fresh_bootstrap, monkeypatch):
    """Upstream dropping _create_embedder must abort startup, not no-op."""
    from neo4j_agent_memory import MemoryClient

    monkeypatch.delattr(MemoryClient, "_create_embedder")

    with pytest.raises(BootstrapError, match="_create_embedder"):
        bootstrap_upstream_patches(force=True)

    # A failed bootstrap must not mark itself as done.
    assert bootstrap_mod._bootstrapped is False


def test_bootstrap_raises_when_factory_target_missing(fresh_bootstrap, monkeypatch):
    """Upstream dropping create_extractor must abort startup, not no-op."""
    import neo4j_agent_memory.extraction.factory as factory_mod

    monkeypatch.delattr(factory_mod, "create_extractor")

    with pytest.raises(BootstrapError, match="create_extractor"):
        bootstrap_upstream_patches(force=True)

    assert bootstrap_mod._bootstrapped is False


def test_extractor_patch_rejects_signature_drift(monkeypatch):
    """An upstream create_extractor with renamed params must fail loudly."""
    import neo4j_agent_memory.extraction.factory as factory_mod
    from agent_memory_mcp.mcp._extractor_patch import patch_extractor_factory

    def create_extractor(config, schema=None, llm=None):  # wrong param names
        return None

    monkeypatch.setattr(factory_mod, "create_extractor", create_extractor)

    with pytest.raises(PatchTargetError, match="signature changed"):
        patch_extractor_factory()


def test_create_mcp_server_goes_through_bootstrap(monkeypatch):
    """create_mcp_server (settings=None path included) must bootstrap."""
    from agent_memory_mcp.mcp.server import create_mcp_server

    spy = MagicMock()
    monkeypatch.setattr(bootstrap_mod, "bootstrap_upstream_patches", spy)

    create_mcp_server()  # settings=None — previously an unpatched path

    spy.assert_called_once()


def test_preconnected_server_goes_through_bootstrap(monkeypatch):
    """Neo4jMemoryMCPServer.__init__ must bootstrap."""
    from agent_memory_mcp.mcp.server import Neo4jMemoryMCPServer

    spy = MagicMock()
    monkeypatch.setattr(bootstrap_mod, "bootstrap_upstream_patches", spy)

    Neo4jMemoryMCPServer(MagicMock())

    spy.assert_called_once()


def test_bootstrap_error_propagates_from_server_construction(
    fresh_bootstrap, monkeypatch
):
    """A patch failure must abort server construction, not degrade silently."""
    from neo4j_agent_memory import MemoryClient

    from agent_memory_mcp.mcp.server import create_mcp_server

    monkeypatch.delattr(MemoryClient, "_create_embedder")

    with pytest.raises(BootstrapError):
        create_mcp_server()

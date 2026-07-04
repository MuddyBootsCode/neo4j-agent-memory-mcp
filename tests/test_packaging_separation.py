"""Tests for package namespace separation (R26).

``agent_memory_mcp`` must NOT shadow the upstream ``neo4j_agent_memory``
package: upstream resolves to site-packages, the local package resolves to
src/, and the upstream modules our patches target remain importable
unmodified.
"""

import os


def test_upstream_extraction_imports_work():
    """Upstream extraction modules must be importable (patch targets)."""
    from neo4j_agent_memory.extraction.base import (
        EntityExtractor,
        ExtractedEntity,
        ExtractionResult,
        NoOpExtractor,
    )

    assert EntityExtractor is not None
    assert ExtractedEntity is not None
    assert ExtractionResult is not None
    assert NoOpExtractor is not None


def test_upstream_factory_importable():
    """Upstream factory must be importable — the extractor patch targets it."""
    from neo4j_agent_memory.extraction.factory import create_extractor

    assert callable(create_extractor)


def test_upstream_not_shadowed_by_local_src():
    """Upstream neo4j_agent_memory must resolve to site-packages, not src/."""
    import agent_memory_mcp
    import neo4j_agent_memory

    upstream_dir = os.path.dirname(os.path.abspath(neo4j_agent_memory.__file__))
    local_dir = os.path.dirname(os.path.abspath(agent_memory_mcp.__file__))
    repo_src = os.path.dirname(local_dir)

    # Upstream must NOT live under this repo's src/ directory.
    assert not upstream_dir.startswith(repo_src + os.sep), (
        f"upstream neo4j_agent_memory resolved inside the repo: {upstream_dir}"
    )
    assert "site-packages" in upstream_dir, (
        f"expected upstream in site-packages, got {upstream_dir}"
    )
    # And the local package must live in src/, not site-packages.
    assert os.path.basename(repo_src) == "src", (
        f"expected agent_memory_mcp under src/, got {local_dir}"
    )

    # Single-path packages on both sides — no __path__ overlay tricks left.
    assert len(list(neo4j_agent_memory.__path__)) == 1
    assert len(list(agent_memory_mcp.__path__)) == 1


def test_local_extraction_modules_do_not_shadow_upstream():
    """agent_memory_mcp.extraction holds only the BAML extensions."""
    import agent_memory_mcp.extraction
    import neo4j_agent_memory.extraction as upstream_ext

    # Local additions importable under the new namespace.
    from agent_memory_mcp.extraction.reasoning_extractor import (  # noqa: F401
        BamlReasoningExtractor,
    )
    from agent_memory_mcp.extraction.unified import (  # noqa: F401
        UnifiedBamlExtractor,
    )

    # Upstream exports still come from upstream, untouched.
    assert hasattr(upstream_ext, "EntityExtractor")
    assert hasattr(upstream_ext, "ExtractionResult")
    assert hasattr(upstream_ext, "create_extractor")
    assert hasattr(upstream_ext, "ExtractionPipeline")

    # The local extraction package must not carry upstream's modules.
    local_dir = os.path.dirname(
        os.path.abspath(agent_memory_mcp.extraction.__file__)
    )
    assert not os.path.exists(os.path.join(local_dir, "factory.py"))
    assert not os.path.exists(os.path.join(local_dir, "base.py"))

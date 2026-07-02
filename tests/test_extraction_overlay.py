"""Tests for extraction subpackage overlay [RFI-I1].

Verifies that the overlay extraction __init__.py correctly extends
__path__ so both overlay modules and base package modules are importable.
"""



def test_base_extraction_imports_still_work():
    """Base package extraction modules must remain importable through overlay."""
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


def test_factory_importable_through_overlay():
    """Factory must be importable — monkey-patch depends on this."""
    from neo4j_agent_memory.extraction.factory import create_extractor

    assert callable(create_extractor)


def test_extraction_package_exports_preserved():
    """All __all__ exports from base extraction package must be available."""
    import neo4j_agent_memory.extraction as ext

    # Core exports that must exist
    assert hasattr(ext, "EntityExtractor")
    assert hasattr(ext, "ExtractedEntity")
    assert hasattr(ext, "ExtractionResult")
    assert hasattr(ext, "NoOpExtractor")
    assert hasattr(ext, "create_extractor")
    assert hasattr(ext, "ExtractionPipeline")


def test_overlay_path_has_both_dirs():
    """__path__ must contain both overlay and installed directories."""
    import neo4j_agent_memory.extraction as ext

    assert len(ext.__path__) >= 2, (
        f"Expected at least 2 entries in __path__, got {len(ext.__path__)}: {ext.__path__}"
    )

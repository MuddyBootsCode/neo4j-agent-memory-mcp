"""BAML-based extraction extensions.

Modules here extend (but do not shadow) the upstream
``neo4j_agent_memory.extraction`` package:

- :mod:`agent_memory_mcp.extraction.unified` — :class:`UnifiedBamlExtractor`,
  the upstream-compatible ``EntityExtractor`` used by the extractor factory
  patch (routes through the coding extraction, preferences only), plus
  ``persist_preferences``.
- :mod:`agent_memory_mcp.extraction.reasoning_extractor` — reasoning chain
  extraction and synthesis.
- :mod:`agent_memory_mcp.extraction.coding` — coding-session
  ``ExtractCodingMemory`` extraction with code-enforced anchor filtering.

Upstream extraction primitives (``EntityExtractor``, ``ExtractionResult``,
``create_extractor``, …) are imported explicitly from
``neo4j_agent_memory.extraction``.
"""

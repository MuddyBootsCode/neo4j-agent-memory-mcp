"""agent_memory_mcp — standalone MCP server for Neo4j Agent Memory.

This package builds on the upstream ``neo4j-agent-memory`` library (pinned in
pyproject.toml) and adds:

- ``mcp/``: a FastMCP-based MCP server (tools, resources, prompts, transports)
- ``extraction/``: unified single-pass BAML extraction + reasoning extraction
- ``temporal/``: fact lifecycle, contradiction detection, temporal queries
- ``baml_client/``: auto-generated BAML client (``uv run baml-cli generate``)

Unlike earlier revisions, this package does NOT share the upstream import
namespace. Upstream modules are imported explicitly, e.g.::

    from neo4j_agent_memory import MemoryClient, MemorySettings

and the pieces of upstream behaviour we extend (Bedrock embedder support,
BAML extractor factory) are applied as explicit, fail-loud monkeypatches via
:mod:`agent_memory_mcp.mcp._bootstrap`.
"""

__version__ = "0.1.0"

"""MCP (Model Context Protocol) server for Neo4j Agent Memory.

Exposes memory capabilities via MCP tools, resources, and prompts
for integration with AI platforms and Cloud API Registry.

Built on FastMCP for decorator-based tool/resource/prompt registration.
"""

from agent_memory_mcp.mcp.server import Neo4jMemoryMCPServer, create_mcp_server

__all__ = [
    "Neo4jMemoryMCPServer",
    "create_mcp_server",
]

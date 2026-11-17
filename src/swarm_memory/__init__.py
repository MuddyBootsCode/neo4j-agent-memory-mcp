"""Swarm memory, MUD-850. Phase 2: hooks that observe and log, and inject nothing.

Standard library only, with no model and no network, because the hook runs as a
new process on every tool call. The old Neo4j and MCP code in
``agent_memory_mcp`` is not used.
"""

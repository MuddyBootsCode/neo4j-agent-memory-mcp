#!/usr/bin/env python
"""Bootstrap script for the local FastMCP-based MCP server.

Adds src/ to sys.path so ``agent_memory_mcp`` is importable when running
straight from a checkout (without an editable install), then delegates to
the server's main() entry point. Equivalent to the installed
``neo4j-memory-mcp`` console script.
"""
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from agent_memory_mcp.mcp.server import main

if __name__ == "__main__":
    main()

"""Deterministic-plane capture for the coding memory graph.

Modules here build Cypher for graph writes driven by git and hooks — no LLM
involved:

- :mod:`agent_memory_mcp.capture.cypher` — pure ``(query, params)`` upsert
  builders for the deterministic plane (``CodeAgent``, ``CodingSession``,
  ``CodeFile``, ``Change``, ``WorkTask``) plus ``anchored_memory_write``,
  which links extracted-plane nodes to files, sessions, and tasks.

Builders return Cypher and parameters only; execution happens through the
server's ``client.graph.execute_write``.
"""

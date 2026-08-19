"""Deterministic-plane capture for the coding memory graph.

Modules here build Cypher for graph writes driven by git and hooks — no LLM
involved:

- :mod:`agent_memory_mcp.capture.cypher` — pure ``(query, params)`` upsert
  builders for the deterministic plane (``CodeAgent``, ``CodingSession``,
  ``CodeFile``, ``Change``, ``WorkTask``) plus ``anchored_memory_write``,
  which links extracted-plane nodes to files, sessions, and tasks.
- :mod:`agent_memory_mcp.capture.git_sweep` — fail-open git collectors
  (branch, repo name, edited files, recent commits, task-key inference)
  that feed the builders; every git failure degrades to an empty value
  because they run inside the prompt-submit hook.

Builders return Cypher and parameters only; execution happens through the
server's ``client.graph.execute_write``.
"""

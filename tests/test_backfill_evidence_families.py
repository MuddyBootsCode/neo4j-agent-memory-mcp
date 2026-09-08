"""The evidence-family backfill keeps every export it writes."""

from __future__ import annotations

import importlib.util
import os
import sys


def _load_script():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "backfill_evidence_families.py")
    spec = importlib.util.spec_from_file_location("backfill_evidence_families", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeGraph:
    def __init__(self, changed: int):
        self.writes = 0
        self._summary = {"lessons": changed, "changed": changed, "guardrails_old": changed,
                         "guardrails_new": 0, "max_old": 9, "max_new": 1}

    async def connect(self):
        pass

    async def close(self):
        pass

    async def execute_read(self, query, params=None):
        if "count(m) AS lessons" in query:
            return [self._summary]
        return [{"eid": f"4:x:{i}", "kind": "Gotcha", "old": 9, "new": 1, "text": "t"} for i in range(self._summary["changed"])]

    async def execute_write(self, query, params=None):
        self.writes += 1
        self.applied = list(params["rows"])
        return [{"n": len(params["rows"])}]


async def test_apply_refuses_to_overwrite_an_existing_export(tmp_path, monkeypatch):
    """A retry with the same --export path must not truncate the rollback
    record of batches that already committed (Codex F5)."""
    mod = _load_script()
    export = tmp_path / "out.jsonl"
    export.write_text('{"eid": "4:x:0", "kind": "Gotcha", "old": 114, "new": 2}\n')
    graph = FakeGraph(changed=1)

    import neo4j_agent_memory.graph.client as gc
    monkeypatch.setattr(gc, "Neo4jClient", lambda cfg: graph)
    monkeypatch.setattr(sys, "argv", ["backfill", "--apply", "--export", str(export)])

    rc = await mod.main()

    assert rc == 2
    assert graph.writes == 0
    assert export.read_text() == '{"eid": "4:x:0", "kind": "Gotcha", "old": 114, "new": 2}\n'


async def test_apply_writes_a_fresh_export_then_updates(tmp_path, monkeypatch):
    mod = _load_script()
    export = tmp_path / "fresh.jsonl"
    graph = FakeGraph(changed=2)

    import neo4j_agent_memory.graph.client as gc
    monkeypatch.setattr(gc, "Neo4jClient", lambda cfg: graph)
    monkeypatch.setattr(sys, "argv", ["backfill", "--apply", "--export", str(export)])

    # After the (fake) update the recount reports nothing left to change.
    calls = {"n": 0}
    real_read = graph.execute_read

    async def read(query, params=None):
        if "count(m) AS lessons" in query:
            calls["n"] += 1
            if calls["n"] > 1:
                return [{**graph._summary, "changed": 0}]
        return await real_read(query, params)

    graph.execute_read = read
    rc = await mod.main()

    assert rc == 0
    assert graph.writes == 1
    assert len(export.read_text().splitlines()) == 2
    # The update is bound to the exported rows and their old values (Codex F7).
    assert [r["eid"] for r in graph.applied] == ["4:x:0", "4:x:1"]
    assert all(r["old"] == 9 and r["new"] == 1 for r in graph.applied)


async def test_rows_that_drifted_after_export_are_left_alone_and_reported(tmp_path, monkeypatch, capsys):
    mod = _load_script()
    export = tmp_path / "drift.jsonl"
    graph = FakeGraph(changed=2)

    async def write(query, params=None):
        graph.writes += 1
        return [{"n": len(params["rows"]) - 1}]  # one row no longer matches its exported old value

    graph.execute_write = write
    import neo4j_agent_memory.graph.client as gc
    monkeypatch.setattr(gc, "Neo4jClient", lambda cfg: graph)
    monkeypatch.setattr(sys, "argv", ["backfill", "--apply", "--export", str(export)])

    rc = await mod.main()

    assert rc == 1
    assert "1 changed under us" in capsys.readouterr().out

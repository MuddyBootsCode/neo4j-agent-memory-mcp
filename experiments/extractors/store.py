"""Variant-scoped Neo4j subgraphs.

Every node written here carries the `:Exp` label and a `variant` property, and
every read filters on that property. Production `:Entity` / `:Preference` nodes
are never touched, and two variants can occupy the database at once so their
results are comparable without a reset between runs.

Labels: (:Exp:ExpEntity) and (:Exp:ExpPreference); edges (:EXP_RELATED).
"""

from __future__ import annotations

import os

from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")


def _password():
    pw = os.environ.get("NEO4J_PASSWORD")
    if pw:
        return pw
    env = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("NEO4J_PASSWORD="):
                v = line.split("=", 1)[1].strip().strip("\"'")
                if v:
                    return v
    return "graphmemory"


class SubGraph:
    def __init__(self, variant):
        self.variant = variant
        self._driver = GraphDatabase.driver(URI, auth=(USER, _password()))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def ensure_schema(self):
        with self._driver.session() as s:
            s.run(
                "CREATE INDEX exp_entity_key IF NOT EXISTS "
                "FOR (e:ExpEntity) ON (e.variant, e.name)"
            )
            s.run(
                "CREATE INDEX exp_pref_key IF NOT EXISTS "
                "FOR (p:ExpPreference) ON (p.variant)"
            )

    def reset(self):
        """Delete only this variant's subgraph."""
        with self._driver.session() as s:
            n = s.run(
                "MATCH (n:Exp {variant: $v}) DETACH DELETE n RETURN count(n) AS n",
                v=self.variant,
            ).single()["n"]
        return n

    def write(self, extraction, source_id):
        """Persist one extraction result into this variant's subgraph.

        Entities MERGE on (variant, name, type) so repeated mentions across
        chunks collapse the way the production dedup pipeline intends. The
        description of the first mention wins; later ones are kept as alternates
        so nothing is silently lost.
        """
        ents = extraction.get("entities") or []
        rels = extraction.get("relations") or []
        prefs = extraction.get("preferences") or []
        names = {e["name"].strip().lower() for e in ents if e.get("name")}

        with self._driver.session() as s:
            for e in ents:
                if not (e.get("name") or "").strip():
                    continue
                s.run(
                    """
                    MERGE (x:Exp:ExpEntity {variant: $v, key: $key})
                    ON CREATE SET x.name = $name, x.type = $type,
                                  x.description = $descr, x.subtype = $subtype,
                                  x.status = $status, x.date = $date,
                                  x.confidence = $conf, x.sources = [$src],
                                  x.alt_descriptions = []
                    ON MATCH  SET x.sources = x.sources + $src,
                                  x.alt_descriptions =
                                    CASE WHEN $descr IS NULL OR $descr = x.description
                                         THEN x.alt_descriptions
                                         ELSE x.alt_descriptions + $descr END
                    """,
                    v=self.variant,
                    key=f"{e['name'].strip().lower()}|{e.get('type', '')}",
                    name=e["name"].strip(),
                    type=e.get("type"),
                    descr=(e.get("description") or "").strip() or None,
                    subtype=e.get("subtype"),
                    status=e.get("status"),
                    date=e.get("date"),
                    conf=float(e.get("confidence") or 0.8),
                    src=source_id,
                )

            for r in rels:
                src_n, tgt_n = r.get("source", ""), r.get("target", "")
                # Drop invented edges, matching production behaviour.
                if src_n.strip().lower() not in names:
                    continue
                if tgt_n.strip().lower() not in names:
                    continue
                s.run(
                    """
                    MATCH (a:ExpEntity {variant: $v}), (b:ExpEntity {variant: $v})
                    WHERE toLower(a.name) = $a AND toLower(b.name) = $b
                    MERGE (a)-[e:EXP_RELATED {variant: $v, type: $t}]->(b)
                    ON CREATE SET e.confidence = $conf
                    """,
                    v=self.variant,
                    a=src_n.strip().lower(),
                    b=tgt_n.strip().lower(),
                    t=str(r.get("relation_type", "RELATED")).upper(),
                    conf=float(r.get("confidence") or 0.7),
                )

            for p in prefs:
                if not (p.get("preference") or "").strip():
                    continue
                s.run(
                    """
                    MERGE (x:Exp:ExpPreference {variant: $v, preference: $pref})
                    ON CREATE SET x.category = $cat, x.context = $ctx,
                                  x.confidence = $conf, x.sources = [$src]
                    ON MATCH  SET x.sources = x.sources + $src
                    """,
                    v=self.variant,
                    pref=p["preference"].strip(),
                    cat=p.get("category"),
                    ctx=p.get("context"),
                    conf=float(p.get("confidence") or 0.7),
                    src=source_id,
                )

        return len(ents), len(rels), len(prefs)

    def records(self):
        """Retrievable records for this variant, as (kind, name, description)."""
        with self._driver.session() as s:
            ents = [
                dict(r)
                for r in s.run(
                    "MATCH (e:ExpEntity {variant: $v}) "
                    "RETURN e.name AS name, e.type AS type, "
                    "e.description AS description ORDER BY e.name",
                    v=self.variant,
                )
            ]
            prefs = [
                dict(r)
                for r in s.run(
                    "MATCH (p:ExpPreference {variant: $v}) "
                    "RETURN p.category AS category, p.preference AS preference, "
                    "p.context AS context ORDER BY p.preference",
                    v=self.variant,
                )
            ]
        return ents, prefs

    def stats(self):
        with self._driver.session() as s:
            row = s.run(
                """
                MATCH (e:ExpEntity {variant: $v})
                WITH count(e) AS entities,
                     count(e.description) AS described
                MATCH (p:ExpPreference {variant: $v})
                WITH entities, described, count(p) AS preferences
                OPTIONAL MATCH ()-[r:EXP_RELATED {variant: $v}]->()
                RETURN entities, described, preferences, count(r) AS relations
                """,
                v=self.variant,
            ).single()
            types = [
                (r["type"], r["n"])
                for r in s.run(
                    "MATCH (e:ExpEntity {variant: $v}) RETURN e.type AS type, "
                    "count(*) AS n ORDER BY n DESC",
                    v=self.variant,
                )
            ]
        return (dict(row) if row else {}), types


def all_variants_present():
    g = SubGraph("_probe")
    try:
        with g._driver.session() as s:
            return [
                r["v"]
                for r in s.run(
                    "MATCH (n:Exp) RETURN DISTINCT n.variant AS v ORDER BY v"
                )
            ]
    finally:
        g.close()

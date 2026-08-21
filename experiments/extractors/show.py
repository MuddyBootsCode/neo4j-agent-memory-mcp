"""Show what each variant stored, side by side.

Named show.py, not inspect.py: this directory goes on sys.path[0], so a module
named inspect shadows the stdlib one that neo4j imports transitively.

The numbers in evaluate.py say which variant retrieves better; this says why,
by showing the text that variant actually put in the graph.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from store import SubGraph, all_variants_present
from variants import VARIANTS, embed_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", dest="variants")
    ap.add_argument("--samples", type=int, default=10)
    a = ap.parse_args()

    present = a.variants or [v for v in all_variants_present() if v in VARIANTS]
    if not present:
        print("no variant subgraphs found — run extract.py first")
        return

    for name in present:
        with SubGraph(name) as g:
            stats, types = g.stats()
            ents, prefs = g.records()
            print("=" * 76)
            print(f"{name}  —  {VARIANTS[name]['blurb']}")
            print(f"  {stats}")
            print(f"  types: {types[:6]}")

            described = [e for e in ents if e.get("description")]
            print(f"\n  entities with a description: {len(described)}/{len(ents)}")
            print("  --- text as the retriever sees it ---")
            for e in ents[: a.samples]:
                print(f"    {embed_text(name, e)[:150]}")
            if prefs:
                print("  --- preferences ---")
                for p in prefs[:3]:
                    print(f"    [{p['category']}] {p['preference'][:110]}")
            print()


if __name__ == "__main__":
    main()

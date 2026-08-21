"""Re-parse traces whose raw response my parser rejected, and store them.

Costs nothing: the model output is already on disk in results/traces/. Only
chunks recorded as json_ok=false are written, so successful chunks are not
double-counted in each node's `sources` list.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baml_runtime as br
from store import SubGraph
from variants import VARIANTS

for variant in VARIANTS:
    path = br.trace_path(variant)
    if not os.path.exists(path):
        continue
    rows = [json.loads(l) for l in open(path)]
    bad = [r for r in rows if r.get("ok") and not r.get("json_ok")]
    if not bad:
        print(f"{variant:18} nothing to repair")
        continue
    fixed = failed = 0
    with SubGraph(variant) as g:
        for r in bad:
            try:
                parsed = br.parse_raw(r.get("raw_response"))
            except Exception as exc:
                failed += 1
                print(f"  {variant} {r['chunk']}: still unparseable — {str(exc)[:80]}")
                continue
            for key in ("entities", "relations", "preferences"):
                parsed.setdefault(key, [])
            g.write(parsed, r["chunk"])
            fixed += 1
        stats, _ = g.stats()
    print(f"{variant:18} repaired {fixed}/{len(bad)} (still failing {failed}) -> {stats}")

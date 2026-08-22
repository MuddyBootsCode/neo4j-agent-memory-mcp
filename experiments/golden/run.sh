#!/usr/bin/env bash
# Full golden run. Env: GOLDEN_RUN (results dir name), GOLDEN_REPOS, GOLDEN_DB,
# GOLDEN_REUSE_DB / GOLDEN_REUSE_SPLIT (skip extraction), GOLDEN_MAX_QUERIES.
set -euo pipefail
cd "$(dirname "$0")"
for s in step1_corpus step2_pool step3_queries step4_label step5_score step6_teardown; do
    echo "=== $s"
    uv run --no-sync --project ../.. python -u "$s.py"
done

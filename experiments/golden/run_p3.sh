#!/usr/bin/env bash
# Two retrieval runs on the p1-capture pool and labels: MiniLM + fusion, then bge-base + fusion.
G=/Users/muddybootscode/Projects/neo4j-agent-memory-mcp/experiments/golden
cd $G
export GOLDEN_REPOS=~/Projects/gradgraph-auth-platform GOLDEN_DB=goldencorpus GOLDEN_POOL_FROM=results/p1-capture/pool.json
export GOLDEN_CODE_REF=$(git -C $G rev-parse --short HEAD)
run() {
  local name=$1; shift
  export GOLDEN_RUN=$name "$@"
  local L=$G/results/$name/run.log
  for s in step1b_materialize step5_score step6_teardown; do
    echo "=== $s" >> $L
    uv run --no-sync --project ../.. python -u $G/$s.py >> $L 2>&1 || { echo "exit=$? at $s" >> $L; return 1; }
  done
  echo "run done" >> $L
}
run p3-bm25 NAM_EMBEDDING_MODEL=all-MiniLM-L6-v2 NAM_EMBEDDING_DIMENSIONS=384 || exit 1
run p3-bge NAM_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 NAM_EMBEDDING_DIMENSIONS=768 || exit 1
echo "all done" >> $G/results/p3-bge/run.log

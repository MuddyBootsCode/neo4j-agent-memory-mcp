#!/usr/bin/env bash
G=/Users/muddybootscode/Projects/neo4j-agent-memory-mcp/experiments/golden
cd $G
export GOLDEN_RUN=p1-capture GOLDEN_REPOS=~/Projects/gradgraph-auth-platform GOLDEN_DB=goldencorpus GOLDEN_REUSE_SPLIT=$G/results/baseline/session_split.json
for s in step1_corpus step2_pool step3_queries; do
  uv run --no-sync --project ../.. python -u $G/$s.py >> $G/results/p1-capture/build.log 2>&1 || { echo "exit=$? at $s" >> $G/results/p1-capture/build.log; exit 1; }
done
echo "build done" >> $G/results/p1-capture/build.log

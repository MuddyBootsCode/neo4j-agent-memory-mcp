// Post-retrieval re-ranking for multi-database fan-out results

class ResultItem {
  id string @description("Result ID")
  content string @description("The result content or summary")
  source_db string @description("Which database this came from")
  result_type string @description("Type: message, entity, preference, trace")
}

class ScoredResult {
  id string @description("Result ID (pass through from input)")
  relevance float @description("Relevance to the original query, 0.0 to 1.0")
  keep bool @description("True if this result is relevant enough to show to the user")
  reasoning string @description("Brief explanation of relevance assessment")
}

class RerankOutput {
  scored_results ScoredResult[] @description("Results scored and filtered, ordered by relevance descending")
  total_input int @description("Number of results received")
  total_kept int @description("Number of results kept after filtering")
}

function RerankResults(query: string, results: ResultItem[]) -> RerankOutput {
  client Bedrock
  prompt #"
    You are a relevance judge. Score each search result for how well it answers
    the user's original query.

    Rules:
    - Score 0.0-1.0 based on direct relevance to the query
    - Set keep=true only for results with relevance >= 0.4
    - A result that partially answers the query is still relevant
    - A result that is tangentially related but not useful should score low
    - Be aggressive about filtering noise — better to show 3 great results than 10 mediocre ones

    ## Original Query
    {{ query }}

    ## Results to Score
    {% for r in results %}
    ### Result {{ loop.index }} (id: {{ r.id }}, from: {{ r.source_db }}, type: {{ r.result_type }})
    {{ r.content }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

// ═══════════════════════════════════════════════════════════════
// Category 5: Reranking (4 tests)
// Validates post-retrieval relevance scoring and filtering
// ═══════════════════════════════════════════════════════════════

test Rerank_HighRelevance {
  functions [RerankResults]
  args {
    query "What actions came out of Monday's standup?"
    results [
      {
        id "1"
        content "Monday standup: Sarah to finish API docs by Wednesday. Marcus to review PR #42. Priya to set up the embedding pipeline staging environment."
        source_db "meetings"
        result_type "message"
      },
      {
        id "2"
        content "Monday standup action items: 1) Sarah - API docs by Wed 2) Marcus - review PR #42 3) Priya - staging env setup"
        source_db "meetings"
        result_type "message"
      }
    ]
  }
  // Expected: Both results score >0.8, keep=true for both
  // Direct match to query about Monday standup actions
}

test Rerank_LowRelevance {
  functions [RerankResults]
  args {
    query "What's Raj's hotel preference?"
    results [
      {
        id "1"
        content "Raj Patel is the CTO of DataVault Solutions based in Denver."
        source_db "neo4j"
        result_type "entity"
      },
      {
        id "2"
        content "DataVault Solutions was founded in 2018 and specializes in data migration."
        source_db "neo4j"
        result_type "entity"
      },
      {
        id "3"
        content "The Denver office is located on Larimer Street."
        source_db "neo4j"
        result_type "entity"
      }
    ]
  }
  // Expected: All results score <0.4, keep=false for all
  // None mention hotel preferences — tangentially related to Raj at best
}

test Rerank_MixedRelevance {
  functions [RerankResults]
  args {
    query "What actions came out of Monday's standup?"
    results [
      {
        id "1"
        content "Monday standup: Sarah to finish API docs by Wednesday. Marcus to review PR #42."
        source_db "meetings"
        result_type "message"
      },
      {
        id "2"
        content "Sarah mentioned she prefers morning meetings over afternoon ones."
        source_db "meetings"
        result_type "preference"
      },
      {
        id "3"
        content "The API documentation project milestone is due Friday."
        source_db "projects"
        result_type "message"
      },
      {
        id "4"
        content "Priya's research on embedding models shows BERT outperforms word2vec for our use case."
        source_db "research"
        result_type "message"
      }
    ]
  }
  // Expected: id=1 scores high (>0.8, keep=true), id=3 moderate (~0.5, keep=true),
  // id=2 low (<0.4, keep=false), id=4 very low (<0.3, keep=false)
  // Should filter noise while keeping partially relevant results
}

test Rerank_NoRelevant {
  functions [RerankResults]
  args {
    query "What's the weather forecast for Austin?"
    results [
      {
        id "1"
        content "Marcus drives up from San Antonio on Thursday mornings."
        source_db "neo4j"
        result_type "message"
      },
      {
        id "2"
        content "The kickoff meeting is at the Sixth Street office in Austin."
        source_db "meetings"
        result_type "message"
      },
      {
        id "3"
        content "Raj's Tesla Model 3 needs the parking garage on 7th for the chargers."
        source_db "neo4j"
        result_type "message"
      }
    ]
  }
  // Expected: All results score <0.4, keep=false, total_kept=0
  // Weather forecasts are outside the knowledge graph's scope entirely
}

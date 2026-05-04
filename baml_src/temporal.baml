// Temporal memory — contradiction detection and temporal extraction

class CandidateFact {
  idx int @description("Index of this fact in the candidates list")
  subject string
  predicate string
  object string
  confidence float
}

class ContradictionResult {
  contradicted_indices int[] @description("Indices of candidate facts that are contradicted by the new fact. Empty if no contradictions.")
  contradiction_type string @description("Type: 'direct_supersession' (same subject, updated value), 'negation' (opposite claim), 'refinement' (more specific version), or 'none'")
  reasoning string @description("Brief explanation of why these facts are contradicted")
}

function DetectContradictions(
  new_fact_subject: string,
  new_fact_predicate: string,
  new_fact_object: string,
  candidates: CandidateFact[]
) -> ContradictionResult {
  client Bedrock
  prompt #"
    You are analyzing whether a NEW FACT contradicts any EXISTING FACTS in a knowledge graph.

    ## NEW FACT
    Subject: {{ new_fact_subject }}
    Predicate: {{ new_fact_predicate }}
    Object: {{ new_fact_object }}

    ## EXISTING CANDIDATE FACTS
    {% for candidate in candidates %}
    [{{ candidate.idx }}] {{ candidate.subject }} —{{ candidate.predicate }}→ {{ candidate.object }} (confidence: {{ candidate.confidence }})
    {% endfor %}

    ## Rules
    - A fact is CONTRADICTED if the new fact makes it no longer true
    - "Alice WORKS_AT Acme" is contradicted by "Alice WORKS_AT Globex" (direct supersession — same subject+predicate, different object)
    - "Project status IS active" is contradicted by "Project status IS completed" (direct supersession)
    - "Alice LIVES_IN Denver" is NOT contradicted by "Alice VISITED Seattle" (different predicates, both can be true)
    - If the new fact is simply more specific (e.g., "Alice WORKS_AT Acme Engineering Division" vs "Alice WORKS_AT Acme"), this is a refinement
    - When in doubt, do NOT mark as contradicted — false negatives are better than false positives

    Return the indices of contradicted facts, the type, and your reasoning.

    {{ ctx.output_format }}
  "#
}

class TemporalExtraction {
  valid_at string? @description("Extracted datetime when fact became true, in ISO 8601. E.g., '2026-03-01T00:00:00Z'. null if not determinable.")
  temporal_qualifier string? @description("Temporal qualifier: 'since', 'until', 'as_of', 'formerly', 'currently', or null")
  is_current_state bool @description("True if this describes the current state of affairs, false if explicitly past tense")
}

function ExtractTemporalContext(
  text: string,
  reference_time: string
) -> TemporalExtraction {
  client Bedrock
  prompt #"
    Extract temporal information from this text. The current reference time is {{ reference_time }}.

    ## Text
    {{ text }}

    ## Rules
    - Convert relative references ("last week", "two months ago") to absolute ISO 8601 dates based on the reference time
    - "since March" → valid_at = "2026-03-01T00:00:00Z", qualifier = "since"
    - "as of yesterday" → valid_at = computed from reference_time, qualifier = "as_of"
    - "used to work at" → is_current_state = false, qualifier = "formerly"
    - "currently manages" → is_current_state = true, qualifier = "currently"
    - If no temporal context is present, return null for valid_at and qualifier, true for is_current_state

    {{ ctx.output_format }}
  "#
}

// Test cases
test DirectSupersession {
  functions [DetectContradictions]
  args {
    new_fact_subject "Alice"
    new_fact_predicate "WORKS_AT"
    new_fact_object "Globex Corporation"
    candidates [
      { idx 0, subject "Alice", predicate "WORKS_AT", object "Acme Inc", confidence 0.95 },
      { idx 1, subject "Alice", predicate "LIVES_IN", object "Denver", confidence 0.9 }
    ]
  }
}

test NoContradiction {
  functions [DetectContradictions]
  args {
    new_fact_subject "Alice"
    new_fact_predicate "VISITED"
    new_fact_object "Seattle"
    candidates [
      { idx 0, subject "Alice", predicate "WORKS_AT", object "Acme Inc", confidence 0.95 },
      { idx 1, subject "Alice", predicate "LIVES_IN", object "Denver", confidence 0.9 }
    ]
  }
}

test TemporalSince {
  functions [ExtractTemporalContext]
  args {
    text "Alice has been working at Globex since March 2026"
    reference_time "2026-03-09T12:00:00Z"
  }
}

test TemporalFormer {
  functions [ExtractTemporalContext]
  args {
    text "Bob used to manage the infrastructure team but left last month"
    reference_time "2026-03-09T12:00:00Z"
  }
}

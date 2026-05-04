// Research vertical ontology

enum ResearchEntityType {
  NOTE @description("A research note, observation, or write-up")
  FINDING @description("A research finding, result, or conclusion")
  SOURCE @description("A source: paper, article, book, dataset, URL")
  TOPIC @description("A research topic, theme, or area of investigation")
  EXPERIMENT @description("An experiment, test, or trial")
}

class ResearchEntity {
  name string @description("Entity name as it appears in text")
  type ResearchEntityType
  status string? @description("Status: draft, validated, refuted, in_progress")
  confidence float @description("Extraction confidence 0.0 to 1.0")
}

class ResearchRelation {
  source string @description("Source entity name")
  target string @description("Target entity name")
  relation_type string @description("Relationship: CITES, SUPPORTS, CONTRADICTS, BUILDS_ON, EXPLORES, PRODUCED_BY, RELATED_TOPIC, VALIDATES")
  confidence float
}

class ResearchExtractionOutput {
  entities ResearchEntity[]
  relations ResearchRelation[]
}

function ExtractResearchEntities(text: string) -> ResearchExtractionOutput {
  client Bedrock
  prompt #"
    Extract research-related entities and relationships from the text.

    Focus on: research notes, findings, sources, topics, and experiments.

    For relations, use types like:
    CITES, SUPPORTS, CONTRADICTS, BUILDS_ON,
    EXPLORES, PRODUCED_BY, RELATED_TOPIC, VALIDATES

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ text }}
  "#
}

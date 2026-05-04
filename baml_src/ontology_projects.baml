// Projects vertical ontology

enum ProjectEntityType {
  PROJECT @description("A project, initiative, or workstream")
  TASK @description("A task, ticket, or work item within a project")
  MILESTONE @description("A project milestone, deadline, or checkpoint")
  DELIVERABLE @description("A deliverable, artifact, or output of project work")
  TEAM @description("A team or group working on a project")
}

class ProjectEntity {
  name string @description("Entity name as it appears in text")
  type ProjectEntityType
  status string? @description("Status: active, completed, blocked, planned, cancelled")
  priority string? @description("Priority: critical, high, medium, low")
  confidence float @description("Extraction confidence 0.0 to 1.0")
}

class ProjectRelation {
  source string @description("Source entity name")
  target string @description("Target entity name")
  relation_type string @description("Relationship: DEPENDS_ON, ASSIGNED_TO, BLOCKED_BY, DELIVERS, PART_OF, OWNS, CONTRIBUTES_TO, TRACKS")
  confidence float
}

class ProjectExtractionOutput {
  entities ProjectEntity[]
  relations ProjectRelation[]
}

function ExtractProjectEntities(text: string) -> ProjectExtractionOutput {
  client Bedrock
  prompt #"
    Extract project-related entities and relationships from the text.

    Focus on: projects, tasks, milestones, deliverables, and teams.

    For relations, use types like:
    DEPENDS_ON, ASSIGNED_TO, BLOCKED_BY, DELIVERS,
    PART_OF, OWNS, CONTRIBUTES_TO, TRACKS

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ text }}
  "#
}

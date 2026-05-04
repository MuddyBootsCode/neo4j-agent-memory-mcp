// Meetings vertical ontology

enum MeetingEntityType {
  MEETING @description("A meeting event: standup, sync, 1:1, kickoff, retrospective, review")
  ATTENDEE @description("A person who attended or was invited to a meeting")
  AGENDA_ITEM @description("A topic or item on the meeting agenda")
  ACTION_ITEM @description("A task or follow-up assigned during a meeting")
  DECISION @description("A decision made during a meeting")
}

class MeetingEntity {
  name string @description("Entity name as it appears in text")
  type MeetingEntityType
  date string? @description("Date/time if mentioned, ISO format preferred")
  status string? @description("Status: scheduled, completed, cancelled, recurring")
  confidence float @description("Extraction confidence 0.0 to 1.0")
}

class MeetingRelation {
  source string @description("Source entity name")
  target string @description("Target entity name")
  relation_type string @description("Relationship: ATTENDED, PRESENTED, ASSIGNED_TO, DISCUSSED, FOLLOW_UP, DECIDED_IN, SCHEDULED_FOR, BLOCKED_BY")
  confidence float
}

class MeetingExtractionOutput {
  entities MeetingEntity[]
  relations MeetingRelation[]
}

function ExtractMeetingEntities(text: string) -> MeetingExtractionOutput {
  client Bedrock
  prompt #"
    Extract meeting-related entities and relationships from the text.

    Focus on: meetings, attendees, agenda items, action items, and decisions.

    For relations, use types like:
    ATTENDED, PRESENTED, ASSIGNED_TO, DISCUSSED, FOLLOW_UP,
    DECIDED_IN, SCHEDULED_FOR, BLOCKED_BY

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ text }}
  "#
}

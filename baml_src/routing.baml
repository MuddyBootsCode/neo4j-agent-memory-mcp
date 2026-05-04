// Database vertical classification for multi-DB routing

enum QueryVertical {
  MEETINGS @alias("k1")
    @description("Meetings, calendars, scheduling, attendees, agendas, standups, syncs, 1:1s, action items from meetings, meeting notes")
  PROJECTS @alias("k2")
    @description("Projects, tasks, milestones, deliverables, sprints, backlogs, dependencies, team assignments, project status, deadlines")
  RESEARCH @alias("k3")
    @description("Research notes, papers, findings, knowledge base, citations, experiments, hypotheses, literature, sources, analysis results")
  GENERAL @alias("k4")
    @description("General memory, personal preferences, facts, people, organizations, locations, conversation history, reasoning traces, anything that doesn't clearly fit other categories")
}

class RoutingTarget {
  vertical QueryVertical @description("The database vertical to query")
  confidence float @description("Confidence this vertical is relevant, 0.0 to 1.0")
  reasoning string @description("Brief explanation of why this vertical is relevant")
}

class RoutingDecision {
  targets RoutingTarget[] @description("Verticals to query, ordered by confidence descending. Only include verticals with confidence > 0.3")
  primary_vertical QueryVertical @description("The single most relevant vertical")
  requires_fanout bool @description("True if multiple verticals must be queried to fully answer the query")
  ambiguous bool @description("True if the query is too vague to route confidently. Set true when no vertical has confidence > 0.6 or the query lacks specificity")
  disambiguation_options string[]? @description("If ambiguous, provide 2-3 concrete interpretations of what the user might mean, each mentioning which vertical it would search. Only populate when ambiguous is true.")
}

function RouteQuery(query: string, context: string?) -> RoutingDecision {
  client Bedrock
  prompt #"
    You are a query router for a knowledge graph with multiple database verticals.
    Analyze the user query and determine which database(s) should be searched.

    Rules:
    - Set requires_fanout to true ONLY when the query genuinely spans multiple domains
    - Confidence reflects how certain you are the vertical contains relevant data
    - Only include verticals with confidence > 0.3
    - GENERAL is the fallback when no specific vertical matches
    - Most queries should target 1-2 verticals, rarely more
    - Set ambiguous to true when the query is vague, lacks context, or could
      reasonably apply to multiple unrelated verticals (e.g., "what did we talk about?",
      "anything new?", "what did Sarah say?")
    - When ambiguous, provide 2-3 specific interpretations the user might mean,
      each mentioning the vertical it would search

    {% if context %}
    ## Recent Context
    {{ context }}
    {% endif %}

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ query }}
  "#
}

function RouteStorage(content: string, memory_type: string, context: string?) -> RoutingDecision {
  client Bedrock
  prompt #"
    You are a storage router for a knowledge graph. Determine which database
    vertical should store this content.

    For storage, pick ONE primary vertical. Only set requires_fanout to true
    if the content genuinely belongs in multiple databases (rare).

    Memory type being stored: {{ memory_type }}

    {% if context %}
    ## Context
    {{ context }}
    {% endif %}

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ content }}
  "#
}

// ═══════════════════════════════════════════════════════════════
// Category 1: Vertical Identification (8 tests)
// Validates that clear queries route to the correct single vertical
// ═══════════════════════════════════════════════════════════════

test MeetingQuery_Clear {
  functions [RouteQuery]
  args {
    query "What was discussed in yesterday's standup?"
    context null
  }
  // Expected: primary_vertical=MEETINGS, ambiguous=false
}

test ProjectQuery_Clear {
  functions [RouteQuery]
  args {
    query "What's the status of the migration project?"
    context null
  }
  // Expected: primary_vertical=PROJECTS, ambiguous=false
}

test ResearchQuery_Clear {
  functions [RouteQuery]
  args {
    query "What papers did we cite in the NLP analysis?"
    context null
  }
  // Expected: primary_vertical=RESEARCH, ambiguous=false
}

test GeneralQuery_Clear {
  functions [RouteQuery]
  args {
    query "What's Marcus's email address?"
    context null
  }
  // Expected: primary_vertical=GENERAL, ambiguous=false
}

test MeetingQuery_Implicit {
  functions [RouteQuery]
  args {
    query "Who attended the sync with DataVault?"
    context null
  }
  // Expected: primary_vertical=MEETINGS, ambiguous=false
  // "sync" and "attended" are strong meeting signals
}

test ProjectQuery_Implicit {
  functions [RouteQuery]
  args {
    query "Which tasks are blocked right now?"
    context null
  }
  // Expected: primary_vertical=PROJECTS, ambiguous=false
  // "tasks" and "blocked" are strong project signals
}

test ResearchQuery_Implicit {
  functions [RouteQuery]
  args {
    query "What experiments validated the embedding approach?"
    context null
  }
  // Expected: primary_vertical=RESEARCH, ambiguous=false
  // "experiments" and "validated" are strong research signals
}

test GeneralQuery_Preference {
  functions [RouteQuery]
  args {
    query "Does Raj prefer Hyatt or Marriott?"
    context null
  }
  // Expected: primary_vertical=GENERAL, ambiguous=false
  // Personal preferences belong in general memory
}

// ═══════════════════════════════════════════════════════════════
// Category 2: Storage Routing (5 tests)
// Validates that content is routed to the correct vertical for storage
// ═══════════════════════════════════════════════════════════════

test StoreMeeting {
  functions [RouteStorage]
  args {
    content "Team standup 3/5: Sarah demoed the new dashboard. Marcus raised a blocker on the API integration. Action item: Sarah to send API specs to Marcus by EOD Thursday."
    memory_type "message"
    context null
  }
  // Expected: primary_vertical=MEETINGS
}

test StoreProject {
  functions [RouteStorage]
  args {
    content "Sprint 12 update: Migration task moved to 'in progress'. Milestone 2 deadline pushed to March 15. The embedding pipeline deliverable is 80% complete. Blocked by: AWS credentials from DataVault."
    memory_type "message"
    context null
  }
  // Expected: primary_vertical=PROJECTS
}

test StoreResearch {
  functions [RouteStorage]
  args {
    content "Literature review: Chen et al. 2024 found that graph-based RAG outperforms vector-only retrieval by 23% on multi-hop questions. This supports our hypothesis that knowledge graphs improve agent memory. Need to replicate experiment with our dataset."
    memory_type "message"
    context null
  }
  // Expected: primary_vertical=RESEARCH
}

test StoreGeneral {
  functions [RouteStorage]
  args {
    content "Raj Patel is the CTO of DataVault Solutions. He's vegetarian and prefers Hyatt hotels. His email is raj.patel@datavaultsolutions.com."
    memory_type "message"
    context null
  }
  // Expected: primary_vertical=GENERAL
  // Personal facts and preferences belong in general memory
}

test StoreAmbiguousContent {
  functions [RouteStorage]
  args {
    content "Talked to the team today about next steps. Things are going well overall."
    memory_type "message"
    context null
  }
  // Expected: primary_vertical=GENERAL (fallback for vague content)
  // Storage should always pick ONE primary, never be ambiguous
}

// ═══════════════════════════════════════════════════════════════
// Category 3: Ambiguity Detection (4 tests)
// Validates that vague queries trigger disambiguation
// ═══════════════════════════════════════════════════════════════

test AmbiguousQuery_Vague {
  functions [RouteQuery]
  args {
    query "anything new?"
    context null
  }
  // Expected: ambiguous=true, disambiguation_options populated
  // Too vague to route confidently to any vertical
}

test AmbiguousQuery_PersonRef {
  functions [RouteQuery]
  args {
    query "What did Sarah say?"
    context null
  }
  // Expected: ambiguous=true
  // Sarah could have said something in a meeting, project update, or research context
}

test AmbiguousQuery_WithContext {
  functions [RouteQuery]
  args {
    query "What did Sarah say?"
    context "Sarah presented research findings at Monday's team meeting about the migration project"
  }
  // Expected: ambiguous=false, requires_fanout likely true
  // Context disambiguates — mentions meeting, research, and project
}

test AmbiguousQuery_Temporal {
  functions [RouteQuery]
  args {
    query "what happened last week?"
    context null
  }
  // Expected: ambiguous=true, disambiguation_options populated
  // Pure temporal reference without domain specificity
}

// ═══════════════════════════════════════════════════════════════
// Category 4: Cross-Domain / Fanout (4 tests)
// Validates that multi-domain queries correctly trigger fanout
// ═══════════════════════════════════════════════════════════════

test CrossDomain_MeetingResearch {
  functions [RouteQuery]
  args {
    query "What research findings were presented in the project kickoff meeting?"
    context null
  }
  // Expected: requires_fanout=true, targets include MEETINGS and RESEARCH
  // Explicitly references both meeting context and research content
}

test CrossDomain_ProjectMeeting {
  functions [RouteQuery]
  args {
    query "What action items came out of the sprint planning meeting?"
    context null
  }
  // Expected: targets include MEETINGS (and possibly PROJECTS)
  // "action items" + "sprint planning" + "meeting" spans meetings/projects
}

test CrossDomain_AllVerticals {
  functions [RouteQuery]
  args {
    query "Give me everything we know about the DataVault migration"
    context null
  }
  // Expected: requires_fanout=true, multiple targets
  // "everything" is a strong fanout signal across all verticals
}

test CrossDomain_ResearchProject {
  functions [RouteQuery]
  args {
    query "Which experiments are blocking the ML pipeline deliverable?"
    context null
  }
  // Expected: requires_fanout=true, targets include RESEARCH and PROJECTS
  // "experiments" → RESEARCH, "deliverable" + "blocking" → PROJECTS
}

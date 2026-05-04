// POLE+O Entity Types
enum EntityType {
  PERSON @description("Individuals mentioned by name. DO NOT extract unnamed roles ('the engineer'), collective nouns ('the team', 'staff'), or pronouns")
  ORGANIZATION @description("Companies, institutions, named groups. DO NOT extract departments ('engineering team'), project names ('migration project'), or informal groups")
  LOCATION @description("Geographic places, named addresses, landmarks. DO NOT extract relative positions ('the office') unless preceded by a proper place name")
  EVENT @description("Specific incidents or calendar events with a date or proper name. DO NOT extract: meetings, calls, reviews, tasks, sprints, deployments, migrations, integrations, or any process noun")
  OBJECT @description("Branded or model-specific items only. DO NOT extract: pipelines, frameworks, approaches, embeddings, APIs, systems, databases, documents, tools, or any technical concept")
}

class ExtractedEntity {
  name string @description("The entity name as it appears in text")
  type EntityType
  subtype string? @description("Optional specific subtype, e.g. VEHICLE, ADDRESS, COMPANY")
  confidence float @description("Extraction confidence from 0.0 to 1.0")
}

class ExtractedRelation {
  source string @description("Source entity name")
  target string @description("Target entity name")
  relation_type string @description("Relationship type, e.g. WORKS_AT, LIVES_IN, OWNS")
  confidence float @description("Relation confidence from 0.0 to 1.0")
}

class ExtractedPreference {
  category string @description("Preference category: food, music, tech, communication, etc.")
  preference string @description("The preference statement")
  context string? @description("When/where this preference applies")
  confidence float @description("Preference confidence from 0.0 to 1.0")
}

class ExtractionOutput {
  entities ExtractedEntity[]
  relations ExtractedRelation[]
  preferences ExtractedPreference[]
}

function ExtractEntities(text: string, entity_types: string) -> ExtractionOutput {
  client Bedrock
  prompt #"
    Extract entities, relationships, and preferences from the following text.

    ## Entity Types (POLE+O Model)
    Extract entities of these types: {{ entity_types }}

    For relations:
    - Identify how entities are connected
    - Use clear relationship types (WORKS_AT, LIVES_IN, OWNS, ATTENDED, KNOWS, etc.)
    - Only include relations between entities in the entities list

    For preferences:
    - User preferences, likes, dislikes, opinions
    - Categories: food, music, communication, style, technology, etc.

    Confidence: 0.0-1.0 based on certainty of extraction

    ## Text to Analyze
    {{ text }}

    {{ ctx.output_format }}
  "#
}

test ProjectKickoffConversation {
  functions [ExtractEntities]
  args {
    entity_types "PERSON, ORGANIZATION, LOCATION, EVENT, OBJECT"
    text #"
      Sarah Chen: Hey Marcus, just got off the call with DataVault Solutions out of Denver. Their CTO, Raj Patel, wants to move forward with the graph migration project. He's flying into Austin next Thursday the 27th for the kickoff meeting at our Sixth Street office.

      Marcus Webb: Great news. Did he mention which systems they're migrating? Last I heard they were running Oracle on an EC2 cluster in us-east-1.

      Sarah Chen: Yeah, Oracle and a legacy SQL Server instance. They also have a Snowflake warehouse they want to keep in place but connect to the graph. Raj said they need everything running before their Q2 board presentation in Chicago on April 15th.

      Marcus Webb: That's tight. I'll need to loop in our ML engineer, Priya Okonkwo — she's been handling the embedding pipeline work out of our Seattle office. Also going to need access to their AWS environment. Can you get us credentials for the DataVault dev account?

      Sarah Chen: Already requested. Raj said his infrastructure lead, Tom Briggs, will send over the IAM role ARN by EOD Friday. Tom's based at their headquarters on Larimer Street in Denver.

      Marcus Webb: Perfect. I'll drive up from my place in San Antonio Thursday morning. I prefer early meetings — before 10am if possible — so I'm not fighting I-35 traffic both ways.

      Sarah Chen: Noted. I'll book the Hyatt Regency on Congress Ave for Raj — he mentioned he prefers Hyatt properties over Marriott when traveling for work. Also he's vegetarian, heads up if we're doing a working lunch.

      Marcus Webb: Good to know. I'll have my Surface Pro and the portable 4K display for the demo. Should we use the Miro board we set up for the WellWise project, or start fresh?

      Sarah Chen: Start fresh — I'll spin up a new one. Actually Raj already uses Miro heavily, said it's his preferred collaboration tool over Figma or Confluence. Oh, and he drives a Tesla Model 3 so if we validate parking, it needs to be the garage on 7th, not the surface lot — the chargers are there.

      Marcus Webb: Got it. I'll send a calendar invite to raj.patel@datavaultsolutions.com and tom.briggs@datavaultsolutions.com. CC me and you. Meeting title: "DataVault Graph Migration Kickoff — Graphable."

      Sarah Chen: Perfect. One more thing — Raj mentioned they had a data breach incident back in January at their Denver HQ that's driving some of this migration urgency. He doesn't want that detail in any written docs, but it's context for why the timeline is so aggressive.

      Marcus Webb: Understood. I'll keep the technical proposal focused on performance and scalability, not the incident. See you Thursday.
    "#
  }
}

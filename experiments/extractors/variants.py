"""Extractor variants, as transforms over production baml_src/extraction.baml.

Each variant changes exactly ONE axis from the `production` control, so an
observed difference is attributable. The previous version of this file declared
entity types as bare strings and silently dropped all 32 `@description(...)`
annotations production carries, which made its control weaker than production
and confounded every result.

Two axes are under test, both living in the .baml text:
  1. the per-type `@description` strings on the EntityType enum
  2. the prompt body of `function ExtractMemory`

Every transform asserts its anchor exists, so a drifted anchor fails loudly
instead of silently producing the control again.
"""

from __future__ import annotations

import re

# ── anchors in baml_src/extraction.baml ──────────────────────────────

ENUM_OPEN = "enum EntityType {"
ENTITY_CLASS = '''class ExtractedEntity {
  name string @description("The entity name as it appears in text")
  type EntityType'''
RULES_BLOCK = """    Rules:
    - A human is always PERSON, even when attending a meeting. A named team is an
      ORGANIZATION. Do not duplicate a person as both PERSON and an attendee type.
    - Capture status / priority / date fields only when explicitly stated.
    - Do NOT extract pronouns, unnamed roles, or vague collective nouns.
"""
RELATIONS_TAIL = "    ACTION_ITEM RESULTED_IN a DECISION, a FINDING SUPPORTS a TOPIC).\n"
DELIVERABLE_LINE = '  DELIVERABLE @description("A concrete artifact or output produced by project work (report, feature, dataset, document)")\n'


def _require(text, anchor, what):
    if anchor not in text:
        raise AssertionError(
            f"variant transform '{what}' could not find its anchor in "
            f"extraction.baml — the production file has drifted, update variants.py"
        )
    return text


# ── axis 1: per-type @description text ───────────────────────────────


def strip_type_descriptions(text):
    """Remove every @description on EntityType values, keeping the values.

    This is what the previous experiment accidentally used as its control.
    Running it deliberately measures what those 32 annotations are worth.
    """
    _require(text, ENUM_OPEN, "strip_type_descriptions")
    start = text.index(ENUM_OPEN)
    end = text.index("\n}", start)
    body = text[start:end]
    stripped = re.sub(r'\s*@description\("(?:[^"\\]|\\.)*"\)', "", body)
    if stripped == body:
        raise AssertionError("strip_type_descriptions removed nothing")
    return text[:start] + stripped + text[end:]


# Sharper guidance for the types that produced the most retrieval noise:
# bare identifiers and sentence fragments stored as entity names.
SHARPENED = {
    "OBJECT": (
        "Branded or model-specific physical/product items only. DO NOT extract "
        "pipelines, frameworks, approaches, APIs, systems, databases, documents, "
        "or technical concepts. DO NOT extract a sentence, a clause, or a "
        "narration fragment — if the name would not fit on a label, it is not an "
        "OBJECT."
    ),
    "TASK": (
        "A task, ticket, or work item within a project (not tied to a specific "
        "meeting — use ACTION_ITEM for meeting follow-ups). Name it by its "
        "identifier or a short noun phrase, e.g. 'GRA-539' or 'apostrophe "
        "backfill fix'. DO NOT name a task with a full sentence describing what "
        "happened."
    ),
    "FINDING": (
        "A research finding, result, or conclusion. Name it as a short noun "
        "phrase, not as the sentence stating it: 'index bloat on migration 072', "
        "not 'We discovered that migration 072 created the view without an "
        "index'. DO NOT extract routine status observations as FINDINGs."
    ),
    "DELIVERABLE": (
        "A concrete artifact or output produced by project work (report, feature, "
        "dataset, document). An artifact identified by a number — a PR, a "
        "migration, a ticket — is only useful later if its name carries the "
        "identifier exactly as written, e.g. 'PR #136', 'migration 072'."
    ),
}


def sharpen_type_descriptions(text):
    """Rewrite selected @description strings with explicit negative examples."""
    out = text
    for value, descr in SHARPENED.items():
        pattern = re.compile(
            r'(\n  ' + value + r' @description\(")(?:[^"\\]|\\.)*("\))'
        )
        if not pattern.search(out):
            raise AssertionError(f"sharpen: no @description found for {value}")
        out = pattern.sub(lambda m: m.group(1) + descr + m.group(2), out, count=1)
    return out


# ── axis 2: the prompt body ──────────────────────────────────────────

WORKED_EXAMPLES = """
    ## Worked examples

    Text: "PR #136 in Grad-Graph/ETL-INFRA is the apostrophe fix under epic
    GRA-564. It's stacked on #92."
    Extract: DELIVERABLE "PR #136", DELIVERABLE "PR #92", PROJECT
    "Grad-Graph/ETL-INFRA", TASK "GRA-564". Relations: PR #136 PART_OF GRA-564,
    PR #136 DEPENDS_ON PR #92.
    Do NOT extract: "the apostrophe fix" (already named by PR #136), "stacked"
    (not a thing).

    Text: "We lost an afternoon because migration 072 created the view without
    an index and nobody caught it until replay lag spiked."
    Extract: DELIVERABLE "migration 072", FINDING "missing index on migration
    072 view".
    Do NOT extract: "We lost an afternoon because migration 072 created the view
    without an index" — that is a sentence, not an entity name.
"""

DESCRIPTION_RULE = """
    ### Every entity needs a description
    `description` is REQUIRED and carries the entity's whole meaning downstream —
    it is the text that makes the entity findable later. Write one sentence that
    stands on its own, for a reader who cannot see this conversation.

    This matters most for identifier-style names, which are otherwise
    indistinguishable from one another:

      name: "PR #136 (Grad-Graph/ETL-INFRA)"
      description: "Fixes apostrophe and ordinal handling in the SOC backfill for
                    epic GRA-564; merged after review."

    A description that only says "a pull request" or restates the number is a
    failed extraction.
"""

WORKFLOW_RULE = """
    Engineering workflow links are how a later question about one artifact
    reaches the rest of its work. Emit them wherever the text supports it: a
    PULL_REQUEST PART_OF an EPIC, a PULL_REQUEST CLOSES an ISSUE, an ISSUE
    PART_OF an EPIC, a PULL_REQUEST DEPENDS_ON another PULL_REQUEST (stacked
    branches), a LESSON LEARNED_FROM whatever produced it.
"""

WORKFLOW_TYPES = '''
  // ── Engineering workflow ───────────────────────────────────────────
  EPIC @description("A named epic: the tracked parent grouping a specific set of tickets and PRs. Use PROJECT for a whole initiative or repo.")
  PULL_REQUEST @description("A pull or merge request. Put the full identifier in `name` (e.g. 'PR #136 (Grad-Graph/ETL-INFRA)'), never a bare number, and what it changes in `description`.")
  ISSUE @description("A tracker ticket referenced by its identifier (e.g. GRA-539). Use TASK for work described without a tracker id.")
  LESSON @description("A durable takeaway from an action or outcome — what to do differently next time. Extract only when the text states the takeaway, not merely the event.")
'''


def add_prompt_examples(text):
    _require(text, RULES_BLOCK, "add_prompt_examples")
    return text.replace(RULES_BLOCK, RULES_BLOCK + WORKED_EXAMPLES, 1)


def add_description_field(text):
    """Add a required description to ExtractedEntity, plus the prompt rule."""
    _require(text, ENTITY_CLASS, "add_description_field")
    out = text.replace(
        ENTITY_CLASS,
        '''class ExtractedEntity {
  name string @description("The entity name as it appears in text")
  description string @description("REQUIRED. One self-contained sentence saying what this entity is and why it mattered here, understandable with no surrounding context. For an identifier-style name (PR #136, GRA-539, migration 072) this sentence is the ONLY thing distinguishing it from every other number.")
  type EntityType''',
        1,
    )
    _require(out, RULES_BLOCK, "add_description_field/rule")
    return out.replace(RULES_BLOCK, RULES_BLOCK + DESCRIPTION_RULE, 1)


def add_workflow_types(text):
    _require(text, DELIVERABLE_LINE, "add_workflow_types")
    out = text.replace(DELIVERABLE_LINE, DELIVERABLE_LINE + WORKFLOW_TYPES, 1)
    _require(out, RELATIONS_TAIL, "add_workflow_types/relations")
    return out.replace(RELATIONS_TAIL, RELATIONS_TAIL + WORKFLOW_RULE, 1)


# ── embedded-text rules ──────────────────────────────────────────────


def _name_only(entity):
    return entity["name"]


def _name_and_description(entity):
    descr = (entity.get("description") or "").strip()
    return f"{entity['name']}: {descr}" if descr else entity["name"]


# ── the variant set ──────────────────────────────────────────────────


def _compose(*fns):
    def run(text):
        for fn in fns:
            text = fn(text)
        return text
    return run


VARIANTS = {
    "production": {
        "blurb": "Unmodified baml_src/extraction.baml — the true control.",
        "transform": lambda text: text,
        "embed_text": _name_only,
        "axis": "control",
    },
    "no-type-descr": {
        "blurb": "All 32 EntityType @description annotations stripped.",
        "transform": strip_type_descriptions,
        "embed_text": _name_only,
        "axis": "type descriptions (removed)",
    },
    "sharp-type-descr": {
        "blurb": "Type descriptions rewritten with explicit negative examples.",
        "transform": sharpen_type_descriptions,
        "embed_text": _name_only,
        "axis": "type descriptions (sharpened)",
    },
    "prompt-examples": {
        "blurb": "Production types; prompt gains worked extraction examples.",
        "transform": add_prompt_examples,
        "embed_text": _name_only,
        "axis": "prompt body",
    },
    "described": {
        "blurb": "Adds a required self-contained description to every entity.",
        "transform": add_description_field,
        "embed_text": _name_and_description,
        "axis": "entity schema",
    },
    "workflow": {
        "blurb": "described + EPIC/PULL_REQUEST/ISSUE/LESSON types and links.",
        "transform": _compose(add_description_field, add_workflow_types),
        "embed_text": _name_and_description,
        "axis": "entity schema + new types",
    },
}


def baml_for(variant, production_text):
    return VARIANTS[variant]["transform"](production_text)


def embed_text(variant, entity):
    return VARIANTS[variant]["embed_text"](entity)

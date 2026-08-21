"""The BAML source for the judge, injected into the runtime file map.

Not written to production ``baml_src/``: the file map handed to
``BamlRuntime.from_files`` is just a dict, so an experiment can add a function
without a codegen step and without production carrying an experiment's types.
The one thing that DOES live in production is the client — ``QwenCoder`` in
``baml_src/clients.baml`` — because a local-model client is worth having
whether or not this experiment survives.
"""

from __future__ import annotations

JUDGE_KEY = "judge.baml"

# Rubric copied verbatim from experiments/extractors/evaluate.py:JUDGE_SYSTEM.
# It has to be identical: the whole point is comparing this model's grades to
# the Claude grades already on disk, and a reworded rubric would make the
# disagreement a prompt artifact rather than a model difference.
# Two functions in one file so a single runtime serves every config: the
# only differences that matter are the return schema and the client options.
JUDGE_BAML = '''
class CandidateVerdict {
  id int @description("The candidate's id, copied exactly from the input")
  grade int @description("0, 1, or 2 — the relevance grade from the rubric")
  keep bool @description("true if this memory should be injected into the assistant's context for this prompt")
  reason string @description("One sentence, at most 30 words, saying why this grade. Name the specific thing that decided it.")
}

class RecallJudgement {
  verdicts CandidateVerdict[] @description("Exactly one entry per candidate, including the 0s")
}

class TerseVerdict {
  id int @description("The candidate's id, copied exactly from the input")
  grade int @description("0, 1, or 2 — the relevance grade from the rubric")
  keep bool @description("true if this memory should be injected into the assistant's context for this prompt")
}

class TerseJudgement {
  verdicts TerseVerdict[] @description("Exactly one entry per candidate, including the 0s")
}

function JudgeRecall(user_prompt: string, candidates: string) -> RecallJudgement {
  client QwenCoder
  prompt #"
    You label retrieval relevance for an AI assistant's memory system.

    The system injects stored memories into a coding assistant's context before it
    answers a user's prompt. Decide whether each memory would genuinely help the
    assistant respond better to that prompt.

    2 = directly informs the answer: a preference the answer should follow, a fact
        the prompt asks about, a constraint that changes the response.
    1 = same topic, useful background, but the answer is substantially the same
        without it.
    0 = unrelated, or so generic it adds nothing. Most memories are 0 for most
        prompts. Merely sharing vocabulary with the prompt is 0 — a record about a
        different PR, ticket, or migration number than the one asked about is 0.

    Be strict. Injected noise wastes context and misleads the model.
    Label every candidate, including the 0s.

    Set keep=true only for grade 2, and for grade 1 when the background is
    specific enough to change how the assistant answers. Never keep a 0.

    The user's prompt:
    ---
    {{ user_prompt }}
    ---

    The candidate memories:
    ---
    {{ candidates }}
    ---

    {{ ctx.output_format }}
  "#
}
function JudgeRecallTerse(user_prompt: string, candidates: string) -> TerseJudgement {
  client QwenCoder
  prompt #"
    You label retrieval relevance for an AI assistant's memory system.

    The system injects stored memories into a coding assistant's context before it
    answers a user's prompt. Decide whether each memory would genuinely help the
    assistant respond better to that prompt.

    2 = directly informs the answer: a preference the answer should follow, a fact
        the prompt asks about, a constraint that changes the response.
    1 = same topic, useful background, but the answer is substantially the same
        without it.
    0 = unrelated, or so generic it adds nothing. Most memories are 0 for most
        prompts. Merely sharing vocabulary with the prompt is 0 — a record about a
        different PR, ticket, or migration number than the one asked about is 0.

    Be strict. Injected noise wastes context and misleads the model.
    Label every candidate, including the 0s.

    Set keep=true only for grade 2, and for grade 1 when the background is
    specific enough to change how the assistant answers. Never keep a 0.

    The user's prompt:
    ---
    {{ user_prompt }}
    ---

    The candidate memories:
    ---
    {{ candidates }}
    ---

    {{ ctx.output_format }}
  "#
}
'''

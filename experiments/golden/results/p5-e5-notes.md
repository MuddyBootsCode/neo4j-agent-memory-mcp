# p5-e5: the error-keyed query set (MUD-460, 2026-09-08)

A second trigger and a second query set, with its own labels. Every other
P5 experiment replays a human prompt; this replays a failure. ExpeRepair
keys retrieval on the latest test feedback (+6.8 on SWE-bench Verified);
cue-anchored memory fires on a non-zero exit rather than on a prompt.
Nothing published measures recall keyed by error text against stored
symptoms — this is that measurement, and it is negative.

## The set

`step3_errors.py` harvested every `is_error` tool result from the 38
p4-live query sessions, dropped sandbox and harness noise (permission
rejections, `command not found`, API 5xx) and repeats of the same first
line: **68 distinct failures**, not the 100 MUD-460 assumed — the corpus
does not contain more. Query text is the exit code and the first 300
characters; ids start at 1000 so a labels file from the prompt set can
never be mistaken for these.

Labelled by Opus 5 against an error-time rubric ("the assistant just saw
this failure; would this lesson help now"): 408 calls, **$13.40**,
19,489 pairs, **84 relevant — 0.43%**. After step1b's holdout dropped 43
lessons from the error queries' own session families, 56 relevant pairs
over 244 lessons remain, 0.82 per query. The prompt set, for comparison,
is 10.33 per query at 3.60%.

## The result

| index | cos20 recall | cos20 P@3 | D precision | D recall | D P@3 | D coverage |
|---|---|---|---|---|---|---|
| full lesson text | 44.6% | 2.5% | 2.9% | 35.7% | 4.9% | 24% |
| symptom only | 44.6% | 2.5% | 3.2% | 39.3% | 3.9% | 24% |
| triggers (E2) | 44.6% | 2.9% | 3.2% | 39.3% | 5.4% | 24% |
| triggers + anchor leg | 44.6% | 2.9% | 3.1% | 37.5% | 3.9% | 22% |

**Do not read the recall column.** 45% of 56 relevant pairs in a top-20 of
244 lessons is what sparsity looks like, not what retrieval working looks
like. The numbers that decide a `PostToolUse` injection are P@3 and
coverage, and they say no: **an error-time injection of three lines would
be right about one time in twenty-five, and three quarters of failures
have nothing relevant in the top ten at all.**

**Nothing on the index side moves it.** Symptom-only vectors, trigger
sentences and the anchor leg all land within a point of the full lesson
text and of each other. That is the tell: the ranker is not what is
failing here. The store simply does not hold a lesson that bears on a
given failure — 0.43% of pairs against the prompt set's 3.60%, an eighth
of the density.

## Why, and what it means for the trigger

Most failures in a real transcript are ordinary mistakes no stored lesson
could have prevented: a string that did not match for `Edit`, a container
that was not running, a glob that matched nothing, a pre-commit hook
reformatting a file. The lessons that *would* help an error are the ones
extracted from that same failure being fixed — and those are exactly what
the holdout removes, which is why 28 of the 84 relevant pairs disappeared
with the 43 dropped lessons. In production there is no holdout, so some of
that would come back, but it comes back as "the session that already
solved this" rather than as transfer to a new failure.

**MUD-460's ship shape is not justified by this measurement.** A
`PostToolUse` hook on `is_error` would fire on every failing tool call —
several a minute on a bad day — to serve something relevant a quarter of
the time and correct at the top about 5% of the time. The prompt-keyed
path serves 80% coverage at 27% precision for the same store.

Worth keeping from it: the query set and the labels (`results/p5-e5/`),
which are reusable evidence for any future error-time work, and the
finding that the gap is corpus density rather than ranking. If error-time
recall is revisited, the thing to change is what capture *stores* about a
failure, not how retrieval reads it.

## Codex review: the truncation defect, and why the conclusion stands

The adversarial review found that the harvest cut each failure to its
first 300 characters, and that 34 of the 68 committed queries hit that
cap. It is a real defect and it is an own-goal: `error_steps` already
middle-truncates to 600 characters, keeping head *and* tail, precisely
because a traceback's diagnosis is its last line. Cutting the head off
again discarded what production had preserved — q1004 lost its "unknown
revision", q1065 its "Theme object has no attribute custom_css". Fixed:
the query is now the failure as the reader returns it, and a regression
pins the tail surviving.

The review's stronger claim, that this "undermines the negative recall
conclusion", does not hold. Split the committed labels by whether the
query was truncated:

| | queries | relevant | per query | have any relevant |
|---|---|---|---|---|
| hit the 300-char cap | 34 | 32 | 0.94 | 62% |
| under the cap | 34 | 24 | 0.71 | 44% |

**The truncated queries are the label-denser half**, not the starved one.
If losing the diagnostic were what made the set sparse, the effect would
run the other way. The sparsity is the corpus: 0.43% of pairs relevant
against the prompt set's 3.60%, and every index-side variant flat within
a point of the others, which is a ranking-independent signal.

The committed `queries.json` was generated before the fix, so it does not
match the current script; regenerating it invalidates the labels and costs
another ~$13 of Opus. The evidence above says that rerun would sharpen the
set without changing its verdict, so it is recorded here as an open option
rather than taken.

**27 of the set's 16,592 pairs carry no verdict** — 0.16%, where the
labeller omitted an id from a chunk. At the set's 0.43% base rate that is
an expected 0.1 relevant pairs missed, immaterial to anything above, and
step5 now reports the number on every run and refuses to score past 1%.

**Regenerating is now safe to do.** The review's next pass found the trap
it would have sprung: step4 resumes on `"<query_id>:<lesson_id>"`, so
regenerating a query under its own id is skipped as already labelled and
then scored against ground truth built for the old text — silently, 34 of
68 queries, no error and wrong numbers. Queries are fingerprinted now
(`query_fingerprints.json`, content hash of the prompt, files and failing
call). step4 drops the labels of any query whose fingerprint moved and
relabels it; step5 refuses to score at all while any label is stale. The
existing p5-e5 labels are fingerprinted against the queries they were
actually made for, so the regeneration is caught the moment it happens.

Same artifact policy as the other P5 runs; `queries.json`, `labels.json`
and `label_usage.json` for the error set live once, under `p5-e5/`.

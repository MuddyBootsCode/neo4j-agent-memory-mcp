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

Same artifact policy as the other P5 runs; `queries.json`, `labels.json`
and `label_usage.json` for the error set live once, under `p5-e5/`.

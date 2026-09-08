# p5-e3: the structured situation card as the query (MUD-458, 2026-09-08)

p3-ctx showed that prepending raw prior turns dilutes the query embedding
(cosine20 recall 39% → 27% on the p1 pool). MUD-458 proposed the
structured alternative: not a transcript window but a card —
`files: <basenames> | last error: <first line> | prompt: <prompt>`, with
the error taken from the failing tool results in the 60 records before the
prompt (`error_steps`, now bounded by `before_line` / `since_line`).

20 of the 100 p4-live queries have a failing tool result inside that
window; MUD-458 estimated 25. The prompt goes into the card whole — the
baseline embeds all of it, so a truncated prompt would be measuring the
truncation — and only the error (25 words) and the file list (4 basenames)
are capped.

| index | query | cos20 recall | cos20 P@5 | D precision | D recall | D coverage |
|---|---|---|---|---|---|---|
| bare | prompt (baseline) | 30.0% | 24.4% | 20.7% | 20.0% | 71% |
| bare | card, no prompt | 17.0% | 9.8% | 8.8% | 8.5% | 39% |
| bare | full card | 23.2% | 16.4% | 16.5% | 16.0% | 66% |
| triggers + leg | prompt | 34.7% | 30.4% | 24.7% | 23.9% | 79% |
| triggers + leg | full card | 30.0% | 22.4% | 18.5% | 17.9% | 73% |

**Dead, and worse than the transcript window it was meant to replace.**
Nearly seven points of ceiling below the bare prompt, and it takes
another 4.7 off the best index too, so it is the query that is being damaged,
not the match.

## What each part costs

cosine20 recall, baseline vs full card, split by what the card could add:

| cell | baseline | full card |
|---|---|---|
| an error inside the window (20) | 32.0% | **11.0%** |
| no error (80) | 29.5% | 26.2% |
| the session edited files (60) | 28.5% | 18.6% |
| edited nothing (40) | 32.1% | 30.0% |

**The error text is what destroys it.** Twenty-five words of stack trace
or shell failure dominate the embedding and the prompt stops being the
query: 32.0% → 11.0% on exactly the queries the error was supposed to
help. The file list costs about ten points where it applies. And the
queries with neither — where the card is the prompt behind a literal
`prompt: ` label — still lose 2.1 points, which is how sensitive a MiniLM
query embedding is to framing text. That last number also bounds the
label confound: dropping the labels could not recover a 7-point deficit.

## What it says about the rest of P5

Two different query-side enrichments have now been measured on this
harness and both are negative: prior turns (p3-ctx, −12) and the
structured card (−7). Everything that has worked has been index-side —
E1's filenames (+5.8) and E2's triggers (+4.7), both of which add to the
*lesson*, where there is room, rather than to the prompt, where there is
not. That is evidence against **E4** (MUD-459, HyDE rewrite fused by RRF)
before its gate model is spent, with one caveat that keeps it alive: E4
fuses a rewrite as a separate leg rather than concatenating it into the
query, so it never dilutes the prompt leg. The dilution finding does not
transfer; the "put more in the query string" finding does.

Same artifact policy as the other P5 runs.

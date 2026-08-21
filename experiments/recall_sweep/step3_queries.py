"""Step 3: build the query set from the later (query) sessions.

Real, human-typed prompts only (see lib.real_user_prompt): filters isMeta,
tool-result turns, synthetic injected content (task notifications, system
reminders, command expansions), and slash commands. Kept prompts are >= 8
words. Capped at 30, sampled evenly across the query sessions (and evenly
within a session when one session dominates, since there is currently only
one query session).

File context: files touched in that session BEFORE the prompt's line index —
easy here because the transcript is already chronological, so
touched_files(path, before_line=idx) does the reconstruction directly. This
is the "ordering is easy" branch from the task description.
"""

from __future__ import annotations

import os

from lib import iter_transcript_lines, load_json, real_user_prompt, sample_evenly, save_json, touched_files

MAX_QUERIES = int(os.environ.get("SWEEP_MAX_QUERIES", "30"))
MIN_WORDS = 8
# Stratified selection: keep every query that config B could possibly answer
# (>=1 already-touched file the corpus also covers), then fill the rest of the
# budget with an even sample of the others. Without this, an even sample over
# a real corpus lands ~6 anchorable queries out of 30 -- too few to compute a
# precision number for anchor-first. The strata are recorded per query so the
# report can break results out instead of averaging a biased set.
STRATIFY = os.environ.get("SWEEP_STRATIFY") == "1"
MAX_ANCHORABLE = int(os.environ.get("SWEEP_MAX_ANCHORABLE", "40"))


def main() -> None:
    split = load_json("session_split.json")
    if split is None:
        raise SystemExit("run step1_split.py first")
    query_sessions = split["query_sessions"]

    if not query_sessions:
        save_json("queries.json", [])
        print("no query sessions available — queries.json is empty")
        return

    per_session_candidates: dict[str, list[dict]] = {}
    for s in query_sessions:
        candidates = []
        for idx, record in iter_transcript_lines(s["path"]):
            text = real_user_prompt(record)
            if text is None:
                continue
            if len(text.split()) < MIN_WORDS:
                continue
            candidates.append({"session": s["session"], "path": s["path"], "line": idx, "prompt": text})
        per_session_candidates[s["session"]] = candidates
        print(f"session {s['session']}: {len(candidates)} candidate prompts (>= {MIN_WORDS} words)")

    total_candidates = sum(len(v) for v in per_session_candidates.values())
    if total_candidates < 8:
        print(
            f"!! SAMPLE SIZE FLAG: only {total_candidates} usable query prompts "
            "found across all query sessions (< 8) — running anyway at this size, "
            "per the task's explicit fallback."
        )

    if STRATIFY:
        corpus_files = set()
        for s in split["corpus_sessions"]:
            corpus_files.update(touched_files(s["path"]))
        print(f"corpus covers {len(corpus_files)} distinct files")

        anchorable, other = [], []
        for cands in per_session_candidates.values():
            for c in cands:
                c["files"] = touched_files(c["path"], before_line=c["line"])
                (anchorable if set(c["files"]) & corpus_files else other).append(c)
        print(f"anchorable candidates: {len(anchorable)} / {total_candidates}")

        chosen = sample_evenly(anchorable, min(len(anchorable), MAX_ANCHORABLE))
        for c in chosen:
            c["anchorable"] = True
        fill = sample_evenly(other, max(0, MAX_QUERIES - len(chosen)))
        for c in fill:
            c["anchorable"] = False
        picked = chosen + fill
    else:
        # Cap allocated evenly across sessions, proportional to each session's pool.
        n_sessions = len(per_session_candidates)
        budgets = {}
        remaining = MAX_QUERIES
        for i, (sess, cands) in enumerate(per_session_candidates.items()):
            # Simple even split across remaining sessions, capped by pool size.
            share = max(1, remaining // (n_sessions - i)) if cands else 0
            take = min(share, len(cands))
            budgets[sess] = take
            remaining -= take

        picked = []
        for sess, cands in per_session_candidates.items():
            for c in sample_evenly(cands, budgets[sess]):
                c["files"] = touched_files(c["path"], before_line=c["line"])
                c["anchorable"] = None
                picked.append(c)

    selected = []
    for qid, c in enumerate(picked):
        selected.append({
            "query_id": qid,
            "session": c["session"],
            "prompt": c["prompt"],
            "line": c["line"],
            "files": c["files"],
            "anchorable": c["anchorable"],
            "file_context": "before_prompt",
        })

    save_json("queries.json", selected)
    n_anchor = sum(1 for q in selected if q["anchorable"])
    print(f"\nselected {len(selected)} queries (cap {MAX_QUERIES})"
          + (f", {n_anchor} anchorable" if STRATIFY else ""))
    for q in selected[:5]:
        print(f"  q{q['query_id']}: [{q['session']}] {q['prompt'][:80]!r} files={len(q['files'])}")


if __name__ == "__main__":
    main()

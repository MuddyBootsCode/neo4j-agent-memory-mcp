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

from lib import iter_transcript_lines, load_json, real_user_prompt, sample_evenly, save_json, touched_files

MAX_QUERIES = 30
MIN_WORDS = 8


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

    selected = []
    qid = 0
    for sess, cands in per_session_candidates.items():
        chosen = sample_evenly(cands, budgets[sess])
        for c in chosen:
            files = touched_files(c["path"], before_line=c["line"])
            selected.append({
                "query_id": qid,
                "session": sess,
                "prompt": c["prompt"],
                "line": c["line"],
                "files": files,
                "file_context": "before_prompt",
            })
            qid += 1

    save_json("queries.json", selected)
    print(f"\nselected {len(selected)} queries (cap {MAX_QUERIES})")
    for q in selected[:5]:
        print(f"  q{q['query_id']}: [{q['session']}] {q['prompt'][:80]!r} files={len(q['files'])}")


if __name__ == "__main__":
    main()

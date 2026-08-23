"""Step 3: the query set — real human prompts from the query sessions.

Stratified 2x2: anchorable (a file touched before the prompt is also in the
pool) x short (< SHORT_WORDS words). Short prompts are the hook's common
case and the recall_sweep dropped them with its >= 8-word filter; here the
floor is 4 words, enough to exclude "ok" / "yes" / "do it".
"""

from __future__ import annotations

import os

from lib import iter_transcript_lines, load_json, real_user_prompt, sample_evenly, save_json, touched_files

MAX_QUERIES = int(os.environ.get("GOLDEN_MAX_QUERIES", "100"))
MIN_WORDS = 4
SHORT_WORDS = 15
MAX_PROMPT_CHARS = 1500  # pasted blocks are not questions


def main() -> None:
    split = load_json("session_split.json")
    pool = load_json("pool.json")
    if split is None or pool is None:
        raise SystemExit("run step1 and step2 first")
    pool_files = {}
    for item in pool:
        pool_files.setdefault(item["repo"], set()).update(item["files"])

    cells: dict[tuple[bool, bool], list[dict]] = {}
    total = 0
    for s in split["query_sessions"]:
        for idx, record in iter_transcript_lines(s["path"]):
            text = real_user_prompt(record)
            if text is None:
                continue
            words = len(text.split())
            if words < MIN_WORDS or len(text) > MAX_PROMPT_CHARS:
                continue
            files = touched_files(s["path"], s["repo_root"], before_line=idx)
            anchorable = bool(set(files) & pool_files.get(s["repo"], set()))
            total += 1
            cells.setdefault((anchorable, words < SHORT_WORDS), []).append({
                "session": s["session"], "repo": s["repo"], "line": idx,
                "prompt": text, "files": files, "anchorable": anchorable, "short": words < SHORT_WORDS,
            })

    per_cell = MAX_QUERIES // 4
    picked: list[dict] = []
    leftovers: list[dict] = []
    for key in ((True, True), (True, False), (False, True), (False, False)):
        cand = cells.get(key, [])
        chosen = sample_evenly(cand, min(per_cell, len(cand)))
        picked += chosen
        rest = [c for c in cand if c not in chosen]
        leftovers += rest
        print(f"cell anchorable={key[0]} short={key[1]}: {len(cand)} candidates, took {len(chosen)}")
    if len(picked) < MAX_QUERIES:
        picked += sample_evenly(leftovers, MAX_QUERIES - len(picked))

    picked.sort(key=lambda c: (c["session"], c["line"]))
    queries = [{"query_id": i, **c} for i, c in enumerate(picked)]
    save_json("queries.json", queries)
    print(f"\n{len(queries)} queries from {total} candidates across {len(split['query_sessions'])} sessions")
    for q in queries[:5]:
        print(f"  q{q['query_id']} [{'A' if q['anchorable'] else '-'}{'s' if q['short'] else 'L'}] {q['prompt'][:70]!r}")


if __name__ == "__main__":
    main()

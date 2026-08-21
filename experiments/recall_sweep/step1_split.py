"""Step 1: list sessions and split chronologically into corpus / query sets.

Writes results/session_split.json. Flags the sample size prominently when it
falls below the design's minimums (>=5 corpus sessions, >=8 query prompts is
checked later once real prompts are extracted).
"""

from __future__ import annotations

from lib import list_sessions, save_json, split_sessions

MAX_CORPUS_SESSIONS = 40


def main() -> None:
    sessions = list_sessions()
    corpus, query = split_sessions(sessions, corpus_fraction=0.6)
    corpus = corpus[:MAX_CORPUS_SESSIONS]

    out = {
        "total_sessions": len(sessions),
        "corpus_sessions": corpus,
        "query_sessions": query,
    }
    save_json("session_split.json", out)

    print(f"total sessions found: {len(sessions)}")
    for s in sessions:
        print(f"  {s['session']}  mtime={s['mtime']:.0f}")
    print(f"corpus sessions ({len(corpus)}):", [s["session"] for s in corpus])
    print(f"query sessions ({len(query)}):", [s["session"] for s in query])
    if len(sessions) < 5:
        print(
            "!! SAMPLE SIZE FLAG: fewer than 5 sessions total exist for this "
            "project — the sweep will run at whatever size exists, per the "
            "task's explicit fallback."
        )


if __name__ == "__main__":
    main()

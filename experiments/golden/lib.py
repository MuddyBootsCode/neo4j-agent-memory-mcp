"""Shared helpers for the golden recall set (MUD-403).

A fixed, committed measurement: real prompts from later sessions, a lesson
pool built by production capture from earlier sessions, and relevance labels
from a model that is not in the pipeline (Claude Opus 5). Every later phase
of the memory plan is scored against it.

Read-only with respect to src/. Imports repo code, never copies it. Writes
only under experiments/golden/results/<run>/ and to the scratch database
named by GOLDEN_DB, which step6 drops.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLING_ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(TOOLING_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _load_dotenv(path: str) -> None:
    """Minimal .env loader: KEY=VALUE lines, no expansion, never overrides."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(TOOLING_ROOT, ".env"))
# Production paths stay local. The Anthropic key in .env is for the labeler
# only; NAM_LLM_PROVIDER=ollama keeps BAML (extraction, gate) off it.
os.environ.setdefault("NAM_LLM_PROVIDER", "ollama")
os.environ.setdefault("NAM_EMBEDDING_PROVIDER", "sentence_transformers")
os.environ.setdefault("BAML_LOG", "warn")

RUN = os.environ.get("GOLDEN_RUN", "baseline")
RESULTS_DIR = os.path.join(HERE, "results", RUN)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Comma-separated repo checkouts whose Claude Code sessions form the corpus.
REPO_ROOTS = [
    os.path.abspath(os.path.expanduser(p))
    for p in os.environ.get(
        "GOLDEN_REPOS", "~/Projects/gradgraph-auth-platform"
    ).split(",")
    if p.strip()
]
INCLUDE_WORKTREES = os.environ.get("GOLDEN_INCLUDE_WORKTREES", "1") == "1"
GOLDEN_DB = os.environ.get("GOLDEN_DB", "goldencorpus")
MAX_CORPUS_SESSIONS = int(os.environ.get("GOLDEN_MAX_CORPUS_SESSIONS", "40"))

LABEL_MODEL = os.environ.get("GOLDEN_LABEL_MODEL", "claude-opus-5")
# Anthropic list price, USD per million tokens, for the cost line in the run record.
PRICE = {
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},
    "claude-sonnet-5": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
}

_WORKTREE_MARKERS = ("/.claude/worktrees/", "/.worktrees/")

_SYNTHETIC_PREFIX_RE = re.compile(
    r"^<(task-notification|system-reminder|command-message|command-name|"
    r"local-command-stdout|user-prompt-submit-hook|automated-reminder|"
    r"command-args|bash-input|bash-stdout|bash-stderr|"
    r"[a-z0-9-]+-command)\b"
)


def result_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


def load_json(name: str, default=None):
    path = result_path(name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(name: str, data) -> None:
    with open(result_path(name), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str, sort_keys=False)


def claude_project_dir(repo_root: str) -> str:
    slug = repo_root.rstrip("/").replace("/", "-").replace(".", "-")
    return os.path.join(os.path.expanduser("~/.claude/projects"), slug)


def repo_name(repo_root: str) -> str:
    return os.path.basename(repo_root.rstrip("/"))


def list_sessions(repo_root: str) -> list[dict]:
    """Human-driven session transcripts for one repo, oldest first."""
    base = claude_project_dir(repo_root)
    dirs = [base]
    if INCLUDE_WORKTREES:
        dirs += sorted(glob.glob(base + "--claude-worktrees-*"))
    sessions = []
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*.jsonl")):
            stem = os.path.splitext(os.path.basename(path))[0]
            sessions.append({
                "session": stem,
                "path": path,
                "mtime": os.path.getmtime(path),
                "repo": repo_name(repo_root),
                "repo_root": repo_root,
            })
    sessions.sort(key=lambda s: s["mtime"])
    return sessions


def split_sessions(sessions: list[dict], corpus_fraction: float = 0.6):
    n = len(sessions)
    if n < 2:
        return sessions, []
    k = max(1, min(round(n * corpus_fraction), n - 1))
    return sessions[:k], sessions[k:]


def repo_relative(path: str, repo_root: str) -> str | None:
    """CodeFile.path for an absolute tool_input.file_path (worktree-aware)."""
    if not isinstance(path, str) or not path:
        return None
    for marker in _WORKTREE_MARKERS:
        at = path.find(marker)
        if at == -1:
            continue
        rest = path[at + len(marker):]
        parts = rest.split("/", 1)
        return parts[1] if len(parts) == 2 and parts[1] else None
    prefix = repo_root.rstrip("/") + "/"
    if path.startswith(prefix):
        return path[len(prefix):]
    if not path.startswith("/"):
        return path
    return None


def iter_transcript_lines(transcript_path: str):
    with open(transcript_path, encoding="utf-8", errors="replace") as fh:
        for idx, raw in enumerate(fh):
            try:
                record = json.loads(raw)
            except Exception:
                continue
            if isinstance(record, dict):
                yield idx, record


def touched_files(transcript_path: str, repo_root: str, *, before_line: int | None = None) -> list[str]:
    """Repo-relative files touched via Edit/Write tool_use, in order, deduped."""
    seen: dict[str, None] = {}
    for idx, record in iter_transcript_lines(transcript_path):
        if before_line is not None and idx >= before_line:
            break
        if record.get("type") != "assistant":
            continue
        content = (record.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") not in ("Edit", "Write"):
                continue
            rel = repo_relative((block.get("input") or {}).get("file_path"), repo_root)
            if rel:
                seen.setdefault(rel, None)
    return list(seen)


def _content_text(content) -> str | None:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        if not content or any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            return None
        texts = [
            b.get("text", "").strip()
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        texts = [t for t in texts if t]
        return " ".join(texts) if texts else None
    return None


def real_user_prompt(record: dict) -> str | None:
    """Human-typed prompt text, or None for meta/synthetic/slash-command turns."""
    if record.get("isMeta") or record.get("type") != "user":
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    text = _content_text(message.get("content"))
    if not text or text.startswith("/") or _SYNTHETIC_PREFIX_RE.match(text):
        return None
    return text


def sample_evenly(items: list, k: int) -> list:
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return list(items)
    step = n / k
    idxs = sorted({int(i * step) for i in range(k)})
    if len(idxs) < k:
        remaining = [i for i in range(n) if i not in idxs]
        idxs = sorted(idxs + remaining[: k - len(idxs)])
    return [items[i] for i in idxs[:k]]


def lesson_id(repo: str, kind: str, canonical_text: str) -> str:
    """Stable pool id: the same lesson text in the same repo gets the same id
    across corpus rebuilds, so labels survive a rebuild that reproduces it.

    Keyed on the canonical text (``lesson_text``), never on what was
    embedded (``embedding_input``) — index-side experiments change the
    latter, and an id that moved with it would detach every label
    (MUD-456).
    """
    h = hashlib.sha1(f"{repo}|{kind}|{canonical_text.strip()}".encode("utf-8"))
    return h.hexdigest()[:12]


def query_fingerprint(query: dict) -> str:
    """Content hash of a query: exactly what a label was made against.

    Labels are keyed ``"<query_id>:<lesson_id>"`` and step4 resumes on that
    key, so a query whose text changes under the same id is skipped as
    already labelled and then scored against ground truth built for the old
    text — silently, with no error and wrong numbers. The fingerprint is
    what makes that detectable (MUD-460).

    Covers everything the labeller sees: the query text, the files, and the
    failing call for an error-keyed set. Not the id, so renumbering a set
    does not invalidate it, and not the file order.
    """
    payload = json.dumps({
        "prompt": query.get("prompt", ""),
        "files": sorted(query.get("files") or []),
        "tool": query.get("tool"),
        "attempt": query.get("attempt"),
    }, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def stale_label_keys(labels: dict, queries: list[dict], stored: dict) -> set[str]:
    """Label keys whose query has changed since it was labelled.

    ``stored`` maps str(query_id) to the fingerprint recorded when the
    labels were written. A query with no stored fingerprint is left alone:
    runs labelled before fingerprints existed are still valid evidence.
    """
    changed = {
        str(q["query_id"]) for q in queries
        if str(q["query_id"]) in stored
        and stored[str(q["query_id"])] != query_fingerprint(q)
    }
    if not changed:
        return set()
    return {key for key in labels if key.split(":", 1)[0] in changed}


def _pool_ids_by_repo(pool: list[dict]) -> dict[str, set[str]]:
    by_repo: dict[str, set[str]] = {}
    for item in pool:
        by_repo.setdefault(item["repo"], set()).add(item["id"])
    return by_repo


def unlabeled_pairs(labels: dict, queries: list[dict], pool: list[dict]) -> int:
    """(query, lesson) pairs in the same repo that carry no verdict.

    A labeller that omits an id, or a relabel that died between deleting
    the old verdicts and writing the new ones, leaves holes here. step5
    would not notice: an unlabelled pair is simply absent from the recall
    denominator and never counts as a hit, so the scores shift quietly
    rather than failing (MUD-460, Codex review).
    """
    by_repo = _pool_ids_by_repo(pool)
    return sum(
        1
        for q in queries
        for lid in by_repo.get(q["repo"], ())
        if f"{q['query_id']}:{lid}" not in labels
    )


def complete_fingerprints(labels: dict, queries: list[dict], pool: list[dict]) -> dict:
    """Fingerprints for the queries whose labels are complete.

    The fingerprint's claim is "these labels judged this text", so writing
    one for a query still being labelled would certify judgments that do
    not exist yet — and an interrupted run would leave the query marked
    current and unlabelled.
    """
    by_repo = _pool_ids_by_repo(pool)
    return {
        str(q["query_id"]): query_fingerprint(q)
        for q in queries
        if all(f"{q['query_id']}:{lid}" in labels for lid in by_repo.get(q["repo"], ()))
    }


def lesson_text(kind: str, props: dict) -> str:
    """The canonical lesson text: what identifies a lesson."""
    from agent_memory_mcp.mcp._coding_tools import memory_embedding_text

    return memory_embedding_text(kind, props)


def embedding_input(kind: str, props: dict, repo: str, files: list[str],
                    triggers: list[str] | None = None) -> str:
    """What the embedder is given for a lesson: the canonical text, plus
    whatever NAM_EMBED_CONTEXT_PREFIX asks for (MUD-456) and the lesson's
    trigger sentences when the run has them (MUD-457)."""
    from agent_memory_mcp.mcp._coding_tools import memory_embedding_input

    return memory_embedding_input(kind, props, repo=repo, files=files, triggers=triggers)


def context_prefix() -> list[str]:
    """The prefix parts this process would embed with, for the run record."""
    from agent_memory_mcp.mcp._coding_tools import CONTEXT_PREFIX

    return list(CONTEXT_PREFIX)


def symptom_only() -> bool:
    """Whether this process embeds symptoms alone (MUD-460)."""
    from agent_memory_mcp.mcp._coding_tools import SYMPTOM_ONLY

    return SYMPTOM_ONLY


async def drop_database(name: str) -> None:
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "graphmemory")),
    )
    try:
        async with driver.session(database="system") as session:
            await session.run(f"DROP DATABASE `{name}` IF EXISTS")
    finally:
        await driver.close()


def session_family(session_id: str) -> str:
    """Parent session of a ``parent:spawn`` subagent id; the id itself
    otherwise (same rule as agent_memory_mcp.capture.cypher)."""
    if ":" in session_id and not session_id.startswith(":"):
        return session_id.split(":", 1)[0]
    return session_id

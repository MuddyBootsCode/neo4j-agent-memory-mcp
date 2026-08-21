"""Render the scored run as a single self-contained HTML page.

    python render.py

Reads results/score.json and results/verdicts.json, inlines both, and writes
results/local-judge.html. No external requests beyond Google Fonts, which is
the one host an Artifact's CSP admits.
"""

from __future__ import annotations

import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


BROWSE = "think-reason"   # the only config carrying both traces and reasons


def payload():
    with open(os.path.join(RESULTS, f"score.{BROWSE}.json")) as fh:
        score = json.load(fh)
    with open(os.path.join(RESULTS, f"verdicts.{BROWSE}.json")) as fh:
        verdicts = json.load(fh)
    with open(os.path.join(RESULTS, "compare.json")) as fh:
        compare = json.load(fh)
    with open(os.path.join(RESULTS, "score.terse-only.json")) as fh:
        terse = json.load(fh)
    ce_path = os.path.join(RESULTS, "crossenc.json")
    crossenc = json.load(open(ce_path)) if os.path.exists(ce_path) else None

    # The described corpus, when it has been run: same 30 prompts, entities
    # carrying a sentence instead of a bare name.
    def _opt(name):
        path = os.path.join(RESULTS, name)
        return json.load(open(path)) if os.path.exists(path) else None

    described = {
        "llm": _opt("score.terse-only.described.json"),
        "ce": _opt("crossenc.described.json"),
    }

    # The terse grade per row, so the browser shows where thinking changed the
    # verdict — that is the whole finding, and it is invisible in aggregate.
    tg = {(r["query_id"], r["id"]): r["qwen"] for r in terse["rows"]}
    for r in score["rows"]:
        r["terse"] = tg.get((r["query_id"], r["id"]))

    calls = {
        k: {
            "thinking": v.get("thinking") or "",
            "duration_ms": v.get("duration_ms"),
            "input_tokens": v.get("input_tokens"),
            "output_tokens": v.get("output_tokens"),
            "attempts": v.get("attempts"),
            "json_ok": v.get("json_ok"),
            "error": v.get("error"),
        }
        for k, v in verdicts.items()
    }
    return {"score": score, "calls": calls, "compare": compare,
            "best": terse["gates"], "crossenc": crossenc,
            "terse_gates": terse["gates"], "described": described}


CSS = """
:root {
  color-scheme: light dark;
  --paper:#EEF1EE; --surface:#FAFCFA; --raised:#FFFFFF;
  --ink:#131714; --muted:#666E67; --line:#D6DCD6; --line-soft:#E4E9E4;
  --accent:#2C6E63; --accent-soft:#DCEAE6;
  --hot:#A6432A; --hot-soft:#F6E3DC;
  --cool:#3A5F94; --cool-soft:#DEE6F3;
  --sans:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --serif:"Newsreader",Georgia,"Times New Roman",serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#0F1310; --surface:#161B17; --raised:#1C221D;
    --ink:#E2E8E2; --muted:#95A095; --line:#2B332C; --line-soft:#222A23;
    --accent:#61B4A4; --accent-soft:#1B322C;
    --hot:#E0866A; --hot-soft:#3A2018;
    --cool:#89A9E0; --cool-soft:#1A2436;
  }
}
:root[data-theme="dark"] {
  --paper:#0F1310; --surface:#161B17; --raised:#1C221D;
  --ink:#E2E8E2; --muted:#95A095; --line:#2B332C; --line-soft:#222A23;
  --accent:#61B4A4; --accent-soft:#1B322C;
  --hot:#E0866A; --hot-soft:#3A2018;
  --cool:#89A9E0; --cool-soft:#1A2436;
}

* { box-sizing:border-box; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1500px; margin:0 auto; padding:40px 28px 80px; }

.eyebrow {
  font-family:var(--mono); font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted);
}
h1 {
  font-family:var(--serif); font-weight:500; font-size:clamp(30px,4vw,46px);
  line-height:1.1; margin:10px 0 12px; text-wrap:balance; letter-spacing:-.01em;
}
.standfirst {
  font-family:var(--serif); font-size:19px; line-height:1.5;
  color:var(--muted); max-width:64ch; margin:0;
}
h2 {
  font-family:var(--serif); font-weight:500; font-size:24px;
  margin:0 0 4px; letter-spacing:-.01em;
}
.section { margin-top:52px; }
.section-note { color:var(--muted); font-size:14px; max-width:70ch; margin:0 0 20px; }

/* ---- verdict band ---- */
.verdict {
  margin-top:34px; background:var(--surface); border:1px solid var(--line);
  border-radius:3px; padding:26px 28px;
  display:grid; gap:26px; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);
}
@media (max-width:900px) { .verdict { grid-template-columns:1fr; } }
.verdict h2 { font-size:20px; }
.answer { font-family:var(--serif); font-size:17px; line-height:1.55; margin:8px 0 0; }
.answer strong { font-weight:600; }

table.gates { width:100%; border-collapse:collapse; font-size:13.5px; }
table.gates th, table.gates td {
  padding:8px 10px; border-bottom:1px solid var(--line-soft); text-align:right;
  font-variant-numeric:tabular-nums;
}
table.gates th { color:var(--muted); font-weight:500; font-size:11px;
  letter-spacing:.08em; text-transform:uppercase; }
table.gates th:first-child, table.gates td:first-child {
  text-align:left; font-family:var(--mono); font-size:12.5px;
}
table.gates tr.best td { background:var(--accent-soft); }
table.gates tr.best td:first-child { font-weight:600; }
.scroll { overflow-x:auto; }

/* ---- stat strip ---- */
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:3px; margin-top:22px; }
.stat { background:var(--surface); padding:14px 16px; }
.stat .k { font-family:var(--mono); font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); }
.stat .v { font-family:var(--serif); font-size:27px; line-height:1.15; margin-top:4px;
  font-variant-numeric:tabular-nums; }
.stat .sub { font-size:12px; color:var(--muted); margin-top:2px; }

/* ---- browser ---- */
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:0 0 16px; }
.chip {
  font-family:var(--mono); font-size:12px; padding:5px 11px; border-radius:2px;
  border:1px solid var(--line); background:var(--surface); color:var(--muted);
  cursor:pointer;
}
.chip[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:var(--paper); }
.chip:focus-visible, .prompt:focus-visible, .think-toggle:focus-visible {
  outline:2px solid var(--accent); outline-offset:2px;
}

.browser { display:grid; grid-template-columns:minmax(0,340px) minmax(0,1fr); gap:20px; }
@media (max-width:1000px) { .browser { grid-template-columns:1fr; } }

.list { border:1px solid var(--line); border-radius:3px; background:var(--surface);
  max-height:78vh; overflow-y:auto; }
.prompt {
  display:block; width:100%; text-align:left; background:none; border:0;
  border-bottom:1px solid var(--line-soft); padding:12px 14px; cursor:pointer;
  color:inherit; font:inherit;
}
.prompt:hover { background:var(--raised); }
.prompt[aria-current="true"] { background:var(--accent-soft); box-shadow:inset 3px 0 0 var(--accent); }
.prompt .t { font-size:13.5px; line-height:1.4; display:-webkit-box;
  -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.prompt .m { display:flex; gap:8px; align-items:center; margin-top:6px;
  font-family:var(--mono); font-size:10.5px; color:var(--muted); }
.bar { display:flex; gap:2px; }
.bar i { width:8px; height:8px; border-radius:1px; background:var(--line); display:block; }
.bar i.agree { background:var(--accent); opacity:.45; }
.bar i.hot { background:var(--hot); }
.bar i.cool { background:var(--cool); }

.detail { border:1px solid var(--line); border-radius:3px; background:var(--surface);
  padding:20px 22px; max-height:78vh; overflow-y:auto; }
.detail .q {
  font-family:var(--serif); font-size:19px; line-height:1.45; margin:0 0 6px;
}
.detail .qm { font-family:var(--mono); font-size:11.5px; color:var(--muted);
  margin-bottom:18px; word-break:break-all; }

.cand { border-top:1px solid var(--line-soft); padding:14px 0; }
.cand.dis-hot { box-shadow:inset 3px 0 0 var(--hot); padding-left:12px; }
.cand.dis-cool { box-shadow:inset 3px 0 0 var(--cool); padding-left:12px; }
.cand-head { display:flex; flex-wrap:wrap; gap:8px; align-items:baseline;
  font-family:var(--mono); font-size:11px; color:var(--muted); }
.tag { padding:2px 7px; border-radius:2px; border:1px solid var(--line); }
.tag.kept { border-color:var(--accent); color:var(--accent); }
.tag.hot { border-color:var(--hot); color:var(--hot); background:var(--hot-soft); }
.tag.cool { border-color:var(--cool); color:var(--cool); background:var(--cool-soft); }
.cand .text { font-family:var(--mono); font-size:12.5px; line-height:1.6; margin:8px 0 0;
  word-wrap:break-word; }
.cand .why { font-size:13.5px; line-height:1.5; margin:8px 0 0; color:var(--ink); }
.cand .why b { font-family:var(--mono); font-size:10.5px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-weight:500; display:block; }

.think-toggle { margin-top:12px; background:none; border:1px solid var(--line);
  border-radius:2px; color:var(--muted); font-family:var(--mono); font-size:11px;
  padding:5px 10px; cursor:pointer; }
.think-toggle:hover { color:var(--ink); }
.think { margin-top:10px; background:var(--raised); border:1px solid var(--line);
  border-radius:2px; padding:14px 16px; font-family:var(--mono); font-size:12px;
  line-height:1.65; white-space:pre-wrap; word-wrap:break-word;
  max-height:460px; overflow-y:auto; color:var(--muted); }
.empty { color:var(--muted); font-size:13.5px; padding:16px 0; }
table.compare td.lbl { text-align:left; font-family:var(--sans); font-size:13px;
  color:var(--muted); }
.code { font-family:var(--mono); font-size:12.5px; }
.cand-head .flip { color:var(--hot); border-bottom:1px solid var(--hot); }

.legend { display:flex; flex-wrap:wrap; gap:16px; margin:0 0 16px;
  font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.legend span { display:flex; align-items:center; gap:6px; }
.legend b { width:10px; height:10px; border-radius:2px; display:block; }

table.conf { border-collapse:collapse; font-family:var(--mono); font-size:12.5px; }
table.conf td, table.conf th { border:1px solid var(--line-soft); padding:8px 14px;
  text-align:center; font-variant-numeric:tabular-nums; }
table.conf th { color:var(--muted); font-weight:500; font-size:11px; }
table.conf td.diag { background:var(--accent-soft); }

footer { margin-top:60px; padding-top:20px; border-top:1px solid var(--line);
  color:var(--muted); font-size:12.5px; font-family:var(--mono); line-height:1.7; }
"""

JS = """
const D = window.__RUN__;
const rows = D.score.rows, calls = D.calls;
const byQuery = {};
rows.forEach(r => { (byQuery[r.query_id] = byQuery[r.query_id] || []).push(r); });
const qids = Object.keys(byQuery).map(Number).sort((a,b) => a-b);

const REL = 1;
const rel = r => (r.claude || 0) >= REL;
const kept = r => r.keep === true;
// Two directions of disagreement, and they are not the same failure.
// hot: the model injects what Claude called noise. cool: it drops what
// Claude called useful. One wastes context, the other loses the answer.
const dis = r => kept(r) && !rel(r) ? 'hot' : (!kept(r) && rel(r) ? 'cool' : null);

let filter = 'all', current = qids[0];

const F = {
  all: () => true,
  dis: r => dis(r) !== null,
  kept: r => kept(r),
  pref: r => r.kind === 'preference',
};

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function renderList() {
  const list = document.getElementById('list');
  list.innerHTML = qids.map(id => {
    const rs = byQuery[id];
    const shown = rs.filter(F[filter]);
    const bar = rs.map(r => {
      const d = dis(r);
      return `<i class="${d || 'agree'}"></i>`;
    }).join('');
    const c = calls[String(id)] || {};
    return `<button class="prompt" data-id="${id}" aria-current="${id === current}"
      ${shown.length ? '' : 'data-empty="1"'}>
      <span class="t">${esc(rs[0].prompt)}</span>
      <span class="m"><span class="bar">${bar}</span>
      <span>${shown.length}/${rs.length}</span>
      <span>${c.duration_ms ? Math.round(c.duration_ms/1000)+'s' : ''}</span></span>
    </button>`;
  }).join('');
  list.querySelectorAll('.prompt').forEach(b =>
    b.addEventListener('click', () => { current = +b.dataset.id; renderList(); renderDetail(); }));
}

function renderDetail() {
  const rs = (byQuery[current] || []).filter(F[filter]);
  const all = byQuery[current] || [];
  const c = calls[String(current)] || {};
  const el = document.getElementById('detail');
  if (!all.length) { el.innerHTML = '<p class="empty">Nothing selected.</p>'; return; }

  const head = `
    <p class="q">${esc(all[0].prompt)}</p>
    <p class="qm">${esc(all[0].project)} · query ${current} ·
      ${c.output_tokens || '?'} output tokens ·
      ${c.duration_ms ? (c.duration_ms/1000).toFixed(1)+'s' : '?'} ·
      ${c.attempts || 1} attempt${(c.attempts || 1) > 1 ? 's' : ''} ·
      json ${c.json_ok ? 'ok' : 'FAILED'}</p>`;

  const think = (c.thinking || '').trim();
  const thinkBlock = think ? `
    <button class="think-toggle" data-think="1" aria-expanded="false">
      Reasoning trace — ${think.length.toLocaleString()} chars</button>
    <div class="think" hidden>${esc(think)}</div>` : '';

  const body = rs.length ? rs.map(r => {
    const d = dis(r);
    const gradeTag = d === 'hot'
      ? '<span class="tag hot">kept · Claude said 0</span>'
      : d === 'cool'
        ? `<span class="tag cool">dropped · Claude said ${r.claude}</span>`
        : `<span class="tag ${kept(r) ? 'kept' : ''}">${kept(r) ? 'kept' : 'dropped'} · agreed</span>`;
    return `<div class="cand ${d ? 'dis-' + d : ''}">
      <div class="cand-head">
        <span>#${r.rank + 1}</span>
        <span>sim ${r.sim.toFixed(3)}</span>
        <span>${esc(r.kind)}</span>
        <span>thinking ${r.qwen == null ? '—' : r.qwen}</span>
        <span${r.terse != null && r.terse !== r.qwen ? ' class="flip"' : ''}>terse ${r.terse == null ? '—' : r.terse}</span>
        <span>claude ${r.claude == null ? '—' : r.claude}</span>
        ${gradeTag}
      </div>
      <p class="text">${esc(r.text)}</p>
      ${r.reason ? `<p class="why"><b>why</b>${esc(r.reason)}</p>` : ''}
    </div>`;
  }).join('') : '<p class="empty">No candidates match this filter for this prompt.</p>';

  el.innerHTML = head + thinkBlock + body;
  const t = el.querySelector('[data-think]');
  if (t) t.addEventListener('click', () => {
    const box = el.querySelector('.think');
    const open = box.hasAttribute('hidden');
    box.toggleAttribute('hidden', !open);
    t.setAttribute('aria-expanded', String(open));
  });
}

document.querySelectorAll('.chip').forEach(chip =>
  chip.addEventListener('click', () => {
    filter = chip.dataset.filter;
    document.querySelectorAll('.chip').forEach(c =>
      c.setAttribute('aria-pressed', String(c.dataset.filter === filter)));
    renderList(); renderDetail();
  }));

renderList(); renderDetail();
"""


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


CORPUS_SECTION = """
  <section class="section">
    <h2>Do entity descriptions help? Not for everyone</h2>
    <p class="section-note">Every record in the production corpus is a bare
    name. In the <span class="code">described</span> corpus each entity carries
    a self-contained sentence. Same 30 prompts, both times.</p>
    <div class="scroll">
      <table class="gates compare">
        <thead>
          <tr><th rowspan="2">gate</th><th colspan="3">production — bare names</th>
              <th colspan="3">described — with sentences</th></tr>
          <tr><th>P</th><th>R</th><th>F1</th><th>P</th><th>R</th><th>F1</th></tr>
        </thead>
        <tbody>{corpus_rows}</tbody>
      </table>
    </div>
    <p class="section-note" style="margin-top:14px">The descriptions help the
    reranker and hurt the language model. The reranker gains what the hypothesis
    predicted — text to match against — and its absolute scores become usable
    for the first time: on bare names every model collapsed to "keep
    everything", while on described corpus a global cutoff holds precision 0.572
    at 2.9 records. The language model moves the other way, precision
    0.742 → 0.523, because it promotes 46 more records off zero. A plausible
    sentence reads as relevant. It is the same over-keeping failure that
    thinking caused.
    <br><br><strong>Read with care:</strong> retrieval ran separately over each
    corpus, so these are not the same 300 records with text added — different
    records were retrieved, and the pools hold 87 and 81 relevant items
    respectively. What compares cleanly is each gate against the cutoff inside
    its own corpus, not one corpus against the other.</p>
  </section>
"""

CE_SECTION = """
  <section class="section">
    <h2>A cross-encoder is not a substitute</h2>
    <p class="section-note">A reranker reads the prompt and one memory together
    in a single pass, so unlike the stored embedding it can compare their
    tokens. That is the same advantage the language model has, at a
    hundredth of the latency. It does not come close to the same result.</p>
    <div class="scroll">
      <table class="gates compare">
        <thead><tr><th>gate</th><th>per query</th><th>inject</th>
          <th>precision</th><th>recall</th><th>F1</th></tr></thead>
        <tbody>{ce_rows}</tbody>
      </table>
    </div>
    <p class="section-note" style="margin-top:14px">All rows are the same 30
    prompts and the same 300 candidates, held at three injected records so
    nothing buys recall with volume. Every reranker beats cosine ordering —
    genuine signal, AUC 0.698 against cosine's 0.642 — and every one lands
    less than halfway to the language model.
    <br><br>The likely reason is what they are being asked to read. A
    cross-encoder is trained on question-and-passage pairs, and 1,259 of the
    1,433 records in this corpus are entities carrying a bare name and no
    description: <span class="code">"Cfo" ruled to SOC 11-1011</span>. There is
    no passage to match against. The language model can reason that GRA-577 is
    not GRA-602; a reranker scoring overlap on an identifier has nothing to
    work with. Testing this on the <span class="code">described</span> corpus is
    seconds of compute and would settle it.</p>
  </section>
"""


def build_html(data):
    s = data["score"]
    g = s["gates"]
    calls = s["calls"]
    a = s["agreement"]

    order = [
        ("threshold_0.50_shipping", "similarity ≥ 0.50 (ships today)"),
        ("threshold_0.38_tuned", "similarity ≥ 0.38 (MUD-386)"),
        ("qwen_keep", "qwen keep flag"),
        ("qwen_grade_ge_1", "qwen grade ≥ 1"),
        ("qwen_grade_eq_2", "qwen grade = 2"),
    ]
    best = max(order, key=lambda kv: g[kv[0]]["f1"])[0]
    gate_rows = "\n".join(
        f'<tr class="{"best" if k == best else ""}">'
        f"<td>{html.escape(label)}</td>"
        f'<td>{g[k]["per_query"]:.2f}</td>'
        f'<td>{fmt(g[k]["precision"])}</td>'
        f'<td>{fmt(g[k]["recall"])}</td>'
        f'<td>{fmt(g[k]["f1"])}</td></tr>'
        for k, label in order
    )

    conf = s["confusion"]
    conf_rows = "\n".join(
        f"<tr><th>claude {ca}</th>"
        + "".join(
            f'<td class="{"diag" if ca == qw else ""}">{conf[str(ca)][str(qw)] if str(ca) in conf else conf[ca][qw]}</td>'
            for qw in (0, 1, 2)
        )
        + "</tr>"
        for ca in (0, 1, 2)
    )

    # The headline is now a config result, not a gate result: the fastest
    # configuration is also the most accurate one, which is the opposite of
    # what the first run implied.
    best_gates = data["best"]
    bk = max(("qwen_keep", "qwen_grade_ge_1", "qwen_grade_eq_2"),
             key=lambda k: best_gates[k]["f1"])
    win = best_gates[bk]
    tuned = g["threshold_0.38_tuned"]
    verdict = (
        f"With thinking disabled and the reason field dropped — the shape a "
        f"production gate would use — the local model reaches "
        f"<strong>F1 {fmt(win['f1'])}</strong> at "
        f"<strong>{win['per_query']:.1f} records per prompt</strong>, against "
        f"<strong>{fmt(tuned['f1'])}</strong> at {tuned['per_query']:.1f} for the "
        f"tuned similarity cutoff. It answers in 4.2 seconds instead of 52, and "
        f"it is <em>more</em> accurate that way, not less."
    )

    median_s = (calls["median_ms"] or 0) / 1000

    ce = data.get("crossenc")
    ce_rows = ""
    if ce:
        tg = data["terse_gates"]["qwen_keep"]
        entries = [(
            "cosine top-3 — what ships, re-shaped",
            "—",
            ce["models"][0]["gates"]["cosine_top_3"]["llm_30"],
        )]
        for m in ce["models"]:
            entries.append((
                m["model"].split("/")[-1] + " top-3",
                f'{m["per_query_ms"]:.0f} ms',
                m["gates"]["top_3"]["llm_30"],
            ))
        entries.append(("qwen-judge terse, keep flag", "4,200 ms", tg))
        ce_rows = "\n".join(
            f'<tr class="{"best" if ms == "4,200 ms" else ""}">'
            f'<td>{html.escape(label)}</td>'
            f'<td>{ms}</td>'
            f'<td>{mm["per_query"]:.2f}</td>'
            f'<td>{fmt(mm["precision"])}</td>'
            f'<td>{fmt(mm["recall"])}</td>'
            f'<td>{fmt(mm["f1"])}</td></tr>'
            for label, ms, mm in entries
        )

    cmp_rows = "\n".join(
        f'<tr class="{"best" if r["config"] == "terse-only" else ""}">'
        f'<td>{html.escape(r["config"])}</td>'
        f'<td class="lbl">{html.escape(r["blurb"])}</td>'
        f'<td>{r["median_s"]:.1f}s</td>'
        f'<td>{r["output_tokens_per_call"]:,}</td>'
        f'<td>{r["json_ok"]}</td>'
        f'<td>{fmt(r["kappa"], 3)}</td>'
        f'<td>{fmt(r["precision"])}</td>'
        f'<td>{fmt(r["recall"])}</td>'
        f'<td>{fmt(r["f1"])}</td></tr>'
        for r in data["compare"]["rows"]
    )

    ce_section = CE_SECTION.replace("{ce_rows}", ce_rows) if ce_rows else ""

    corpus_section = ""
    d = data.get("described") or {}
    if d.get("llm") and d.get("ce") and ce:
        def _best(bundle):
            return max(bundle["models"],
                       key=lambda m: m["gates"]["top_3"]["llm_30"]["f1"])
        pairs_ = [
            ("cosine top-3 — what ships",
             ce["models"][0]["gates"]["cosine_top_3"]["llm_30"],
             d["ce"]["models"][0]["gates"]["cosine_top_3"]["llm_30"]),
            ("best reranker top-3",
             _best(ce)["gates"]["top_3"]["llm_30"],
             _best(d["ce"])["gates"]["top_3"]["llm_30"]),
            ("qwen-judge terse, keep flag",
             data["terse_gates"]["qwen_keep"],
             d["llm"]["gates"]["qwen_keep"]),
        ]
        corpus_rows = "\n".join(
            f'<tr class="{"best" if "qwen" in label else ""}">'
            f'<td>{html.escape(label)}</td>'
            f'<td>{fmt(a["precision"])}</td><td>{fmt(a["recall"])}</td><td>{fmt(a["f1"])}</td>'
            f'<td>{fmt(b["precision"])}</td><td>{fmt(b["recall"])}</td><td>{fmt(b["f1"])}</td>'
            f'</tr>'
            for label, a, b in pairs_
        )
        corpus_section = CORPUS_SECTION.replace("{corpus_rows}", corpus_rows)

    return f"""<title>Local Judge Readout</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header>
    <p class="eyebrow">qwen3.6-35b-a3b · local · {s['queries']} prompts · {s['candidates']} candidates</p>
    <h1>What a local model says about the memories it is handed</h1>
    <p class="standfirst">Recall ships with no reasoning in it at all — a cosine
    cutoff decides what enters the context. This run puts a 35B model on the
    machine in that seat, over the same candidates, and reads back why it kept
    what it kept.</p>
  </header>

  <div class="verdict">
    <div>
      <h2>Does a local judge beat the cutoff?</h2>
      <p class="answer">{verdict}</p>
      <p class="section-note" style="margin-top:14px">Scored against the
      {s['claude_relevant']} of {s['candidates']} candidates Claude graded
      relevant during the variant sweep. Recall is bounded by the candidate
      pool — a gate cannot inject what retrieval never surfaced, so this
      measures the gate, not the retriever.</p>
    </div>
    <div class="scroll">
      <table class="gates">
        <thead><tr><th>gate</th><th>inject/prompt</th><th>precision</th><th>recall</th><th>F1</th></tr></thead>
        <tbody>{gate_rows}</tbody>
      </table>
    </div>
  </div>

  <section class="section" style="margin-top:34px">
    <h2>Three configurations, identical candidates</h2>
    <p class="section-note">Thinking and the written justification are separate
    costs, so they are measured separately. Dropping thinking was meant to buy
    latency at some price in accuracy. It bought both.</p>
    <div class="scroll">
      <table class="gates compare">
        <thead><tr><th>config</th><th>what changed</th><th>median</th>
          <th>out tokens</th><th>schema</th><th>κ</th>
          <th>precision</th><th>recall</th><th>F1</th></tr></thead>
        <tbody>{cmp_rows}</tbody>
      </table>
    </div>
    <p class="section-note" style="margin-top:14px">Gate is <span class="code">grade ≥ 1</span>
    throughout. The reasoning made the model <strong>generous</strong>: thinking
    through each candidate, it called 150 of 300 records irrelevant where Claude
    called 213 irrelevant. Told to answer directly, it called 203 irrelevant.
    Sixty-five records it had talked itself into keeping went back to zero, and
    precision moved 0.497 → 0.711.</p>
  </section>

  <div class="stats">
    <div class="stat"><div class="k">binary agreement</div>
      <div class="v">{fmt(a['binary'], 2)}</div>
      <div class="sub">relevant vs not, both judges</div></div>
    <div class="stat"><div class="k">exact agreement</div>
      <div class="v">{fmt(a['exact'], 2)}</div>
      <div class="sub">same 0/1/2 grade</div></div>
    <div class="stat"><div class="k">cohen κ</div>
      <div class="v">{fmt(a['kappa'], 2)}</div>
      <div class="sub">chance-corrected</div></div>
    <div class="stat"><div class="k">schema compliance</div>
      <div class="v">{calls['json_ok']}/{calls['total']}</div>
      <div class="sub">{calls['missing_ids']} skipped, {calls['hallucinated_ids']} invented ids</div></div>
    <div class="stat"><div class="k">median call</div>
      <div class="v">{median_s:.0f}s</div>
      <div class="sub">{calls['output_tokens']:,} output tokens total</div></div>
    <div class="stat"><div class="k">reasoning captured</div>
      <div class="v">{calls['thinking_chars'] / 1000:.0f}k</div>
      <div class="sub">characters of thinking</div></div>
  </div>

  <section class="section">
    <h2>Every prompt, every candidate</h2>
    <p class="section-note">The disagreements are the part worth reading. A
    record marked red was injected by the model over Claude's zero; a record
    marked blue was dropped despite Claude rating it useful. The first wastes
    context, the second loses the answer.</p>

    <div class="legend">
      <span><b style="background:var(--accent);opacity:.45"></b> both judges agree</span>
      <span><b style="background:var(--hot)"></b> qwen kept, claude said 0</span>
      <span><b style="background:var(--cool)"></b> qwen dropped, claude said 1 or 2</span>
      <span>underlined <em>terse</em> = the grade changed when thinking was off</span>
    </div>

    <div class="toolbar">
      <button class="chip" data-filter="all" aria-pressed="true">all candidates</button>
      <button class="chip" data-filter="dis" aria-pressed="false">disagreements only</button>
      <button class="chip" data-filter="kept" aria-pressed="false">kept only</button>
      <button class="chip" data-filter="pref" aria-pressed="false">preferences only</button>
    </div>

    <div class="browser">
      <div class="list" id="list"></div>
      <div class="detail" id="detail"></div>
    </div>
  </section>

  {ce_section}

  {corpus_section}

  <section class="section">
    <h2>Where the two judges part</h2>
    <p class="section-note">Rows are Claude's grade, columns are the local
    model's. The diagonal is agreement.</p>
    <div class="scroll">
      <table class="conf">
        <thead><tr><th></th><th>qwen 0</th><th>qwen 1</th><th>qwen 2</th></tr></thead>
        <tbody>{conf_rows}</tbody>
      </table>
    </div>
  </section>

  <footer>
    model qwen-judge (derived from rafw007/Qwen3.6-35B-A3B-mlx-claude-coder-abliterated, red-team system prompt removed)<br>
    corpus: production variant, {s['candidates']} candidates from top-10 MiniLM retrieval · labels from the extractor variant sweep<br>
    run through the real BAML runtime via the QwenCoder client; reasoning read from the Collector's HTTP response
  </footer>
</div>
<script>window.__RUN__ = {json.dumps(data)};</script>
<script>{JS}</script>
"""


def main():
    data = payload()
    out = os.path.join(RESULTS, "local-judge.html")
    with open(out, "w") as fh:
        fh.write(build_html(data))
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

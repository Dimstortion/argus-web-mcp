"""Score native WebSearch/WebFetch vs Argus from the captured JSON files.
usage: score_native.py <workdir> <repo>"""
import json
import pathlib
import re
import statistics as st
import sys
from urllib.parse import urlsplit

S = pathlib.Path(sys.argv[1])
REPO = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(REPO / "benchmark"))
from run_compare import title_overlap  # noqa: E402
from scorer import token_f1  # noqa: E402


def load(name):
    p = S / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def host(u):
    h = urlsplit(u).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 2) if xs else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 2) if xs else None


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 2) if xs else None


out = []
P = out.append

# ---------------------------------------------------------------- search
ns, ad = load("native_search.json"), load("argus_direct.json")
if ns and ad:
    nat = {r["id"]: r for r in ns}
    rows = []
    for a in ad["search"]:
        n = nat.get(a["id"])
        if not n:
            continue
        text = "\n".join(t["text"] for t in n["tool_results"])
        m = re.search(r"Links: (\[.*?\])\n", text, re.S)
        links = []
        if m:
            try:
                links = json.loads(m.group(1))
            except Exception:
                links = []
        n_dom = {host(x.get("url", "")) for x in links}
        a_dom = {host(x.get("url", "")) for x in a["results"]}
        rows.append({
            "id": a["id"], "cat": a["category"],
            "n_links": len(links), "n_chars": len(text), "n_err": any(t["is_error"] for t in n["tool_results"]) or not links,
            "n_api_s": n.get("tool_s") if n.get("tool_s") is not None else (n["dur_api_ms"] or 0) / 1000, "n_cost": n["cost"],
            "n_top3": mean([title_overlap(a["query"], x.get("title", "")) for x in links[:3]]) or 0,
            "n_snippet": 0,
            "a_hits": a["n"], "a_chars": sum(len(x.get("snippet") or "") + len(x.get("title") or "") for x in a["results"]),
            "a_s": a["secs"], "a_cache": a["from_cache"], "a_err": bool(a["error"]) or a["n"] == 0,
            "a_top3": mean([title_overlap(a["query"], x.get("title", "")) for x in a["results"][:3]]) or 0,
            "a_snippet": sum(1 for x in a["results"] if x.get("snippet")),
            "a_engines": len(a.get("engines") or []),
            "dom_overlap": len(n_dom & a_dom), "n_dom": len(n_dom), "a_dom": len(a_dom),
        })
    P("## 1. Search: native WebSearch vs Argus `search` (24 queries, 8 categories)\n")
    P("| metric | native WebSearch | Argus search |\n|---|---|---|")
    P(f"| queries answered (no error, >0 results) | {sum(not r['n_err'] for r in rows)}/{len(rows)} | {sum(not r['a_err'] for r in rows)}/{len(rows)} |")
    P(f"| results per query (mean) | {mean([r['n_links'] for r in rows])} | {mean([r['a_hits'] for r in rows])} |")
    P(f"| unique domains per query (mean) | {mean([r['n_dom'] for r in rows])} | {mean([r['a_dom'] for r in rows])} |")
    P(f"| per-result snippet | none (links + one model-written summary) | {mean([r['a_snippet'] for r in rows])} of {mean([r['a_hits'] for r in rows])} hits |")
    P(f"| payload chars per query (mean) | {mean([r['n_chars'] for r in rows])} | {mean([r['a_chars'] for r in rows])} |")
    P(f"| top-3 title/query overlap (relevance proxy) | {mean([r['n_top3'] for r in rows])} | {mean([r['a_top3'] for r in rows])} |")
    P(f"| latency p50 / p95 (s) | {med([r['n_api_s'] for r in rows])} / {q([r['n_api_s'] for r in rows], .95)} (tool_use -> tool_result inside Claude Code) | {med([r['a_s'] for r in rows])} / {q([r['a_s'] for r in rows], .95)} (MCP round trip to the VPS) |")
    P(f"| served from cache | n/a | {sum(1 for r in rows if r['a_cache'])}/{len(rows)} |")
    P(f"| engines answering (mean) | 1 (undisclosed) | {mean([r['a_engines'] for r in rows])} |")
    P(f"| domain overlap native vs Argus (mean shared domains) | {mean([r['dom_overlap'] for r in rows])} | same |")
    P("| $ per query (tool only) | $0.01 search fee + result tokens | $0 (self-hosted) |")
    P("\nPer category (results / domains / top-3 overlap):\n")
    P("| category | native | Argus |\n|---|---|---|")
    for cat in sorted({r["cat"] for r in rows}):
        rs = [r for r in rows if r["cat"] == cat]
        P(f"| {cat} | {mean([r['n_links'] for r in rs])} / {mean([r['n_dom'] for r in rs])} / {mean([r['n_top3'] for r in rs])} | {mean([r['a_hits'] for r in rs])} / {mean([r['a_dom'] for r in rs])} / {mean([r['a_top3'] for r in rs])} |")
    P("")
    # research depth
    rr = ad["research"]
    P("## 2. Depth: one Argus `research(deep)` call per query (what native search cannot return)\n")
    P("| metric | Argus research(deep, 5 sources) |\n|---|---|")
    P(f"| queries with >=1 full source | {sum(1 for r in rr if r['n'] >= 1)}/{len(rr)} |")
    P(f"| sources per query (mean) | {mean([r['n'] for r in rr])} |")
    P(f"| full-content words per query (mean / median) | {mean([r['words'] for r in rr])} / {med([r['words'] for r in rr])} |")
    P(f"| latency p50 / p95 (s) | {med([r['secs'] for r in rr])} / {q([r['secs'] for r in rr], .95)} |")
    P(f"| failed source fetches (total) | {sum(r['failed'] for r in rr)} |")
    P("")

# ---------------------------------------------------------------- fetch/read
nf = load("native_fetch.json")
if nf and ad:
    nat = {r["id"]: r for r in nf}
    rows = []
    for a in ad["read"]:
        n = nat.get(a["id"])
        if not n:
            continue
        text = "\n".join(t["text"] for t in n["tool_results"])
        err = any(t["is_error"] for t in n["tool_results"]) or not text.strip()
        row = {"id": a["id"], "cat": a["category"], "url": a["url"],
               "n_words": len(text.split()), "n_err": err, "n_api_s": n.get("tool_s") if n.get("tool_s") is not None else (n["dur_api_ms"] or 0) / 1000, "n_cost": n["cost"],
               "a_words": a["words"], "a_err": bool(a["error"]) or a["words"] == 0, "a_s": a["secs"], "a_path": a["render_path"],
               "a_meta": sum(1 for k in ("title", "author", "published", "lang", "site") if (a.get("metadata") or {}).get(k) or (k == "title")),
               "n_text": text, "a_text": a["content"]}
        gold = REPO / "benchmark" / "gold" / f"{a['id']}.md"
        if gold.exists():
            g = gold.read_text(encoding="utf-8")
            row["gold_words"] = len(g.split())
            row["n_f1"] = round(token_f1(text, g)[2] if isinstance(token_f1(text, g), tuple) else token_f1(text, g), 3)
            row["a_f1"] = round(token_f1(a["content"], g)[2] if isinstance(token_f1(a["content"], g), tuple) else token_f1(a["content"], g), 3)
        rows.append(row)
    P("## 3. Read: native WebFetch vs Argus `read` (16 URLs: docs, long-form, news, forum, PDF, JS)\n")
    P("| metric | native WebFetch | Argus read |\n|---|---|---|")
    P(f"| URLs returned with content | {sum(not r['n_err'] for r in rows)}/{len(rows)} | {sum(not r['a_err'] for r in rows)}/{len(rows)} |")
    P(f"| words returned per URL (mean / median) | {mean([r['n_words'] for r in rows])} / {med([r['n_words'] for r in rows])} | {mean([r['a_words'] for r in rows])} / {med([r['a_words'] for r in rows])} |")
    P(f"| latency p50 / p95 (s) | {med([r['n_api_s'] for r in rows])} / {q([r['n_api_s'] for r in rows], .95)} (tool_use -> tool_result inside Claude Code) | {med([r['a_s'] for r in rows])} / {q([r['a_s'] for r in rows], .95)} (cold, cache-busted) |")
    P(f"| structured metadata (title/author/date/lang/site) | none | {mean([r['a_meta'] for r in rows])} of 5 fields populated |")
    P("| how the text is produced | page -> markdown -> a small model rewrites it against the prompt (lossy, non-verbatim) | deterministic extraction, full content, no model |")
    P("| caching | 15 min per URL, not controllable | per-source TTL, `from_cache` + `stale` flagged |")
    gold_rows = [r for r in rows if "a_f1" in r]
    if gold_rows:
        P("\nAgainst curated gold main-text (token F1, higher is better):\n")
        P("| item | gold words | native words | native F1 | Argus words | Argus F1 |\n|---|---|---|---|---|---|")
        for r in gold_rows:
            P(f"| {r['id']} | {r['gold_words']} | {r['n_words']} | {r['n_f1']} | {r['a_words']} | {r['a_f1']} |")
        P(f"| **mean** | | | **{mean([r['n_f1'] for r in gold_rows])}** | | **{mean([r['a_f1'] for r in gold_rows])}** |")
    P("\nPer URL:\n")
    P("| id | category | native words | Argus words | Argus path | Argus s | native s |\n|---|---|---|---|---|---|---|")
    for r in rows:
        P(f"| {r['id']} | {r['cat']} | {'ERR' if r['n_err'] else r['n_words']} | {'ERR' if r['a_err'] else r['a_words']} | {r['a_path']} | {r['a_s']} | {round(r['n_api_s'], 1)} |")
    P("")

# ---------------------------------------------------------------- agent level
fw = load("4way.json")
if fw:
    recs = fw if isinstance(fw, list) else fw.get("records", fw.get("rows", []))
    from run_4way import aggregate_condition, render_report
    by = {}
    for cond in ("claude-native", "claude-argus"):
        rs = [r for r in recs if r.get("condition") == cond]
        if rs:
            by[cond] = aggregate_condition(rs)
    P("## 4. Agent level: `claude -p` end to end (24 scenarios, same prompt, same model)\n")
    P("Condition `claude-native` = WebSearch + WebFetch only; `claude-argus` = Argus MCP (research/search/read) only. Cost is the CLI-reported figure.\n")
    P(render_report(by, recs))
    P("")

(S / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))

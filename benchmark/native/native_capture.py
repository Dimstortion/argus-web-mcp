"""Capture the RAW output of Claude Code's native WebSearch / WebFetch tools.

A headless `claude -p` run (haiku, cheap: the model only relays) with stream-json output
exposes every tool_result block verbatim, so what is measured is the tool's own output,
not a model's paraphrase of it. Resumable: ids already in the output file are skipped.
usage: native_capture.py <workdir> search|fetch  (workdir holds compare24.json + testset.yaml; cwd/ is an empty dir the CLI runs in)
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import yaml

S = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
CWD = S / "cwd"
CWD.mkdir(exist_ok=True)
OUT = S / f"native_{MODE}.json"
CLAUDE = shutil.which("claude")
ENV = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
FETCH_PROMPT = ("Return the complete main content of this page as markdown, verbatim and "
                "complete. Do not summarize, shorten, or omit sections.")


def _ms(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts) if ts > 1e11 else ts * 1000
    from datetime import datetime
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp() * 1000


def run(prompt: str, tool: str) -> dict:
    cmd = [CLAUDE, "-p", prompt, "--setting-sources", "", "--disable-slash-commands",
           "--tools", tool, "--model", "claude-haiku-5-5", "--output-format", "stream-json",
           "--verbose", "--max-turns", "3"]
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=CWD, env=ENV, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                           timeout=300)
        out, err, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, err, code = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout", -1
    rec = {"wall_s": round(time.perf_counter() - t, 1), "exit": code, "tool_results": [],
           "tool_uses": [], "usage": None, "cost": None, "dur_api_ms": None, "tool_s": None,
           "err": (err or "")[-300:] if code else None}
    for line in out.splitlines():
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") == "result":
            rec["usage"] = o.get("usage")
            rec["cost"] = o.get("total_cost_usd")
            rec["dur_api_ms"] = o.get("duration_api_ms")
        if o.get("type") in ("user", "assistant"):
            ts = o.get("timestamp")
            for c in (o.get("message") or {}).get("content", []):
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_result":
                    cont = c.get("content")
                    text = cont if isinstance(cont, str) else "\n".join(
                        x.get("text", "") for x in cont if isinstance(x, dict))
                    rec["tool_results"].append({"text": text, "is_error": bool(c.get("is_error"))})
                    if ts and rec.get("_t_use") and rec["tool_s"] is None:
                        rec["tool_s"] = round((_ms(ts) - rec["_t_use"]) / 1000, 2)  # tool_use -> tool_result
                if c.get("type") == "tool_use":
                    rec["tool_uses"].append(c.get("input"))
                    if ts and not rec.get("_t_use"):
                        rec["_t_use"] = _ms(ts)
    rec.pop("_t_use", None)
    return rec


if MODE == "search":
    items = [(s["id"], s["query"], s["category"]) for s in json.loads((S / "compare24.json").read_text(encoding="utf-8"))]
    tool = "WebSearch"
    def prompt(x):
        return (f"Use the WebSearch tool exactly once with the query: {x}\n"
                "After the tool returns, reply with exactly: done")
else:
    ts = yaml.safe_load(open(S / "testset.yaml", encoding="utf-8"))
    items = [(u["id"], u["url"], u.get("category", "")) for u in ts["urls"]]
    tool = "WebFetch"
    def prompt(x):
        return (f'Use the WebFetch tool exactly once on the URL {x} with this prompt: "{FETCH_PROMPT}"\n'
                "After the tool returns, reply with exactly: done")

done = {}
if OUT.exists():
    done = {r["id"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))}
for i, (iid, x, cat) in enumerate(items, 1):
    if iid in done and done[iid]["tool_results"]:
        continue
    rec = run(prompt(x), tool)
    rec.update({"id": iid, "input": x, "category": cat})
    done[iid] = rec
    OUT.write_text(json.dumps(list(done.values()), indent=1), encoding="utf-8")
    n = len(rec["tool_results"])
    print(f"{i}/{len(items)} {iid} results={n} chars={sum(len(r['text']) for r in rec['tool_results'])} "
          f"tool={rec['tool_s']}s api={rec['dur_api_ms']} wall={rec['wall_s']} cost={rec['cost']}", flush=True)
print("done", OUT)

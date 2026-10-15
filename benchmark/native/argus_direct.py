"""Direct Argus tool measurements over MCP (the live VPS, bearer from ~/.claude.json):
search + research(deep) per query, read per URL. Timings are client-side wall clock.
usage: argus_direct.py <workdir> <repo>
"""
import asyncio
import json
import pathlib
import sys
import time

import yaml

S = pathlib.Path(sys.argv[1])
REPO = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(REPO / "benchmark"))
from fastmcp import Client  # noqa: E402
from fastmcp.client.transports import StreamableHttpTransport  # noqa: E402
from run_4way import _argus_url_and_token, argus_mcp_config  # noqa: E402

url, token = _argus_url_and_token(argus_mcp_config())
OUT = S / "argus_direct.json"
TS = f"argus_bench={int(time.time())}"  # cache-buster for read (Argus caches by URL)


async def call(c, name, args):
    t = time.perf_counter()
    try:
        r = await asyncio.wait_for(c.call_tool(name, args, raise_on_error=False), 200)
        d = json.loads(r.content[0].text)
    except Exception as e:  # noqa: BLE001
        d = {"error": f"client:{e!r}"}
    return d, round(time.perf_counter() - t, 2)


async def main():
    queries = json.loads((S / "compare24.json").read_text(encoding="utf-8"))
    ts = yaml.safe_load(open(S / "testset.yaml", encoding="utf-8"))
    res = {"search": [], "research": [], "read": []}
    async with Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {token}"})) as c:
        for i, q in enumerate(queries, 1):
            d, dt = await call(c, "search", {"query": q["query"], "count": 10})
            res["search"].append({"id": q["id"], "category": q["category"], "query": q["query"], "secs": dt,
                                  "n": len(d.get("results", [])), "from_cache": d.get("from_cache"),
                                  "degraded": d.get("degraded"), "engines": d.get("engines_used"),
                                  "error": d.get("error"), "results": d.get("results", [])})
            d, dt = await call(c, "research", {"query": q["query"], "mode": "deep", "max_sources": 5})
            srcs = d.get("sources", [])
            res["research"].append({"id": q["id"], "category": q["category"], "query": q["query"], "secs": dt,
                                    "n": len(srcs), "words": sum(s.get("word_count") or len((s.get("content") or "").split()) for s in srcs),
                                    "failed": len(d.get("failed", [])), "from_cache": d.get("from_cache"),
                                    "error": d.get("error"), "urls": [s.get("url") for s in srcs]})
            print(f"{i}/{len(queries)} {q['id']} search {res['search'][-1]['n']} hits {dt}s | research {len(srcs)} src {res['research'][-1]['words']} words {res['research'][-1]['secs']}s", flush=True)
            OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        for i, u in enumerate(ts["urls"], 1):
            sep = "&" if "?" in u["url"] else "?"
            d, dt = await call(c, "read", {"url": u["url"] + sep + TS})
            content = d.get("content") or ""
            res["read"].append({"id": u["id"], "category": u.get("category"), "url": u["url"], "secs": dt,
                                "status": d.get("status"), "words": len(content.split()), "chars": len(content),
                                "render_path": d.get("render_path"), "from_cache": d.get("from_cache"),
                                "error": d.get("error"), "code": d.get("code"), "metadata": d.get("metadata"),
                                "content": content})
            print(f"read {i}/{len(ts['urls'])} {u['id']} status {d.get('status')} words {len(content.split())} path {d.get('render_path')} {dt}s err {d.get('error')}", flush=True)
            OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("done", OUT)

asyncio.run(main())

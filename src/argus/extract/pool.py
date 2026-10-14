"""``extract_article`` off the event loop AND off the GIL.

trafilatura's pruning is pure Python, so threads never ran pages in parallel: 10 pages
took 10.5 s sequentially, 10.7 s on 2 threads and 14.0 s on 4 (GIL thrash), against
6.3 s on 2 processes (measured 2026-10-15). ``research``/``batch_read`` fan out 5-8
extractions at once, so on the 2-vCPU VPS they were slower than serial and starved the
loop. Workers are spawned (never forked: the parent runs ONNX and Chromium threads) and
import only the extract package. ``ARGUS_EXTRACT_WORKERS=0`` keeps the thread path
(tests, tiny hosts).
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import functools
import multiprocessing
import os

from .article import extract_article

_pool: cf.ProcessPoolExecutor | None = None


def workers() -> int:
    return int(os.environ.get("ARGUS_EXTRACT_WORKERS", str(min(2, os.cpu_count() or 1))))


def _get_pool() -> cf.ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = cf.ProcessPoolExecutor(
            workers(), mp_context=multiprocessing.get_context("spawn")
        )
    return _pool


async def extract_async(html: str, url: str, **kw):
    """``extract_article`` on a worker process (or a thread when workers() == 0)."""
    if workers() <= 0:
        return await asyncio.to_thread(extract_article, html, url, **kw)
    fn = functools.partial(extract_article, html, url, **kw)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_get_pool(), fn)
    except cf.process.BrokenProcessPool:  # a worker died (OOM kill): rebuild once, retry
        shutdown()
        return await loop.run_in_executor(_get_pool(), fn)


def warm() -> None:
    """Spawn the workers now so the first read does not pay their import (~1 s each)."""
    if workers() > 0:
        for _ in range(workers()):
            _get_pool().submit(extract_article, "<p>warm</p>", "https://localhost/warm")


def shutdown() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None

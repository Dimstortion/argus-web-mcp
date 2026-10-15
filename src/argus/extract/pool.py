"""``extract_article`` off the event loop AND off the GIL.

trafilatura's pruning is pure Python, so threads never ran pages in parallel: 10 pages
took 10.5 s sequentially, 10.7 s on 2 threads and 14.0 s on 4 (GIL thrash), against
6.3 s on 2 processes (measured 2026-10-15). ``research``/``batch_read`` fan out 5-8
extractions at once, so on the 2-vCPU VPS they were slower than serial and starved the
loop. Workers are spawned (never forked: the parent runs ONNX and Chromium threads) and
import only the extract package. ``ARGUS_EXTRACT_WORKERS=0`` keeps the thread path
(tests, tiny hosts). Spawn re-imports the main module in each worker, so a script that
calls into Argus needs an ``if __name__ == "__main__"`` guard (uvicorn has one) or that
setting; without either, every extraction fails as ``extract_failed``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import functools
import logging
import multiprocessing
import os

from ..config import _int
from .article import extract_article

logger = logging.getLogger(__name__)
_pool: cf.ProcessPoolExecutor | None = None
# Typical pages take 0.35-1.5 s. A worker stuck past this is killed: cancelling the await
# never stops a process, and two wedged workers would otherwise queue every later read.
_EXTRACT_TIMEOUT_S = 30.0


def workers() -> int:
    return _int("ARGUS_EXTRACT_WORKERS", min(2, os.cpu_count() or 1))


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
    pool = _get_pool()
    try:
        return await asyncio.wait_for(loop.run_in_executor(pool, fn), _EXTRACT_TIMEOUT_S)
    except TimeoutError:
        logger.warning("extraction of %s exceeded %gs; recycling workers", url, _EXTRACT_TIMEOUT_S)
        _retire(pool, kill=True)
        return extract_article("", url, **kw)  # same shape, empty content -> empty_content
    except cf.process.BrokenProcessPool:  # a worker died (OOM kill): rebuild once, retry
        _retire(pool)
        pool = _get_pool()
        try:
            return await loop.run_in_executor(pool, fn)
        except (cf.process.BrokenProcessPool, OSError):  # cannot spawn here: serve anyway
            logger.warning("extraction pool unavailable; extracting on a thread")
            _retire(pool)
            return await asyncio.to_thread(extract_article, html, url, **kw)


def _retire(pool: cf.ProcessPoolExecutor, kill: bool = False) -> None:
    """Drop ``pool`` only if it is still the current one: every caller in flight sees the
    same broken pool, and the second must reuse the first's replacement, not kill it.
    ``kill`` ends its workers too (a wedged one ignores shutdown); callers still waiting
    on it get BrokenProcessPool and take the rebuild path."""
    if _pool is pool:
        if kill:
            for proc in list((pool._processes or {}).values()):  # no public API before 3.14
                proc.kill()
        shutdown()


def warm() -> None:
    """Spawn the workers now so the first read does not pay their import (~1 s each)."""
    if workers() > 0:
        for _ in range(workers()):
            fut = _get_pool().submit(extract_article, "<p>warm</p>", "https://localhost/warm")
            fut.add_done_callback(_log_warm)


def _log_warm(fut: cf.Future) -> None:
    if (exc := fut.exception()) is not None:  # the first read would hit the same error
        logger.warning("extraction worker warm-up failed: %r", exc)


def shutdown() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None

"""extract_async: same result as extract_article, on a real spawned worker and on the
thread path, and a dead pool is rebuilt instead of failing every later read."""


import asyncio
import concurrent.futures as cf

import pytest

from argus.extract import pool
from argus.extract.article import extract_article

HTML = ("<html><body><article><h1>Title</h1>" + "<p>Real paragraph text here.</p>" * 5
        + "</article></body></html>")
URL = "https://example.com/a"


@pytest.fixture(autouse=True)
def _fresh_pool():
    pool.shutdown()
    yield
    pool.shutdown()


async def test_thread_path_when_workers_is_zero(monkeypatch):
    monkeypatch.setenv("ARGUS_EXTRACT_WORKERS", "0")
    assert await pool.extract_async(HTML, URL) == extract_article(HTML, URL)
    assert pool._pool is None


async def test_process_path_matches_inline(monkeypatch):
    monkeypatch.setenv("ARGUS_EXTRACT_WORKERS", "1")
    pool.warm()
    got = await pool.extract_async(HTML, URL, fmt="text")
    assert got == extract_article(HTML, URL, fmt="text")
    assert got["content"] and got["format"] == "text"


async def test_broken_pool_is_rebuilt_once_for_all_callers_in_flight(monkeypatch):
    monkeypatch.setenv("ARGUS_EXTRACT_WORKERS", "1")
    pool.warm()
    first = pool._get_pool()
    first._broken = "killed"  # what a worker OOM kill leaves behind: every submit raises
    made = []
    real = pool.cf.ProcessPoolExecutor

    class Counted(real):
        def __init__(self, *a, **k):
            made.append(self)
            super().__init__(*a, **k)

    monkeypatch.setattr(pool.cf, "ProcessPoolExecutor", Counted)
    # research/batch_read: several extractions hit the same dead pool at once
    got = await asyncio.gather(*(pool.extract_async(HTML, URL) for _ in range(3)))
    assert all(g["content"] for g in got)
    assert len(made) == 1 and pool._pool is made[0]  # one replacement, nobody killed it


async def test_pool_that_cannot_be_rebuilt_falls_back_to_a_thread(monkeypatch):
    monkeypatch.setenv("ARGUS_EXTRACT_WORKERS", "1")

    class Dead(cf.ProcessPoolExecutor):
        def submit(self, *a, **k):
            raise cf.process.BrokenProcessPool("no workers here")

    monkeypatch.setattr(pool, "_get_pool", lambda: Dead(1))
    assert (await pool.extract_async(HTML, URL))["content"] == extract_article(HTML, URL)["content"]


def test_warm_logs_a_failed_worker(caplog):
    fut = cf.Future()
    fut.set_exception(ImportError("no trafilatura"))
    with caplog.at_level("WARNING", logger="argus.extract.pool"):
        pool._log_warm(fut)
    assert "warm-up failed" in caplog.text


def test_bad_worker_count_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ARGUS_EXTRACT_WORKERS", "two")
    assert pool.workers() >= 1


async def test_wedged_extraction_is_killed_and_the_pool_recovers(monkeypatch):
    monkeypatch.setenv("ARGUS_EXTRACT_WORKERS", "1")
    monkeypatch.setattr(pool, "_EXTRACT_TIMEOUT_S", 0.0)  # every extraction "hangs"
    first = pool._get_pool()
    got = await pool.extract_async(HTML, URL)
    assert got["content"] == "" and pool._pool is None  # empty result, workers recycled
    monkeypatch.setattr(pool, "_EXTRACT_TIMEOUT_S", 30.0)
    assert (await pool.extract_async(HTML, URL))["content"]
    assert pool._pool is not first

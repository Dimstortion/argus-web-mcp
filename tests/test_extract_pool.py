"""extract_async: same result as extract_article, on a real spawned worker and on the
thread path, and a dead pool is rebuilt instead of failing every later read."""


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


async def test_broken_pool_is_rebuilt_once(monkeypatch):
    monkeypatch.setenv("ARGUS_EXTRACT_WORKERS", "1")
    pool.warm()
    first = pool._get_pool()
    first._broken = "killed"  # what a worker OOM kill leaves behind: every submit raises
    got = await pool.extract_async(HTML, URL)
    assert got["content"] and pool._pool is not first

"""The paced tab pool: pacing, tab ceiling, reuse, and the shared 429 backoff.

The whole point of this module is browser BEHAVIOUR — how many tabs open, whether they
are reused, how long we wait between loads — so these tests drive the real `drain` with a
fake fetch and a fake clock. A test that mocked `drain` itself would assert nothing about
the thing that went wrong in production.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from xbrain.refetch_pool import (
    DEFAULT_TABS,
    MAX_TABS,
    PAUSE_MAX_MS,
    PAUSE_MIN_MS,
    MAX_RATE_LIMIT_BACKOFFS,
    RefetchRateLimited,
    backoff_ms,
    clamp_tabs,
    drain,
    pause_ms,
)


class Recorder:
    """A fake fetch + clock that records which tab handled what, and every wait."""

    def __init__(self, texts: dict[str, str | None] | None = None) -> None:
        self.texts = texts or {}
        self.by_tab: dict[int, list[str]] = {}
        self.waits: list[float] = []
        self.results: list[tuple[str, str | None]] = []

    async def fetch(self, index: int, url: str) -> str | None:
        self.by_tab.setdefault(index, []).append(url)
        # Yield to the loop, because a real `page.goto` always suspends on I/O. Without
        # this the fake never suspends, so the first worker drains the whole queue before
        # any other is scheduled — an artefact of the fake, not of the pool.
        await asyncio.sleep(0)
        return self.texts.get(url, f"body of {url}")

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)

    def on_result(self, url: str, text: str | None) -> None:
        self.results.append((url, text))


def run(urls, *, tabs=1, rec=None, **kwargs):
    rec = rec or Recorder()
    asyncio.run(
        drain(
            urls,
            tabs=tabs,
            fetch=rec.fetch,
            sleep=rec.sleep,
            on_result=rec.on_result,
            rng=random.Random(0),
            **kwargs,
        )
    )
    return rec


# --- the tab ceiling -------------------------------------------------------------


def test_clamp_tabs_defaults_and_never_exceeds_the_ceiling():
    assert clamp_tabs(None) == DEFAULT_TABS
    assert clamp_tabs(2) == 2
    assert clamp_tabs(MAX_TABS) == MAX_TABS
    assert clamp_tabs(MAX_TABS + 50) == MAX_TABS


def test_clamp_tabs_floors_at_one_so_a_bad_value_never_makes_a_pool_that_does_nothing():
    """0 tabs would drain nothing and exit 0 — indistinguishable from "everything failed"."""
    assert clamp_tabs(0) == 1
    assert clamp_tabs(-3) == 1


def test_the_pool_opens_no_more_workers_than_the_ceiling_even_when_asked_for_more():
    rec = run([f"u{i}" for i in range(20)], tabs=99)
    assert len(rec.by_tab) <= MAX_TABS


# --- tab REUSE: the regression this module exists for ----------------------------


def test_one_tab_handles_every_url_in_sequence_rather_than_one_tab_per_url():
    """The bug: `browser_text_fetcher` did new_page()/close() per item — 749 tabs in a row.

    A worker owns its tab for the whole run, so with one tab every URL must arrive at
    index 0, in order.
    """
    urls = ["a", "b", "c", "d"]
    rec = run(urls, tabs=1)
    assert rec.by_tab == {0: urls}


def test_every_url_is_visited_exactly_once_across_the_pool():
    urls = [f"u{i}" for i in range(31)]
    rec = run(urls, tabs=MAX_TABS)
    visited = [u for tab in rec.by_tab.values() for u in tab]
    assert sorted(visited) == sorted(urls)
    assert len(visited) == len(urls)


def test_the_work_actually_spreads_across_the_tabs_instead_of_one_tab_taking_it_all():
    """Asking for N tabs must buy N tabs making progress, not one tab and N-1 idle."""
    rec = run([f"u{i}" for i in range(40)], tabs=MAX_TABS)
    assert len(rec.by_tab) == MAX_TABS
    assert all(len(handled) > 1 for handled in rec.by_tab.values())


def test_on_result_fires_once_per_url_including_the_ones_that_yielded_nothing():
    """A failed re-fetch must still be counted, or the checkpoint cadence drifts."""
    rec = run(["a", "b"], tabs=1, rec=Recorder({"a": None}))
    assert rec.results == [("a", None), ("b", "body of b")]


def test_a_tab_that_raises_does_not_kill_the_pool_and_reports_no_text():
    class Exploding(Recorder):
        async def fetch(self, index: int, url: str) -> str | None:
            if url == "boom":
                raise RuntimeError("navigation failed")
            return await super().fetch(index, url)

    rec = run(["a", "boom", "c"], tabs=1, rec=Exploding())
    assert dict(rec.results) == {"a": "body of a", "boom": None, "c": "body of c"}


# --- pacing ----------------------------------------------------------------------


def test_pause_is_random_inside_the_configured_window():
    rng = random.Random(1234)
    values = {pause_ms(rng) for _ in range(200)}
    assert all(PAUSE_MIN_MS <= v <= PAUSE_MAX_MS for v in values)
    # A constant delay is as mechanical as no delay: the window must actually be used.
    assert len(values) > 50


def test_a_pause_separates_consecutive_loads_on_the_same_tab():
    rec = run(["a", "b", "c"], tabs=1)
    assert len(rec.waits) == 2  # 3 loads, 2 gaps
    assert all(PAUSE_MIN_MS / 1000 <= w <= PAUSE_MAX_MS / 1000 for w in rec.waits)


def test_the_first_load_of_each_tab_waits_for_nothing():
    """One item must not cost a 30 s wait before a single byte is fetched."""
    assert run(["only"], tabs=1).waits == []
    assert run(["a", "b", "c"], tabs=3).waits == []


def test_backoff_is_random_inside_its_own_window():
    rng = random.Random(7)
    values = {backoff_ms(rng) for _ in range(100)}
    assert all(60_000 <= v <= 180_000 for v in values)
    assert len(values) > 20


# --- the shared 429 backoff ------------------------------------------------------


def test_a_429_parks_the_pool_for_a_backoff_and_then_carries_on():
    flag = {"limited": True}
    rec = Recorder()
    asyncio.run(
        drain(
            ["a", "b"],
            tabs=1,
            fetch=rec.fetch,
            sleep=rec.sleep,
            on_result=rec.on_result,
            rate_limited=lambda: flag["limited"],
            clear_rate_limit=lambda: flag.update(limited=False),
            rng=random.Random(0),
        )
    )
    assert flag["limited"] is False
    # The first wait is the backoff (>= 60 s), not an ordinary 5-30 s page pause.
    assert rec.waits[0] >= 60
    assert len(rec.results) == 2


def test_one_429_costs_ONE_backoff_even_with_several_tabs_running():
    """Three tabs seeing one rate limit must not spend three separate backoffs."""
    flag = {"limited": True}
    rec = Recorder()
    asyncio.run(
        drain(
            [f"u{i}" for i in range(9)],
            tabs=3,
            fetch=rec.fetch,
            sleep=rec.sleep,
            on_result=rec.on_result,
            rate_limited=lambda: flag["limited"],
            clear_rate_limit=lambda: flag.update(limited=False),
            rng=random.Random(0),
        )
    )
    assert len([w for w in rec.waits if w >= 60]) == 1


def test_the_pool_gives_up_rather_than_keep_poking_a_rate_limited_endpoint():
    """Past the budget it raises: pushing through a 429 is what escalates to a ban."""
    rec = Recorder()
    with pytest.raises(RefetchRateLimited):
        asyncio.run(
            drain(
                [f"u{i}" for i in range(50)],
                tabs=1,
                fetch=rec.fetch,
                sleep=rec.sleep,
                on_result=rec.on_result,
                rate_limited=lambda: True,  # never clears
                clear_rate_limit=lambda: None,
                rng=random.Random(0),
            )
        )
    assert len([w for w in rec.waits if w >= 60]) == MAX_RATE_LIMIT_BACKOFFS


def test_results_gathered_before_the_rate_limit_are_kept_by_the_caller():
    """The caller checkpoints from `on_result`, so whatever it saw survives the raise."""
    calls = {"n": 0}

    def limited() -> bool:
        # Clean for the first two items, then limited forever.
        calls["n"] += 1
        return calls["n"] > 2

    rec = Recorder()
    with pytest.raises(RefetchRateLimited):
        asyncio.run(
            drain(
                [f"u{i}" for i in range(30)],
                tabs=1,
                fetch=rec.fetch,
                sleep=rec.sleep,
                on_result=rec.on_result,
                rate_limited=limited,
                clear_rate_limit=lambda: None,
                rng=random.Random(0),
            )
        )
    assert len(rec.results) >= 2

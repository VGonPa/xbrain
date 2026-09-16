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
from playwright._impl._errors import TargetClosedError

from xbrain.refetch_pool import (
    DEFAULT_TABS,
    MAX_TABS,
    PAUSE_MAX_MS,
    PAUSE_MIN_MS,
    MAX_RATE_LIMIT_BACKOFFS,
    RATE_LIMIT_BACKOFF_MIN_MS,
    RefetchBrowserClosed,
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


# --- the navigation gate: no tab starts a load while a 429 is unresolved ------------


class Interleaved(Recorder):
    """A fake whose loads and waits take TIME, so the tabs genuinely overlap.

    `Recorder.sleep` returns without suspending, so under it a backoff never overlaps any
    other tab's work and a gate that leaks cannot be seen. Here time is event-loop ticks:
    a load is one, a page pause `PAUSE_TICKS`, a backoff `BACKOFF_TICKS` — long enough for
    every other tab to come round to its next load while the backoff is still running.
    """

    PAUSE_TICKS = 2
    BACKOFF_TICKS = 20

    def __init__(self, *, throttled_url: str, late: bool) -> None:
        super().__init__()
        self.throttled_url = throttled_url
        self.late = late
        self.limited = False
        self.backing_off = False
        # (url, 429 pending, backoff running) at the instant each load began.
        self.starts: list[tuple[str, bool, bool]] = []
        self.events: list[tuple[str, asyncio.Task | None]] = []

    async def fetch(self, index: int, url: str) -> str | None:
        self.starts.append((url, self.limited, self.backing_off))
        self.events.append(("load", asyncio.current_task()))
        if url == self.throttled_url:
            if self.late:
                asyncio.ensure_future(self._late_429())
            else:
                self.limited = True
        return await super().fetch(index, url)

    async def _late_429(self) -> None:
        # Production reads each response in a task its listener schedules, so the flag
        # rises AFTER `goto` has returned — by then the tab can already be in its pause.
        await asyncio.sleep(0)
        self.limited = True

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        backoff = seconds >= RATE_LIMIT_BACKOFF_MIN_MS / 1000
        self.events.append(("backoff" if backoff else "pause", asyncio.current_task()))
        if backoff:
            self.backing_off = True
        for _ in range(self.BACKOFF_TICKS if backoff else self.PAUSE_TICKS):
            await asyncio.sleep(0)
        if backoff:
            self.backing_off = False
            self.events.append(("backoff_end", asyncio.current_task()))

    def rate_limited(self) -> bool:
        return self.limited

    def clear_rate_limit(self) -> None:
        self.limited = False


def run_interleaved(urls, *, tabs, rec):
    return run(
        urls,
        tabs=tabs,
        rec=rec,
        rate_limited=rec.rate_limited,
        clear_rate_limit=rec.clear_rate_limit,
    )


@pytest.mark.parametrize(
    ("tabs", "urls", "throttled", "late"),
    [
        # Tab 1's load draws the 429; tab 0 takes the backoff; tab 1 must not load again
        # until it is over. Before: the backoff CLEARED the flag and then slept, so tab 1
        # read "not limited" and loaded in the middle of it.
        (2, ["a", "b", "c", "d", "e", "f"], "b", False),
        # The 429 is read after `goto` returned, while the tab is pausing. Before: the flag
        # was polled BEFORE the pause, so the tab woke up and loaded straight into it.
        (1, ["a", "b", "c"], "a", True),
    ],
    ids=["429-during-another-tabs-load", "429-read-after-the-load-returned"],
)
def test_no_tab_starts_a_load_while_a_429_is_pending_or_being_backed_off(
    tabs, urls, throttled, late
):
    rec = Interleaved(throttled_url=throttled, late=late)
    run_interleaved(urls, tabs=tabs, rec=rec)

    # The scenario must actually have hit a backoff, or "nothing leaked" is vacuous.
    assert len([w for w in rec.waits if w >= RATE_LIMIT_BACKOFF_MIN_MS / 1000]) == 1
    assert sorted(url for url, _ in rec.results) == sorted(urls)
    leaked = [start for start in rec.starts if start[1] or start[2]]
    assert leaked == [], "a tab loaded while a 429 was pending or a backoff was running"


def test_tabs_held_by_a_backoff_pause_again_instead_of_all_loading_the_instant_it_ends():
    """Every tab parked at the gate is released at the same moment the backoff ends.

    Loading them all in that tick would answer a rate limit with a synchronised burst of
    `tabs` requests — the exact pattern the pacing exists to avoid. Each must pause first.
    """
    rec = Interleaved(throttled_url="u0", late=False)
    run_interleaved([f"u{i}" for i in range(9)], tabs=3, rec=rec)

    end = max(i for i, (kind, _) in enumerate(rec.events) if kind == "backoff_end")
    after = rec.events[end + 1 :]
    resumed = {task for kind, task in after if kind == "load"}
    assert len(resumed) == 3, "every tab must have loaded again after the backoff"
    for task in resumed:
        first_load = next(i for i, (kind, t) in enumerate(after) if kind == "load" and t is task)
        assert ("pause", task) in after[:first_load]


# --- a closed browser is a stop, not a run of empty results -----------------------


def test_a_closed_browser_stops_the_pool_instead_of_reporting_the_rest_as_empty():
    """The operator closes the headful window while tab 0 loads `c`.

    From then on every call on every tab raises Playwright's own `TargetClosedError`. The
    pool used to swallow each one as "this post yielded nothing" and keep going: every
    remaining item was counted as attempted, and the run finished as if it had worked.
    """

    class BrowserClosed(Recorder):
        closed = False

        async def fetch(self, index: int, url: str) -> str | None:
            if url == "c":
                self.closed = True
            if self.closed:
                raise TargetClosedError()
            return await super().fetch(index, url)

    rec = BrowserClosed()
    with pytest.raises(RefetchBrowserClosed) as excinfo:
        run(["a", "b", "c", "d", "e", "f"], tabs=2, rec=rec)

    assert isinstance(excinfo.value.__cause__, TargetClosedError)
    # What finished before the closure reached the caller (which checkpoints from it), and
    # nothing after it was reported as an empty result.
    assert rec.results == [("a", "body of a"), ("b", "body of b")]


def test_a_browser_closed_mid_run_checkpoints_the_repairs_and_fails_the_run(monkeypatch, tmp_path):
    """The same closure through `refetch_full_texts_pooled`, the function the CLI calls.

    Before the fix it returned "1 repaired" and `refetch-truncated` exited 0, so an operator
    who had closed the browser was told the run had completed.
    """
    from contextlib import asynccontextmanager
    from datetime import datetime, timezone

    from tests.test_fetch_x import _tweet_detail_payload
    from xbrain import cli, fetch_x, refetch_pool
    from xbrain.models import Author, Item

    monkeypatch.setattr(refetch_pool, "PAUSE_MIN_MS", 0)
    monkeypatch.setattr(refetch_pool, "PAUSE_MAX_MS", 0)

    class TweetDetail:
        status = 200
        url = "https://x.com/i/api/graphql/q/TweetDetail"

        async def json(self) -> dict:
            return _tweet_detail_payload()

    class Page:
        def __init__(self) -> None:
            self.listeners: list = []

        def on(self, event: str, listener) -> None:
            self.listeners.append(listener)

        async def goto(self, url: str, wait_until: str | None = None) -> None:
            if url.endswith("/status/200"):
                raise TargetClosedError()
            for listener in self.listeners:
                listener(TweetDetail())

        async def wait_for_timeout(self, ms: float) -> None:
            for _ in range(3):
                await asyncio.sleep(0)

        async def close(self) -> None:
            return None

    class Context:
        async def new_page(self) -> Page:
            return Page()

    @asynccontextmanager
    async def fake_x_context_async(storage_state_path, headless=False):
        yield Context()

    monkeypatch.setattr(fetch_x, "x_context_async", fake_x_context_async)

    def mk(rest_id: str, text: str) -> Item:
        return Item(
            id=rest_id,
            source="bookmark",
            url=f"https://x.com/bob/status/{rest_id}",
            author=Author(handle="bob", name="Bob"),
            text=text,
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            captured_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )

    repaired, lost = mk("100", "The truncated post, now"), mk("200", "Cut off at the")
    checkpointed: list[str] = []

    with pytest.raises(RefetchBrowserClosed) as excinfo:
        fetch_x.refetch_full_texts_pooled(
            {"100": repaired, "200": lost},
            [repaired, lost],
            tmp_path / "state.json",
            tabs=1,
            checkpoint=lambda: checkpointed.append(repaired.text),
        )

    assert checkpointed[-1] == "The truncated post, now complete."
    assert lost.text == "Cut off at the"
    # The CLI turns this into `Error: …` + exit 1, not a traceback and not exit 0.
    assert isinstance(excinfo.value, cli._OPERATOR_ERRORS)

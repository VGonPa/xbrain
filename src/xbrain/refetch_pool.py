"""A paced pool of reused browser tabs for the truncated-text re-fetch.

`browser_text_fetcher` opened a tab, navigated, and closed it — **per item**. Over the
749 truncated items in the real store that is 749 tab open/close cycles in a row, with
no pause between them: a visible storm of windows, and a request pattern no human
produces. X does not have to ban the account for that to hurt; it only has to start
serving the rate-limited timeline, and then the run silently repairs nothing.

This module replaces that with the shape a person actually has open: a handful of tabs,
each NAVIGATED IN PLACE from one post to the next, with a random pause between loads.
Nothing here opens a browser per item, and nothing opens more than `MAX_TABS`.

The concurrency core (`drain`) takes its fetch, its pause and its clock as arguments, so
the ordering and pacing rules are tested without a browser. Production binds them to
Playwright in `xbrain.fetch_x`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)

# Deliberately slow, human-paced. A real reader lingers on a post; the previous
# implementation navigated as fast as Chromium could load, which is the single most
# obvious bot signal a scroll-free workload emits. The window is wide on purpose:
# a CONSTANT delay is just as mechanical as no delay at all.
PAUSE_MIN_MS = 5_000
PAUSE_MAX_MS = 30_000

# A person has a few tabs open, not fifty. More tabs is more parallel load against the
# same account at the same instant, which is exactly the burst the pacing exists to
# avoid — so this is a hard ceiling, not a default to raise under pressure.
DEFAULT_TABS = 3
MAX_TABS = 4

# When X answers with HTTP 429 the only safe move is to stop poking it. Same budget and
# the same randomized stretch the timeline scroll already uses (`extract.extractor`):
# pushing through a rate limit is what escalates to a suspension.
RATE_LIMIT_BACKOFF_MIN_MS = 60_000
RATE_LIMIT_BACKOFF_MAX_MS = 180_000
MAX_RATE_LIMIT_BACKOFFS = 3


class RefetchRateLimited(RuntimeError):
    """X kept rate-limiting the re-fetch past the backoff budget.

    Raised instead of grinding on: the caller has already checkpointed every repair made
    so far, so stopping loses nothing and protects the account. Resume later.
    """


def clamp_tabs(requested: int | None) -> int:
    """The number of tabs to actually open: `DEFAULT_TABS` when unset, never > `MAX_TABS`.

    Clamps rather than rejects, and clamps at BOTH ends — a caller passing 0 or -1 must
    not produce a pool that silently does nothing, which would look exactly like a run
    where every item failed.
    """
    if requested is None:
        return DEFAULT_TABS
    return max(1, min(int(requested), MAX_TABS))


def pause_ms(rng: random.Random | None = None) -> int:
    """One random inter-load pause, in milliseconds, uniform over the configured window."""
    return (rng or random).randint(PAUSE_MIN_MS, PAUSE_MAX_MS)


def backoff_ms(rng: random.Random | None = None) -> int:
    """One random rate-limit backoff, in milliseconds."""
    return (rng or random).randint(RATE_LIMIT_BACKOFF_MIN_MS, RATE_LIMIT_BACKOFF_MAX_MS)


async def drain(
    urls: Iterable[str],
    *,
    tabs: int,
    fetch: Callable[[int, str], Awaitable[str | None]],
    sleep: Callable[[float], Awaitable[None]],
    on_result: Callable[[str, str | None], None],
    rate_limited: Callable[[], bool] = lambda: False,
    clear_rate_limit: Callable[[], None] = lambda: None,
    rng: random.Random | None = None,
) -> None:
    """Pull `urls` through `tabs` workers, pausing between loads on the same tab.

    Each worker owns ONE tab for the whole run and navigates it in place, which is the
    entire point: `fetch` is handed the worker's index, never a fresh tab.

    `on_result` is called from the event loop, in completion order, once per URL —
    including for a URL that yielded nothing, so the caller can count attempts and
    checkpoint on a cadence rather than guessing which ones ran.

    The pause happens BEFORE each load except a worker's first, so a run of one item
    costs no wait, and no two workers are ever deliberately synchronised.

    `rate_limited` is polled between items. When it answers true every worker parks for a
    randomized backoff and the flag is cleared; past `MAX_RATE_LIMIT_BACKOFFS` the whole
    pool raises `RefetchRateLimited` instead of continuing to poke a limited endpoint.
    """
    queue: asyncio.Queue[str] = asyncio.Queue()
    for url in urls:
        queue.put_nowait(url)
    backoffs = 0
    lock = asyncio.Lock()

    async def worker(index: int) -> None:
        nonlocal backoffs
        first = True
        while True:
            try:
                url = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            if rate_limited():
                async with lock:
                    # Re-check inside the lock: the first worker through clears the flag,
                    # and the rest must not each spend a separate backoff for one 429.
                    if rate_limited():
                        if backoffs >= MAX_RATE_LIMIT_BACKOFFS:
                            raise RefetchRateLimited(
                                f"X devolvió 429 tras {backoffs} backoffs — re-fetch "
                                "detenido; las reparaciones hechas ya están guardadas."
                            )
                        backoffs += 1
                        wait = backoff_ms(rng)
                        logger.warning(
                            "X devolvió 429 (rate limit) — esperando %.0fs antes de seguir.",
                            wait / 1000,
                        )
                        clear_rate_limit()
                        await sleep(wait / 1000)

            if not first:
                await sleep(pause_ms(rng) / 1000)
            first = False

            try:
                text = await fetch(index, url)
            except RefetchRateLimited:
                raise
            except Exception:  # noqa: BLE001 - one bad post must not kill the pool
                logger.debug("refetch: la pestaña %d falló en %s", index, url, exc_info=True)
                text = None
            on_result(url, text)

    await asyncio.gather(*(worker(i) for i in range(clamp_tabs(tabs))))

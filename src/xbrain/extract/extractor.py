"""Orchestrate scrolling X and intercepting GraphQL responses."""

from __future__ import annotations

import logging
import random
from datetime import datetime

from playwright.sync_api import BrowserContext, Response

from xbrain.extract.browser import is_logged_out
from xbrain.extract.graphql import parse_tweets
from xbrain.models import Item, SourceName

logger = logging.getLogger(__name__)

_OPERATIONS = {"bookmark": "Bookmarks", "own_tweet": "UserTweets"}
# Deliberately slow, human-paced scrolling — avoids X rate-limiting / account bans.
_SETTLE_MS = 6000
_SCROLL_PAUSE_MIN_MS = 5000
_SCROLL_PAUSE_MAX_MS = 12000
_MAX_IDLE_SCROLLS = 4
# When X answers a GraphQL call with HTTP 429 ("rate limit exceeded"), pause for
# a randomized stretch before scrolling again, and give up after a few backoffs
# rather than hammering — pushing through a rate-limit is what escalates to a ban.
_RATE_LIMIT_BACKOFF_MIN_MS = 60_000
_RATE_LIMIT_BACKOFF_MAX_MS = 180_000
_MAX_RATE_LIMIT_BACKOFFS = 3


def rate_limit_decision(*, new_hits: bool, backoffs_done: int, max_backoffs: int) -> str:
    """Decide how to react when scrolling sees X's 429 rate-limit responses.

    Pure helper — the unit-tested core of the anti-ban backoff. Returns:
    - ``"scroll"`` when no fresh 429 arrived since the last check (carry on);
    - ``"abort"`` once we've already backed off ``max_backoffs`` times — stop
      rather than keep poking a rate-limited endpoint and risk a suspension;
    - ``"backoff"`` when a fresh 429 arrived and backoff budget remains.
    """
    if not new_hits:
        return "scroll"
    if backoffs_done >= max_backoffs:
        return "abort"
    return "backoff"


def collect_new_items(
    responses: list[dict], source: SourceName, known_ids: set[str]
) -> tuple[list[Item], bool]:
    """Parse responses into items, flagging when a known id is reached.

    Pure function — no browser. This is the unit-tested core of extraction.
    """
    new_items: list[Item] = []
    hit_known = False
    for response in responses:
        for item in parse_tweets(response, source):
            if item.id in known_ids:
                hit_known = True
            else:
                new_items.append(item)
    return new_items, hit_known


def extract_source(
    context: BrowserContext,
    source: SourceName,
    url: str,
    known_ids: set[str],
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Item]:
    """Scroll an X page, intercept GraphQL responses, return new items.

    Stops when a known id is reached (incremental) or no new responses arrive
    after `_MAX_IDLE_SCROLLS` scrolls (end of timeline).
    """
    operation = _OPERATIONS[source]
    captured: list[dict] = []
    # Mutable counter shared with the response callback: how many 429s X has
    # returned on GraphQL calls so far. A dict sidesteps `nonlocal` in the hook.
    rate_limit = {"hits": 0}
    page = context.new_page()

    def on_response(response: Response) -> None:
        if "/graphql/" not in response.url:
            return
        if response.status == 429:
            rate_limit["hits"] += 1
            return
        if operation in response.url:
            try:
                captured.append(response.json())
            except Exception:  # noqa: BLE001 - ignore non-JSON / partial bodies
                pass

    page.on("response", on_response)
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(_SETTLE_MS)
    if is_logged_out(page.url):
        page.close()
        raise RuntimeError("Sesión de X caducada. Ejecuta `xbrain login`.")

    idle = 0
    last_count = 0
    handled_rate_limits = 0
    backoffs = 0
    while idle < _MAX_IDLE_SCROLLS:
        _, hit_known = collect_new_items(captured, source, known_ids)
        if hit_known:
            break
        action = rate_limit_decision(
            new_hits=rate_limit["hits"] > handled_rate_limits,
            backoffs_done=backoffs,
            max_backoffs=_MAX_RATE_LIMIT_BACKOFFS,
        )
        if action == "abort":
            logger.warning(
                "X sigue rate-limiteando (%s × 429) tras %s backoffs — paro para "
                "no arriesgar la cuenta. Reanuda la extracción más tarde.",
                rate_limit["hits"],
                backoffs,
            )
            break
        if action == "backoff":
            handled_rate_limits = rate_limit["hits"]
            backoffs += 1
            wait_ms = random.randint(_RATE_LIMIT_BACKOFF_MIN_MS, _RATE_LIMIT_BACKOFF_MAX_MS)
            logger.warning(
                "X devolvió 429 (rate limit) — esperando %.0fs antes de seguir.",
                wait_ms / 1000,
            )
            page.wait_for_timeout(wait_ms)
            continue
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(random.randint(_SCROLL_PAUSE_MIN_MS, _SCROLL_PAUSE_MAX_MS))
        if len(captured) == last_count:
            idle += 1
        else:
            idle = 0
            last_count = len(captured)

    page.close()
    new_items, _ = collect_new_items(captured, source, known_ids)
    return _filter_in_range(new_items, since, until)


def _filter_in_range(
    items: list[Item], since: datetime | None, until: datetime | None
) -> list[Item]:
    """Keep items within [since, until] (inclusive), de-duplicated by id.

    Pure helper — the date-window filter applied after scrolling. An open bound
    (`None`) is unconstrained on that side; first-seen wins on duplicate ids.
    """
    in_range: dict[str, Item] = {}
    for item in items:
        if since and item.created_at < since:
            continue
        if until and item.created_at > until:
            continue
        in_range.setdefault(item.id, item)
    return list(in_range.values())

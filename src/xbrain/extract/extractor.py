"""Orchestrate scrolling X and intercepting GraphQL responses."""

from __future__ import annotations

import random
from datetime import datetime

from playwright.sync_api import BrowserContext, Response

from xbrain.extract.browser import is_logged_out
from xbrain.extract.graphql import parse_tweets
from xbrain.models import Item, SourceName

_OPERATIONS = {"bookmark": "Bookmarks", "own_tweet": "UserTweets"}
# Deliberately slow, human-paced scrolling — avoids X rate-limiting / account bans.
_SETTLE_MS = 6000
_SCROLL_PAUSE_MIN_MS = 5000
_SCROLL_PAUSE_MAX_MS = 12000
_MAX_IDLE_SCROLLS = 4


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
    page = context.new_page()

    def on_response(response: Response) -> None:
        if "/graphql/" in response.url and operation in response.url:
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
    while idle < _MAX_IDLE_SCROLLS:
        _, hit_known = collect_new_items(captured, source, known_ids)
        if hit_known:
            break
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(random.randint(_SCROLL_PAUSE_MIN_MS, _SCROLL_PAUSE_MAX_MS))
        if len(captured) == last_count:
            idle += 1
        else:
            idle = 0
            last_count = len(captured)

    page.close()
    new_items, _ = collect_new_items(captured, source, known_ids)
    in_range: dict[str, Item] = {}
    for item in new_items:
        if since and item.created_at < since:
            continue
        if until and item.created_at > until:
            continue
        in_range.setdefault(item.id, item)
    return list(in_range.values())

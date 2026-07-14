"""Parse X (Twitter) internal GraphQL responses into Item objects."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlparse

from xbrain.extract.video import build_video_media
from xbrain.models import (
    Author,
    Content,
    ContentSource,
    ContentSourceFailure,
    ContentSourceSuccess,
    FailureReason,
    Item,
    Link,
    Media,
    MediaEntry,
    SourceName,
    ThreadInfo,
)

logger = logging.getLogger(__name__)

X_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
_NESTED_TWEET_KEYS = ("quoted_status_result", "retweeted_status_result")
# x.com ⇄ twitter.com host aliases, folded to `x.com` when deduping links so a
# non-canonical variant of an article URL matches the synthesized one.
_X_HOST_ALIASES = frozenset(
    {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}
)


def parse_tweets(response: dict[str, Any], source: SourceName) -> list[Item]:
    """Extract every timeline tweet from one X GraphQL response.

    Walks the response tree recursively looking for `tweet_results` blocks,
    which is how X wraps timeline tweets in both the Bookmarks and UserTweets
    operations. Anchoring on the key name (stable) rather than a fixed path
    keeps the parser resilient to X restructuring the surrounding envelope.
    """
    items: list[Item] = []
    seen: set[str] = set()
    for result in _find_tweet_results(response):
        tweet = _unwrap(result)
        if tweet is None:
            continue
        rest_id = tweet.get("rest_id")
        if not rest_id or rest_id in seen:
            continue
        item = _tweet_to_item(tweet, source)
        if item is not None:
            seen.add(str(rest_id))
            items.append(item)
    return items


def _find_tweet_results(obj: Any) -> Iterator[dict[str, Any]]:
    """Yield every top-level `tweet_results.result` dict in the tree.

    Skips quoted/retweeted sub-trees so a nested hydrated tweet is not
    surfaced as a standalone timeline item.
    """
    if isinstance(obj, dict):
        results = obj.get("tweet_results")
        if isinstance(results, dict) and isinstance(results.get("result"), dict):
            yield results["result"]
        for key, value in obj.items():
            if key not in _NESTED_TWEET_KEYS:
                yield from _find_tweet_results(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _find_tweet_results(value)


def iter_tweet_payloads(response: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield `(rest_id, full result subtree)` for every tweet in one GraphQL response.

    The persistence seam for `payloads.save_payload`. We keep each tweet's WHOLE subtree
    (legacy, note_tweet, user, entities, quoted_status_result, card) and drop only the
    timeline envelope — cursors, pagination, instructions — which carries no item data and
    would otherwise be duplicated once per tweet in the response.
    """
    for result in _find_tweet_results(response):
        tweet = _unwrap(result)
        rest_id = tweet.get("rest_id") if tweet else None
        if tweet and rest_id:
            yield str(rest_id), tweet


def _dig(obj: Any, *keys: str) -> dict[str, Any]:
    """Walk nested dict keys, returning {} on any missing/null/non-dict node."""
    for key in keys:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj if isinstance(obj, dict) else {}


def _unwrap(result: dict[str, Any]) -> dict[str, Any] | None:
    """Unwrap the TweetWithVisibilityResults envelope if present."""
    typename = result.get("__typename")
    if typename == "TweetWithVisibilityResults":
        inner = result.get("tweet")
        return inner if isinstance(inner, dict) else None
    if typename == "Tweet":
        return result
    return result if "legacy" in result else None


# X truncates a long post at 280 characters of prose and appends a t.co self-link.
_TRAILING_TCO = re.compile(r"https://t\.co/\w+\s*$")
# A sentence that really ended does so on punctuation. Anything else at the cap is a cut.
_TERMINAL = (".", "!", "?", "…", '"', "”", ")", ":", ";", "»")
# The observed cut lands at 270–290 chars of body (the store's median body is 279).
_TRUNCATION_FLOOR = 265


def looks_truncated(text: str, links: list[str]) -> bool:
    """True when `text` bears X's 280-character truncation signature.

    The mechanical marks: ~280 characters of prose, ending mid-word with no terminal
    punctuation, optionally followed by an appended `t.co` self-link that is NOT among the
    tweet's own `links` (X adds it; the tweet never contained it).

    Deliberately biased towards FLAGGING: a detector that silently passes a genuinely
    truncated tweet reproduces the very bug it exists to catch — the generator gets half a
    sentence and finishes it. A false flag costs a re-fetch; a missed one costs a
    fabrication in the knowledge base.
    """
    body = text.rstrip()
    match = _TRAILING_TCO.search(body)
    if match and match.group().strip() not in links:
        body = body[: match.start()].rstrip()
    return len(body) >= _TRUNCATION_FLOOR and not body.endswith(_TERMINAL)


def items_needing_refetch(store: dict[str, Item]) -> list[Item]:
    """Every stored item whose text was truncated at ingest.

    These were extracted before the `note_tweet` fix, so their long-form body was never
    read. The raw GraphQL payloads are NOT persisted (they are captured in-flight during
    the browser session), so repairing them REQUIRES a re-fetch from X — this is not a
    free re-parse of data we already hold.
    """
    return [
        item
        for item in store.values()
        if looks_truncated(item.text, [link.url for link in item.links])
    ]


def _tweet_text(tweet: dict[str, Any], legacy: dict[str, Any]) -> str:
    """The tweet's FULL text — the long-form body when X provides one.

    `legacy.full_text` is capped at 280 characters: X truncates a long post there and
    appends a t.co self-link, so the stored text ends mid-word ("…solve literally any").
    The generator is then handed half a sentence and told to summarise it, and it completes
    the sentence itself — a fabrication WE cause, at ingest. The real body lives in
    `note_tweet.note_tweet_results.result.text`.

    An empty long-form body falls back to `full_text`: shipping a truncated tweet is bad,
    shipping an EMPTY one is worse.
    """
    # `_dig` coerces a non-dict leaf to {}, so walk to the parent and read the leaf here.
    note = _dig(tweet, "note_tweet", "note_tweet_results", "result").get("text")
    if isinstance(note, str) and note.strip():
        return note
    return legacy.get("full_text", "")


def _tweet_to_item(tweet: dict[str, Any], source: SourceName) -> Item | None:
    legacy = tweet.get("legacy")
    rest_id = tweet.get("rest_id")
    if not isinstance(legacy, dict) or not rest_id:
        return None
    author = _extract_author(tweet)
    if author is None:
        return None
    quoted_id = legacy.get("quoted_status_id_str") or _quoted_id(tweet)
    return Item(
        id=str(rest_id),
        source=source,
        url=f"https://x.com/{author.handle}/status/{rest_id}",
        author=author,
        text=_tweet_text(tweet, legacy),
        created_at=_parse_x_date(legacy.get("created_at")),
        captured_at=datetime.now(timezone.utc),
        media=_extract_media(legacy),
        links=_extract_links(legacy, tweet),
        quoted_id=quoted_id,
        thread=_thread_info(legacy),
        # The quoted post rides in the SAME payload — parsed here, at no network cost.
        # A tweet that quotes nothing keeps `content=None`: an empty `Content` would
        # read as "already fetched" to `fetch._should_refetch`.
        content=_quoted_content(tweet, quoted_id),
    )


def _extract_author(tweet: dict[str, Any]) -> Author | None:
    user = _dig(tweet, "core", "user_results", "result")
    legacy = _dig(user, "legacy")
    core = _dig(user, "core")  # newer responses moved name/handle here
    handle = legacy.get("screen_name") or core.get("screen_name")
    name = legacy.get("name") or core.get("name")
    if not handle:
        return None
    return Author(handle=handle, name=name or handle)


def _extract_links(legacy: dict[str, Any], tweet: dict[str, Any]) -> list[Link]:
    """Every link on a tweet: the text URLs in `entities.urls` plus the
    synthesized canonical link for a directly-bookmarked long-form Article.

    The Article link is appended only when the tweet carries an Article entity
    and the URL is not already present (dedup against `entities.urls`), so a
    tweet that merely *links* an Article never double-adds it.
    """
    links: list[Link] = []
    for entry in legacy.get("entities", {}).get("urls", []):
        expanded = entry.get("expanded_url")
        if expanded:
            links.append(Link(url=expanded, domain=urlparse(expanded).netloc))
    _append_article_link(links, tweet)
    return links


def _canonical_url_key(url: str) -> str:
    """A scheme/host/trailing-slash-insensitive key for deduping links.

    Folds the x.com ⇄ twitter.com host aliases to `x.com` and drops the scheme
    and a trailing slash, so a non-canonical variant of an article URL already
    in `entities.urls` (e.g. `http://twitter.com/i/article/<id>/`) compares
    equal to the synthesized `https://x.com/i/article/<id>` link — avoiding a
    redundant re-fetch of the same Article.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in _X_HOST_ALIASES:
        host = "x.com"
    return f"{host}{parsed.path.rstrip('/')}"


def _append_article_link(links: list[Link], tweet: dict[str, Any]) -> None:
    """Append the synthesized Article link unless a canonical-equivalent URL is
    already present (dedup against `entities.urls`), so a tweet that merely
    *links* an Article — in any host/scheme/slash variant — never double-adds it.
    """
    article = _extract_article_link(tweet)
    if article is None:
        return
    seen = {_canonical_url_key(link.url) for link in links}
    if _canonical_url_key(article.url) not in seen:
        links.append(article)


def _extract_article_link(tweet: dict[str, Any]) -> Link | None:
    """Synthesize the canonical `/i/article/<id>` Link for a directly-bookmarked
    X long-form Article, or None when the tweet carries no Article entity.

    X attaches a long-form Article to its tweet result as an `article` block:
    `tweet["article"]["article_results"]["result"]` carries the Article's
    numeric `rest_id`. We anchor on those stable key names via `_dig` (the same
    null-safe walk `_extract_author` uses) rather than a fixed path, so an X
    shape drift degrades to None (no link) instead of mis-parsing into a wrong
    link. The `https://x.com/i/article/<rest_id>` URL is chosen so the existing
    `is_x_url` + `_classify_x_url` routing already fires the rendered-fetch path
    for it — no change to `fetch_x`.

    NOTE: the `article.article_results.result.rest_id` key path is pinned
    against a CONSTRUCTED fixture (see `tests/test_graphql.py`), not a recorded
    live payload; validate it against a real bookmarked-Article GraphQL
    response before production reliance (RFC #39 open-Q #4). X may ALSO surface
    an Article via a `card`/`unified_card` variant — not handled here (it
    degrades safely to no link); fold that path into the same open-Q #4
    real-payload validation before building it.
    """
    result = _dig(tweet, "article", "article_results", "result")
    # Only synthesize for an actual Article result — reject e.g. a Card that
    # happens to carry a rest_id. Allow a missing __typename for real-payload
    # flexibility (the parser still gates on the rest_id shape below).
    if result.get("__typename") not in (None, "Article"):
        return None
    rest_id = result.get("rest_id")
    # A valid X article id is a numeric string; anything else (missing, empty,
    # None, a dict/list, or non-numeric text) degrades to no link — this also
    # kills the garbage-URL vector of interpolating a non-scalar rest_id.
    if not (isinstance(rest_id, str) and rest_id.isdigit()):
        return None
    return Link(url=f"https://x.com/i/article/{rest_id}", domain="x.com")


def _extract_media(legacy: dict[str, Any]) -> list[MediaEntry]:
    entries = legacy.get("extended_entities", {}).get("media") or legacy.get("entities", {}).get(
        "media", []
    )
    media: list[MediaEntry] = []
    for entry in entries:
        if entry.get("type") in ("video", "animated_gif"):
            video = build_video_media(entry)
            if video is not None:
                media.append(video)
            continue
        url = entry.get("media_url_https") or entry.get("expanded_url")
        if url:
            media.append(Media(type="photo", url=url))
    return media


def _quoted_id(tweet: dict[str, Any]) -> str | None:
    quoted = _dig(tweet, "quoted_status_result", "result")
    rest_id = quoted.get("rest_id")
    return str(rest_id) if rest_id else None


def _quoted_result(tweet: dict[str, Any]) -> dict[str, Any]:
    """The hydrated quoted post X embeds in the timeline entry, or `{}`.

    X nests it at the tweet's top level; some payload shapes carry it under
    `legacy` instead, so both are probed (a miss degrades to `{}` → a failure
    source, never a wrong parse).
    """
    return _dig(tweet, "quoted_status_result", "result") or _dig(
        tweet, "legacy", "quoted_status_result", "result"
    )


def _quoted_content(tweet: dict[str, Any], quoted_id: str | None) -> Content | None:
    """The quoted post as a `ContentSource`, or None when the tweet quotes nothing.

    **No network call.** X embeds the quoted post — its body AND its author — in the
    very timeline payload we already capture, so this is a pure re-parse of bytes we
    hold. Before this, only `quoted_id` was kept and the body was dropped, leaving
    the generator a bare reaction ("Read this and you'll understand") and nothing to
    ground it in.

    Failure taxonomy — a quote we cannot read is a FAILURE state, never silence:
    `not_found` (X tombstoned it: deleted), `forbidden` (`TweetUnavailable`:
    protected, suspended or blocked), `empty_content` (X sent the id but hydrated no
    post, or hydrated one with no body/author). Each keeps `quoted_id` addressable in
    the URL, and each leaves `quoted_content_unfetched(item)` True — so #86's
    `content NOT fetched` marker still fires and the generator is still forbidden to
    invent the post it cannot see.
    """
    if not quoted_id:
        return None
    result = _quoted_result(tweet)
    quoted = _unwrap(result) if result else None
    source: ContentSource = _quoted_failure(result, quoted_id, served=quoted is not None)
    if quoted is not None:
        # The BODY is what makes the quote evidence. An author X did not hydrate costs us
        # the attribution — `quoted_attribution` then names nobody — but it must not cost
        # us the body too: dropping the cure because one field is missing would be the
        # bug wearing a different hat.
        author = _extract_author(quoted)
        text = _dig(quoted, "legacy").get("full_text") or ""
        rest_id = quoted.get("rest_id") or quoted_id
        handle = author.handle if author else "i"
        if text:
            source = ContentSourceSuccess(
                kind="quoted_tweet",
                url=f"https://x.com/{handle}/status/{rest_id}",
                # No title: a post has none, and `notes_io.note_title` takes the first
                # source that carries one — borrowing the field for the author would
                # rename the note after the account it quotes.
                text=text,
                author=author,
                attempts=1,
            )
    return Content(fetched_at=datetime.now(timezone.utc), sources=[source])


# X's `__typename` for a quoted post it will not serve → our failure reason.
_QUOTED_FAILURE_REASONS: dict[str, FailureReason] = {
    "TweetTombstone": "not_found",  # deleted by its author, or by X
    "TweetUnavailable": "forbidden",  # protected / suspended / blocked
}


def _quoted_failure(
    result: dict[str, Any], quoted_id: str, *, served: bool
) -> ContentSourceFailure:
    """Why the quoted post is not readable — a record of what X ACTUALLY did.

    Three different facts, and they must not be flattened into one: X tombstoned the post
    (`not_found`), X refused it (`forbidden`), or X gave us nothing usable
    (`empty_content`). That last bucket splits again — `served` distinguishes a post X
    HANDED US that simply carries no text (a photo/video quote: X served it, there is
    just nothing to read) from one X never hydrated at all. Writing "X did not serve the
    quoted post" over a media-only quote would be a false statement about the world,
    stored in the very evidence base whose purpose is to stop us inventing facts.
    """
    typename = str(result.get("__typename") or "")
    reason = _QUOTED_FAILURE_REASONS.get(typename, "empty_content")
    if reason == "empty_content" and served:
        error = "the quoted post carries no text (media-only)"
    else:
        error = f"X did not serve the quoted post ({typename or 'not hydrated'})"
    return ContentSourceFailure(
        kind="quoted_tweet",
        # The id-only permalink: with no author handle there is no /<handle>/status/
        # form, and x.com/i/status/<id> resolves to the same post.
        url=f"https://x.com/i/status/{quoted_id}",
        failure_reason=reason,
        error=error,
        attempts=1,
    )


def _thread_info(legacy: dict[str, Any]) -> ThreadInfo | None:
    """Detect a self-thread from X's `self_thread` marker in the legacy block."""
    self_thread = legacy.get("self_thread")
    if isinstance(self_thread, dict) and self_thread.get("id_str"):
        return ThreadInfo(root_id=str(self_thread["id_str"]))
    return None


def _parse_x_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.strptime(value, X_DATE_FORMAT)
    except ValueError:
        logger.warning("unparseable X date %r, using now()", value)
        return datetime.now(timezone.utc)

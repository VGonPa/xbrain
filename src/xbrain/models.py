"""Data models for the XBrain store."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter

logger = logging.getLogger(__name__)

# The set of enrichment executor names — one source of truth shared by the
# data model, the config loader and the enrichment phase.
ExecutorName = Literal["manual", "api", "claude-code"]

# The set of item source names — one source of truth shared by the data model
# and the GraphQL parser.
SourceName = Literal["bookmark", "own_tweet"]

# Categorised reasons a content fetch can fail — structured evidence so a
# broken link is demonstrable, not assumed (design §4).
FailureReason = Literal[
    "not_found",
    "forbidden",
    "paywall",
    "timeout",
    "dns_error",
    "js_required",
    "empty_content",
    "unknown_error",  # catch-all for uncategorised failures (e.g. an extractor
    # exception we did not classify). Transient by default — `_should_refetch`
    # in fetch.py treats it as retry-worthy on the next run, mirroring the
    # pre-#20 behaviour where `failure_reason=None` meant transient.
]

# The set of content-source kinds — one source of truth shared by the data
# model, the fetch stage and the wiki renderer.
ContentKind = Literal["external_article", "x_article", "thread", "quoted_tweet"]


class Author(BaseModel):
    """The X account that authored an item."""

    handle: str
    name: str


class Link(BaseModel):
    """One external URL extracted from an item's text."""

    url: str
    domain: str


class Media(BaseModel):
    """One photo or video attached to an item."""

    type: Literal["photo", "video"]
    url: str


class ThreadInfo(BaseModel):
    """Marker that an item is part of a multi-tweet thread."""

    is_thread: bool = True
    root_id: str
    position: int | None = None


class ContentSourceSuccess(BaseModel):
    """A fetched article whose body was successfully extracted.

    The success variant of the `ContentSource` tagged union. `text` is
    required — a success without text is not a success — and the type
    system enforces this at construction time.
    """

    outcome: Literal["success"] = "success"
    kind: ContentKind
    url: str
    title: str | None = None
    text: str
    http_status: int | None = None
    # extraction attempts: 1 = single pass, 2 = + Firecrawl fallback;
    # 0 only on pre-Fase-2 records.
    attempts: int = 0


class ContentSourceFailure(BaseModel):
    """A fetched article whose body could not be extracted.

    The failure variant of the `ContentSource` tagged union — structured
    broken-link evidence so the wiki can render a ``⚠ Enlace roto`` line
    rather than pretending the link was never there (design §4).
    `failure_reason` is required: a failure without a reason is not
    demonstrable evidence.
    """

    outcome: Literal["failure"] = "failure"
    kind: ContentKind
    url: str
    failure_reason: FailureReason
    error: str | None = None
    http_status: int | None = None
    # extraction attempts: 1 = single pass, 2 = + Firecrawl fallback;
    # 0 only on pre-Fase-2 records.
    attempts: int = 0


def _normalise_legacy_content_source(value: Any) -> Any:
    """Map the legacy ``{ok: bool, ...}`` shape to the tagged-union shape.

    Older ``data/items.json`` records (pre-#20) carry ``ok: True`` /
    ``ok: False`` instead of ``outcome: "success"`` / ``outcome: "failure"``.
    The mapping is one-to-one:

    - ``ok=True`` (success)  → ``outcome="success"``
    - ``ok=False`` (failure) → ``outcome="failure"``

    Records that already carry ``outcome`` are returned unchanged. Records
    that have neither discriminator are rejected — silently inventing one
    would mask data corruption.

    Fields irrelevant to the new variant (e.g. ``title`` / ``text`` on the
    failure variant) are dropped during normalisation so the resulting dict
    matches the variant's declared fields exactly. This is purely defensive
    — extra fields on a pydantic model are ignored by default, but stripping
    them up front keeps the on-the-wire shape clean once the record is
    re-dumped.
    """
    if not isinstance(value, dict):
        return value
    if "outcome" in value:
        return value
    if "ok" not in value:
        # Include enough context to find the offending record in a big file.
        url = value.get("url", "<unknown URL>")
        raise ValueError(
            f"ContentSource record missing both 'outcome' and 'ok' "
            f"discriminator (url={url!r}); the record cannot be safely "
            "categorised as success or failure."
        )
    payload = {k: v for k, v in value.items() if k != "ok"}
    payload["outcome"] = "success" if value["ok"] else "failure"
    if payload["outcome"] == "success":
        # success has no failure_reason / error — drop if present so the
        # re-dumped record is clean (pydantic ignores extras anyway).
        payload.pop("failure_reason", None)
        payload.pop("error", None)
    else:
        # failure has no title / text
        payload.pop("title", None)
        payload.pop("text", None)
        # Legacy records sometimes recorded a failure (`ok=False`) with no
        # categorised `failure_reason` (e.g. an HTTP 429 that the old code
        # did not map). The new variant requires the field — bucket those
        # under `unknown_error` (a transient retry-worthy reason added in
        # the #20 review pass, see `xbrain.fetch._TRANSIENT_FAILURES`).
        # `unknown_error` is preferable to `timeout` here because the
        # actual cause is unknown — "timeout" would be a lie that hides
        # 429s, SSL handshake failures, and other distinct error modes.
        if payload.get("failure_reason") in (None, ""):
            payload["failure_reason"] = "unknown_error"
            logger.warning(
                "Legacy ContentSource without failure_reason bucketed as "
                "'unknown_error' (url=%s). The next `fetch_pending` run will "
                "retry it; use `--force` to suppress the retry.",
                value.get("url", "<unknown URL>"),
            )
    return payload


# The persisted ContentSource type — a discriminated union over the success
# and failure variants, wrapped in an outer `BeforeValidator` that normalises
# the legacy `ok: bool` records on read so existing `data/items.json` files
# keep working.
#
# The wrapping is layered on purpose: the `BeforeValidator` must run BEFORE
# pydantic dispatches on the `outcome` discriminator. If both annotations were
# on the same `Annotated`, the discriminator check would run first and reject
# legacy records that carry `ok` instead of `outcome`. The outer Annotated
# guarantees the right ordering.
_ContentSourceTagged = Annotated[
    Union[ContentSourceSuccess, ContentSourceFailure],
    Field(discriminator="outcome"),
]
ContentSource = Annotated[
    _ContentSourceTagged,
    BeforeValidator(_normalise_legacy_content_source),
]


# A TypeAdapter is the documented pydantic-v2 entry point for validating /
# dumping a discriminated-union *type alias* (since the alias itself is not a
# class with `.model_validate`). Tests use this; production code goes through
# `Item` and `Content` which carry the union as a field.
ContentSourceAdapter: TypeAdapter[Union[ContentSourceSuccess, ContentSourceFailure]] = TypeAdapter(
    ContentSource
)


class Content(BaseModel):
    """The fetched article(s) attached to an item, with their fetch timestamp."""

    fetched_at: datetime
    sources: list[ContentSource] = Field(default_factory=list)


class Enrichment(BaseModel):
    """LLM-generated summary and topic assignment for an item."""

    enriched_at: datetime
    executor: ExecutorName
    summary: str | None = None
    primary_topic: str | None = None
    topics: list[str] = Field(default_factory=list)
    user_notes: str | None = None


class Topic(BaseModel):
    """One entry of the induced topic vocabulary (data/vocab.yaml)."""

    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    description: str


class TopicPage(BaseModel):
    """One synthesized topic-page overview, persisted in data/topics.json.

    `post_count_at_synth` records how many posts the topic had when the overview
    was synthesized — comparing it to the live count derives staleness without a
    stored flag that could desync.
    """

    slug: str
    overview: str
    notes: list[str] = Field(default_factory=list)
    synthesized_at: datetime
    post_count_at_synth: int


class Item(BaseModel):
    """One captured X post (bookmark or own tweet) with all its derived data."""

    id: str
    source: SourceName
    url: str
    author: Author
    text: str
    created_at: datetime
    captured_at: datetime
    media: list[Media] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    quoted_id: str | None = None
    thread: ThreadInfo | None = None
    content: Content | None = None
    enriched: Enrichment | None = None
    bookmark_folder: str | None = None


class SourceCursor(BaseModel):
    """Per-source extractor cursor: where we left off last run."""

    last_seen_id: str | None = None
    last_run: datetime | None = None


class ArchiveImport(BaseModel):
    """Marker recording a one-off X archive import."""

    file: str
    at: datetime


class State(BaseModel):
    """Top-level extractor state persisted in `data/state.json`."""

    bookmarks: SourceCursor = Field(default_factory=SourceCursor)
    own_tweets: SourceCursor = Field(default_factory=SourceCursor)
    archive_imported: ArchiveImport | None = None

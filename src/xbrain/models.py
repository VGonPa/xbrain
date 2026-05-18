"""Data models for the XBrain store."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

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
]


class Author(BaseModel):
    handle: str
    name: str


class Link(BaseModel):
    url: str
    domain: str


class Media(BaseModel):
    type: Literal["photo", "video"]
    url: str


class ThreadInfo(BaseModel):
    is_thread: bool = True
    root_id: str
    position: int | None = None


class ContentSource(BaseModel):
    kind: Literal["external_article", "x_article", "thread", "quoted_tweet"]
    url: str
    title: str | None = None
    text: str | None = None
    ok: bool = True
    error: str | None = None
    http_status: int | None = None
    failure_reason: FailureReason | None = None
    # extraction attempts: 1 = single pass, 2 = + Firecrawl fallback; 0 only on pre-Fase-2 records
    attempts: int = 0


class Content(BaseModel):
    fetched_at: datetime
    sources: list[ContentSource] = Field(default_factory=list)


class Enrichment(BaseModel):
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
    last_seen_id: str | None = None
    last_run: datetime | None = None


class ArchiveImport(BaseModel):
    file: str
    at: datetime


class State(BaseModel):
    bookmarks: SourceCursor = Field(default_factory=SourceCursor)
    own_tweets: SourceCursor = Field(default_factory=SourceCursor)
    archive_imported: ArchiveImport | None = None

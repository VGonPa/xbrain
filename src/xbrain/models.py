"""Data models for the XBrain store."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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


class Content(BaseModel):
    fetched_at: datetime
    sources: list[ContentSource] = Field(default_factory=list)


class Enrichment(BaseModel):
    enriched_at: datetime
    executor: Literal["manual", "api", "claude-code"]
    summary: str | None = None
    primary_topic: str | None = None
    topics: list[str] = Field(default_factory=list)
    user_notes: str | None = None


class Topic(BaseModel):
    """One entry of the induced topic vocabulary (data/vocab.yaml)."""
    slug: str
    description: str


class Item(BaseModel):
    id: str
    source: Literal["bookmark", "own_tweet"]
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

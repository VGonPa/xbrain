"""Data models for the XBrain store."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from xbrain.i18n import SUPPORTED_LANGUAGES

# Type alias mirroring `i18n.SUPPORTED_LANGUAGES` so the data layer rejects
# unknown languages at construction time. `Literal[*tuple]` unpacking is
# Python 3.11+ — we are on 3.12 — and keeps the source of truth in `i18n`.
SupportedLanguage = Literal[*SUPPORTED_LANGUAGES]  # type: ignore[valid-type]

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
# model, the fetch stage and the wiki renderer. `x_video` is manufactured by the
# `digest-video` stage (#44): a bookmarked video's transcript is attached as a
# `ContentSourceSuccess(kind="x_video")` so the existing enrich → topics →
# generate pipeline consumes it exactly like an article body.
ContentKind = Literal["external_article", "x_article", "thread", "quoted_tweet", "x_video"]


class Author(BaseModel):
    """The X account that authored an item."""

    handle: str
    name: str


class Link(BaseModel):
    """One external URL extracted from an item's text."""

    url: str
    domain: str


# Categorised reasons a photo download can fail — mirrors the design of
# `FailureReason` for content fetches. The transient subset
# (`_TRANSIENT_MEDIA_FAILURES` in `xbrain.media`) is retried on the next
# `xbrain media` run; permanent reasons stay as-is unless `--force` is passed.
MediaFailureReason = Literal[
    "http_4xx",  # permanent: dead URL / cdn-removed media
    "http_5xx",  # transient: server-side, may succeed on retry
    "timeout",  # transient: network blip / cdn slow path
    "format_error",  # permanent: bytes downloaded but Pillow rejected them
    "unknown_error",  # bare-except bucket; transient by default (mirrors fetch.py)
]


class _MediaPhotoBase(BaseModel):
    """Common fields for the three photo variants.

    `type` is preserved for wire-compatibility with legacy records that
    used the flat `{type, url}` shape — a re-dump after migration still
    carries it. The discriminator is `kind` (see the variant subclasses)
    — globally unique across photo + video variants because the
    `state="pending"` shape would otherwise collide between
    `MediaPhotoPending` and `MediaVideoPending`.
    """

    type: Literal["photo"] = "photo"
    url: str


class MediaPhotoPending(_MediaPhotoBase):
    """A photo URL captured at extract time, download not attempted yet.

    This is the initial state for every photo entry the extractor or the
    archive importer creates. `xbrain media` walks the store and tries to
    advance each pending entry to either `MediaPhotoDownloaded` (bytes on
    disk) or `MediaPhotoFailed` (categorised failure).
    """

    kind: Literal["photo_pending"] = "photo_pending"


def _reject_local_path_traversal(value: str) -> str:
    """Reject absolute paths and `..` components on a `local_path`.

    Shared by `MediaPhotoDownloaded` and `MediaPhotoDescribed` so the
    on-disk path contract has one source of truth. `local_path` is
    joined onto `data/media/` at render and download time. A persisted
    record like ``"/etc/passwd"`` or ``"../../x"`` would let a poisoned
    `items.json` exfiltrate bytes outside the media root. The downloader
    never builds such a path, but the persisted store is on-disk plain
    JSON the user can edit — defence in depth at the type boundary is
    cheap.

    The two variants share the validator via this free function rather
    than via class inheritance: the rest of the codebase relies on
    `isinstance(entry, MediaPhotoDownloaded)` meaning "exactly the
    Downloaded state" in many call sites (the photo states form a
    tagged union, not a Liskov hierarchy).
    """
    if value.startswith("/") or value.startswith("\\"):
        raise ValueError(f"local_path must be relative, got {value!r}")
    # Normalise separators before scanning components so a Windows-style
    # path persisted on a foreign machine is still caught.
    for part in value.replace("\\", "/").split("/"):
        if part == "..":
            raise ValueError(f"local_path must not contain '..' components: {value!r}")
    return value


def _require_utc_aware(field_name: str, value: datetime) -> datetime:
    """Reject naive timestamps — every persisted instant must be UTC-aware.

    Shared by every photo-variant timestamp (`downloaded_at`,
    `described_at`, `last_attempt_at`) so a hand-edited naive record on
    a foreign machine does not silently miscompare against
    `now(timezone.utc)`. We do not coerce — that would mask the bug.
    """
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware, got naive {value!r}")
    return value


class MediaPhotoDownloaded(_MediaPhotoBase):
    """A photo successfully downloaded; the bytes exist at `local_path`.

    `local_path` is relative to `data/media/` so it can be moved across
    machines without rewriting the store. Dimensions and byte size are
    captured for completeness — they let `xbrain diff` answer "did the
    download cascade pick a smaller size on a re-run?" without re-reading
    every file.

    Dimensions and byte size are `gt=0`: a zero-pixel or zero-byte
    "downloaded" photo is semantically illegal — Pillow validation in
    `xbrain.media._decode_image` rules out the dim=0 case at the seam,
    and the type constraint pins it at the data layer too.

    `downloaded_at` MUST be timezone-aware (UTC) so a hand-edited naive
    timestamp does not silently miscompare against `now(timezone.utc)`.
    """

    kind: Literal["photo_downloaded"] = "photo_downloaded"
    local_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bytes_size: int = Field(gt=0)
    downloaded_at: datetime

    @field_validator("local_path")
    @classmethod
    def _validate_local_path(cls, value: str) -> str:
        _ = cls  # required by @field_validator+@classmethod; placate vulture
        return _reject_local_path_traversal(value)

    @field_validator("downloaded_at")
    @classmethod
    def _validate_downloaded_at(cls, value: datetime) -> datetime:
        _ = cls
        return _require_utc_aware("downloaded_at", value)


class MediaPhotoFailed(_MediaPhotoBase):
    """A photo download attempted and failed (categorised).

    Mirrors `ContentSourceFailure`: `failure_reason` is required so a
    failure is always demonstrable evidence (no Optional, no silent loss).
    `attempts` counts how many `xbrain media` runs have tried this URL —
    each transient retry bumps it. `attempts` is `ge=1`: a "failed but
    never attempted" record is semantically nonsense, and the downloader
    increments before producing any `MediaPhotoFailed`.

    `last_attempt_at` MUST be timezone-aware (UTC) — same contract as
    every other persisted instant on the photo variants.
    """

    kind: Literal["photo_failed"] = "photo_failed"
    failure_reason: MediaFailureReason
    error: str | None = None
    attempts: int = Field(ge=1)
    last_attempt_at: datetime

    @field_validator("last_attempt_at")
    @classmethod
    def _validate_last_attempt_at(cls, value: datetime) -> datetime:
        _ = cls
        return _require_utc_aware("last_attempt_at", value)


class MediaPhotoDescribed(_MediaPhotoBase):
    """A downloaded photo that has also been described by a vision LLM.

    The terminal state for a content-bearing photo: the bytes still live
    at `local_path` (every `MediaPhotoDownloaded` invariant carries over
    verbatim — `local_path` + dimensions + `bytes_size` + `downloaded_at`
    are re-declared so the field shape is wire-identical) AND a vision
    pass has classified the image and written a short prose description.
    Decorative photos (avatars, reaction memes, abstract backgrounds)
    reach this variant too — they carry `is_decorative=True` and an
    empty `description`, so downstream callers can filter them out
    without re-classifying.

    Why NOT inherit from `MediaPhotoDownloaded`? The photo states form a
    pydantic tagged union, not a Liskov class hierarchy. 25+ call sites
    use `isinstance(entry, MediaPhotoDownloaded)` to mean "exactly the
    Downloaded state" (eligibility checks, diff counters, generator
    routing). Inheritance would silently re-match every Described entry
    against those checks. We dedupe the field-traversal validator and
    the UTC-aware validator via free functions instead.

    `description_version` is the rubric/version tag the description was
    produced under. Bumping the configured version invalidates existing
    entries: the next `xbrain describe` run treats them as stale and
    re-describes (no `--force` needed). `description_lang` is typed as
    `SupportedLanguage` so an unknown language is rejected at
    construction — the type alias is derived from `i18n.SUPPORTED_LANGUAGES`
    (single source of truth).

    `description` is empty for decorative photos by contract — enforced
    by a model-level validator below so a hand-constructed variant
    cannot violate the invariant. The vision rubric returns an empty
    string in the decorative case and a refusal (faces, NSFW) is also
    bucketed as decorative with empty description.

    `described_at` and `downloaded_at` MUST be timezone-aware (UTC).
    """

    kind: Literal["photo_described"] = "photo_described"
    local_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bytes_size: int = Field(gt=0)
    downloaded_at: datetime
    is_decorative: bool
    description: str
    description_lang: SupportedLanguage
    description_version: str
    described_at: datetime

    @field_validator("local_path")
    @classmethod
    def _validate_local_path(cls, value: str) -> str:
        _ = cls
        return _reject_local_path_traversal(value)

    @field_validator("downloaded_at")
    @classmethod
    def _validate_downloaded_at(cls, value: datetime) -> datetime:
        _ = cls
        return _require_utc_aware("downloaded_at", value)

    @field_validator("described_at")
    @classmethod
    def _validate_described_at(cls, value: datetime) -> datetime:
        _ = cls
        return _require_utc_aware("described_at", value)

    @model_validator(mode="after")
    def _decorative_implies_empty_description(self) -> "MediaPhotoDescribed":
        """A decorative photo MUST have an empty description (rubric contract).

        Enforced at the type boundary so downstream callers can rely on
        `is_decorative => not description` without re-validating. The
        producer (`describe._apply_judgment`) already overwrites the
        description to "" on decorative judgments; this is the
        defence-in-depth for hand-edited records.
        """
        if self.is_decorative and self.description != "":
            raise ValueError(
                f"is_decorative=True requires an empty description, got {self.description!r}"
            )
        return self


class MediaVideoPending(BaseModel):
    """A video captured but not downloaded.

    `url` is the playable stream extracted from `video_info.variants` — the
    highest-bitrate progressive mp4 when X offers one, else the HLS (`.m3u8`)
    manifest. It is NOT the poster image; that is kept separately in
    `thumbnail_url` so a note can still show a still while the bytes are
    pending.

    `bitrate` (of the chosen mp4) and `duration_millis` let a download
    pre-flight estimate total size (bitrate × duration) without fetching a
    byte. All three are optional: legacy records carry only `url`, and an
    HLS-only variant has no bitrate. Note that a real X mp4 can report
    `bitrate: 0` (animated GIFs always do), so a downstream size estimator
    must treat `bitrate in (None, 0)` as "unknown" — fall back to a HEAD
    request's `Content-Length` — NOT as "0 bytes". Videos stay in this state until a
    future iteration adds the download (direct mp4 via httpx; HLS via
    ffmpeg). The variant is in the union from day one so the wire shape does
    not change when video download lands.
    """

    type: Literal["video"] = "video"
    kind: Literal["video_pending"] = "video_pending"
    url: str
    thumbnail_url: str | None = None
    bitrate: int | None = None
    duration_millis: int | None = None


class MediaVideoDownloaded(BaseModel):
    """A video successfully downloaded; the mp4 bytes exist at `local_path`.

    The downloaded counterpart of `MediaVideoPending`, produced by
    `xbrain download-videos` for a real progressive **mp4** stream (HLS
    `.m3u8` manifests need ffmpeg and are deferred to a follow-up — they
    stay `MediaVideoPending`). The playable `url`, the poster `thumbnail_url`
    and the `bitrate` / `duration_millis` are carried over verbatim from the
    pending entry so the record stays self-describing (and a `--force`
    re-download has everything it needs without re-capturing from X). The
    carried fields are optional for the same reason they are on
    `MediaVideoPending`: a legacy or hand-built pending entry may lack them.

    `local_path` is relative to `data/media/` (no absolute path, no `..`
    traversal — the shared `_reject_local_path_traversal` validator) so the
    store can move across machines. `bytes_size` is `gt=0`: a zero-byte
    "downloaded" video is semantically illegal — the downloader records the
    size only after a non-empty write. Unlike a photo there are no width /
    height — a video is not decoded, only its bytes are written.

    Why NOT inherit from `MediaVideoPending`? The media states form a
    pydantic tagged union, not a Liskov hierarchy — call sites use
    `isinstance(entry, MediaVideoPending)` to mean "exactly the pending
    state" (the download eligibility walk, the diff counters, the generator
    routing). Inheritance would silently re-match a downloaded entry against
    those checks. The carried fields are re-declared (mirroring how the photo
    Downloaded / Described variants re-declare their shared fields).

    `downloaded_at` MUST be timezone-aware (UTC) — same contract as every
    other persisted instant on the media variants.
    """

    type: Literal["video"] = "video"
    kind: Literal["video_downloaded"] = "video_downloaded"
    url: str
    thumbnail_url: str | None = None
    bitrate: int | None = None
    duration_millis: int | None = None
    local_path: str
    bytes_size: int = Field(gt=0)
    downloaded_at: datetime

    @field_validator("local_path")
    @classmethod
    def _validate_local_path(cls, value: str) -> str:
        _ = cls  # required by @field_validator+@classmethod; placate vulture
        return _reject_local_path_traversal(value)

    @field_validator("downloaded_at")
    @classmethod
    def _validate_downloaded_at(cls, value: datetime) -> datetime:
        _ = cls
        return _require_utc_aware("downloaded_at", value)


class MediaVideoFailed(BaseModel):
    """A video download attempted and failed (categorised).

    The failure off-ramp from `MediaVideoPending`, mirroring
    `MediaPhotoFailed`: `failure_reason` (reusing `MediaFailureReason`) is
    required so a failure is always demonstrable evidence, and `attempts`
    (`ge=1`) counts how many `xbrain download-videos` runs have tried this
    url — each transient retry bumps it. The source `url` plus the carried
    `thumbnail_url` / `bitrate` / `duration_millis` ride along so a retry has
    everything it needs without re-capturing from X.

    `last_attempt_at` MUST be timezone-aware (UTC) — same contract as the
    other persisted instants on the media variants.
    """

    type: Literal["video"] = "video"
    kind: Literal["video_failed"] = "video_failed"
    url: str
    thumbnail_url: str | None = None
    bitrate: int | None = None
    duration_millis: int | None = None
    failure_reason: MediaFailureReason
    error: str | None = None
    attempts: int = Field(ge=1)
    last_attempt_at: datetime

    @field_validator("last_attempt_at")
    @classmethod
    def _validate_last_attempt_at(cls, value: datetime) -> datetime:
        _ = cls
        return _require_utc_aware("last_attempt_at", value)


def _normalise_legacy_media(value: Any) -> Any:
    """Migrate the legacy ``{type, url}`` shape to the tagged union.

    The legacy shape (the original `Media(type=..., url=...)` BaseModel)
    is mapped one-to-one:

    - ``{"type": "photo", "url": ...}`` → `MediaPhotoPending` payload.
    - ``{"type": "video", "url": ...}`` → `MediaVideoPending` payload.

    Records that already carry a ``kind`` field are passed through
    unchanged — they are either fresh (extract-time) or already in the
    tagged-union shape (`photo_pending` / `photo_downloaded` /
    `photo_failed` / `photo_described` / `video_pending`). A record with
    neither `kind` nor a recognised `type` is passed through unchanged
    so pydantic raises a clean discriminator error rather than this
    validator inventing a state. The described variant has no legacy
    shape — it can only originate from a vision-describe run.
    """
    if not isinstance(value, dict):
        return value
    if "kind" in value:
        return value
    type_value = value.get("type")
    if type_value == "photo":
        return {**value, "kind": "photo_pending"}
    if type_value == "video":
        return {**value, "kind": "video_pending"}
    return value


# The persisted media type — a discriminated union over the four variants,
# wrapped in an outer `BeforeValidator` that promotes the legacy
# `{type, url}` shape on read. Same layering rationale as `ContentSource`
# (see the long comment above): the discriminator check must run AFTER
# the legacy normaliser, otherwise legacy records get rejected.
_MediaTagged = Annotated[
    Union[
        MediaPhotoPending,
        MediaPhotoDownloaded,
        MediaPhotoFailed,
        MediaPhotoDescribed,
        MediaVideoPending,
        MediaVideoDownloaded,
        MediaVideoFailed,
    ],
    Field(discriminator="kind"),
]
MediaEntry = Annotated[
    _MediaTagged,
    BeforeValidator(_normalise_legacy_media),
]


# TypeAdapter for tests / ad-hoc validation of a single entry outside an
# `Item` context (mirrors `ContentSourceAdapter`).
MediaEntryAdapter: TypeAdapter[
    Union[
        MediaPhotoPending,
        MediaPhotoDownloaded,
        MediaPhotoFailed,
        MediaPhotoDescribed,
        MediaVideoPending,
        MediaVideoDownloaded,
        MediaVideoFailed,
    ]
] = TypeAdapter(MediaEntry)


def Media(  # noqa: N802  -- factory keeps the legacy PascalCase call site
    *, type: Literal["photo", "video"], url: str
) -> MediaPhotoPending | MediaVideoPending:
    """Backward-compatible factory matching the pre-tagged-union constructor.

    The previous `Media` class was a flat `BaseModel(type, url)`. The
    extractor (`extract/graphql.py`) and the archive importer
    (`archive.py`) still call `Media(type="photo", url=...)` directly —
    this factory keeps those call sites working by returning the
    appropriate variant. Photo URLs become `MediaPhotoPending` (the
    initial state); video URLs become `MediaVideoPending`.

    TODO: when the LLM-description phase migrates `extract/graphql.py`
    and `archive.py` to construct the variants directly, drop this
    factory.
    """
    if type == "photo":
        return MediaPhotoPending(url=url)
    return MediaVideoPending(url=url)


class ThreadInfo(BaseModel):
    """Marker that an item is part of a multi-tweet thread."""

    is_thread: bool = True
    root_id: str
    position: int | None = None


class VideoFrame(BaseModel):
    """One key-frame slide extracted from a video by `digest-video --frames` (#44 PR4).

    The visual layer of a slide-heavy talk: a scene-change key frame's
    `timestamp` (seconds into the video, `ge=0`), the relative `local_path` of
    its downscaled image under `data/media/` (`<item-id>/frames/<n>.<ext>`), and
    the `description` produced by the EXTERNAL vision step. `generate` mirrors the
    image into the vault's `_media/` tree and embeds it exactly like a downloaded
    photo, with the description as a caption. It is carried on the `x_video`
    `ContentSourceSuccess.frames` list (additive — see there).

    `local_path` reuses the shared `_reject_local_path_traversal` validator (no
    absolute path, no `..`) so a hand-edited store cannot aim a slide embed at
    bytes outside the media root — the same defence the photo variants carry.
    """

    timestamp: float = Field(ge=0)
    local_path: str
    description: str = ""

    @field_validator("local_path")
    @classmethod
    def _validate_local_path(cls, value: str) -> str:
        _ = cls  # required by @field_validator+@classmethod; placate vulture
        return _reject_local_path_traversal(value)


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
    # Video-transcript metadata for `kind="x_video"` sources (#44). Both are
    # optional + default to None so every EXISTING (article) record LOADS
    # unchanged — `has_speech is None` marks a non-video source. (A re-dump does
    # add `has_speech: null` / `language: null` to legacy sources — a one-time,
    # backward-compatible additive churn, not a load-breaking change.) For an
    # `x_video` source `has_speech=False` is the no-speech marker (empty `text`,
    # never a failure), and `language` is the detected transcript language.
    has_speech: bool | None = None
    language: str | None = None
    # Key-frame slides for `kind="x_video"` sources when `digest-video --frames`
    # ran (#44 PR4). Optional + additive (defaults to `[]`), so every EXISTING
    # record LOADS unchanged — a pre-PR4 `x_video` source (and every article
    # source) simply carries no frames. A re-dump adds `frames: []`, the same
    # one-time backward-compatible additive churn as `has_speech`/`language`. Each
    # `VideoFrame` embeds into the note like a downloaded photo (see `VideoFrame`).
    frames: list[VideoFrame] = Field(default_factory=list)


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
    media: list[MediaEntry] = Field(default_factory=list)
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

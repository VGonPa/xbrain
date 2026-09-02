"""The knowledge read models — the logical view a consumer sees (spec §3.2).

These do NOT duplicate `xbrain.models`. `Item`, `Content`, `Enrichment`, `Topic` and
`TopicPage` remain the store's write models and are untouched by this layer; the classes
here are the READ projection an external model receives, and they exist because a consumer
must never have to know the shape of `items.json`, hunt for a markdown heading, or guess
which account wrote a quoted tweet (spec §3.1).

THREE CONSTRAINTS CARRY THE WEIGHT, and each is asserted by the suite:

* `frozen=True` — a surface whose text can be mutated after emission has a fingerprint that
  no longer means anything, and "the retrieved text is verbatim with respect to that
  surface" (spec §3.8) stops being checkable;
* `extra="forbid"` — spec §7.1 freezes these shapes per envelope (today `SearchResponse` "2",
  `EvidenceBundle` "1", the graph envelope "1") so the CLI and MCP adapters cannot drift; a
  model that swallows unknown keys lets a producer add a field no consumer ever sees;
* fingerprints are pattern-constrained to lowercase sha256 hex, the same defence
  `VerificationVerdict.output_fingerprint` already carries.

WHAT IS DELIBERATELY ABSENT: `KnowledgeSurface` has no `verification` field. A persisted
verdict could never be invalidated when the verdict changed, because `surface_fingerprint`
hashes (version, type, origin, text) and not the verdict — so a FAIL revoked by
`verify --audit` would keep being served as the PASS it used to be. Verification is
hydrated from the LIVE store at response time, applying the same freshness check `generate`
already applies. `SurfaceType` lives here, with the fields that reference it; the maps that
say which kind produces which surface live in `surfaces.py`, next to the emitter that uses
them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xbrain.knowledge.provenance import Origin, TrustClass
from xbrain.models import Author, ContentKind, FailureReason, SourceName, Verdict

# Every kind of text this layer can surface. Spec §4, one row per physical surface.
SurfaceType = Literal[
    "post",
    "external_article",
    "x_article",
    "thread",
    "quoted_post",
    "video_transcript",
    "video_frame",
    "image_description",
    "summary",
    "video_digest",
    "topic_description",
    "topic_overview",
    "topic_note",
    "user_note",
]

# Warnings a surface can carry about itself (spec §3.4, last bullet).
SurfaceWarning = Literal["stale", "machine_generated", "unverified", "truncated"]

# Why a link in `item.links` has no recovered body.
#
# FIVE values, not the four the plan sketched, and the fifth is the honest one. `timeout`,
# `dns_error` and `unknown_error` are neither HTTP refusals nor extraction failures: they
# are "the fetch failed", which is exactly the distinction `executors.api._FAILURE_CLAUSE`
# already draws for the LLM surfaces. Squeezing a DNS failure into `http_error` to keep a
# four-value enum would report a fact about the page that nobody measured.
UnfetchedReason = Literal[
    "not_attempted",  # no fetch was ever recorded for this URL
    "http_error",  # the server refused: 404, 403, paywall
    "not_extractable",  # downloaded, but no article body could be extracted
    "blocked_interstitial",  # a cookie/consent or login wall came back instead of an article
    "fetch_failed",  # timeout, DNS, or an uncategorised extractor error
]

# TOTAL over `models.FailureReason`, asserted by the suite. A `.get(..., default)` here
# would silently bucket a newly added store failure reason into whichever branch the
# default happened to name — the class of silent mislabelling this layer exists to avoid.
UNFETCHED_REASON_BY_FAILURE: dict[FailureReason, UnfetchedReason] = {
    "not_found": "http_error",
    "forbidden": "http_error",
    "paywall": "http_error",
    "js_required": "not_extractable",
    "empty_content": "not_extractable",
    "blocked_interstitial": "blocked_interstitial",
    "timeout": "fetch_failed",
    "dns_error": "fetch_failed",
    "unknown_error": "fetch_failed",
}

_FROZEN = ConfigDict(frozen=True, extra="forbid")
_SHA256 = r"^[0-9a-f]{64}$"


class DerivedText(BaseModel):
    """A piece of DERIVED text with its provenance attached in the SAME object.

    This is the shape that satisfies invariant 2 of spec §3.7 structurally rather than by
    convention: the text cannot be read without the `origin` that qualifies it, because the
    two arrive together. It is why `SearchResult.summary`, `TopicRecord.overview` and the
    topic notes are nested models rather than bare strings — spec §3.6 is explicit that a
    topic's description, overview and notes *are derived layers WITH THEIR OWN provenance*,
    and a bare `str` field would strip exactly that.

    `verification_status` is hydrated from the live store when the text is a verified output
    (M5), and is `None` when there is no CURRENT verdict — never a stale one, and never a
    reassuring default.
    """

    model_config = _FROZEN

    text: str
    origin: Origin
    verification_status: Verdict | None = None


class Locator(BaseModel):
    """Where the text lives in the original data. Only `kind` is required.

    A locator is what makes a claim checkable by hand: given one, a reader can open the
    store and find the exact bytes the answer quoted. Every positional field is optional
    because the kinds differ in what "position" even means — an `item_text` surface has no
    source index, a topic note has no character range.

    `article_block` is a RESERVED kind with no producer today (m18). Measured 2026-08-31:
    250 `ArticleImageBlock`s in the corpus, **0 with a description**, so there is no text to
    index. When `describe` starts describing them they arrive as `image_description`
    carrying `kind="article_block"` and a `block_index`. The slot is declared so the hole is
    visibly a known gap rather than an oversight.
    """

    model_config = _FROZEN

    kind: Literal[
        "item_text",
        "content_source",
        "media",
        "video_frame",
        "article_block",
        "enrichment",
        "topic_page",
        "vocab",
    ]
    source_index: int | None = None
    content_kind: ContentKind | None = None
    url: str | None = None
    media_index: int | None = None
    frame_index: int | None = None
    frame_timestamp: float | None = None
    block_index: int | None = None
    note_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None


class SourceFailure(BaseModel):
    """A content fetch that was ATTEMPTED and failed — structured, not prose.

    Spec §4: a fetch failure emits no text but `get` must report that the source failed and
    why. Distinct from `UnfetchedLink`: this one records an attempt.
    """

    model_config = _FROZEN

    kind: ContentKind
    url: str
    failure_reason: FailureReason
    error: str | None = None
    http_status: int | None = None


class UnfetchedLink(BaseModel):
    """A URL in `item.links` with no recovered body — metadata, never evidence (m7).

    Spec §4 says these URLs *are metadata, not textual evidence*, and that the consumer must
    SEE them together with the reason there is no body — the same thing
    `unfetched_links_note` already tells the LLM surfaces (PR-I).

    There is deliberately NO text field. Naming the cause never licenses describing the
    content, and the absence of the field is the guardrail: nothing can put a body here.
    """

    model_config = _FROZEN

    url: str
    reason: UnfetchedReason
    detail: str | None = None


class KnowledgeSurface(BaseModel):
    """One semantic unit of text on an item or topic, before chunking (spec §3.2).

    `attribution` is what sustains invariant 3 of spec §3.7: on a `quoted_post` it is the
    THIRD PARTY's author, never the item's. That is the whole #86 attribution rule
    expressed as a field — the poster is not the author of what they quote.

    `producer` and `produced_at` answer spec §3.4's *method or component that produced it*
    and *instant of capture/generation*. They are populated from data already in the store
    (`enriched.executor`, `MediaPhotoDescribed.description_version`/`described_at`,
    `TopicPage.synthesized_at`, `content.fetched_at`) and are `None` where the format
    genuinely does not record it — which INCLUDES `video_transcript` and `video_frame`
    (F7-7, round 08): the `x_video` source records neither the transcriber nor the vision
    command that wrote the text. Until round 08 those two surfaces carried the command
    CONFIGURED when the surface was emitted; measured on the real corpus, changing
    `[transcribe].command` to `whisper-large-v3` made `get` serve `producer:
    whisper-large-v3` for a transcript parakeet wrote, text and fingerprints identical — a
    provenance claim the store cannot back, and spec §3.4 says the unknown is not filled in
    by intuition. It is `None` now, the origin (`asr`/`vlm`, `machine_generated`) is still
    declared, and the honest fix — stamping the producer on the source when `digest-video`
    attaches the transcript, as `caption_contract` does for frames — is a store change
    outside Plan 02, recorded as an open issue for Plan 03.
    """

    model_config = _FROZEN

    surface_id: str
    owner_type: Literal["item", "topic"]
    owner_id: str
    surface_type: SurfaceType
    text: str
    title: str | None = None
    origin: Origin
    trust_class: TrustClass
    derived: bool
    attribution: Author | None = None
    producer: str | None = None
    produced_at: datetime | None = None
    locator: Locator
    fingerprint: str = Field(pattern=_SHA256)
    language: str | None = None
    warnings: tuple[SurfaceWarning, ...] = ()


class KnowledgeChunk(BaseModel):
    """An indexable fragment of a surface — the unit lexical and vector search score.

    `text` is VERBATIM with respect to the surface: `surface.text[char_start:char_end] ==
    chunk.text` for every chunk, which is the operational definition of spec §3.8's
    verifiability claim and is property-tested over the whole fixture corpus.

    `title` travels WITH the chunk (m6). Spec §4 says article titles accompany their chunks;
    with the title only on the surface, a `SearchMatch` on chunk 7 of a long article would
    reach the consumer as an orphan paragraph. It is accompanying metadata, not a chunk of
    its own, so it adds nothing to the indexed corpus.
    """

    model_config = _FROZEN

    chunk_id: str
    surface_id: str
    owner_type: Literal["item", "topic"]
    owner_id: str
    surface_type: SurfaceType
    text: str
    title: str | None = None
    chunk_index: int
    char_start: int
    char_end: int
    origin: Origin
    trust_class: TrustClass
    derived: bool
    attribution: Author | None = None
    topics: tuple[str, ...] = ()
    url: str | None = None
    language: str | None = None
    fingerprint: str = Field(pattern=_SHA256)


class KnowledgeItem(BaseModel):
    """The stable item, as a consumer sees it (spec §3.2).

    `failed_sources` and `unfetched_links` are two fields on purpose (m7): "we tried and it
    failed" and "there is no body for this URL" are different facts, and collapsing them
    makes a link nobody attempted indistinguishable from one that returned a 404.
    """

    model_config = _FROZEN

    item_id: str
    source: SourceName
    url: str
    author: Author
    created_at: datetime
    captured_at: datetime
    primary_topic: str | None = None
    topics: tuple[str, ...] = ()
    available_surfaces: tuple[SurfaceType, ...] = ()
    content_kinds: tuple[ContentKind, ...] = ()
    failed_sources: tuple[SourceFailure, ...] = ()
    unfetched_links: tuple[UnfetchedLink, ...] = ()
    note_path: str | None = None
    bookmark_folder: str | None = None
    warnings: tuple[str, ...] = ()


class TopicRecord(BaseModel):
    """The common view of a topic across vocabulary, assignments and synthesis (spec §3.6).

    `stale` is derived by the emitter from the live primary-post count against
    `TopicPage.post_count_at_synth` — the same derivation `topics.topics_needing_synth`
    performs. It is carried here so consumers do not each re-derive it; `TopicPage` itself
    still stores the count rather than a flag, because a stored flag desyncs.
    """

    model_config = _FROZEN

    topic_id: str
    slug: str
    description: DerivedText
    overview: DerivedText | None = None
    notes: tuple[DerivedText, ...] = ()
    primary_item_ids: tuple[str, ...] = ()
    secondary_item_ids: tuple[str, ...] = ()
    synthesized_at: datetime | None = None
    post_count_at_synth: int | None = None
    stale: bool = True
    vocab_fingerprint: str = Field(pattern=_SHA256)
    synthesis_fingerprint: str | None = Field(default=None, pattern=_SHA256)

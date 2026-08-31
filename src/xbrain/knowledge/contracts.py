"""The external response schemas, frozen at `schema_version: "1"` (spec §7).

FROZEN NOW, ON PURPOSE, before the services that fill them exist. Spec §7.1 says CLI JSON
and MCP are two adapters over ONE service and must not implement two formats — and the way
two formats appear is never a decision: the second adapter is written months later, against
whatever the first happened to emit. Freezing the model first makes the second adapter a
consumer rather than an author.

It is also why `producer`, `produced_at` and the eight filters are all here in version 1.
Adding a field afterwards is exactly the incompatible change the freeze is meant to prevent.

THE NAMING RULE, AND WHY IT IS ENFORCED BY TOTALITY (m12). Invariant 2 of spec §3.7: nothing
is called `text` without `origin` beside it. A walker that scans a JSON payload and fails on
a text field with no `origin` sibling cannot be written honestly — `query`, `url`, `handle`,
`name` and `title` are text too, so it either fails always or grows a list of ad-hoc
exceptions, which is a second copy of the rule wearing the costume of a check. Instead the
DECLARED `str` fields are partitioned into two frozensets and the partition is asserted
total: adding a text field without deciding which side it is on goes red.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo

from xbrain.knowledge.models import (
    DerivedText,
    KnowledgeChunk,
    KnowledgeItem,
    KnowledgeSurface,
    Locator,
    SourceFailure,
    SurfaceType,
    TopicRecord,
    UnfetchedLink,
)
from xbrain.knowledge.provenance import Origin, TrustClass
from xbrain.models import Author, ContentKind, SourceName, VerificationVerdict

_FROZEN = ConfigDict(frozen=True, extra="forbid")

# The retrieval strategies spec §5.7 requires to be separable. Declared in full here even
# though this PR ships only `lexical`, because the enum is part of the frozen contract:
# adding a member later would change the response schema.
Strategy = Literal["lexical", "vector", "hybrid", "hybrid_graph"]

# Which generator produced a match. `graph` is declared for Plan 04 for the same reason.
Channel = Literal["lexical", "vector", "graph"]


class SearchMatch(BaseModel):
    """One chunk that matched, with why it matched and where it came from (spec §5.3).

    `matched_by`, `lexical_rank` and `vector_rank` are kept because spec §5.3 requires the
    fusion to preserve the EXPLANATION of a result. `score` is a ranking signal and is
    documented as such — it is deliberately not presented as a probability, because a fused
    rank has no calibrated scale.
    """

    model_config = _FROZEN

    chunk_id: str
    surface_type: SurfaceType
    origin: Origin
    trust_class: TrustClass
    derived: bool
    excerpt: str
    attribution: Author | None = None
    matched_by: tuple[Channel, ...] = ()
    lexical_rank: int | None = None
    vector_rank: int | None = None
    score: float = 0.0
    locator: Locator


class SearchResult(BaseModel):
    """One item's matches, grouped (spec §5.4).

    Grouping by item is what stops a long transcript filling the top ten with ten adjacent
    windows of itself. `verify_with` answers spec §5.4's *indication of what to ask `get` for
    in order to verify*: when a match landed on a summary, this names the underlying source
    that can actually settle the claim.
    """

    model_config = _FROZEN

    rank: int
    item_id: str
    url: str
    author: Author
    created_at: datetime
    summary: DerivedText | None = None
    topics: tuple[str, ...] = ()
    matches: tuple[SearchMatch, ...] = ()
    available_surfaces: tuple[SurfaceType, ...] = ()
    verify_with: tuple[SurfaceType, ...] = ()


class SearchFilters(BaseModel):
    """The EIGHT minimum filters of spec §7.2 — not six.

    `content_kinds` and `has_surfaces` come from no existing column and need their own
    plumbing in Plan 02. They are declared here anyway because declaring six would freeze a
    contract the next plan could not satisfy without a version bump; committing to eight is
    the honest version of the same decision.
    """

    model_config = _FROZEN

    created_from: datetime | None = None
    created_to: datetime | None = None
    source: SourceName | None = None
    author: str | None = None
    topics: tuple[str, ...] = ()
    content_kinds: tuple[ContentKind, ...] = ()
    origins: tuple[Origin, ...] = ()
    has_surfaces: tuple[SurfaceType, ...] = ()


class IndexStatusRef(BaseModel):
    """What the response says about the index that answered it (spec §5.6, §9.3).

    TWO SIGNALS WITH TWO NAMES (B3). `corrupt_chunks_excluded` counts rows whose fingerprint
    does not recompute — corruption, or a row written by a different chunker version. That is
    an INTERNAL consistency check and it does not detect the failure that actually happens:
    *you ran `enrich` and did not reindex*. That one is a `degraded` flag
    (`"index_behind_store"`), raised by comparing the manifest's cheap store signal against
    the current file.

    The previous name for the counter was `stale_chunks_excluded`, which sounded like the
    second and measured the first. One name per fact, because a counter that answers a
    different question from the one its name asks is how a wrong number gets quoted with
    confidence.
    """

    model_config = _FROZEN

    manifest_version: str
    built_at: datetime | None = None
    corrupt_chunks_excluded: int = 0
    degraded: tuple[str, ...] = ()


class SearchResponse(BaseModel):
    """The `search` envelope (spec §7.2)."""

    model_config = _FROZEN

    schema_version: Literal["1"] = "1"
    query: str
    strategy: Strategy
    filters: SearchFilters
    index: IndexStatusRef
    results: tuple[SearchResult, ...] = ()
    truncated: bool = False
    cursor: str | None = None


class EvidenceBundle(BaseModel):
    """The `get` response — the unit a model reasons over (spec §3.2, §7.3).

    It contains the item, its topics and the requested surfaces, each with provenance. It
    contains NO conclusion written by xbrain: spec §0.2 is explicit that the primary output
    is structured evidence and that the consuming model does the reasoning.

    `verification` is hydrated from the live store at response time (M5). A verdict copied
    onto a `KnowledgeSurface` at emission would be frozen there and would keep asserting a
    PASS that the current output never earned.
    """

    model_config = _FROZEN

    schema_version: Literal["1"] = "1"
    item: KnowledgeItem
    topics: tuple[TopicRecord, ...] = ()
    surfaces: tuple[KnowledgeSurface, ...] = ()
    chunks: tuple[KnowledgeChunk, ...] = ()
    failures: tuple[SourceFailure, ...] = ()
    unfetched_links: tuple[UnfetchedLink, ...] = ()
    verification: dict[str, VerificationVerdict] = Field(default_factory=dict)
    truncated: bool = False
    cursor: str | None = None


class GraphNode(BaseModel):
    """A node of the minimal graph — `item:<id>` or `topic:<slug>` (spec §6.2)."""

    model_config = _FROZEN

    node_id: str
    node_type: Literal["item", "topic"]
    label: str | None = None


class GraphEdge(BaseModel):
    """An edge, with the METHOD that created it and the items that support it (spec §6.2).

    `supporting_item_ids` and `method` are required by spec §3.7.10 and §6.4: a
    `CO_OCCURS_WITH` edge means *these topics were assigned together to these items of
    xbrain*, and NOT *these concepts are causally related in the world*. Carrying the support
    ids is what lets a consumer check which of the two it is looking at.
    """

    model_config = _FROZEN

    source: str
    target: str
    relation: Literal["HAS_PRIMARY_TOPIC", "HAS_TOPIC", "CO_OCCURS_WITH"]
    method: str
    weight: float = 0.0
    shared_items: int = 0
    supporting_item_ids: tuple[str, ...] = ()


class GraphPath(BaseModel):
    """An explicit path from a seed to an expanded node (spec §6.3)."""

    model_config = _FROZEN

    nodes: tuple[str, ...]
    edges: tuple[GraphEdge, ...] = ()


class GraphExpansionResponse(BaseModel):
    """The `graph_expand` envelope (spec §7.4)."""

    model_config = _FROZEN

    schema_version: Literal["1"] = "1"
    seeds: tuple[str, ...] = ()
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    paths: tuple[GraphPath, ...] = ()


# Every model whose declared text fields the partition below must cover.
#
# `Author` is here although it comes from `xbrain.models` and carries no corpus content
# today. It is reachable from the contract twice — `SearchResult.author` and
# `KnowledgeSurface.attribution`, the second being the attribution rule itself — and leaving
# it out put its `str` fields outside the partition entirely, so a text field added to it
# later (a bio, a display note) would join the contract with nobody deciding whether it needs
# an origin, and the totality test that exists to force that decision would stay green.
CONTRACT_MODELS: tuple[type[BaseModel], ...] = (
    Author,
    DerivedText,
    SearchMatch,
    SearchResult,
    SearchFilters,
    IndexStatusRef,
    SearchResponse,
    EvidenceBundle,
    GraphNode,
    GraphEdge,
    GraphPath,
    GraphExpansionResponse,
    KnowledgeSurface,
    KnowledgeChunk,
    KnowledgeItem,
    TopicRecord,
    UnfetchedLink,
    SourceFailure,
    Locator,
)


def is_str_field(field: FieldInfo) -> bool:
    """True for a `str` or `str | None` field — NOT for a nested model or a collection.

    Deliberately shallow. The partition below is a partition over `str` FIELDS, and the one
    apparent exception (`SearchResult.summary`) is not an exception at all: it is a nested
    `DerivedText`, whose own `text` field IS in the partition and which carries its `origin`
    in the same object. Nesting content together with its provenance is the correct way to
    satisfy invariant 2, not a way around it. If a future change needed to cover nested
    content carriers too, the honest move is to redefine THIS function to descend into them
    and say so here — not to smuggle a model into a set of string fields.
    """
    annotation = field.annotation
    if annotation is str:
        return True
    if get_origin(annotation) is None:
        return False
    args = set(get_args(annotation))
    return args == {str, type(None)}


# Text fields that TRANSPORT CORPUS CONTENT and therefore require provenance beside them.
TEXT_FIELDS_REQUIRING_ORIGIN: frozenset[tuple[str, str]] = frozenset(
    {
        ("DerivedText", "text"),
        ("SearchMatch", "excerpt"),
        ("KnowledgeSurface", "text"),
        ("KnowledgeChunk", "text"),
    }
)

# Text fields that are metadata, identifiers, or an echo of the request. None of them can
# carry a claim, so none of them needs an origin: a URL is topic signal and never a name
# (the rule `evidence.py` had to learn the hard way), a handle is an identifier, and a title
# names a work rather than asserting anything about it.
TEXT_FIELDS_WITHOUT_ORIGIN: frozenset[tuple[str, str]] = frozenset(
    {
        ("Author", "handle"),
        ("Author", "name"),
        ("SearchResponse", "query"),
        ("SearchResponse", "cursor"),
        ("SearchResult", "item_id"),
        ("SearchResult", "url"),
        ("SearchMatch", "chunk_id"),
        ("SearchFilters", "author"),
        ("IndexStatusRef", "manifest_version"),
        ("EvidenceBundle", "cursor"),
        ("GraphNode", "node_id"),
        ("GraphNode", "label"),
        ("GraphEdge", "source"),
        ("GraphEdge", "target"),
        ("GraphEdge", "method"),
        ("KnowledgeSurface", "surface_id"),
        ("KnowledgeSurface", "owner_id"),
        ("KnowledgeSurface", "title"),
        ("KnowledgeSurface", "producer"),
        ("KnowledgeSurface", "fingerprint"),
        ("KnowledgeSurface", "language"),
        ("KnowledgeChunk", "chunk_id"),
        ("KnowledgeChunk", "surface_id"),
        ("KnowledgeChunk", "owner_id"),
        ("KnowledgeChunk", "title"),
        ("KnowledgeChunk", "url"),
        ("KnowledgeChunk", "language"),
        ("KnowledgeChunk", "fingerprint"),
        ("KnowledgeItem", "item_id"),
        ("KnowledgeItem", "url"),
        ("KnowledgeItem", "primary_topic"),
        ("KnowledgeItem", "note_path"),
        ("KnowledgeItem", "bookmark_folder"),
        ("TopicRecord", "topic_id"),
        ("TopicRecord", "slug"),
        ("TopicRecord", "vocab_fingerprint"),
        ("TopicRecord", "synthesis_fingerprint"),
        ("UnfetchedLink", "url"),
        ("UnfetchedLink", "detail"),
        ("SourceFailure", "url"),
        ("SourceFailure", "error"),
        ("Locator", "url"),
    }
)

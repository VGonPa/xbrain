"""Reproducible identity for surfaces and chunks (spec §3.3).

Spec §3.3 requires ids that are stable while owner, surface and chunker do not change; free
of collisions between a quoted tweet, a thread, an article and a video; deterministic in
order; reversible from a chunk back to its surface and owner; and explicitly migrated when
the chunker version changes.

    topic_id   = "topic:<slug>"
    surface_id = "<owner_type>:<owner_id>:<surface_type>:<source_key>"
    chunk_id   = "<surface_id>:<chunk_index>:<chunker_version>"

THE DECISION THAT MATTERS: a content source's key is `sha1(kind\\0url)[:12]`, NOT its index
in `content.sources`. `fetch` rewrites that list on every re-capture, so an index-keyed id
would repoint stored chunks at a different body the first time two sources swapped places —
and it would fail silently, because the id would still resolve, just to the wrong text. The
kind is hashed alongside the URL because an `x_article` and an `external_article` can point
at the same canonical URL.

The `#n` suffix disambiguates a repeated `(kind, url)` WITHIN one item. Measured on the
corpus (2026-08-31): **0 items** have one today, so it is a safe failure path rather than
the normal case — implemented and tested anyway, because 119 items already carry more than
one source and the failure mode (two surfaces collapsing onto one id) is silent. The first
occurrence never carries a suffix, so a later duplicate cannot renumber the original.
"""

from __future__ import annotations

import hashlib
import re

from xbrain.knowledge.provenance import Origin
from xbrain.models import ContentSourceSuccess, Item

# Bumping either version invalidates everything derived under the old one. They are hashed
# INTO the fingerprints (and the chunker version into the chunk id itself), so a rebuild
# after a bump writes new ids rather than overwriting old ones with differently-cut text.
SURFACE_VERSION = "xbrain-knowledge-surface/v1"
CHUNKER_VERSION = "xbrain-knowledge-chunker/v1"

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# The key of a surface that can occur at most once per owner (spec §3.3, first row).
SINGLETON_SOURCE_KEY = "0"


class IdError(ValueError):
    """An id could not be built from the values given — a malformed slug, typically."""


def _sha1_12(*parts: str) -> str:
    """The first 12 hex chars of `sha1` over NUL-joined parts.

    NUL-joined for the same reason `verification.contract_fingerprint` does it: so two arms
    cannot concatenate into a third value. Truncated to 12 because this is a namespacing
    key inside an id, not a collision-resistance guarantee for adversarial input — and a
    64-char hash in every id would make the ids unreadable in a log for no gain.
    """
    return hashlib.sha1("\0".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def topic_id(slug: str) -> str:
    """`topic:<slug>`.

    The slug is validated against the same pattern `models.Topic` enforces. A slug
    containing a colon would make the id ambiguous to parse back, and spec §3.3 requires
    reverse resolution — so the ambiguity is rejected here rather than pushed onto every
    consumer.
    """
    if not _SLUG_RE.match(slug):
        raise IdError(f"not a topic slug: {slug!r}")
    return f"topic:{slug}"


def surface_id(owner_type: str, owner_id: str, surface_type: str, source_key: str) -> str:
    """`<owner_type>:<owner_id>:<surface_type>:<source_key>` (spec §3.3)."""
    return f"{owner_type}:{owner_id}:{surface_type}:{source_key}"


def chunk_id(surface: str, chunk_index: int, *, chunker_version: str = CHUNKER_VERSION) -> str:
    """`<surface_id>:<chunk_index>:<chunker_version>` (spec §3.3).

    The version is an argument defaulting to the module constant, so a test can assert the
    migration property without monkeypatching a global.
    """
    return f"{surface}:{chunk_index}:{chunker_version}"


def content_source_key(kind: str, url: str, *, occurrence: int = 0) -> str:
    """The `source_key` of one content source, with its occurrence ordinal applied."""
    base = _sha1_12(kind, url)
    return base if occurrence == 0 else f"{base}#{occurrence}"


def content_source_keys(item: Item) -> dict[int, str]:
    """`{index in content.sources: source_key}` for every SUCCESS source on the item.

    Keyed by the REAL index so a `locator.source_index` built from it points at the right
    entry even when failures sit in between — failures carry no text, emit no surface, and
    therefore appear in neither the keys nor the values.

    Duplicate `(kind, url)` pairs are numbered by order of appearance. That order comes
    from the list, which `fetch` may rewrite; two entries that share a `(kind, url)` are
    however indistinguishable by every field this key is built from, so a swap between them
    cannot change the mapping's value set.
    """
    keys: dict[int, str] = {}
    seen: dict[tuple[str, str], int] = {}
    for index, source in enumerate(item.content.sources if item.content else []):
        if not isinstance(source, ContentSourceSuccess):
            continue
        pair = (source.kind, source.url)
        occurrence = seen.get(pair, 0)
        seen[pair] = occurrence + 1
        keys[index] = content_source_key(source.kind, source.url, occurrence=occurrence)
    return keys


def video_frame_source_key(video_source_key: str, frame_index: int) -> str:
    """`<the video's source_key>.f<frame index>` (spec §3.3, `video_frame` row).

    Derived FROM the video's key rather than hashed independently, so a frame id says which
    video it came from without a lookup, and re-capturing the video (a new URL, hence a new
    key) visibly renames its frames instead of silently reusing the old ids.
    """
    return f"{video_source_key}.f{frame_index}"


def surface_fingerprint(
    surface_type: str, origin: Origin | str, text: str, *, surface_version: str = SURFACE_VERSION
) -> str:
    """sha256 of `(emitter version, surface type, origin, text)`.

    The type and the origin are hashed ALONGSIDE the text because the same string can
    legitimately appear under two surfaces — a summary that quotes the post verbatim — and
    those two are not interchangeable evidence. A fingerprint over text alone would call
    them the same content and let the index dedupe away the distinction provenance exists
    to preserve.
    """
    return _sha256("\0".join([surface_version, surface_type, str(origin), text]))


def chunk_fingerprint(
    surface: str, chunk_index: int, text: str, *, chunker_version: str = CHUNKER_VERSION
) -> str:
    """sha256 of `(chunker version, surface_id, chunk index, text)`.

    Spec §5.2: *the algorithm version is part of the fingerprint*, so changing the chunking
    forces the affected chunks to be regenerated instead of being served under ids whose
    text no longer matches how they were cut.
    """
    return _sha256("\0".join([chunker_version, surface, str(chunk_index), text]))


def _sha256(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

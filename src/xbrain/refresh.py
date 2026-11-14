"""Backfill the playable video URL + metadata onto already-stored items.

Why this stage exists
---------------------
`xbrain extract` is incremental: `extract_source` stops as soon as it reaches a
known id, and `store.merge_items` "adds, never overwrites". So the video
records captured before the playable-URL work landed — effectively the whole
existing corpus — still hold the **poster image** in `MediaVideoPending.url`,
with `bitrate` and `duration_millis` unset. A normal `extract` run will never
revisit them, so they stay poster-era forever.

`xbrain refresh-media` re-captures the full X history (logged in, with no
skip-known) and rewrites the VIDEO media on items already in the store, in
place: each poster-era `MediaVideoPending` is swapped for the freshly-parsed
one that carries the playable stream URL + bitrate + duration. Photos and every
enrichment / description / fetch field are left exactly as they are — the
backfill touches video entries and nothing else.

This module is pure: no browser, no I/O. The CLI (`cli._run_refresh_media`)
drives the live capture, the auto-snapshot and the persistence; here we only
transform an in-memory store against freshly-parsed items and estimate the
eventual download size without fetching a byte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from xbrain.executors.api import quoted_source
from xbrain.models import (
    QUOTED_CONTENT_KINDS,
    Author,
    Content,
    ContentSource,
    ContentSourceSuccess,
    Item,
    MediaEntry,
    MediaVideoPending,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# bitrate is bits/second and duration is milliseconds, so
# bytes = bitrate * (duration_millis / 1000) / 8 = bitrate * duration_millis / 8000.
_BITS_PER_BYTE_TIMES_MILLIS_PER_SECOND = 8000


@dataclass
class RefreshReport:
    """Counts emitted by `refresh_video_media` for the CLI summary line.

    - ``items_seen`` — fresh items whose id is already in the store: the
      population the backfill can act on. Fresh items not in the store are
      skipped (backfill only touches known ids) and never counted here.
    - ``items_refreshed`` — store items that had at least one video entry
      replaced this run.
    - ``videos_updated`` — individual `MediaVideoPending` entries swapped in
      place across all items.
    - ``items_with_video_not_seen`` — store items that still hold a video entry
      but were NOT re-seen in this capture (still poster-era; X did not surface
      them in the scroll). A diagnostic for "how much is left to backfill".
    """

    items_seen: int = 0
    items_refreshed: int = 0
    videos_updated: int = 0
    items_with_video_not_seen: int = 0


def _is_real_stream(video: MediaVideoPending) -> bool:
    """True when a fresh video is a real playable stream, not a poster fallback.

    `build_video_media` falls back to the poster image (``url ==
    media_url_https == thumbnail_url``) when X serves no usable
    `video_info.variants` — a drift symptom. A real stream's ``url`` is the
    mp4/HLS, which differs from the poster ``thumbnail_url``. Using this as the
    upgrade discriminator stops a drifted re-capture from DEGRADING an
    already-good record back to a poster (this is the repo's first overwriting
    path, so the guard matters).
    """
    return video.url != video.thumbnail_url


def _rebuild_media(
    existing: list[MediaEntry], fresh_videos: list[MediaVideoPending]
) -> tuple[list[MediaEntry], int]:
    """Replace each video entry positionally; keep every photo entry verbatim.

    Walks ``existing`` in order. The i-th `MediaVideoPending` is replaced by the
    i-th entry of ``fresh_videos`` **only when that fresh entry is a real stream**
    (see `_is_real_stream`); a poster-fallback fresh entry, or no fresh
    counterpart at all (fewer fresh videos than store videos), leaves the
    existing record untouched — never degraded, never dropped. Photo variants
    (Pending / Downloaded / Failed / Described) always pass through unchanged,
    preserving their download and description state. Returns the rebuilt list
    and the number of real replacements made.
    """
    rebuilt: list[MediaEntry] = []
    replaced = 0
    fresh_iter = iter(fresh_videos)
    for entry in existing:
        if not isinstance(entry, MediaVideoPending):
            rebuilt.append(entry)
            continue
        fresh = next(fresh_iter, None)
        if fresh is not None and _is_real_stream(fresh):
            rebuilt.append(fresh)
            replaced += 1
        else:
            rebuilt.append(entry)  # no upgrade available → keep the existing record
    return rebuilt, replaced


def _apply_fresh_videos(store_item: Item, fresh: Item, report: RefreshReport) -> None:
    """Swap ``store_item``'s video entries for ``fresh``'s; tally onto ``report``.

    A fresh item with no video leaves the store item untouched (the common case
    for a photo-only or text post re-seen in the scroll).
    """
    fresh_videos = [m for m in fresh.media if isinstance(m, MediaVideoPending)]
    if not fresh_videos:
        return
    rebuilt, replaced = _rebuild_media(store_item.media, fresh_videos)
    if replaced:
        store_item.media = rebuilt
        report.items_refreshed += 1
        report.videos_updated += replaced


def _count_video_items_not_seen(store: dict[str, Item], seen_ids: set[str]) -> int:
    """Store items that still hold a video entry but were NOT in this capture."""
    return sum(
        1
        for item_id, item in store.items()
        if item_id not in seen_ids and any(isinstance(m, MediaVideoPending) for m in item.media)
    )


def refresh_video_media(store: dict[str, Item], fresh_items: list[Item]) -> RefreshReport:
    """Backfill freshly-captured video media onto matching store items in place.

    For each fresh item whose id is in the store and that carries at least one
    video entry, the store item's ``media`` list is rebuilt: every existing
    `MediaVideoPending` is replaced positionally by the corresponding fresh
    video entry, and every photo entry is kept exactly as-is. Fresh items not in
    the store are skipped (backfill only touches known items); store items not
    re-seen, and fresh items with no video, leave the store untouched.

    Mutates ``store`` in place and returns a `RefreshReport` with the counts.
    """
    # First fresh entry wins on a duplicate id (an item captured from two
    # sources, e.g. a bookmark of one's own tweet) — mirrors extract_source.
    fresh_by_id: dict[str, Item] = {}
    for item in fresh_items:
        fresh_by_id.setdefault(item.id, item)

    report = RefreshReport()
    for fresh_id, fresh in fresh_by_id.items():
        store_item = store.get(fresh_id)
        if store_item is None:
            continue
        report.items_seen += 1
        _apply_fresh_videos(store_item, fresh, report)

    report.items_with_video_not_seen = _count_video_items_not_seen(store, set(fresh_by_id))
    return report


def estimate_download_size(store: dict[str, Item]) -> tuple[int, int, int]:
    """Estimate the total bytes to download for every stored video — no network.

    Sums ``bitrate * duration_millis / 1000 / 8`` (bits/s × seconds ÷ 8) over
    every `MediaVideoPending` in the store. A video whose ``bitrate`` is ``None``
    or ``0`` (animated GIFs always report ``0``), or whose ``duration_millis`` is
    missing, is UNKNOWN: excluded from the byte sum and counted separately —
    never treated as 0 bytes. Returns ``(estimated_bytes, n_estimable,
    n_unknown)``. Downloads nothing.
    """
    estimated_bytes = 0
    n_estimable = 0
    n_unknown = 0
    for item in store.values():
        for entry in item.media:
            if not isinstance(entry, MediaVideoPending):
                continue
            bitrate = entry.bitrate
            duration = entry.duration_millis
            if not bitrate or duration is None:
                n_unknown += 1
                continue
            estimated_bytes += bitrate * duration // _BITS_PER_BYTE_TIMES_MILLIS_PER_SECOND
            n_estimable += 1
    return estimated_bytes, n_estimable, n_unknown


@dataclass
class QuotedBackfillReport:
    """Counts emitted by `backfill_quoted_sources` for the CLI summary line.

    - ``items_seen`` — fresh items whose id is already in the store.
    - ``sources_attached`` — stored quote-tweets that GAINED a `quoted_tweet` source.
    - ``readable`` / ``unreadable`` — of those, the ones that carry the quoted body
      vs the ones X would not serve (deleted, protected, not hydrated). Both are
      progress: an unreadable quote is recorded as a demonstrable failure, which is
      what keeps the `content NOT fetched` marker honest.
    - ``already_present`` — already had a readable quote; left untouched (idempotent).
    - ``quoted_items_not_seen`` — stored quote-tweets STILL holding only a `quoted_id`
      because X did not re-surface them in this capture. The honest diagnostic: how
      much of the corpus this run could not repair.
    """

    items_seen: int = 0
    sources_attached: int = 0
    readable: int = 0
    unreadable: int = 0
    already_present: int = 0
    quoted_items_not_seen: int = 0


def _quoted_source(item: Item) -> ContentSource | None:
    """The item's `quoted_tweet` source in either variant (readable or failed).

    The either-variant selector, for deciding what is RECORDED. "Is there readable
    evidence?" is a different question with exactly one answer in this codebase —
    `executors.api.quoted_source`, which every LLM surface reads — so it is imported,
    not re-implemented here.
    """
    if item.content is None:
        return None
    return next((s for s in item.content.sources if s.kind in QUOTED_CONTENT_KINDS), None)


def _quoted_evidence(item: Item) -> tuple[str, Author | None] | None:
    """Exactly what the LLM surfaces READ from the quoted post, or None.

    Every quoted surface — the api prompt's body + label, the worksheet's `quoted_text` +
    `quoted_attribution`, the judge's `[Quoted post — …]` block, and the `NOT fetched`
    marker's on/off — is a function of `quoted_source(item)`, i.e. of exactly this pair.
    So this IS the generator's view, and comparing it before/after answers the only
    question that matters: did anything the model will read actually change?
    """
    source = quoted_source(item)
    return None if source is None else (source.text, source.author)


def _readable_evidence(source: ContentSource) -> tuple[str, Author | None] | None:
    """`_quoted_evidence` for a source about to be attached (it replaces any prior one)."""
    if isinstance(source, ContentSourceSuccess) and source.text:
        return (source.text, source.author)
    return None


def _attach_quoted(item: Item, source: ContentSource, now: datetime) -> bool:
    """Put `source` on `item`, replacing any prior `quoted_tweet` and keeping the rest.
    Return True iff the LLM-visible evidence changed (and `fetched_at` was advanced).

    **The clock moves only when the EVIDENCE moves.** `enrich._needs_reenrichment` keys
    on `content.fetched_at > enriched_at`, and `verify` fingerprints the OUTPUT — so an
    unconditional bump would, on every single run: re-run the model on a byte-identical
    prompt, non-deterministically rewrite a good summary, AND staleness-invalidate every
    persisted verdict badge on the item. A deleted or protected quote re-attaches
    identically on each re-capture, so that churn would be permanent. This is bug #44,
    and `fetch.fetch_item` / `fetch_x._attach_x_sources` both guard it (via
    `_sources_materially_equal`) with a comment describing this exact failure.

    The guard here is STRICTER than `_sources_materially_equal`: identical sources
    obviously leave the evidence identical, but so does replacing one *unreadable* record
    with a different unreadable one (`not_found` → `forbidden`, a changed `error`
    string). No LLM surface reads those fields — the `NOT fetched` marker fires either
    way — so there is nothing to re-generate. Record the failure; do not move the clock.
    """
    changed = _quoted_evidence(item) != _readable_evidence(source)
    kept = (
        [s for s in item.content.sources if s.kind not in QUOTED_CONTENT_KINDS]
        if item.content
        else []
    )
    if item.content is not None:
        fetched_at = now if changed else item.content.fetched_at
    else:
        # No content yet. Starting the clock at `now()` for a record no generator will
        # read would itself trip `_needs_reenrichment` — the churn, from a standing
        # start. Anchor at capture time instead: nothing was fetched, and the item's own
        # capture is when we learned what we know.
        fetched_at = now if changed else item.captured_at
    item.content = Content(fetched_at=fetched_at, sources=[*kept, source])
    return changed


def backfill_quoted_sources(
    store: dict[str, Item], fresh_items: list[Item], *, now: Callable[[], datetime] = _utcnow
) -> QuotedBackfillReport:
    """Attach the quoted post — parsed from a fresh capture — onto already-stored items.

    **No per-item fetch.** X embeds the quoted post in the same timeline payload as the
    tweet quoting it, so one re-capture of the history (the `refresh-media` harness,
    with no skip-known) carries every quoted body and author we need. This function is
    the pure merge: `extract.graphql` already did the parsing.

    Only `quoted_tweet` sources are touched — an article body, a transcript, a thread
    and every enrichment/description field are preserved. Idempotent: an item that
    already holds a READABLE quote is left alone (no needless re-enrichment), while a
    recorded FAILURE is upgraded if X serves the post this time.
    """
    report = QuotedBackfillReport()
    seen: set[str] = set()
    for fresh in fresh_items:
        stored = store.get(fresh.id)
        if stored is None:
            continue  # backfill only touches known ids; adding items is `extract`'s job
        seen.add(fresh.id)
        report.items_seen += 1
        incoming = _quoted_source(fresh)
        if incoming is not None:
            _merge_quoted(stored, incoming, report, now)
    report.quoted_items_not_seen = sum(
        1
        for item in store.values()
        if item.quoted_id and item.id not in seen and quoted_source(item) is None
    )
    return report


def _merge_quoted(
    stored: Item,
    incoming: ContentSource,
    report: QuotedBackfillReport,
    now: Callable[[], datetime],
) -> None:
    """Merge one freshly-captured quoted source onto its stored item, tallying `report`.

    Skipped when the item already holds a READABLE quote (there is nothing better to
    gain), and when the incoming record is byte-identical to the one we hold — a
    re-capture re-serves the same deleted post on every run, and rewriting an identical
    failure would churn the store for nothing.
    """
    if quoted_source(stored) is not None or _quoted_source(stored) == incoming:
        report.already_present += 1
        return
    _attach_quoted(stored, incoming, now())
    report.sources_attached += 1
    if isinstance(incoming, ContentSourceSuccess):
        report.readable += 1
    else:
        report.unreadable += 1


def quoted_source_from_item(quoted: Item) -> ContentSourceSuccess:
    """The `quoted_tweet` source for a quoted post we ALREADY hold as an item.

    Its body and its author come straight off the stored item — the same fields the
    GraphQL parser would produce from a fresh capture, because they came from exactly
    that parser when the quoted post was itself captured.
    """
    return ContentSourceSuccess(
        kind="quoted_tweet",
        url=quoted.url,
        text=quoted.text,
        author=quoted.author,
        attempts=1,
    )


def backfill_quoted_from_store(
    store: dict[str, Item], *, now: Callable[[], datetime] = _utcnow
) -> QuotedBackfillReport:
    """Attach the quoted post to every quote-tweet whose quoted post is ALREADY in the
    store. Pure, offline, **zero network calls**.

    A quote-tweet's `quoted_id` frequently names another item we captured in its own
    right — measured on the real store: 199 of 762 (26.1%). For those the evidence has
    been sitting in `items.json` the whole time, one dict lookup away, while the
    generator was left to invent it. This is the free half of the repair; the remaining
    quote-tweets need `refresh-quoted` (a re-capture) to reach their quoted post.

    A `quoted_id` we do NOT hold is left untouched — NOT stamped as a failure. The post
    is very likely alive; we simply never captured it, and a re-capture can still bring
    it in. Stamping it `not_found` would be inventing a fact about X, which is the class
    of bug this whole effort exists to kill.
    """
    report = QuotedBackfillReport()
    for item in store.values():
        if not item.quoted_id:
            continue
        if quoted_source(item) is not None:
            report.already_present += 1
            continue
        # `quoted_id == item.id` would make the item its own evidence — a self-quote is
        # not a thing X produces, but a corrupt record must not become a citation loop.
        quoted = store.get(item.quoted_id) if item.quoted_id != item.id else None
        if quoted is None or not quoted.text:
            report.quoted_items_not_seen += 1
            continue
        _attach_quoted(item, quoted_source_from_item(quoted), now())
        report.sources_attached += 1
        report.readable += 1
    return report

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

from dataclasses import dataclass

from xbrain.models import Item, MediaEntry, MediaVideoPending

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


def _rebuild_media(
    existing: list[MediaEntry], fresh_videos: list[MediaVideoPending]
) -> tuple[list[MediaEntry], int]:
    """Replace each video entry positionally; keep every photo entry verbatim.

    Walks ``existing`` in order. The i-th `MediaVideoPending` is replaced by the
    i-th entry of ``fresh_videos``; photo variants (Pending / Downloaded /
    Failed / Described) pass through untouched, preserving their download and
    description state. A store video with no fresh counterpart (fewer fresh
    videos than store videos) is left as-is — never dropped. Returns the rebuilt
    list and the number of replacements made.
    """
    rebuilt: list[MediaEntry] = []
    replaced = 0
    fresh_iter = iter(fresh_videos)
    for entry in existing:
        if isinstance(entry, MediaVideoPending):
            fresh = next(fresh_iter, None)
            if fresh is None:
                rebuilt.append(entry)
            else:
                rebuilt.append(fresh)
                replaced += 1
        else:
            rebuilt.append(entry)
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

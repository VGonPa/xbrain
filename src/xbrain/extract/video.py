"""Build a playable `MediaVideoPending` from an X video/animated_gif entry.

Shared by both extraction paths — the live GraphQL parser
(`extract/graphql.py`) and the X data-archive importer (`archive.py`) —
because the archive JSON carries the same
`extended_entities.media[].video_info.variants` shape as the live response.
Centralising the variant selection here keeps the two paths from drifting:
before this module, only the GraphQL path captured the playable stream while
the archive path silently stored the poster image.
"""

from __future__ import annotations

from typing import Any

from xbrain.models import MediaVideoPending


def select_variant(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the playable variant from a media entry's `video_info.variants`.

    X serves a video as several `variants`: progressive `video/mp4` files
    (one per bitrate, each a complete downloadable file) plus an
    `application/x-mpegURL` HLS manifest. Prefer the highest-bitrate mp4 (a
    single downloadable file); fall back to the HLS manifest when no mp4 is
    offered. Returns None when there are no usable variants.

    The bitrate key treats both a missing `bitrate` and an explicit
    `"bitrate": null` as 0 — X drifts, so a variant carrying `null` must not
    crash the `max` (None is not orderable against int).
    """
    variants = entry.get("video_info", {}).get("variants", [])
    mp4s = [v for v in variants if v.get("content_type") == "video/mp4" and v.get("url")]
    if mp4s:
        return max(mp4s, key=lambda v: v.get("bitrate") or 0)
    return next((v for v in variants if v.get("url")), None)


def build_video_media(entry: dict[str, Any]) -> MediaVideoPending | None:
    """Build a `MediaVideoPending` from a video/animated_gif media entry.

    Stores the playable stream URL (never the poster), keeps the poster as
    `thumbnail_url`, and records the chosen mp4's bitrate plus the clip
    duration so a later download can estimate size without fetching bytes.
    Falls back to the poster (then `expanded_url`) only when no playable
    variant exists, so a malformed entry is surfaced rather than dropped.
    Returns None when there is no usable URL at all.
    """
    variant = select_variant(entry)
    url = (variant or {}).get("url") or entry.get("media_url_https") or entry.get("expanded_url")
    if not url:
        return None
    return MediaVideoPending(
        url=url,
        thumbnail_url=entry.get("media_url_https"),
        bitrate=(variant or {}).get("bitrate"),
        duration_millis=entry.get("video_info", {}).get("duration_millis"),
    )

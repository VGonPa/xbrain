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
from urllib.parse import urlparse

from xbrain.models import MediaVideoPending, TranscriptFormat


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


_TRANSCRIPT_FORMAT_BY_SUFFIX: tuple[tuple[str, TranscriptFormat], ...] = (
    (".vtt", "vtt"),
    (".srt", "srt"),
    (".txt", "text"),
    (".json", "json"),
)


def _infer_transcript_format(url: str, content_type: str | None = None) -> TranscriptFormat:
    """Infer a small text-track format from URL/content-type metadata."""
    lowered_type = (content_type or "").lower()
    if "vtt" in lowered_type or "webvtt" in lowered_type:
        return "vtt"
    if "srt" in lowered_type or "subrip" in lowered_type:
        return "srt"
    if "json" in lowered_type:
        return "json"
    if lowered_type.startswith("text/"):
        return "text"
    path = urlparse(url).path.lower()
    for suffix, fmt in _TRANSCRIPT_FORMAT_BY_SUFFIX:
        if path.endswith(suffix):
            return fmt
    return "unknown"


def _candidate_url(node: dict[str, Any]) -> str | None:
    """Return a likely caption URL from a caption-ish metadata node."""
    for key in ("url", "transcript_url", "subtitle_url", "caption_url", "vtt_url", "srt_url"):
        value = node.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def _caption_nodes(obj: Any) -> list[dict[str, Any]]:
    """Find caption/subtitle/text-track nodes in a drift-tolerant X media payload."""
    found: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = key.lower()
            caption_key = any(
                token in lowered
                for token in ("caption", "subtitle", "subtitles", "transcript", "text_track")
            )
            if caption_key:
                if isinstance(value, dict):
                    found.append(value)
                elif isinstance(value, list):
                    found.extend(node for node in value if isinstance(node, dict))
            if isinstance(value, (dict, list)):
                found.extend(_caption_nodes(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_caption_nodes(value))
    return found


def _best_transcript_track(
    entry: dict[str, Any],
) -> tuple[str, str | None, TranscriptFormat] | None:
    """Return the best available caption/text-track URL from a media entry."""
    for node in _caption_nodes(entry):
        url = _candidate_url(node)
        if not url:
            continue
        language = (
            node.get("language")
            or node.get("language_code")
            or node.get("lang")
            or node.get("locale")
        )
        content_type = node.get("content_type") or node.get("mime_type") or node.get("type")
        return (
            url,
            str(language) if language else None,
            _infer_transcript_format(url, str(content_type) if content_type else None),
        )
    return None


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
    transcript = _best_transcript_track(entry)
    return MediaVideoPending(
        url=url,
        thumbnail_url=entry.get("media_url_https"),
        bitrate=(variant or {}).get("bitrate"),
        duration_millis=entry.get("video_info", {}).get("duration_millis"),
        transcript_url=transcript[0] if transcript else None,
        transcript_language=transcript[1] if transcript else None,
        transcript_format=transcript[2] if transcript else None,
    )

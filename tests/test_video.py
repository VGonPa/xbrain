# tests/test_video.py
"""Unit tests for the shared video-variant helper.

The selection/build behaviour is also exercised end-to-end via
`test_graphql.py` and `test_archive.py`; these pin the helper directly so a
regression localises here rather than in a caller.
"""

from xbrain.extract.video import build_video_media, select_variant
from xbrain.models import MediaVideoPending


def _entry(variants: list[dict], *, duration_millis: object = 30000) -> dict:
    """A media entry with a fixed poster and the given `video_info.variants`."""
    video_info: dict = {"variants": variants}
    if duration_millis is not None:
        video_info["duration_millis"] = duration_millis
    return {
        "type": "video",
        "media_url_https": "https://pbs.twimg.com/poster.jpg",
        "video_info": video_info,
    }


_MP4_LOW = {"bitrate": 256000, "content_type": "video/mp4", "url": "https://v/low.mp4"}
_MP4_HIGH = {"bitrate": 2176000, "content_type": "video/mp4", "url": "https://v/high.mp4"}
_HLS = {"content_type": "application/x-mpegURL", "url": "https://v/play.m3u8"}


def test_select_variant_prefers_highest_bitrate_mp4():
    assert select_variant(_entry([_MP4_LOW, _MP4_HIGH, _HLS])) is _MP4_HIGH


def test_select_variant_falls_back_to_hls_when_no_mp4():
    assert select_variant(_entry([_HLS])) is _HLS


def test_select_variant_returns_none_when_no_usable_variant():
    assert select_variant(_entry([])) is None
    assert select_variant(_entry([{"content_type": "video/mp4"}])) is None  # no url


def test_select_variant_treats_null_and_missing_bitrate_as_zero():
    missing = {"content_type": "video/mp4", "url": "https://v/missing.mp4"}
    null = {"bitrate": None, "content_type": "video/mp4", "url": "https://v/null.mp4"}
    real = {"bitrate": 832000, "content_type": "video/mp4", "url": "https://v/real.mp4"}
    # Must not raise (None is not orderable against int) and the real one wins.
    assert select_variant(_entry([missing, null, real])) is real


def test_build_video_media_captures_stream_poster_bitrate_duration():
    entry = build_video_media(_entry([_MP4_HIGH]))
    assert isinstance(entry, MediaVideoPending)
    assert entry.url == "https://v/high.mp4"
    assert entry.thumbnail_url == "https://pbs.twimg.com/poster.jpg"
    assert entry.bitrate == 2176000
    assert entry.duration_millis == 30000


def test_build_video_media_falls_back_to_poster_when_no_variant():
    entry = build_video_media(_entry([]))
    assert isinstance(entry, MediaVideoPending)
    assert entry.url == "https://pbs.twimg.com/poster.jpg"
    assert entry.thumbnail_url == "https://pbs.twimg.com/poster.jpg"
    assert entry.bitrate is None


def test_build_video_media_returns_none_when_no_url_at_all():
    assert build_video_media({"type": "video", "video_info": {"variants": []}}) is None

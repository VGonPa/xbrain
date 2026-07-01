# tests/test_models.py
from datetime import datetime, timezone

from xbrain.models import Author, Item, Link, State


def test_item_round_trips_through_json():
    item = Item(
        id="123",
        source="bookmark",
        url="https://x.com/foo/status/123",
        author=Author(handle="foo", name="Foo Bar"),
        text="hello world",
        created_at=datetime(2026, 5, 10, 14, 23, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, 9, 0, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/a", domain="example.com")],
    )
    restored = Item.model_validate_json(item.model_dump_json())
    assert restored == item
    assert restored.content is None
    assert restored.enriched is None


def test_state_defaults_are_empty_cursors():
    state = State()
    assert state.bookmarks.last_seen_id is None
    assert state.own_tweets.last_seen_id is None
    assert state.archive_imported is None


def test_enrichment_has_primary_topic_and_no_note_worthiness():
    from datetime import datetime, timezone
    from xbrain.models import Enrichment

    e = Enrichment(
        enriched_at=datetime.now(timezone.utc),
        executor="api",
        summary="resumen",
        primary_topic="ai-coding",
        topics=["ai-coding", "ai-and-work"],
    )
    assert e.primary_topic == "ai-coding"
    assert not hasattr(e, "note_worthiness")


def test_topic_model_holds_slug_and_description():
    from xbrain.models import Topic

    t = Topic(slug="ai-coding", description="Using LLMs to write software.")
    assert t.slug == "ai-coding"


def test_topic_rejects_non_kebab_case_slug():
    import pytest
    from pydantic import ValidationError

    from xbrain.models import Topic

    for bad in ["AI Coding", "ai_coding", "-ai", "ai-", "ai--coding"]:
        with pytest.raises(ValidationError):
            Topic(slug=bad, description="d")


def test_item_has_optional_bookmark_folder():
    from datetime import datetime, timezone
    from xbrain.models import Author, Item

    base = dict(
        id="1",
        source="bookmark",
        url="https://x.com/a/status/1",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    assert Item(**base).bookmark_folder is None
    assert Item(**base, bookmark_folder="AI papers").bookmark_folder == "AI papers"


def test_content_source_failure_variant_carries_failure_evidence():
    from xbrain.models import ContentSourceFailure

    src = ContentSourceFailure(
        kind="external_article",
        url="https://example.com/x",
        http_status=404,
        failure_reason="not_found",
        attempts=2,
        error="HTTP 404",
    )
    assert src.outcome == "failure"
    assert src.http_status == 404
    assert src.failure_reason == "not_found"
    assert src.attempts == 2


def test_content_source_loads_legacy_ok_true_shape():
    """A pre-#20 record with `ok=True` must read into the success variant."""
    from xbrain.models import ContentSourceAdapter, ContentSourceSuccess

    src = ContentSourceAdapter.validate_python(
        {
            "kind": "external_article",
            "url": "https://e.com/x",
            "ok": True,
            "title": "T",
            "text": "body",
            "http_status": 200,
            "failure_reason": None,
            "error": None,
            "attempts": 1,
        }
    )
    assert isinstance(src, ContentSourceSuccess)
    assert src.outcome == "success"
    assert src.text == "body"
    assert src.http_status == 200


def test_content_source_loads_legacy_ok_false_shape():
    """A pre-#20 record with `ok=False` must read into the failure variant."""
    from xbrain.models import ContentSourceAdapter, ContentSourceFailure

    src = ContentSourceAdapter.validate_python(
        {
            "kind": "external_article",
            "url": "https://e.com/x",
            "ok": False,
            "title": None,
            "text": None,
            "http_status": 404,
            "failure_reason": "not_found",
            "error": "HTTP 404",
            "attempts": 2,
        }
    )
    assert isinstance(src, ContentSourceFailure)
    assert src.outcome == "failure"
    assert src.failure_reason == "not_found"
    assert src.error == "HTTP 404"


def test_content_source_dumps_with_outcome_discriminator():
    """A success variant dumps with `outcome: "success"` and NO `ok` field."""
    from xbrain.models import ContentSourceSuccess

    src = ContentSourceSuccess(kind="external_article", url="u", text="t")
    dumped = src.model_dump(mode="json")
    assert dumped["outcome"] == "success"
    assert "ok" not in dumped


def test_content_source_failure_dumps_with_outcome_discriminator():
    """A failure variant dumps with `outcome: "failure"` and NO `ok` field."""
    from xbrain.models import ContentSourceFailure

    src = ContentSourceFailure(kind="external_article", url="u", failure_reason="not_found")
    dumped = src.model_dump(mode="json")
    assert dumped["outcome"] == "failure"
    assert "ok" not in dumped


def test_content_source_rejects_record_with_neither_discriminator():
    """Silently inventing `outcome` would mask data corruption — reject loudly."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import ContentSourceAdapter

    with pytest.raises(ValidationError):
        ContentSourceAdapter.validate_python({"kind": "external_article", "url": "u"})


def test_content_source_success_requires_text():
    """The whole point of the refactor: a success without text is a type error.

    Pydantic raises a ValidationError at construction; mypy raises an error
    statically (see tests/test_type_safety.py).
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import ContentSourceSuccess

    with pytest.raises(ValidationError):
        ContentSourceSuccess(kind="external_article", url="u")  # missing `text`


def test_content_source_failure_requires_failure_reason():
    """Symmetric: a failure without a `failure_reason` is not demonstrable evidence."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import ContentSourceFailure

    with pytest.raises(ValidationError):
        ContentSourceFailure(kind="external_article", url="u")  # missing `failure_reason`


def test_content_source_legacy_failure_without_reason_buckets_as_transient():
    """A pre-#20 record with `ok=False, failure_reason=None` (e.g. HTTP 429 that
    the old code did not categorise) migrates losslessly: the `error` text is
    preserved, and `failure_reason` is bucketed under `unknown_error` (a
    transient retry-worthy reason added in the #20 review pass) so:

    1. The wiki still renders a broken-link line.
    2. The next `fetch_pending` run auto-retries the record (issue #19
       retries `timeout`/`dns_error`/`unknown_error`), giving it one chance
       to land on a proper category rather than staying invisibly stuck.

    `unknown_error` is preferred over `timeout` for honesty — "timeout"
    would mislabel 429s / SSL handshake failures / other distinct error
    modes that the legacy code dumped without a reason.
    """
    from xbrain.models import ContentSourceAdapter, ContentSourceFailure

    src = ContentSourceAdapter.validate_python(
        {
            "kind": "external_article",
            "url": "https://e.com/throttled",
            "ok": False,
            "title": None,
            "text": None,
            "http_status": 429,
            "failure_reason": None,
            "error": "HTTP 429: Too Many Requests",
            "attempts": 1,
        }
    )
    assert isinstance(src, ContentSourceFailure)
    assert src.failure_reason == "unknown_error"
    assert src.error == "HTTP 429: Too Many Requests"
    assert src.http_status == 429


# ------------------------------------------------------------ x_video content source


def test_x_video_content_source_round_trips():
    """An `x_video` `ContentSourceSuccess` (transcript as `text`) round-trips.

    PR2 (#44) attaches a video transcript to the item as a content source with
    the new `kind="x_video"` discriminator plus the optional `has_speech` /
    `language` markers, so `generate`/`enrich` can consume it via the existing
    `ContentSource` union. A dump → re-parse must preserve every field.
    """
    from xbrain.models import ContentSourceAdapter, ContentSourceSuccess

    src = ContentSourceSuccess(
        kind="x_video",
        url="https://video.twimg.com/amplify_video/123/vid/720/A.mp4?tag=16",
        title="A great talk",
        text="hello, this is the transcript",
        has_speech=True,
        language="en",
    )
    restored = ContentSourceAdapter.validate_python(src.model_dump(mode="json"))
    assert isinstance(restored, ContentSourceSuccess)
    assert restored.kind == "x_video"
    assert restored.text == "hello, this is the transcript"
    assert restored.has_speech is True
    assert restored.language == "en"
    assert restored.title == "A great talk"


def test_x_video_no_speech_source_carries_empty_text_and_marker():
    """A silent / no-speech video is a SUCCESS source with empty text +
    `has_speech=False` — never a failure. The marker lets `generate` render a
    "silent video" line and `enrich` skip it, without inferring from empty text."""
    from xbrain.models import ContentSourceSuccess

    src = ContentSourceSuccess(kind="x_video", url="https://v/x.mp4", text="", has_speech=False)
    assert src.text == ""
    assert src.has_speech is False


def test_content_source_success_defaults_speech_fields_to_none():
    """The new fields are optional + default to None so an EXISTING (article)
    record loads unchanged — back-compat: no `has_speech`/`language` in the
    legacy shape means `None`, and `has_speech is None` distinguishes a
    non-video source from a transcribed one."""
    from xbrain.models import ContentSourceAdapter, ContentSourceSuccess

    legacy = ContentSourceAdapter.validate_python(
        {"kind": "external_article", "url": "u", "ok": True, "title": "T", "text": "body"}
    )
    assert isinstance(legacy, ContentSourceSuccess)
    assert legacy.has_speech is None
    assert legacy.language is None


def test_content_kind_includes_x_video():
    """`x_video` is a member of the `ContentKind` literal (one source of truth)."""
    from typing import get_args

    from xbrain.models import ContentKind

    assert "x_video" in get_args(ContentKind)


def test_media_legacy_photo_shape_migrates_to_pending():
    """A legacy ``{type: "photo", url}`` record reads as MediaPhotoPending.

    The on-disk shape before the tagged union was a flat ``Media(type, url)``.
    The `_normalise_legacy_media` BeforeValidator promotes it to the new
    tagged-union shape on read, so existing ``data/items.json`` files keep
    working without a manual migration step.
    """
    from xbrain.models import MediaEntryAdapter, MediaPhotoPending

    entry = MediaEntryAdapter.validate_python(
        {"type": "photo", "url": "https://pbs.twimg.com/media/X.jpg"}
    )
    assert isinstance(entry, MediaPhotoPending)
    assert entry.url == "https://pbs.twimg.com/media/X.jpg"
    assert entry.kind == "photo_pending"
    assert entry.type == "photo"


def test_media_legacy_video_shape_migrates_to_video_pending():
    """A legacy ``{type: "video", url}`` record reads as MediaVideoPending."""
    from xbrain.models import MediaEntryAdapter, MediaVideoPending

    entry = MediaEntryAdapter.validate_python(
        {"type": "video", "url": "https://video.twimg.com/x.mp4"}
    )
    assert isinstance(entry, MediaVideoPending)
    assert entry.url == "https://video.twimg.com/x.mp4"
    assert entry.kind == "video_pending"


def test_media_new_tagged_shape_passes_through():
    """A record that already carries `kind` is preserved verbatim.

    Round-trip dumps from the new variants must read back as the same
    variant — a freshly-downloaded photo on the live store must NOT be
    re-bucketed as pending.
    """
    from datetime import datetime, timezone

    from xbrain.models import MediaEntryAdapter, MediaPhotoDownloaded

    payload = {
        "kind": "photo_downloaded",
        "type": "photo",
        "url": "https://pbs.twimg.com/media/X.jpg",
        "local_path": "123/0.jpg",
        "width": 1200,
        "height": 800,
        "bytes_size": 99000,
        "downloaded_at": datetime(2026, 5, 24, tzinfo=timezone.utc).isoformat(),
    }
    entry = MediaEntryAdapter.validate_python(payload)
    assert isinstance(entry, MediaPhotoDownloaded)
    assert entry.local_path == "123/0.jpg"
    assert entry.width == 1200


def test_media_photo_described_round_trips_through_json():
    """A `MediaPhotoDescribed` payload reads back as the same variant.

    Exercises the discriminator path: the new `photo_described` kind
    must match exactly one variant. A round-trip dump must NOT collapse
    back to `MediaPhotoDownloaded` (which carries the same on-disk
    fields minus the description payload).
    """
    from datetime import datetime, timezone

    from xbrain.models import MediaEntryAdapter, MediaPhotoDescribed

    payload = {
        "kind": "photo_described",
        "type": "photo",
        "url": "https://pbs.twimg.com/media/X.jpg",
        "local_path": "123/0.jpg",
        "width": 1200,
        "height": 800,
        "bytes_size": 99000,
        "downloaded_at": datetime(2026, 5, 24, tzinfo=timezone.utc).isoformat(),
        "is_decorative": False,
        "description": "A chart showing model accuracy by parameter count.",
        "description_lang": "English",
        "description_version": "v1",
        "described_at": datetime(2026, 5, 24, 12, tzinfo=timezone.utc).isoformat(),
    }
    entry = MediaEntryAdapter.validate_python(payload)
    assert isinstance(entry, MediaPhotoDescribed)
    assert entry.is_decorative is False
    assert entry.description.startswith("A chart")
    assert entry.description_lang == "English"
    assert entry.description_version == "v1"
    # Re-dump and re-parse: variant must survive verbatim.
    restored = MediaEntryAdapter.validate_python(entry.model_dump(mode="json"))
    assert isinstance(restored, MediaPhotoDescribed)
    assert restored == entry


def test_media_photo_described_decorative_carries_empty_description():
    """Decorative entries store an empty description by contract.

    The vision rubric returns empty for decorative; refusals (faces, NSFW)
    are bucketed as decorative + empty. The model accepts this explicitly
    — there is no `gt=0` length constraint on `description`.
    """
    from datetime import datetime, timezone

    from xbrain.models import MediaPhotoDescribed

    entry = MediaPhotoDescribed(
        url="https://pbs.twimg.com/media/X.jpg",
        local_path="123/0.jpg",
        width=400,
        height=400,
        bytes_size=12000,
        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        is_decorative=True,
        description="",
        description_lang="English",
        description_version="v1",
        described_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
    )
    assert entry.is_decorative is True
    assert entry.description == ""


def test_media_photo_described_rejects_absolute_local_path():
    """Path-traversal defence carries over from the downloaded variant —
    the bytes referenced by `local_path` are inherited from the prior
    state, so the same hardening applies.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDescribed

    with pytest.raises(ValidationError):
        MediaPhotoDescribed(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="/etc/passwd",
            width=1,
            height=1,
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            is_decorative=False,
            description="x",
            description_lang="English",
            description_version="v1",
            described_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_photo_described_rejects_parent_traversal_local_path():
    """Same defence as the downloaded variant — `..` in `local_path` is rejected."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDescribed

    with pytest.raises(ValidationError):
        MediaPhotoDescribed(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="../escape/x.jpg",
            width=1,
            height=1,
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            is_decorative=False,
            description="x",
            description_lang="English",
            description_version="v1",
            described_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_photo_described_rejects_naive_described_at():
    """`described_at` must be timezone-aware (UTC); a naive datetime is rejected.

    The UTC-aware invariant is enforced by a field validator so a
    hand-edited `items.json` entry cannot smuggle a local-time datetime
    past the type boundary. Naive timestamps cause downstream UTC math
    (eligibility checks, sort orders) to drift silently.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDescribed

    with pytest.raises(ValidationError):
        MediaPhotoDescribed(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="123/0.jpg",
            width=10,
            height=10,
            bytes_size=100,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            is_decorative=False,
            description="hello",
            description_lang="English",
            description_version="v1",
            described_at=datetime(2026, 5, 24),  # naive — must fail
        )


def test_media_photo_described_rejects_unsupported_description_lang():
    """`description_lang` must be in `SUPPORTED_LANGUAGES`; others are rejected.

    The type alias is derived from `i18n.SUPPORTED_LANGUAGES` so the
    `Literal[...]` validator rejects unknown languages at construction.
    Prevents an out-of-band language tag from polluting the vault.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDescribed

    with pytest.raises(ValidationError):
        MediaPhotoDescribed(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="123/0.jpg",
            width=10,
            height=10,
            bytes_size=100,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            is_decorative=False,
            description="hello",
            description_lang="Klingon",  # not in SUPPORTED_LANGUAGES
            description_version="v1",
            described_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_photo_described_rejects_decorative_with_nonempty_description():
    """`is_decorative=True` implies `description == ""` — model-validator enforces.

    Defence-in-depth for hand-edited records: the producer
    (`describe._apply_judgment`) already blanks the description on
    decorative judgments, but a hand-written entry that violates the
    invariant must still be rejected at the type boundary so downstream
    callers can rely on `is_decorative => not description` unconditionally.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDescribed

    with pytest.raises(ValidationError):
        MediaPhotoDescribed(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="123/0.jpg",
            width=10,
            height=10,
            bytes_size=100,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            is_decorative=True,
            description="should be empty when decorative",  # violates invariant
            description_lang="English",
            description_version="v1",
            described_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_discriminator_rejects_unknown_kind():
    """Silently inventing a variant would mask data corruption — reject loudly."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaEntryAdapter

    with pytest.raises(ValidationError):
        MediaEntryAdapter.validate_python(
            {"kind": "photo_in_orbit", "type": "photo", "url": "https://pbs.twimg.com/X.jpg"}
        )


def test_media_photo_downloaded_rejects_absolute_local_path():
    """An absolute `local_path` would let a poisoned items.json exfiltrate
    bytes outside `data/media/`. The validator must reject it at construction.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDownloaded

    with pytest.raises(ValidationError):
        MediaPhotoDownloaded(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="/etc/passwd",
            width=1,
            height=1,
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_photo_downloaded_rejects_parent_traversal_local_path():
    """A `local_path` containing `..` would let a poisoned items.json escape
    the media root via path concatenation. The validator must reject it.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDownloaded

    with pytest.raises(ValidationError):
        MediaPhotoDownloaded(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="../x/y.jpg",
            width=1,
            height=1,
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_photo_downloaded_rejects_zero_bytes_size():
    """A zero-byte download is nonsense — the downloader records bytes_size
    after writing, so a value of 0 either means a failed write was persisted
    as success or the record was hand-edited. Reject at the boundary.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDownloaded

    with pytest.raises(ValidationError):
        MediaPhotoDownloaded(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="123/0.jpg",
            width=1,
            height=1,
            bytes_size=0,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_photo_downloaded_rejects_zero_width():
    """A zero-width image cannot have been decoded — reject as data corruption."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDownloaded

    with pytest.raises(ValidationError):
        MediaPhotoDownloaded(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="123/0.jpg",
            width=0,
            height=1,
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_photo_downloaded_rejects_zero_height():
    """A zero-height image cannot have been decoded — reject as data corruption."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoDownloaded

    with pytest.raises(ValidationError):
        MediaPhotoDownloaded(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="123/0.jpg",
            width=1,
            height=0,
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_photo_failed_rejects_zero_attempts():
    """A `MediaPhotoFailed` with `attempts=0` is semantically nonsense — the
    downloader increments `attempts` before constructing any failed record,
    so 0 cannot occur naturally. The validator must reject it.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoFailed

    with pytest.raises(ValidationError):
        MediaPhotoFailed(
            url="https://pbs.twimg.com/media/X.jpg",
            failure_reason="not_found",
            attempts=0,
            last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_failed_requires_failure_reason():
    """Symmetric with ContentSourceFailure: a failure without a reason is a
    type error. The validator constructs the variant via pydantic, which
    flags the missing required field.
    """
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaPhotoFailed

    with pytest.raises(ValidationError):
        MediaPhotoFailed(url="u")  # missing failure_reason and last_attempt_at


def test_media_factory_returns_pending_variant_for_photo():
    """The backward-compat `Media(type=..., url=...)` factory must yield a
    `MediaPhotoPending` so the extractor and archive callsites keep
    building items the tagged union accepts.
    """
    from xbrain.models import Media, MediaPhotoPending

    entry = Media(type="photo", url="https://pbs.twimg.com/media/X.jpg")
    assert isinstance(entry, MediaPhotoPending)
    assert entry.kind == "photo_pending"


def test_media_factory_returns_video_pending_for_video():
    """The factory routes `type="video"` to `MediaVideoPending` — video
    bytes are not downloaded yet.
    """
    from xbrain.models import Media, MediaVideoPending

    entry = Media(type="video", url="https://video.twimg.com/x.mp4")
    assert isinstance(entry, MediaVideoPending)
    assert entry.kind == "video_pending"


def test_item_with_legacy_media_loads_into_pending_variant():
    """A persisted Item with the legacy media shape migrates on read.

    Exercises the full path: the Item validator hits MediaEntry's
    BeforeValidator on each element of the `media` list. This is the
    invariant the wire-shape compatibility rests on.
    """
    from datetime import datetime, timezone

    from xbrain.models import Item, MediaPhotoPending, MediaVideoPending

    payload = {
        "id": "1",
        "source": "bookmark",
        "url": "https://x.com/a/status/1",
        "author": {"handle": "a", "name": "A"},
        "text": "t",
        "created_at": datetime(2026, 5, 1, tzinfo=timezone.utc).isoformat(),
        "captured_at": datetime(2026, 5, 16, tzinfo=timezone.utc).isoformat(),
        "media": [
            {"type": "photo", "url": "https://pbs.twimg.com/media/X.jpg"},
            {"type": "video", "url": "https://video.twimg.com/x.mp4"},
        ],
        "links": [],
    }
    item = Item.model_validate(payload)
    assert len(item.media) == 2
    assert isinstance(item.media[0], MediaPhotoPending)
    assert isinstance(item.media[1], MediaVideoPending)


def test_topic_page_model_round_trips():
    from datetime import datetime, timezone

    from xbrain.models import TopicPage

    page = TopicPage(
        slug="ai-coding",
        overview="Resumen del tema.",
        notes=["Nota uno.", "Nota dos."],
        synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        post_count_at_synth=42,
    )
    restored = TopicPage.model_validate(page.model_dump(mode="json"))
    assert restored.slug == "ai-coding"
    assert restored.post_count_at_synth == 42
    assert restored.notes == ["Nota uno.", "Nota dos."]


def test_media_video_pending_carries_thumbnail_and_size_metadata():
    """MediaVideoPending can carry the poster thumbnail plus the bitrate and
    duration used to estimate download size — without fetching anything."""
    from xbrain.models import MediaVideoPending

    entry = MediaVideoPending(
        url="https://video.twimg.com/high.mp4?tag=12",
        thumbnail_url="https://pbs.twimg.com/poster.jpg",
        bitrate=2176000,
        duration_millis=30000,
    )

    assert entry.thumbnail_url == "https://pbs.twimg.com/poster.jpg"
    assert entry.bitrate == 2176000
    assert entry.duration_millis == 30000


def test_media_video_pending_metadata_optional_for_legacy_records():
    """A bare video_pending (url only) still loads — the new fields default to
    None so existing items.json records need no migration."""
    from xbrain.models import MediaEntryAdapter, MediaVideoPending

    entry = MediaEntryAdapter.validate_python(
        {"kind": "video_pending", "type": "video", "url": "https://video.twimg.com/x.mp4"}
    )

    assert isinstance(entry, MediaVideoPending)
    assert entry.thumbnail_url is None
    assert entry.bitrate is None
    assert entry.duration_millis is None


# ------------------------------------------------ video download/failed variants


def test_media_video_downloaded_round_trips_through_json():
    """A `MediaVideoDownloaded` payload reads back as the same variant.

    Exercises the discriminator: the new `video_downloaded` kind must match
    exactly one variant and survive a dump → re-parse round-trip verbatim
    (the live store re-dumps every record on each save).
    """
    from datetime import datetime, timezone

    from xbrain.models import MediaEntryAdapter, MediaVideoDownloaded

    payload = {
        "kind": "video_downloaded",
        "type": "video",
        "url": "https://video.twimg.com/ext_tw_video/1/vid/1280x720/A.mp4?tag=12",
        "thumbnail_url": "https://pbs.twimg.com/poster.jpg",
        "bitrate": 2176000,
        "duration_millis": 30000,
        "local_path": "123/0.mp4",
        "bytes_size": 8160000,
        "downloaded_at": datetime(2026, 5, 24, tzinfo=timezone.utc).isoformat(),
    }
    entry = MediaEntryAdapter.validate_python(payload)
    assert isinstance(entry, MediaVideoDownloaded)
    assert entry.local_path == "123/0.mp4"
    assert entry.bytes_size == 8160000
    assert entry.bitrate == 2176000
    assert entry.thumbnail_url == "https://pbs.twimg.com/poster.jpg"
    restored = MediaEntryAdapter.validate_python(entry.model_dump(mode="json"))
    assert isinstance(restored, MediaVideoDownloaded)
    assert restored == entry


def test_media_video_downloaded_carried_fields_optional():
    """thumbnail_url / bitrate / duration_millis default to None so an mp4
    downloaded from a legacy (thumbnail-less) pending entry still validates."""
    from datetime import datetime, timezone

    from xbrain.models import MediaVideoDownloaded

    entry = MediaVideoDownloaded(
        url="https://video.twimg.com/x.mp4",
        local_path="9/0.mp4",
        bytes_size=12,
        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )
    assert entry.thumbnail_url is None
    assert entry.bitrate is None
    assert entry.duration_millis is None
    assert entry.type == "video"


def test_media_video_downloaded_rejects_absolute_local_path():
    """An absolute `local_path` would let a poisoned items.json write bytes
    outside `data/media/` — reuse of the shared no-traversal validator."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaVideoDownloaded

    with pytest.raises(ValidationError):
        MediaVideoDownloaded(
            url="https://video.twimg.com/x.mp4",
            local_path="/etc/passwd",
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_video_downloaded_rejects_parent_traversal_local_path():
    """A `local_path` containing `..` must be rejected at construction."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaVideoDownloaded

    with pytest.raises(ValidationError):
        MediaVideoDownloaded(
            url="https://video.twimg.com/x.mp4",
            local_path="../x/y.mp4",
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_video_downloaded_rejects_zero_bytes_size():
    """A zero-byte download is nonsense — bytes_size is gt=0."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaVideoDownloaded

    with pytest.raises(ValidationError):
        MediaVideoDownloaded(
            url="https://video.twimg.com/x.mp4",
            local_path="123/0.mp4",
            bytes_size=0,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_video_downloaded_rejects_naive_downloaded_at():
    """A naive timestamp would miscompare against now(timezone.utc) — reject."""
    import pytest
    from datetime import datetime
    from pydantic import ValidationError

    from xbrain.models import MediaVideoDownloaded

    with pytest.raises(ValidationError):
        MediaVideoDownloaded(
            url="https://video.twimg.com/x.mp4",
            local_path="123/0.mp4",
            bytes_size=1,
            downloaded_at=datetime(2026, 5, 24),  # naive
        )


def test_media_video_failed_round_trips_and_carries_fields():
    """A `MediaVideoFailed` round-trips and carries the source url + metadata
    so a retry has everything it needs without re-capturing."""
    from datetime import datetime, timezone

    from xbrain.models import MediaEntryAdapter, MediaVideoFailed

    payload = {
        "kind": "video_failed",
        "type": "video",
        "url": "https://video.twimg.com/x.mp4",
        "thumbnail_url": "https://pbs.twimg.com/poster.jpg",
        "bitrate": 1000000,
        "duration_millis": 5000,
        "failure_reason": "http_5xx",
        "error": "HTTP 503",
        "attempts": 2,
        "last_attempt_at": datetime(2026, 5, 24, tzinfo=timezone.utc).isoformat(),
    }
    entry = MediaEntryAdapter.validate_python(payload)
    assert isinstance(entry, MediaVideoFailed)
    assert entry.failure_reason == "http_5xx"
    assert entry.attempts == 2
    assert entry.thumbnail_url == "https://pbs.twimg.com/poster.jpg"
    restored = MediaEntryAdapter.validate_python(entry.model_dump(mode="json"))
    assert isinstance(restored, MediaVideoFailed)
    assert restored == entry


def test_media_video_failed_rejects_zero_attempts():
    """`attempts=0` is nonsense — the downloader increments before failing."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaVideoFailed

    with pytest.raises(ValidationError):
        MediaVideoFailed(
            url="https://video.twimg.com/x.mp4",
            failure_reason="http_4xx",
            attempts=0,
            last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )


def test_media_video_failed_requires_failure_reason():
    """A failure without a reason is a type error — symmetric with photos."""
    import pytest
    from pydantic import ValidationError

    from xbrain.models import MediaVideoFailed

    with pytest.raises(ValidationError):
        MediaVideoFailed(url="https://video.twimg.com/x.mp4")  # missing required fields


def test_media_video_failed_rejects_naive_last_attempt_at():
    """A naive last_attempt_at must be rejected (utc-aware contract)."""
    import pytest
    from datetime import datetime
    from pydantic import ValidationError

    from xbrain.models import MediaVideoFailed

    with pytest.raises(ValidationError):
        MediaVideoFailed(
            url="https://video.twimg.com/x.mp4",
            failure_reason="timeout",
            attempts=1,
            last_attempt_at=datetime(2026, 5, 24),  # naive
        )

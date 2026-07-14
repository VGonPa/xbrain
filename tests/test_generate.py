# tests/test_generate.py
from datetime import datetime, timezone
from pathlib import Path

from xbrain.generate import generate
from xbrain.models import (
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
)
from xbrain.notes_io import slugify


def _item(item_id: str, with_link: bool, text: str | None = None) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text if text is not None else f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/p", domain="example.com")] if with_link else [],
    )


def test_generate_creates_index_log_and_only_link_notes(tmp_path: Path):
    store = {"1": _item("1", with_link=True), "2": _item("2", with_link=False)}
    generate(store, tmp_path)
    assert (tmp_path / "_index.md").exists()
    assert (tmp_path / "log.md").exists()
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 1


def test_regeneration_preserves_user_content_after_marker(tmp_path: Path):
    store = {"1": _item("1", with_link=True)}
    generate(store, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    note.write_text(note.read_text(encoding="utf-8") + "MI ANOTACION", encoding="utf-8")
    generate(store, tmp_path)
    assert "MI ANOTACION" in note.read_text(encoding="utf-8")


def test_log_lists_every_item(tmp_path: Path):
    store = {"1": _item("1", with_link=True), "2": _item("2", with_link=False)}
    generate(store, tmp_path)
    log = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "Note 1" in log
    assert "Note 2" in log


def test_generate_renders_downloaded_photo_as_obsidian_embed(tmp_path: Path):
    """A `MediaPhotoDownloaded` becomes a `![[_media/<id>/<n>.<ext>]]` embed.

    The bytes are copied from `media_root` into the vault's `_media/`
    subdirectory at render time, so the resulting vault is self-contained
    and the embed resolves without user configuration.
    """
    from xbrain.models import MediaPhotoDownloaded

    media_root = tmp_path / "media"
    photo_dir = media_root / "1"
    photo_dir.mkdir(parents=True)
    (photo_dir / "0.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    item = _item("1", with_link=True)
    item.media = [
        MediaPhotoDownloaded(
            url="https://pbs.twimg.com/media/A.png",
            local_path="1/0.png",
            width=10,
            height=8,
            bytes_size=12,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )
    ]

    output_dir = tmp_path / "vault"
    generate({"1": item}, output_dir, media_root=media_root)
    note = next((output_dir / "items").glob("*.md"))
    body = note.read_text(encoding="utf-8")
    assert "![[_media/1/0.png]]" in body
    # The file got mirrored into the vault.
    assert (output_dir / "_media" / "1" / "0.png").exists()


def test_generate_renders_failed_photo_as_warning(tmp_path: Path):
    """A `MediaPhotoFailed` becomes a one-line ⚠ warning carrying URL + reason."""
    from xbrain.models import MediaPhotoFailed

    item = _item("1", with_link=True)
    item.media = [
        MediaPhotoFailed(
            url="https://pbs.twimg.com/media/dead.png",
            failure_reason="http_4xx",
            attempts=1,
            last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )
    ]

    generate({"1": item}, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    body = note.read_text(encoding="utf-8")
    assert "⚠" in body
    assert "https://pbs.twimg.com/media/dead.png" in body
    # The `http_4xx` reason maps to "URL no encontrada (HTTP 4xx)" via
    # `_FAILURE_ES_MEDIA`. The whole rendered phrase must be present —
    # asserting one substring or the other is brittle.
    assert "URL no encontrada (HTTP 4xx)" in body


def test_generate_skips_pending_photo_silently(tmp_path: Path):
    """A `MediaPhotoPending` produces NO output — `xbrain media` is the seam.

    A pending photo is not an error; it just means the next `xbrain media`
    run will pick it up. Surfacing it in the note as "still pending" would
    be noise. The note is therefore identical to one without any media.
    """
    from xbrain.models import MediaPhotoPending

    item = _item("1", with_link=True)
    item.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/pending.png")]

    generate({"1": item}, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    body = note.read_text(encoding="utf-8")
    assert "pending.png" not in body
    assert "⚠" not in body
    assert "🎥" not in body


def test_generate_renders_video_pending_as_clickable_play_link(tmp_path: Path):
    """The video URL is now the playable mp4 (not the poster), so render it as a
    labelled clickable link, flagged as pending local download."""
    from xbrain.models import MediaVideoPending

    item = _item("1", with_link=True)
    item.media = [MediaVideoPending(url="https://video.twimg.com/high.mp4?tag=12")]

    generate({"1": item}, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    body = note.read_text(encoding="utf-8")
    assert "[Ver vídeo](https://video.twimg.com/high.mp4?tag=12)" in body
    assert "pendiente de descarga" in body


def test_generate_renders_downloaded_video_as_obsidian_embed(tmp_path: Path):
    """A `MediaVideoDownloaded` becomes a `![[_media/<id>/<n>.mp4]]` embed and the
    mp4 bytes are mirrored into the vault's `_media/` tree (self-contained vault),
    exactly like a downloaded photo."""
    from xbrain.models import MediaVideoDownloaded

    media_root = tmp_path / "media"
    video_dir = media_root / "1"
    video_dir.mkdir(parents=True)
    (video_dir / "0.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake")

    item = _item("1", with_link=True)
    item.media = [
        MediaVideoDownloaded(
            url="https://video.twimg.com/x.mp4",
            thumbnail_url="https://pbs.twimg.com/poster.jpg",
            bitrate=2176000,
            duration_millis=30000,
            local_path="1/0.mp4",
            bytes_size=20,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )
    ]

    output_dir = tmp_path / "vault"
    generate({"1": item}, output_dir, media_root=media_root)
    body = next((output_dir / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "![[_media/1/0.mp4]]" in body
    assert (output_dir / "_media" / "1" / "0.mp4").exists()


def test_generate_renders_failed_video_as_warning(tmp_path: Path):
    """A `MediaVideoFailed` becomes a one-line ⚠ warning carrying URL + reason."""
    from xbrain.models import MediaVideoFailed

    item = _item("1", with_link=True)
    item.media = [
        MediaVideoFailed(
            url="https://video.twimg.com/dead.mp4",
            failure_reason="http_5xx",
            attempts=1,
            last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )
    ]

    generate({"1": item}, tmp_path)
    body = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "⚠" in body
    assert "https://video.twimg.com/dead.mp4" in body
    assert "error del servidor (HTTP 5xx)" in body


def test_generate_media_only_item_gets_a_note(tmp_path: Path):
    """An item with only media (no link, no enrichment) is note-worthy.

    Previously `_has_note(item)` only returned True for items with links or
    enrichment, so a photo-only tweet was invisible. This test pins the
    current behaviour: a photo-only item gets its note rendered.
    """
    from xbrain.models import MediaPhotoDownloaded

    media_root = tmp_path / "media"
    (media_root / "1").mkdir(parents=True)
    (media_root / "1" / "0.png").write_bytes(b"PNGfake")

    item = _item("1", with_link=False)
    item.media = [
        MediaPhotoDownloaded(
            url="https://pbs.twimg.com/media/A.png",
            local_path="1/0.png",
            width=4,
            height=3,
            bytes_size=7,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )
    ]
    generate({"1": item}, tmp_path, media_root=media_root)
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 1


def test_generate_media_block_precedes_links_section(tmp_path: Path):
    """Photos render in the Tweet section, ahead of `## Enlaces`.

    The intended read order: tweet text → photos → links → external content.
    """
    from xbrain.models import MediaPhotoDownloaded

    media_root = tmp_path / "media"
    (media_root / "1").mkdir(parents=True)
    (media_root / "1" / "0.png").write_bytes(b"PNGfake")
    item = _item("1", with_link=True)
    item.media = [
        MediaPhotoDownloaded(
            url="u",
            local_path="1/0.png",
            width=4,
            height=3,
            bytes_size=7,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )
    ]

    generate({"1": item}, tmp_path, media_root=media_root)
    body = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    embed_idx = body.find("![[_media/1/0.png]]")
    enlaces_idx = body.find("## Enlaces")
    assert embed_idx != -1
    assert enlaces_idx != -1
    assert embed_idx < enlaces_idx


def test_generate_renders_multiple_photos_inline(tmp_path: Path):
    """All photos render — no cap (per spec decision: "ALL photos inline")."""
    from xbrain.models import MediaPhotoDownloaded

    media_root = tmp_path / "media"
    (media_root / "1").mkdir(parents=True)
    for n in range(4):
        (media_root / "1" / f"{n}.png").write_bytes(b"PNGfake")

    item = _item("1", with_link=True)
    item.media = [
        MediaPhotoDownloaded(
            url=f"u{n}",
            local_path=f"1/{n}.png",
            width=4,
            height=3,
            bytes_size=7,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )
        for n in range(4)
    ]

    generate({"1": item}, tmp_path, media_root=media_root)
    body = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    for n in range(4):
        assert f"![[_media/1/{n}.png]]" in body


def test_generate_handles_missing_media_bytes_gracefully(tmp_path: Path):
    """A `MediaPhotoDownloaded` whose file vanished still renders the embed.

    Missing bytes on disk is a manual-cleanup edge case. We log a warning
    and still emit the embed — Obsidian shows it as a broken image, which
    is the right user signal that the bytes are gone.
    """
    from xbrain.models import MediaPhotoDownloaded

    media_root = tmp_path / "media"
    # No file on disk.
    item = _item("1", with_link=True)
    item.media = [
        MediaPhotoDownloaded(
            url="u",
            local_path="1/0.png",
            width=4,
            height=3,
            bytes_size=7,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )
    ]

    # The generator must not raise — missing bytes is recoverable.
    generate({"1": item}, tmp_path, media_root=media_root)
    body = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "![[_media/1/0.png]]" in body


def test_generate_works_without_media_root_argument(tmp_path: Path):
    """`media_root=None` is the legacy code path — pending photos stay silent.

    Backward compat: callers that haven't been updated still work. Failed
    and video-pending entries still render (URL is in the data), only the
    mirror-to-vault step is skipped.
    """
    from xbrain.models import MediaPhotoFailed, MediaPhotoPending

    item = _item("1", with_link=True)
    item.media = [
        MediaPhotoPending(url="https://pbs.twimg.com/media/A.png"),
        MediaPhotoFailed(
            url="https://pbs.twimg.com/media/B.png",
            failure_reason="http_4xx",
            attempts=1,
            last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        ),
    ]
    generate({"1": item}, tmp_path)  # no media_root
    body = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    # Pending: silent
    assert "A.png" not in body
    # Failed: still rendered (URL is in the data, no file needed)
    assert "B.png" in body


def test_slugify_handles_edge_cases():
    slug = slugify("Café del Día")
    assert slug == slug.lower()
    assert slug.isascii()
    assert slug == "cafe-del-dia"
    assert slugify("") == "item"
    assert slugify("!!!") == "item"
    long_slug = slugify("a" * 200)
    assert len(long_slug) <= 60


def test_regeneration_replaces_generated_block(tmp_path: Path):
    generate({"1": _item("1", with_link=True, text="Original text")}, tmp_path)
    generate({"1": _item("1", with_link=True, text="Updated text")}, tmp_path)
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 1  # the stale-title note is migrated, not orphaned
    content = notes[0].read_text(encoding="utf-8")
    assert "Updated text" in content
    assert "Original text" not in content


def test_missing_end_marker_preserves_file(tmp_path: Path):
    store = {"1": _item("1", with_link=True)}
    generate(store, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    note.write_text("contenido del usuario sin marcadores", encoding="utf-8")
    generate(store, tmp_path)
    assert "contenido del usuario sin marcadores" in note.read_text(encoding="utf-8")


def test_note_filenames_do_not_collide(tmp_path: Path):
    store = {
        "1001": _item("1001", with_link=True, text="Mismo titulo"),
        "2002": _item("2002", with_link=True, text="Mismo titulo"),
    }
    generate(store, tmp_path)
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 2


def test_note_filenames_unique_for_similar_ids(tmp_path: Path):
    # Same date, identical slug, ids sharing their last 6 characters.
    # Keying filenames on the full id keeps both notes distinct.
    store = {
        "1000001": _item("1000001", with_link=True, text="Mismo titulo"),
        "2000001": _item("2000001", with_link=True, text="Mismo titulo"),
    }
    generate(store, tmp_path)
    notes = list((tmp_path / "items").glob("*.md"))
    assert len({note.name for note in notes}) == 2


def test_note_has_frontmatter(tmp_path: Path):
    generate({"1": _item("1", with_link=True)}, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    content = note.read_text(encoding="utf-8")
    assert "id:" in content
    assert "source:" in content
    assert "tags: [x-knowledge" in content


def test_frontmatter_includes_topics_and_folder_as_tags(tmp_path):
    from datetime import datetime, timezone
    from xbrain.generate import generate
    from xbrain.models import Author, Enrichment, Item, Link

    item = Item(
        id="1",
        source="bookmark",
        url="https://x.com/a/status/1",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://arxiv.org/abs/1", domain="arxiv.org")],
        bookmark_folder="AI papers",
        enriched=Enrichment(
            enriched_at=datetime.now(timezone.utc),
            executor="api",
            summary="s",
            primary_topic="ai-coding",
            topics=["ai-coding", "ai-and-work"],
        ),
    )
    generate({"1": item}, tmp_path)
    note = next((tmp_path / "items").glob("*-1.md")).read_text(encoding="utf-8")
    assert "ai-coding" in note and "ai-and-work" in note
    assert "ai-papers" in note  # folder, slugified, as a tag
    assert "bookmark_folder: AI papers" in note


def test_generate_since_until_filters_item_notes(tmp_path: Path):
    old_item = _item("1", with_link=True, text="Old note")
    old_item.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new_item = _item("2", with_link=True, text="New note")
    new_item.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = {"1": old_item, "2": new_item}
    generate(store, tmp_path, since=datetime(2023, 1, 1, tzinfo=timezone.utc))
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 1
    assert "New note" in notes[0].read_text(encoding="utf-8")
    log = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "Old note" in log
    assert "New note" in log


def test_failure_es_covers_every_failure_reason():
    from typing import get_args

    from xbrain.generate import _FAILURE_ES
    from xbrain.models import FailureReason

    assert set(_FAILURE_ES) == set(get_args(FailureReason))


def test_note_renders_broken_link_evidence(tmp_path):
    from datetime import datetime, timezone

    from xbrain.generate import generate

    item = Item(
        id="1",
        source="bookmark",
        url="https://x.com/a/status/1",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://dead.example.com/p", domain="dead.example.com")],
        content=Content(
            fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            sources=[
                ContentSourceFailure(
                    kind="external_article",
                    url="https://dead.example.com/p",
                    http_status=404,
                    failure_reason="not_found",
                )
            ],
        ),
    )
    generate({"1": item}, tmp_path)
    note = next((tmp_path / "items").glob("*-1.md")).read_text(encoding="utf-8")
    assert "Enlace roto" in note
    assert "HTTP 404" in note
    assert "2026-05-17" in note


# --------------------------------------------------------------------- i18n


def _enriched_item(item_id: str = "9") -> Item:
    """An item with enrichment + a fetched article — exercises every header."""
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Body {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/p", domain="example.com")],
        enriched=Enrichment(
            enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            executor="api",
            summary="Resumen del item.",
            primary_topic="ai-coding",
            topics=["ai-coding", "software"],
        ),
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.com/p",
                    title="Article title",
                    text="Article body",
                )
            ],
        ),
    )


def test_generate_english_headers_by_default(tmp_path: Path):
    """Default language English: headers and labels read in English."""
    generate({"9": _enriched_item()}, tmp_path)
    index = (tmp_path / "_index.md").read_text(encoding="utf-8")
    assert "## Summary" in index
    assert "## Topics" in index
    assert "## Resumen" not in index
    assert "## Temas" not in index

    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "**Topics:**" in note
    assert "## Content: Article title" in note
    assert "**Temas:**" not in note
    assert "## Contenido:" not in note


def test_generate_spanish_headers_when_requested(tmp_path: Path):
    """Explicit Spanish: every code-generated header reads in Spanish."""
    generate({"9": _enriched_item()}, tmp_path, output_language="Spanish")
    index = (tmp_path / "_index.md").read_text(encoding="utf-8")
    assert "## Resumen" in index
    assert "## Temas" in index
    assert "## Summary" not in index

    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "**Temas:**" in note
    assert "## Contenido: Article title" in note
    assert "**Topics:**" not in note


def test_generate_rejects_unsupported_language(tmp_path: Path):
    """A bogus language must surface as ValueError, not silent default."""
    import pytest

    with pytest.raises(ValueError, match="Klingon"):
        generate({"9": _enriched_item()}, tmp_path, output_language="Klingon")


# ----------------------------------------------------------- x_video digest (#44)


def _video_item(
    item_id: str = "7",
    *,
    text: str,
    has_speech: bool = True,
    title: str | None = "The Talk",
    frames: list | None = None,
    digest: str = "",
) -> Item:
    """An enriched video bookmark carrying an `x_video` transcript source (#44)."""
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text="watch this talk",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        enriched=Enrichment(
            enriched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            executor="api",
            summary="A crisp summary of the talk.",
            primary_topic="ai-coding",
            topics=["ai-coding"],
        ),
        content=Content(
            fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/v",
                    title=title,
                    text=text,
                    has_speech=has_speech,
                    frames=frames or [],
                    digest=digest,
                )
            ],
        ),
    )


def test_generate_renders_video_digest_section(tmp_path: Path):
    """A with-speech `x_video` source renders a `## Video digest` section carrying
    the title + the transcript-derived text — the #44 payoff."""
    generate({"7": _video_item(text="The transcript body of the talk.")}, tmp_path)
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "## Video digest: The Talk" in note
    assert "The transcript body of the talk." in note
    # NOT rendered under the generic article `## Content:` heading
    assert "## Content: The Talk" not in note


def test_generate_renders_silent_video_line_for_no_speech(tmp_path: Path):
    """A no-speech `x_video` source renders a one-line silent-video note, not an
    empty `## Video digest` block."""
    generate({"7": _video_item(text="", has_speech=False)}, tmp_path)
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "silent video" in note.lower() or "sin voz" in note.lower()
    assert "## Video digest" not in note


def test_generate_video_digest_is_stable_on_regen(tmp_path: Path):
    """Regenerating a video note is deterministic and preserves the user tail."""
    store = {"7": _video_item(text="Stable transcript body.")}
    generate(store, tmp_path)
    note_path = next((tmp_path / "items").glob("*.md"))
    note_path.write_text(note_path.read_text(encoding="utf-8") + "MI NOTA", encoding="utf-8")
    first = note_path.read_text(encoding="utf-8")
    generate(store, tmp_path)
    second = note_path.read_text(encoding="utf-8")
    assert first == second
    assert "MI NOTA" in second


def test_generate_video_digest_spanish_header(tmp_path: Path):
    """The digest heading is localised alongside the other wiki headers."""
    generate(
        {"7": _video_item(text="Cuerpo de la transcripción.")}, tmp_path, output_language="Spanish"
    )
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "## Video digest:" not in note
    assert "## Resumen del vídeo: The Talk" in note  # the localised heading IS rendered
    assert "Cuerpo de la transcripción." in note


def test_generate_renders_video_digest_prose_when_present(tmp_path: Path):
    """When the `x_video` source carries a long-form `digest`, it renders as the
    body of the `## Video digest` section — the readable headline of the video."""
    generate(
        {"7": _video_item(text="raw transcript body", digest="This talk explains scaling laws.")},
        tmp_path,
    )
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "## Video digest: The Talk" in note
    assert "This talk explains scaling laws." in note


def test_generate_video_digest_demotes_raw_to_details_when_digest_present(tmp_path: Path):
    """With a digest, the raw transcript + frame embeds are demoted into a collapsible
    `<details>` below the digest prose — evidence kept, noise hidden. The digest prose
    sits ABOVE the `<details>`, the transcript + embeds INSIDE it."""
    generate(
        {
            "7": _video_item(
                text="raw transcript body", frames=_frames(), digest="A concise digest."
            )
        },
        tmp_path,
    )
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "<details>" in note and "</details>" in note
    digest_idx = note.index("A concise digest.")
    details_idx = note.index("<details>")
    assert digest_idx < details_idx  # digest prose is the headline, above the raw dump
    inside = note[details_idx : note.index("</details>")]
    assert "raw transcript body" in inside
    assert "![[_media/7/frames/0.png]]" in inside


def test_generate_video_digest_no_details_when_digest_absent(tmp_path: Path):
    """Fallback: with no digest, the section is unchanged — transcript rendered inline,
    no `<details>` wrapper. Safe to ship the render change before any digest exists."""
    generate({"7": _video_item(text="inline transcript body")}, tmp_path)
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "inline transcript body" in note
    assert "<details>" not in note


def test_generate_video_digest_evidence_label_is_localised(tmp_path: Path):
    """The `<details>` summary label follows the configured output language."""
    generate(
        {"7": _video_item(text="cuerpo", digest="Un resumen conciso.")},
        tmp_path,
        output_language="Spanish",
    )
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "transcripción" in note.lower()  # the ES evidence label


# ----------------------------------------------------------- x_video slide frames (#44 PR4)


def _frames():
    from xbrain.models import VideoFrame

    return [
        VideoFrame(timestamp=12.0, local_path="7/frames/0.png", description="A title slide."),
        VideoFrame(timestamp=95.0, local_path="7/frames/1.png", description="A code slide."),
    ]


def test_generate_embeds_and_mirrors_slide_frames(tmp_path: Path):
    """A slide-heavy `x_video` source (#44 PR4) embeds each kept slide into the
    digest section the SAME way downloaded photos are embedded — `![[_media/...]]`
    — with its vision description as a caption, and the image is mirrored from
    `media_root` into the vault's `_media/` tree so the embed resolves."""
    media_root = tmp_path / "media"
    (media_root / "7" / "frames").mkdir(parents=True)
    (media_root / "7" / "frames" / "0.png").write_bytes(b"\x89PNG slide0")
    (media_root / "7" / "frames" / "1.png").write_bytes(b"\x89PNG slide1")

    store = {"7": _video_item(text="The talk transcript.", frames=_frames())}
    generate(store, tmp_path, media_root=media_root)
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")

    assert "## Video digest: The Talk" in note
    assert "The talk transcript." in note
    assert "![[_media/7/frames/0.png]]" in note  # embedded like a photo
    assert "![[_media/7/frames/1.png]]" in note
    assert "A title slide." in note  # the vision description as caption
    # mirrored into the self-contained vault _media/ tree
    assert (tmp_path / "_media" / "7" / "frames" / "0.png").exists()
    assert (tmp_path / "_media" / "7" / "frames" / "1.png").exists()


def test_generate_no_frames_note_is_unchanged(tmp_path: Path):
    """The visual layer is inert without frames: a with-speech `x_video` source with
    an EMPTY `frames` list renders byte-identically to the pre-PR4 digest section —
    no stray embed lines appear on the default (non-`--frames`) path."""
    without = _video_item(text="Same transcript body.")
    with_empty = _video_item(text="Same transcript body.", frames=[])
    generate({"7": without}, tmp_path / "a")
    generate({"7": with_empty}, tmp_path / "b")
    note_a = next((tmp_path / "a" / "items").glob("*.md")).read_text(encoding="utf-8")
    note_b = next((tmp_path / "b" / "items").glob("*.md")).read_text(encoding="utf-8")
    assert note_a == note_b
    assert "_media" not in note_a  # no embed lines when there are no frames


def test_generate_silent_slide_deck_embeds_frames(tmp_path: Path):
    """A SILENT slide deck (no speech but with frames) renders the digest heading +
    the slide embeds — the visual layer is where a screen-only video's content
    lives — instead of the bare silent-video line."""
    media_root = tmp_path / "media"
    (media_root / "7" / "frames").mkdir(parents=True)
    (media_root / "7" / "frames" / "0.png").write_bytes(b"\x89PNG s0")
    (media_root / "7" / "frames" / "1.png").write_bytes(b"\x89PNG s1")

    store = {"7": _video_item(text="", has_speech=False, frames=_frames())}
    generate(store, tmp_path, media_root=media_root)
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "## Video digest: The Talk" in note
    assert "![[_media/7/frames/0.png]]" in note
    assert "silent video" not in note.lower()


def test_generate_frames_embed_without_media_root(tmp_path: Path):
    """Like photos, the frame embed lines render even without `media_root` (the URL
    is in the data; only the mirrored bytes are missing) — a missing mirror renders
    a broken embed, not a crash."""
    store = {"7": _video_item(text="body", frames=_frames())}
    generate(store, tmp_path)  # no media_root
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "![[_media/7/frames/0.png]]" in note


def test_slide_caption_collapses_internal_newlines(tmp_path: Path):
    """A multi-line vision description must render as ONE valid `> ...` blockquote
    caption: internal newlines are collapsed to spaces so the second line can't
    spill OUT of the blockquote (an Obsidian caption is a single line)."""
    from xbrain.generate import _slide_embed_lines
    from xbrain.models import VideoFrame

    frames = [
        VideoFrame(
            timestamp=1.0,
            local_path="7/frames/0.png",
            description="First line of the slide.\nSecond line of the slide.",
        )
    ]
    lines = _slide_embed_lines(frames)
    # Every emitted line is truly a single line — none carries an internal newline
    # that would break the blockquote when the note is joined with "\n".
    assert all("\n" not in line for line in lines)
    assert "> First line of the slide. Second line of the slide." in lines


def test_generate_silent_line_wins_over_stale_text_when_no_speech(tmp_path: Path):
    """Defensive: a malformed third-party transcriber that reports `has_speech=False`
    yet leaves non-empty `text` must still render the SILENT line — the tweet-signal
    truth — not the stale/contradictory transcript text (and no digest heading)."""
    generate({"7": _video_item(text="stale", has_speech=False, frames=[])}, tmp_path)
    note = next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")
    assert "silent video" in note.lower() or "sin voz" in note.lower()
    assert "stale" not in note
    assert "## Video digest" not in note


# --------------------------------------------------------------- topic_style


def _hashtag_item() -> Item:
    """Enriched item with two topics — exercises the topic-line rendering."""
    return Item(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="alice", name="Alice"),
        text="Body 42",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/p", domain="example.com")],
        enriched=Enrichment(
            enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            executor="api",
            summary="Resumen.",
            primary_topic="ai-coding",
            topics=["ai-coding", "software-engineering"],
        ),
    )


def test_generate_renders_topics_as_wikilinks_by_default(tmp_path: Path):
    """Default `topic_style` keeps the byte-for-byte current wikilink form."""
    generate({"42": _hashtag_item()}, tmp_path)
    note = next((tmp_path / "items").glob("*-42.md")).read_text(encoding="utf-8")
    assert "**Topics:** [[ai-coding]] · [[software-engineering]]" in note
    assert "#ai-coding" not in note
    assert "#software-engineering" not in note


def test_generate_renders_topics_as_hashtags_when_requested(tmp_path: Path):
    """`topic_style="hashtag"` emits Obsidian tags space-separated on the line."""
    generate({"42": _hashtag_item()}, tmp_path, topic_style="hashtag")
    note = next((tmp_path / "items").glob("*-42.md")).read_text(encoding="utf-8")
    assert "**Topics:** #ai-coding #software-engineering" in note
    assert "[[ai-coding]]" not in note
    assert "[[software-engineering]]" not in note
    # Orthogonality invariant: frontmatter `tags:` are unchanged across modes.
    # Both slugs must still be present in the frontmatter as native Obsidian tags.
    assert "tags: [x-knowledge, ai-coding, software-engineering]" in note


def test_generate_hashtag_mode_does_not_affect_index_or_topic_page_lists(tmp_path: Path):
    """Hashtag mode is in-body-only — the `_index.md` ## Topics section stays wikilinks."""
    generate({"42": _hashtag_item()}, tmp_path, topic_style="hashtag")
    index = (tmp_path / "_index.md").read_text(encoding="utf-8")
    # The index ranks topics with wikilink-plus-count — independent of topic_style.
    assert "[[ai-coding]] (1)" in index
    assert "[[software-engineering]] (1)" in index
    # And the index never carries the in-body `**Topics:**` line at all.
    assert "**Topics:**" not in index


def test_generate_rejects_unknown_topic_style(tmp_path: Path):
    """Unknown topic_style at the generator boundary surfaces a ValueError."""
    import pytest

    with pytest.raises(ValueError, match="topic_style"):
        generate({"42": _hashtag_item()}, tmp_path, topic_style="bogus")


# ------------------------------------------------- x_article blogpost render (#39 PR5)

_ART_TS = datetime(2026, 5, 24, tzinfo=timezone.utc)


def _article_downloaded_image(index: int = 0, *, alt: str | None = None, item_id: str = "99"):
    """An `ArticleImageBlock` in the `MediaPhotoDownloaded` state.

    `local_path` carries the `article/` namespace exactly as PR4 stores it
    (`<id>/article/<n>.<ext>`) — the STORED path the renderer/mirror reuse
    verbatim (no per-source index recompute).
    """
    from xbrain.models import ArticleImageBlock, MediaPhotoDownloaded

    return ArticleImageBlock(
        media=MediaPhotoDownloaded(
            url=f"https://pbs.twimg.com/media/art{index}.png",
            local_path=f"{item_id}/article/{index}.png",
            width=10,
            height=8,
            bytes_size=12,
            downloaded_at=_ART_TS,
        ),
        alt=alt,
    )


def _article_item(*, blocks, item_id: str = "99", title: str | None = "The Article", text=None):
    """A bookmarked X Article item carrying an ordered `x_article` body.

    Mirrors the shape PR2/PR3 produce: a synthesised `/i/article/<id>` link (so
    the item gets a note) plus a single `x_article` `ContentSourceSuccess` whose
    `text` is the flattened concatenation of the `ArticleTextBlock` texts (the
    PR1 invariant). Pass an explicit `text` for the empty-`blocks` fallback path.
    """
    from xbrain.models import ArticleTextBlock

    if text is None:
        text = "".join(b.text for b in blocks if isinstance(b, ArticleTextBlock))
    url = f"https://x.com/i/article/{item_id}"
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text="check out my article",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url=url, domain="x.com")],
        content=Content(
            fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_article",
                    url=url,
                    title=title,
                    text=text,
                    http_status=200,
                    attempts=1,
                    blocks=blocks,
                )
            ],
        ),
    )


def test_generate_renders_article_as_ordered_blogpost(tmp_path: Path):
    """An `x_article` with ordered text→image→text blocks renders as a blogpost:
    the paragraphs and the inline `![[_media/<id>/article/<n>]]` embed interleave
    IN ORDER under `## Content: <title>`, the alt is the caption, and the bytes are
    mirrored into the vault's `_media/<id>/article/` namespace."""
    from xbrain.models import ArticleTextBlock

    media_root = tmp_path / "media"
    (media_root / "99" / "article").mkdir(parents=True)
    (media_root / "99" / "article" / "0.png").write_bytes(b"\x89PNG art0")

    blocks = [
        ArticleTextBlock(text="First paragraph."),
        _article_downloaded_image(0, alt="A diagram"),
        ArticleTextBlock(text="\n\nSecond paragraph."),
    ]
    output_dir = tmp_path / "vault"
    generate({"99": _article_item(blocks=blocks)}, output_dir, media_root=media_root)
    body = next((output_dir / "items").glob("*-99.md")).read_text(encoding="utf-8")

    assert "## Content: The Article" in body
    assert "First paragraph." in body
    assert "Second paragraph." in body
    assert "![[_media/99/article/0.png]]" in body
    assert "> A diagram" in body  # the alt-text is the caption
    # Order: text, then image, then text.
    i_first = body.index("First paragraph.")
    i_img = body.index("![[_media/99/article/0.png]]")
    i_second = body.index("Second paragraph.")
    assert i_first < i_img < i_second
    # The baked `\n\n` separator on the non-first text block was stripped — no
    # stray blank line leaks before the second paragraph.
    assert "\n\n\nSecond paragraph." not in body
    # Bytes mirrored into the self-contained vault under the `article/` namespace.
    assert (output_dir / "_media" / "99" / "article" / "0.png").exists()


def test_generate_renders_image_only_article(tmp_path: Path):
    """An image-only Article (blocks = [image], text == "") still renders the
    heading + the embed — no empty body, no crash."""
    media_root = tmp_path / "media"
    (media_root / "99" / "article").mkdir(parents=True)
    (media_root / "99" / "article" / "0.png").write_bytes(b"\x89PNG only")

    output_dir = tmp_path / "vault"
    generate(
        {"99": _article_item(blocks=[_article_downloaded_image(0)])},
        output_dir,
        media_root=media_root,
    )
    body = next((output_dir / "items").glob("*-99.md")).read_text(encoding="utf-8")
    assert "## Content: The Article" in body
    assert "![[_media/99/article/0.png]]" in body


def test_generate_renders_leading_and_trailing_image_article(tmp_path: Path):
    """A [image, text, image] Article renders both embeds around the body IN ORDER."""
    from xbrain.models import ArticleTextBlock

    media_root = tmp_path / "media"
    (media_root / "99" / "article").mkdir(parents=True)
    (media_root / "99" / "article" / "0.png").write_bytes(b"\x89PNG a0")
    (media_root / "99" / "article" / "1.png").write_bytes(b"\x89PNG a1")

    blocks = [
        _article_downloaded_image(0),
        ArticleTextBlock(text="Body between images."),
        _article_downloaded_image(1),
    ]
    output_dir = tmp_path / "vault"
    generate({"99": _article_item(blocks=blocks)}, output_dir, media_root=media_root)
    body = next((output_dir / "items").glob("*-99.md")).read_text(encoding="utf-8")
    i0 = body.index("![[_media/99/article/0.png]]")
    i_txt = body.index("Body between images.")
    i1 = body.index("![[_media/99/article/1.png]]")
    assert i0 < i_txt < i1
    assert (output_dir / "_media" / "99" / "article" / "1.png").exists()


def test_generate_article_pending_image_is_silent(tmp_path: Path):
    """A pending inline image renders nothing (no embed, no warning) — a future
    `xbrain media` run advances it — while the surrounding text still renders."""
    from xbrain.models import ArticleImageBlock, ArticleTextBlock, MediaPhotoPending

    blocks = [
        ArticleTextBlock(text="Body."),
        ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/p.png")),
    ]
    generate({"99": _article_item(blocks=blocks)}, tmp_path)  # no media_root
    body = next((tmp_path / "items").glob("*-99.md")).read_text(encoding="utf-8")
    assert "Body." in body
    assert "_media/99/article" not in body
    assert "⚠" not in body
    # No embed of any kind and no dead URL-filename slips through for a pending
    # image (matching the stronger tweet-photo sibling).
    assert "![[" not in body
    assert "p.png" not in body


def test_generate_image_only_all_pending_article_emits_no_bare_heading(tmp_path: Path):
    """An image-only Article whose sole block is a PENDING image (the normal
    post-`fetch`/pre-`media` state) renders NOTHING for the source — no bare
    `## Content:` heading over an empty body, no embed, no crash — mirroring how
    `_video_digest_lines` avoids an empty digest block."""
    from xbrain.models import ArticleImageBlock, MediaPhotoPending

    blocks = [ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/p.png"))]
    generate({"99": _article_item(blocks=blocks)}, tmp_path)  # no media_root
    body = next((tmp_path / "items").glob("*-99.md")).read_text(encoding="utf-8")
    assert "## Content:" not in body  # no bare heading when the body is empty
    assert "![[" not in body
    assert "p.png" not in body


def test_generate_article_video_variant_image_is_skipped_not_crash(tmp_path: Path, caplog):
    """An `ArticleImageBlock` wrapping a video variant (nominally admitted by the
    `MediaEntry` union, never emitted by the PR3 producer) is logged and skipped —
    no embed, no crash — pinning the graceful defensive branch."""
    import logging as _logging

    from xbrain.models import ArticleImageBlock, ArticleTextBlock, MediaVideoPending

    blocks = [
        ArticleTextBlock(text="Body."),
        ArticleImageBlock(media=MediaVideoPending(url="https://video.twimg.com/v.mp4")),
    ]
    with caplog.at_level(_logging.WARNING, logger="xbrain.generate"):
        generate({"99": _article_item(blocks=blocks)}, tmp_path)
    body = next((tmp_path / "items").glob("*-99.md")).read_text(encoding="utf-8")
    assert "Body." in body  # the surrounding text still renders
    assert "![[" not in body  # no embed emitted for the video variant
    assert "v.mp4" not in body
    assert "unexpected MediaVideoPending media variant" in caplog.text


def test_generate_article_failed_image_renders_warning(tmp_path: Path):
    """A failed inline image renders a one-line ⚠ note carrying reason + URL —
    visible evidence, never a silent drop, never a crash."""
    from xbrain.models import ArticleImageBlock, ArticleTextBlock, MediaPhotoFailed

    blocks = [
        ArticleTextBlock(text="Body."),
        ArticleImageBlock(
            media=MediaPhotoFailed(
                url="https://pbs.twimg.com/media/dead.png",
                failure_reason="http_4xx",
                attempts=1,
                last_attempt_at=_ART_TS,
            )
        ),
    ]
    generate({"99": _article_item(blocks=blocks)}, tmp_path)
    body = next((tmp_path / "items").glob("*-99.md")).read_text(encoding="utf-8")
    assert "⚠" in body
    assert "https://pbs.twimg.com/media/dead.png" in body
    assert "URL no encontrada (HTTP 4xx)" in body


def test_generate_article_described_image_renders_caption(tmp_path: Path):
    """A described inline image renders the embed + its vision description as a
    `> …` caption (the model admits a described article image even though PR5's
    producer only emits pending → downloaded)."""
    from xbrain.models import ArticleImageBlock, MediaPhotoDescribed

    media_root = tmp_path / "media"
    (media_root / "99" / "article").mkdir(parents=True)
    (media_root / "99" / "article" / "0.png").write_bytes(b"\x89PNG d0")

    block = ArticleImageBlock(
        media=MediaPhotoDescribed(
            url="https://pbs.twimg.com/media/d.png",
            local_path="99/article/0.png",
            width=10,
            height=8,
            bytes_size=12,
            downloaded_at=_ART_TS,
            is_decorative=False,
            description="A labeled architecture diagram.",
            description_lang="English",
            description_version="v1",
            described_at=_ART_TS,
        ),
        alt=None,
    )
    generate({"99": _article_item(blocks=[block])}, tmp_path, media_root=media_root)
    body = next((tmp_path / "items").glob("*-99.md")).read_text(encoding="utf-8")
    assert "![[_media/99/article/0.png]]" in body
    assert "> A labeled architecture diagram." in body
    assert (tmp_path / "_media" / "99" / "article" / "0.png").exists()


def test_generate_article_empty_blocks_falls_back_to_text(tmp_path: Path):
    """An `x_article` with EMPTY `blocks` (trafilatura-only fallback, or a pre-#39
    record) renders the plain `source.text` block — byte-for-byte the pre-PR5
    heading+text shape, no article-image namespace, no empty heading."""
    item = _article_item(blocks=[], title="Fallback Art", text="Plain trafilatura body.")
    generate({"99": item}, tmp_path)
    body = next((tmp_path / "items").glob("*-99.md")).read_text(encoding="utf-8")
    # Exact pre-PR5 fallback shape: `## Content: <title>` + blank + text + blank.
    assert "## Content: Fallback Art\n\nPlain trafilatura body.\n" in body
    assert "_media/99/article" not in body


def test_generate_article_blogpost_is_stable_on_regen(tmp_path: Path):
    """Regenerating a blogpost note is deterministic and preserves the user tail."""
    media_root = tmp_path / "media"
    (media_root / "99" / "article").mkdir(parents=True)
    (media_root / "99" / "article" / "0.png").write_bytes(b"\x89PNG s0")
    from xbrain.models import ArticleTextBlock

    blocks = [ArticleTextBlock(text="Body."), _article_downloaded_image(0, alt="cap")]
    store = {"99": _article_item(blocks=blocks)}
    output_dir = tmp_path / "vault"
    generate(store, output_dir, media_root=media_root)
    note_path = next((output_dir / "items").glob("*-99.md"))
    note_path.write_text(note_path.read_text(encoding="utf-8") + "MI NOTA", encoding="utf-8")
    first = note_path.read_text(encoding="utf-8")
    generate(store, output_dir, media_root=media_root)
    second = note_path.read_text(encoding="utf-8")
    assert first == second
    assert "MI NOTA" in second


def test_generate_article_missing_bytes_renders_broken_embed(tmp_path: Path):
    """A downloaded inline image whose bytes vanished still renders the embed
    (a broken image is the right signal) — the mirror skips it, no crash."""
    store = {"99": _article_item(blocks=[_article_downloaded_image(0)])}
    media_root = tmp_path / "media"  # exists but empty — no bytes to mirror
    media_root.mkdir()
    output_dir = tmp_path / "vault"
    generate(store, output_dir, media_root=media_root)
    body = next((output_dir / "items").glob("*-99.md")).read_text(encoding="utf-8")
    assert "![[_media/99/article/0.png]]" in body
    assert not (output_dir / "_media" / "99" / "article" / "0.png").exists()


def test_generate_article_embed_without_media_root(tmp_path: Path):
    """The embed line renders even without `media_root` (the path is in the data;
    only the mirrored bytes are missing) — parity with the photo/frame convention."""
    generate({"99": _article_item(blocks=[_article_downloaded_image(0)])}, tmp_path)
    body = next((tmp_path / "items").glob("*-99.md")).read_text(encoding="utf-8")
    assert "![[_media/99/article/0.png]]" in body


def test_article_blocks_lines_strips_baked_separator(tmp_path: Path):
    """`_article_blocks_lines` strips the baked `\\n\\n` separator off each
    non-first text block so no rendered line carries it (interleaved rendering
    must not re-emit the flatten separator as a stray blank line)."""
    from xbrain.generate import _article_blocks_lines
    from xbrain.i18n import strings_for
    from xbrain.models import ArticleTextBlock

    blocks = [ArticleTextBlock(text="Para one."), ArticleTextBlock(text="\n\nPara two.")]
    text = "".join(b.text for b in blocks)
    source = ContentSourceSuccess(
        kind="x_article", url="https://x.com/i/article/99", title="T", text=text, blocks=blocks
    )
    lines = _article_blocks_lines(source, strings_for("English"))
    assert "\n\nPara two." not in lines  # the baked separator did not survive
    assert "Para two." in lines
    # Every rendered line is a single physical line (no internal newline that
    # would desync the `"\n".join` in `_render_note`).
    assert all("\n" not in line for line in lines)


# ----------------------------------------------------- verification badges (#79, staleness-aware)


def _badge_item(item_id: str = "9", *, summary: str = "A crisp summary.") -> Item:
    """An enriched (non-video) item for verification-badge tests."""
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text="a tweet",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        enriched=Enrichment(
            enriched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            executor="api",
            summary=summary,
            primary_topic="ai-coding",
            topics=["ai-coding", "software-engineering"],
        ),
    )


def _stamp(
    item: Item,
    target: str,
    verdict: str,
    *,
    flags: list[str] | None = None,
    language: str = "English",
) -> None:
    """Stamp a CURRENT verification verdict onto `item` — current under the whole judging
    CONTRACT (the output it judged, the source it read, the rubric it applied), not just
    the output text."""
    from xbrain.models import VerificationVerdict
    from xbrain.verification import contract_fingerprint, fingerprint_output

    fp = fingerprint_output(item, target)
    contract = contract_fingerprint(item, target, language)
    assert fp is not None and contract is not None
    item.verification[target] = VerificationVerdict(
        verdict=verdict,
        faithfulness="FAIL" if verdict == "FAIL" else "PASS",
        adherence="REVIEW" if verdict == "REVIEW" else "PASS",
        output_fingerprint=fp,
        contract_fingerprint=contract,
        verified_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        flags=flags or [],
    )


def _note_body(tmp_path: Path) -> str:
    return next((tmp_path / "items").glob("*.md")).read_text(encoding="utf-8")


def test_generate_badges_a_current_summary_fail(tmp_path: Path):
    item = _badge_item()
    _stamp(item, "summary", "FAIL", flags=["unsupported number"])
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    assert "❌" in body
    assert "Verification: FAIL" in body
    assert "unsupported number" in body


def test_generate_badges_a_current_review_with_warning_emoji(tmp_path: Path):
    item = _badge_item()
    _stamp(item, "summary", "REVIEW")
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    assert "⚠️" in body
    assert "Verification: REVIEW" in body


def test_generate_does_not_badge_a_stale_verdict(tmp_path: Path):
    """THE core correctness test: a verdict stored against the OLD summary is silently
    NOT badged once the summary is re-generated (fingerprint no longer matches) — a fixed
    output never shows a ❌."""
    item = _badge_item(summary="Original summary that FAILED.")
    _stamp(item, "summary", "FAIL", flags=["unsupported number"])
    # The summary is re-generated (fixed) AFTER the verdict was stored.
    item.enriched.summary = "Corrected, faithful summary."
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    assert "❌" not in body
    assert "Verification: FAIL" not in body
    assert "Corrected, faithful summary." in body  # the current output still renders


def test_generate_does_not_badge_a_pass(tmp_path: Path):
    item = _badge_item()
    _stamp(item, "summary", "PASS")
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    assert "Verification:" not in body


def test_generate_legacy_item_without_verification_renders_no_badge(tmp_path: Path):
    item = _badge_item()
    assert item.verification == {}
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    assert "Verification:" not in body
    assert "A crisp summary." in body


def test_generate_badges_topics_verdict(tmp_path: Path):
    item = _badge_item()
    _stamp(item, "topics", "FAIL", flags=["wrong topic"])
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    assert "Verification: FAIL" in body
    assert "wrong topic" in body


def test_generate_badges_digest_verdict_near_video_header(tmp_path: Path):
    item = _video_item(item_id="12", text="Transcript body.", digest="A long readable digest.")
    _stamp(item, "digest", "FAIL", flags=["hallucinated claim"])
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    header_idx = body.index("## Video digest")
    badge_idx = body.index("Verification: FAIL")
    assert badge_idx > header_idx
    # The badge sits right by the header, above the digest body.
    assert badge_idx < body.index("A long readable digest.")


def test_generate_spanish_badge_label_is_localized(tmp_path: Path):
    item = _badge_item()
    _stamp(item, "summary", "FAIL", flags=["número no soportado"], language="Spanish")
    generate({item.id: item}, tmp_path, output_language="Spanish")
    body = _note_body(tmp_path)
    assert "Verificación: FALLA" in body


def test_generate_badge_collapses_newline_in_flag_issue(tmp_path: Path):
    """A multi-line flag issue must not break out of the single-line `> …` blockquote —
    internal newlines are collapsed to spaces (mirrors the frame-caption invariant)."""
    item = _badge_item()
    _stamp(item, "summary", "FAIL", flags=["line one\nline two"])
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    badge_line = next(line for line in body.splitlines() if "Verification: FAIL" in line)
    assert badge_line.startswith(">")  # still one blockquote line
    assert "line one line two" in badge_line


def test_generate_no_badge_when_output_fixed_before_write(tmp_path: Path):
    """End-to-end (#79): a FAIL judged on summary "A", the summary regenerated to "B" BEFORE
    `--write-verdicts`, must never badge "B" — the stored (judged) fingerprint is of "A", so
    `generate` on "B" finds a mismatch. Uses the real write path with a judged fingerprint."""
    from xbrain.verification import (
        apply_verdicts_to_store,
        contract_fingerprint,
        fingerprint_output,
    )

    item = _badge_item(summary="A - judged and FAILED.")
    judged_fp = fingerprint_output(item, "summary")
    judged_contract = contract_fingerprint(item, "summary", "English")
    item.enriched.summary = "B - fixed before the write."  # regenerated in the window
    store = {item.id: item}
    result = apply_verdicts_to_store(
        store,
        [
            {
                "item_id": item.id,
                "target": "summary",
                "verdict": "FAIL",
                "faithfulness": "FAIL",
                "adherence": "PASS",
                "flags": [],
            }
        ],
        {(item.id, "summary"): judged_fp},
        {(item.id, "summary"): judged_contract},
    )
    assert result.written == 1
    assert store[item.id].verification["summary"].output_fingerprint == judged_fp
    generate(store, tmp_path)
    body = _note_body(tmp_path)
    assert "❌" not in body and "Verification: FAIL" not in body
    assert "B - fixed before the write." in body  # the current output still renders


def test_generate_does_not_badge_a_verdict_judged_against_a_DIFFERENT_SOURCE(tmp_path: Path):
    """The output is untouched, but the evidence under it changed (a new frame description
    landed). The judge never saw that source, so its verdict says nothing about this
    output-plus-source pair. No badge."""
    from xbrain.models import Content, ContentSourceSuccess, VideoFrame

    item = _badge_item()
    item.content = Content(
        fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        sources=[
            ContentSourceSuccess(
                kind="x_video",
                url="https://x.com/v",
                title="A talk",
                text="the transcript body",
                has_speech=True,
                frames=[
                    VideoFrame(timestamp=0.0, local_path="9/frames/0.png", description="A chart.")
                ],
            )
        ],
    )
    _stamp(item, "summary", "FAIL", flags=["unsupported number"])
    source = item.content.sources[0]
    source.frames = [
        *source.frames,
        VideoFrame(timestamp=9.0, local_path="9/frames/9.png", description="A brand-new slide."),
    ]
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    assert "❌" not in body
    assert "Verification: FAIL" not in body


def test_generate_does_not_badge_a_verdict_judged_under_an_OLD_RUBRIC(tmp_path: Path, monkeypatch):
    """#86 rewrote `rubric-verify.md` without touching one output character, and every
    stored verdict still painted. This is the arm that closes it."""
    from xbrain import rubrics as rubrics_mod
    from xbrain.verification import rubric_digest

    item = _badge_item()
    _stamp(item, "summary", "FAIL", flags=["unsupported number"])

    original = rubrics_mod._RUBRICS_DIR
    shadow = tmp_path / "rubrics"
    shadow.mkdir()
    for rubric in original.glob("*.md"):
        shadow.joinpath(rubric.name).write_text(
            rubric.read_text(encoding="utf-8"), encoding="utf-8"
        )
    verify = shadow / "rubric-verify.md"
    verify.write_text(
        verify.read_text(encoding="utf-8") + "\nA RULE THE JUDGE NEVER SAW.\n", encoding="utf-8"
    )
    monkeypatch.setattr(rubrics_mod, "_RUBRICS_DIR", shadow)
    rubric_digest.cache_clear()
    try:
        generate({item.id: item}, tmp_path / "out")
        body = next((tmp_path / "out" / "items").glob("*.md")).read_text(encoding="utf-8")
    finally:
        rubric_digest.cache_clear()
    assert "❌" not in body
    assert "Verification: FAIL" not in body


def test_generate_does_not_badge_a_LEGACY_verdict_with_no_contract(tmp_path: Path):
    """Every verdict stored before this change was judged under a contract we cannot
    reconstruct. It is not grandfathered in: it paints NOTHING."""
    from xbrain.models import VerificationVerdict
    from xbrain.verification import fingerprint_output

    item = _badge_item()
    item.verification["summary"] = VerificationVerdict(
        verdict="FAIL",
        faithfulness="FAIL",
        output_fingerprint=fingerprint_output(item, "summary"),
        verified_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        flags=["unsupported number"],
    )  # no contract_fingerprint — the old shape
    generate({item.id: item}, tmp_path)
    body = _note_body(tmp_path)
    assert "❌" not in body
    assert "Verification: FAIL" not in body

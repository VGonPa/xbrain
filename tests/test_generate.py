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


def test_generate_renders_video_pending_as_placeholder(tmp_path: Path):
    """A `MediaVideoPending` becomes a 🎥 placeholder carrying the URL.

    Video bytes are not downloaded yet. The URL is the only evidence we
    have — surface it so the reader can click through to X.
    """
    from xbrain.models import MediaVideoPending

    item = _item("1", with_link=True)
    item.media = [MediaVideoPending(url="https://video.twimg.com/x.mp4")]

    generate({"1": item}, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    body = note.read_text(encoding="utf-8")
    assert "🎥" in body
    assert "https://video.twimg.com/x.mp4" in body


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

# tests/test_worksheet.py
import json
from datetime import datetime, timezone

from xbrain.models import Author, Item, Link, Topic
from xbrain.worksheet import export_worksheet, import_worksheet


def _item(item_id: str) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="post text",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://arxiv.org/abs/1", domain="arxiv.org")],
        bookmark_folder="AI papers",
    )


VOCAB = [Topic(slug="misc", description="Noise.")]


def test_export_worksheet_writes_items_vocab_and_rubrics(tmp_path):
    path = tmp_path / "ws.json"
    export_worksheet([_item("1"), _item("2")], VOCAB, path, "claude-code", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {it["item_id"] for it in data["items"]} == {"1", "2"}
    assert data["items"][0]["bookmark_folder"] == "AI papers"
    assert data["items"][0]["links"][0]["domain"] == "arxiv.org"
    assert "topics" in data["rubrics"]
    assert [t["slug"] for t in data["vocab"]] == ["misc"]
    assert data["judgments"] == []
    # The rubrics shipped in the worksheet must already have `{language}`
    # substituted — the Claude Code session reads them as-is.
    assert "{language}" not in data["rubrics"]["summary"]
    assert "**Language:** English" in data["rubrics"]["summary"]


def test_export_worksheet_records_executor(tmp_path):
    path = tmp_path / "ws.json"
    export_worksheet([_item("1")], VOCAB, path, "manual", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["executor"] == "manual"


def test_import_worksheet_reads_filled_judgments(tmp_path):
    path = tmp_path / "ws.json"
    export_worksheet([_item("1")], VOCAB, path, "claude-code", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["judgments"] = [
        {"item_id": "1", "summary": "s", "primary_topic": "misc", "topics": ["misc"]}
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    executor, judgments = import_worksheet(path)
    assert executor == "claude-code"
    assert judgments[0]["item_id"] == "1"


def test_import_worksheet_reads_executor_back(tmp_path):
    path = tmp_path / "ws.json"
    export_worksheet([_item("1")], VOCAB, path, "manual", "English")
    executor, judgments = import_worksheet(path)
    assert executor == "manual"
    assert judgments == []


def test_import_worksheet_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        import_worksheet(tmp_path / "nope.json")


def test_import_worksheet_rejects_non_list_judgments(tmp_path):
    import pytest

    # A worksheet whose `judgments` is not a list (e.g. an object) is a clean
    # up-front error, not an obscure failure when the loop tries to iterate it.
    path = tmp_path / "ws.json"
    path.write_text(json.dumps({"judgments": {}}), encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        import_worksheet(path)
    assert "must be a list" in str(exc_info.value)


def _described_photo(*, description: str, decorative: bool = False):
    """Build a `MediaPhotoDescribed` for the worksheet image tests (#34).

    Mirrors the api-executor test helper: `MediaPhotoDescribed` enforces
    `is_decorative => description == ""` at the model layer, so the
    `description` argument is dropped in the decorative branch.
    """
    from xbrain.models import MediaPhotoDescribed

    return MediaPhotoDescribed(
        url="https://pbs.twimg.com/media/X.jpg",
        local_path="1/0.jpg",
        width=4,
        height=3,
        bytes_size=512,
        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        is_decorative=decorative,
        description="" if decorative else description,
        description_lang="English",
        description_version="v1",
        described_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
    )


def _photo_item(item_id: str, *photos) -> Item:
    """An item carrying described photos (#34)."""
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="post text",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        media=list(photos),
    )


def test_export_worksheet_includes_image_descriptions(tmp_path):
    """A content-bearing described photo surfaces under `image_descriptions` so
    the manual/claude-code enrich track sees the same visual signal the api
    track injects as `Images in this post:` — closing the #34 gap."""
    path = tmp_path / "ws.json"
    export_worksheet(
        [_photo_item("1", _described_photo(description="A chart of MMLU scores."))],
        VOCAB,
        path,
        "manual",
        "English",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["image_descriptions"] == ["A chart of MMLU scores."]


def test_export_worksheet_omits_decorative_image_descriptions(tmp_path):
    """A decorative-only item carries no image descriptions — avatars / reaction
    memes are filtered at the same seam the api path uses, so no topic noise.

    Note: this exercises the filter's decorative exclusion only *indirectly*. The
    model invariant `is_decorative => description == ""` (enforced by
    `MediaPhotoDescribed`) makes a decorative-photo-with-nonempty-description
    unconstructable, so the filter's `not is_decorative` clause and its
    empty-description backstop can't be told apart by a test — either one alone
    excludes this photo. Both clauses are kept for defence in depth.
    """
    path = tmp_path / "ws.json"
    export_worksheet(
        [_photo_item("1", _described_photo(description="ignored", decorative=True))],
        VOCAB,
        path,
        "manual",
        "English",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["image_descriptions"] == []


def test_export_worksheet_omits_image_descriptions_when_no_described_photos(tmp_path):
    """An item with no described photos is unchanged: an empty image list.

    Regression guard so an item that never went through `xbrain describe`
    exports a clean, empty `image_descriptions` rather than a missing key.
    """
    path = tmp_path / "ws.json"
    export_worksheet([_item("1")], VOCAB, path, "manual", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["image_descriptions"] == []


def _video_item(
    item_id: str,
    *,
    text: str,
    has_speech: bool = True,
    frame_descs: tuple[str, ...] = (),
) -> Item:
    """An item carrying an `x_video` transcript + key-frame content source (#44).

    `frame_descs` populate the `x_video` `frames` list (slides/screens shown in the
    video), a different field from the `MediaPhotoDescribed` photos on `item.media`.
    """
    from xbrain.models import Content, ContentSourceSuccess, VideoFrame

    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="watch this",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/v",
                    text=text,
                    has_speech=has_speech,
                    frames=[
                        VideoFrame(
                            timestamp=float(i),
                            local_path=f"{item_id}/frames/{i}.png",
                            description=d,
                        )
                        for i, d in enumerate(frame_descs)
                    ],
                )
            ],
        ),
    )


def test_export_worksheet_carries_video_transcript_in_its_own_field(tmp_path):
    """A video transcript is exported under `video_transcript`, NOT `article` —
    so the manual/claude-code enrich track sees it as a transcript."""
    path = tmp_path / "ws.json"
    export_worksheet(
        [_video_item("1", text="deep talk on retrieval")], VOCAB, path, "manual", "English"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["items"][0]
    assert entry["video_transcript"] == "deep talk on retrieval"
    assert entry["article"] is None


def test_export_worksheet_omits_no_speech_video_transcript(tmp_path):
    """A no-speech video contributes no transcript text to the worksheet."""
    path = tmp_path / "ws.json"
    export_worksheet(
        [_video_item("1", text="", has_speech=False)], VOCAB, path, "manual", "English"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["video_transcript"] is None


def test_export_worksheet_carries_the_full_video_transcript(tmp_path):
    """The worksheet carries the FULL transcript — unlike the `api` prompt, which is
    truncated to `TRANSCRIPT_CHAR_LIMIT`. The worksheet is judged by a full-context
    agent (not a bounded per-item model call), so it reads the whole talk and its
    summary/topics are not front-biased to the first ~13 min. The two engines
    legitimately diverge on input size (see `worksheet._video_transcript`)."""
    from xbrain.rubrics import TRANSCRIPT_CHAR_LIMIT

    long_text = "word " * TRANSCRIPT_CHAR_LIMIT  # >> the api cap
    path = tmp_path / "ws.json"
    export_worksheet([_video_item("1", text=long_text)], VOCAB, path, "manual", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    transcript = data["items"][0]["video_transcript"]
    assert transcript == long_text  # full, untruncated
    assert "transcript truncated" not in transcript


def test_export_worksheet_includes_video_frame_descriptions(tmp_path):
    """Key-frame descriptions surface under `video_frame_descriptions` so the
    manual/claude-code enrich track sees what the video SHOWS (slides/screens),
    mirroring the api path's `Video frames` section."""
    path = tmp_path / "ws.json"
    export_worksheet(
        [
            _video_item(
                "1",
                text="a talk",
                frame_descs=("A title slide: 'Scaling'.", "A chart of loss curves."),
            )
        ],
        VOCAB,
        path,
        "manual",
        "English",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["video_frame_descriptions"] == [
        "A title slide: 'Scaling'.",
        "A chart of loss curves.",
    ]


def test_export_worksheet_frame_descriptions_present_when_no_speech(tmp_path):
    """A mute slide/screen-share video (has_speech=False) contributes its frame
    descriptions even with no transcript — the whole point of feeding frames."""
    path = tmp_path / "ws.json"
    export_worksheet(
        [
            _video_item(
                "1",
                text="",
                has_speech=False,
                frame_descs=("A slide: Postgres vs DynamoDB latency.",),
            )
        ],
        VOCAB,
        path,
        "manual",
        "English",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["items"][0]
    assert entry["video_frame_descriptions"] == ["A slide: Postgres vs DynamoDB latency."]
    assert entry["video_transcript"] is None  # no speech → no transcript


def test_export_worksheet_omits_video_frame_descriptions_when_none(tmp_path):
    """A video without described frames exports a clean empty list, not a missing key."""
    path = tmp_path / "ws.json"
    export_worksheet([_video_item("1", text="a talk")], VOCAB, path, "manual", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["video_frame_descriptions"] == []


def test_export_worksheet_notes_unfetched_links(tmp_path):
    """A linking item with no fetched content carries an explicit guardrail note,
    so the enrich agent never guesses the linked content from the URL/domain."""
    path = tmp_path / "ws.json"
    export_worksheet([_item("1")], VOCAB, path, "claude-code", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    note = data["items"][0]["unfetched_links_note"]
    assert note is not None
    assert "NOT fetched" in note


def test_export_worksheet_no_unfetched_note_when_article_fetched(tmp_path):
    from xbrain.models import Content, ContentSourceSuccess

    item = _item("1")
    item.content = Content(
        fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        sources=[
            ContentSourceSuccess(
                kind="external_article",
                url="https://arxiv.org/abs/1",
                title="Paper",
                text="the fetched body",
            )
        ],
    )
    path = tmp_path / "ws.json"
    export_worksheet([item], VOCAB, path, "claude-code", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["unfetched_links_note"] is None


def test_export_worksheet_no_unfetched_note_without_links(tmp_path):
    item = _item("1")
    item.links = []
    path = tmp_path / "ws.json"
    export_worksheet([item], VOCAB, path, "claude-code", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["unfetched_links_note"] is None


def _exported(item, tmp_path) -> dict:
    """Export one item and return its worksheet entry."""
    path = tmp_path / "ws.json"
    export_worksheet([item], VOCAB, path, "claude-code", "English")
    return json.loads(path.read_text(encoding="utf-8"))["items"][0]


def _with_sources(item, *sources):
    from xbrain.models import Content

    item.content = Content(
        fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc), sources=list(sources)
    )
    return item


def _success(kind: str, text: str, **extra):
    from xbrain.models import ContentSourceSuccess

    return ContentSourceSuccess(kind=kind, url="https://x.com/a/status/1", text=text, **extra)


def test_export_worksheet_thread_text_is_a_thread_not_an_article(tmp_path):
    """F1: a thread's own text must NOT be exported as the fetched `article` — the
    enrich agent would read the poster's own words as a downloaded linked page."""
    item = _with_sources(_item("1"), _success("thread", "1/ my thread\n\n2/ about agents"))
    entry = _exported(item, tmp_path)
    assert entry["article"] is None
    assert entry["thread"] == "1/ my thread\n\n2/ about agents"
    # …and the link nobody fetched is still flagged.
    assert entry["unfetched_links_note"] is not None


def test_export_worksheet_partial_fetch_notes_the_missing_link(tmp_path):
    """F5: 2 links, 1 fetched — the article is exported AND the note states the counts."""
    from xbrain.models import Link

    item = _item("1")
    item.links = [
        Link(url="https://arxiv.org/abs/1", domain="arxiv.org"),
        Link(url="https://t.co/x", domain="time.com"),
    ]
    _with_sources(item, _success("external_article", "the fetched body", title="Paper"))
    entry = _exported(item, tmp_path)
    assert entry["article"] == "the fetched body"
    assert "1 of 2" in entry["unfetched_links_note"]


def test_export_worksheet_marks_an_unfetched_quoted_post(tmp_path):
    """F3: the quoted post is never fetched — the worksheet says so, so the agent
    does not invent the shared content the summary rubric asks it to describe."""
    item = _item("1")
    item.quoted_id = "999"
    entry = _exported(item, tmp_path)
    assert entry["quoted_content_note"] is not None
    assert "NOT fetched" in entry["quoted_content_note"]


def test_export_worksheet_no_quoted_note_without_a_quote(tmp_path):
    assert _exported(_item("1"), tmp_path)["quoted_content_note"] is None

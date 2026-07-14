# tests/test_executors_api.py
from datetime import datetime, timezone

import pytest

from xbrain.executors.api import (
    QUOTED_CONTENT_UNFETCHED_NOTE,
    ApiExecutor,
    _user_prompt,
    links_content_unfetched,
    unfetched_links_note,
)
from xbrain.models import Author, Content, ContentSourceSuccess, Item, Link, Topic

from tests.conftest import FakeAnthropic


def _item(item_id: str, **extra) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="un post sobre LLMs",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        **extra,
    )


VOCAB = [
    Topic(slug="ai-coding", description="LLMs writing software."),
    Topic(slug="misc", description="Posts that do not fit a topic."),
]


def test_api_executor_returns_one_judgment_per_item():
    payload = {"summary": "r", "primary_topic": "ai-coding", "topics": ["ai-coding"]}
    client = FakeAnthropic([payload, payload])
    ex = ApiExecutor(model="claude-haiku-4-5-20251001", output_language="English", client=client)
    out = ex.enrich_items([_item("1"), _item("2")], VOCAB)
    assert {j.item_id for j in out} == {"1", "2"}
    assert len(client.messages.calls) == 2


def test_api_executor_substitutes_language_in_system_prompt():
    """Regression guard: the system prompt must ship to the LLM with
    `{language}` substituted. If a refactor calls `load_rubric` without
    `language=`, the placeholder leaks and we get wrong-language output.
    """
    payload = {"summary": "r", "primary_topic": "misc", "topics": ["misc"]}
    client = FakeAnthropic([payload])
    ApiExecutor(model="m", output_language="Spanish", client=client).enrich_items(
        [_item("1")], VOCAB
    )
    system = client.messages.calls[0]["system"]
    assert "{language}" not in system
    assert "**Language:** Spanish" in system


def test_api_executor_sends_the_configured_model():
    client = FakeAnthropic([{"summary": "r", "primary_topic": "misc", "topics": ["misc"]}])
    ApiExecutor(model="claude-sonnet-4-6", output_language="English", client=client).enrich_items(
        [_item("1")], VOCAB
    )
    assert client.messages.calls[0]["model"] == "claude-sonnet-4-6"


def test_user_prompt_includes_link_domains_and_folder():
    item = _item(
        "1",
        links=[Link(url="https://arxiv.org/abs/1", domain="arxiv.org")],
        bookmark_folder="AI papers",
    )
    prompt = _user_prompt(item, VOCAB)
    assert "arxiv.org" in prompt
    assert "AI papers" in prompt


def test_user_prompt_includes_folder_when_no_links():
    item = _item("1", bookmark_folder="AI papers")
    prompt = _user_prompt(item, VOCAB)
    assert "AI papers" in prompt
    assert not item.links


def test_user_prompt_includes_link_domains_when_no_folder():
    item = _item("1", links=[Link(url="https://arxiv.org/abs/1", domain="arxiv.org")])
    prompt = _user_prompt(item, VOCAB)
    assert "arxiv.org" in prompt
    assert not item.bookmark_folder


def test_api_executor_skips_wrong_shape_response(capsys):
    # A response that is valid JSON but not a judgment object must be skipped
    # with a warning, not silently become an empty enrichment.
    client = FakeAnthropic(
        [
            {"not": "a judgment"},
            {"summary": "r", "primary_topic": "misc", "topics": ["misc"]},
        ]
    )
    ex = ApiExecutor(model="m", output_language="English", client=client)
    out = ex.enrich_items([_item("1"), _item("2")], VOCAB)
    assert {j.item_id for j in out} == {"2"}  # item 1 skipped
    err = capsys.readouterr().err
    assert "enrichment failed for item 1" in err


def test_api_executor_skips_item_on_api_failure(capsys):
    # A transient API failure on one item must not abort the whole batch.
    from anthropic import APIError

    client = FakeAnthropic(
        [
            APIError("503 service unavailable", request=None, body=None),
            {"summary": "r", "primary_topic": "misc", "topics": ["misc"]},
        ]
    )
    ex = ApiExecutor(model="m", output_language="English", client=client)
    out = ex.enrich_items([_item("1"), _item("2")], VOCAB)
    assert {j.item_id for j in out} == {"2"}
    captured = capsys.readouterr().err
    assert "enrichment failed for item 1" in captured
    # Partial-failure summary line is visible on stderr
    assert "SUMMARY: enriched: 1, failed: 1" in captured


def test_api_executor_raises_when_all_items_fail():
    """An API key revocation / total outage must surface as non-zero exit, not
    silent empty result. The CLI's _handle_cli_errors catches RuntimeError."""
    import pytest
    from anthropic import APIError

    client = FakeAnthropic(
        [
            APIError("401 unauthorized", request=None, body=None),
            APIError("401 unauthorized", request=None, body=None),
        ]
    )
    ex = ApiExecutor(model="m", output_language="English", client=client)
    with pytest.raises(RuntimeError, match="All 2 items failed enrichment"):
        ex.enrich_items([_item("1"), _item("2")], VOCAB)


def test_api_executor_propagates_programmer_bugs():
    """`AttributeError` and other non-recoverable exceptions must NOT be
    swallowed — the developer needs to see the traceback.
    """
    import pytest

    class _Boom:
        class messages:  # noqa: N801
            @staticmethod
            def create(*_args, **_kwargs):
                raise AttributeError("programmer bug — undefined attribute")

    ex = ApiExecutor(model="m", output_language="English", client=_Boom())
    with pytest.raises(AttributeError, match="programmer bug"):
        ex.enrich_items([_item("1")], VOCAB)


def test_api_executor_propagates_keyboard_interrupt():
    """Ctrl-C must NOT be swallowed by the recoverable-errors tuple. The
    narrow catch uses Exception subclasses; KeyboardInterrupt inherits from
    BaseException and falls through. This is the property of Python, but
    pin it as a regression test — a future refactor that switches the
    tuple to BaseException would silently break Ctrl-C without any failing
    test.
    """
    import pytest

    class _CtrlC:
        class messages:  # noqa: N801
            @staticmethod
            def create(*_args, **_kwargs):
                raise KeyboardInterrupt

    ex = ApiExecutor(model="m", output_language="English", client=_CtrlC())
    with pytest.raises(KeyboardInterrupt):
        ex.enrich_items([_item("1")], VOCAB)


def test_api_executor_emits_no_summary_on_total_failure(capsys):
    """The all-failed branch raises before printing the summary — there is
    no `SUMMARY: enriched: 0, ...` line on a total-failure run. The raised
    RuntimeError is the signal."""
    import pytest
    from anthropic import APIError

    client = FakeAnthropic([APIError("503", request=None, body=None)])
    ex = ApiExecutor(model="m", output_language="English", client=client)
    with pytest.raises(RuntimeError):
        ex.enrich_items([_item("1")], VOCAB)
    assert "SUMMARY:" not in capsys.readouterr().err


def test_api_executor_emits_no_summary_when_all_succeed(capsys):
    """No failures, no noise: a clean batch stays silent on stderr."""
    payload = {"summary": "r", "primary_topic": "misc", "topics": ["misc"]}
    client = FakeAnthropic([payload])
    ex = ApiExecutor(model="m", output_language="English", client=client)
    ex.enrich_items([_item("1")], VOCAB)
    assert "SUMMARY:" not in capsys.readouterr().err


def _described_photo(*, description: str, decorative: bool = False):
    """Build a `MediaPhotoDescribed` for prompt-integration tests.

    `MediaPhotoDescribed` enforces `is_decorative => description == ""`
    at the model layer; this helper honours that contract by forcing an
    empty description when `decorative=True`, so the caller's `description`
    argument is silently dropped in the decorative branch.
    """
    from datetime import datetime, timezone

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


def test_user_prompt_includes_content_bearing_image_descriptions():
    """Non-decorative described photos must surface as `Images in this post:` lines."""
    item = _item("1", media=[_described_photo(description="A chart of MMLU scores.")])
    prompt = _user_prompt(item, VOCAB)
    assert "Images in this post:" in prompt
    assert "A chart of MMLU scores." in prompt


def test_user_prompt_excludes_decorative_image_descriptions():
    """Decorative photos must NOT appear in the prompt — pure noise."""
    item = _item("1", media=[_described_photo(description="ignored", decorative=True)])
    prompt = _user_prompt(item, VOCAB)
    assert "Images in this post:" not in prompt


def test_user_prompt_omits_images_section_when_no_described_photos():
    """Items without any described photos must NOT have the images section.

    Otherwise the LLM sees a hint of missing context and may hallucinate
    visual evidence that does not exist. Regression guard.
    """
    item = _item("1")  # no media
    prompt = _user_prompt(item, VOCAB)
    assert "Images in this post:" not in prompt


def test_user_prompt_image_descriptions_precede_links_and_article():
    """Image section sits between the post body and the links/article."""
    from xbrain.models import Link

    item = _item(
        "1",
        media=[_described_photo(description="A diagram of GraphQL caching.")],
        links=[Link(url="https://example.com/x", domain="example.com")],
    )
    prompt = _user_prompt(item, VOCAB)
    image_idx = prompt.index("Images in this post:")
    links_idx = prompt.index("Links in the post")
    assert image_idx < links_idx


def _video_item(item_id: str, *, text: str = "a talk about scaling laws", has_speech: bool = True):
    """An item whose only content is an `x_video` transcript source (#44)."""
    from xbrain.models import Content, ContentSourceSuccess

    return _item(
        item_id,
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/a/status/1/video/1",
                    title="A great talk",
                    text=text,
                    has_speech=has_speech,
                )
            ],
        ),
    )


def test_user_prompt_includes_video_transcript_section():
    """An `x_video` transcript surfaces under a labelled `Video transcript:` block."""
    prompt = _user_prompt(_video_item("1"), VOCAB)
    assert "Video transcript:" in prompt
    assert "a talk about scaling laws" in prompt


def test_user_prompt_omits_video_transcript_when_no_speech():
    """A no-speech (has_speech=False, empty) transcript adds nothing — no section."""
    prompt = _user_prompt(_video_item("1", text="", has_speech=False), VOCAB)
    assert "Video transcript:" not in prompt


def test_user_prompt_video_transcript_not_relabelled_as_article():
    """The transcript must render as `Video transcript:`, never mislabelled as a
    `Linked article` (would tell the LLM the wrong content type)."""
    prompt = _user_prompt(_video_item("1"), VOCAB)
    assert "Linked article" not in prompt


def test_user_prompt_truncates_a_long_video_transcript():
    """A 72-min-talk-scale transcript is capped so one item can't blow the prompt."""
    from xbrain.rubrics import TRANSCRIPT_CHAR_LIMIT

    long_text = "word " * (TRANSCRIPT_CHAR_LIMIT)  # >> the cap
    prompt = _user_prompt(_video_item("1", text=long_text), VOCAB)
    assert "transcript truncated" in prompt
    assert len(prompt) < len(long_text)


def test_user_prompt_video_transcript_sits_between_images_and_links_and_article():
    """The `Video transcript:` block is spliced AFTER the image descriptions and
    BEFORE the links/article — mirroring the image-ordering guard so an accidental
    reorder in `_user_prompt` is caught (the transcript is post content, read in the
    same natural order as images: post body → images → transcript → links → article)."""
    from xbrain.models import Content, ContentSourceSuccess, Link

    item = _item(
        "1",
        media=[_described_photo(description="A diagram of the training loop.")],
        links=[Link(url="https://example.com/x", domain="example.com")],
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/a/status/1/video/1",
                    title="A great talk",
                    text="a talk about scaling laws",
                    has_speech=True,
                ),
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.com/x",
                    title="An article",
                    text="the article body",
                ),
            ],
        ),
    )
    prompt = _user_prompt(item, VOCAB)
    image_idx = prompt.index("Images in this post:")
    transcript_idx = prompt.index("Video transcript:")
    links_idx = prompt.index("Links in the post")
    article_idx = prompt.index("Linked article")
    assert image_idx < transcript_idx < links_idx
    assert transcript_idx < article_idx


def _video_item_with_frames(
    item_id: str,
    *,
    text: str = "a talk about scaling laws",
    has_speech: bool = True,
    frame_descs: tuple[str, ...] = (),
):
    """An `x_video` item carrying key-frame descriptions (slides/screens shown).

    Frame descriptions ride on the `x_video` `ContentSourceSuccess.frames` list,
    a different field from the `MediaPhotoDescribed` photo descriptions on
    `item.media` — so a slide/screen-share video contributes visual topic signal
    even with `has_speech=False` (no transcript).
    """
    from xbrain.models import Content, ContentSourceSuccess, VideoFrame

    frames = [
        VideoFrame(timestamp=float(i), local_path=f"1/frames/{i}.png", description=desc)
        for i, desc in enumerate(frame_descs)
    ]
    return _item(
        item_id,
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/a/status/1/video/1",
                    title="A great talk",
                    text=text,
                    has_speech=has_speech,
                    frames=frames,
                )
            ],
        ),
    )


def test_user_prompt_includes_video_frame_descriptions():
    """Key-frame descriptions surface under a labelled `Video frames` block so the
    LLM reads what the video SHOWS (slides/screens), not just what it says."""
    item = _video_item_with_frames(
        "1", frame_descs=("A title slide: 'Scaling Laws'.", "A chart of loss vs compute.")
    )
    prompt = _user_prompt(item, VOCAB)
    assert "Video frames" in prompt
    assert "A title slide: 'Scaling Laws'." in prompt
    assert "A chart of loss vs compute." in prompt


def test_user_prompt_omits_video_frames_when_none():
    """A video with no described frames must NOT have the frames section — otherwise
    the LLM sees a hint of missing visual context and may hallucinate slides."""
    prompt = _user_prompt(_video_item_with_frames("1", frame_descs=()), VOCAB)
    assert "Video frames" not in prompt


def test_user_prompt_includes_frames_even_when_no_speech():
    """THE slide-deck case: a mute screen-share video (has_speech=False, empty text)
    still contributes its frame descriptions as topic signal, even though it has no
    transcript. This is the whole reason frames feed enrich."""
    item = _video_item_with_frames(
        "1",
        text="",
        has_speech=False,
        frame_descs=("A slide comparing Postgres vs DynamoDB latency.",),
    )
    prompt = _user_prompt(item, VOCAB)
    assert "Video frames" in prompt
    assert "Postgres vs DynamoDB" in prompt
    assert "Video transcript:" not in prompt  # no speech → no transcript block


def test_user_prompt_bounds_the_video_frames_section():
    """A 60-frame deck can't blow the per-item prompt: the frames block is capped at
    `FRAME_DESC_CHAR_LIMIT` and signposts the cut."""

    big = tuple(f"Frame {i}: " + ("x" * 500) for i in range(60))  # >> the cap
    prompt = _user_prompt(_video_item_with_frames("1", frame_descs=big), VOCAB)
    assert "Video frames" in prompt
    # The section is bounded: not every 500-char frame made it in.
    assert len(prompt) < sum(len(d) for d in big)
    assert "frames omitted" in prompt


def test_user_prompt_video_frames_sit_after_transcript_before_links():
    """Order: post → images → transcript → frames → links → article. Voice then
    slides, both before the links/article, so the LLM reads the video in one place."""
    from xbrain.models import Content, ContentSourceSuccess, Link, VideoFrame

    item = _item(
        "1",
        links=[Link(url="https://example.com/x", domain="example.com")],
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/a/status/1/video/1",
                    title="A great talk",
                    text="a talk about scaling laws",
                    has_speech=True,
                    frames=[
                        VideoFrame(
                            timestamp=1.0, local_path="1/frames/0.png", description="A title slide."
                        )
                    ],
                ),
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.com/x",
                    title="An article",
                    text="the article body",
                ),
            ],
        ),
    )
    prompt = _user_prompt(item, VOCAB)
    transcript_idx = prompt.index("Video transcript:")
    frames_idx = prompt.index("Video frames")
    links_idx = prompt.index("Links in the post")
    article_idx = prompt.index("Linked article")
    assert transcript_idx < frames_idx < links_idx
    assert frames_idx < article_idx


# -------------------------------------------------- unfetched-content guardrails


def _linked_item(*, links: int = 1, sources: list[ContentSourceSuccess] | None = None, **extra):
    """An item with `links` outbound links and an optional content source list."""
    return _item(
        "1",
        links=[Link(url=f"https://t.co/{i}", domain=f"site{i}.com") for i in range(links)],
        content=(
            Content(fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc), sources=list(sources))
            if sources is not None
            else None
        ),
        **extra,
    )


def _source(kind: str, text: str = "some body", **extra) -> ContentSourceSuccess:
    return ContentSourceSuccess(kind=kind, url="https://x/1", text=text, **extra)


# The predicate is the WHOLE guardrail: when it is False no surface is marked, so
# every content shape it can meet is pinned here. Only a fetched LINKED page
# (`external_article` / `x_article`) counts as "the link was fetched" — a `thread`
# is the item's OWN text, a `quoted_tweet` is another post, an `x_video` is a
# manufactured transcript. None of them is evidence an outbound link was read.
@pytest.mark.parametrize(
    ("links", "kinds", "expected"),
    [
        (0, [], False),  # no links → nothing to guard
        (1, [], True),  # links, content is None → nothing fetched
        (1, ["external_article"], False),  # the link WAS fetched
        (1, ["x_article"], False),  # an X longform article counts too
        (1, ["thread"], True),  # F1: a thread is the item's own text
        (1, ["quoted_tweet"], True),  # a quoted post is not the linked page
        (1, ["x_video"], True),  # a transcript is not the linked page
        (1, ["thread", "x_video"], True),  # F1: still nothing linked was fetched
        (2, ["external_article"], True),  # F5: 1 of 2 links fetched → still flagged
        (2, ["external_article", "x_article"], False),  # both links fetched
        (1, ["external_article", "thread"], False),  # article + own thread → fetched
    ],
)
def test_links_content_unfetched_only_counts_fetched_linked_pages(links, kinds, expected):
    item = _linked_item(links=links, sources=[_source(k) for k in kinds] if kinds else None)
    assert links_content_unfetched(item) is expected


def test_links_content_unfetched_ignores_a_textless_source():
    """A success source with empty text (a no-speech video) is not fetched content."""
    item = _linked_item(sources=[_source("x_video", text="", has_speech=False)])
    assert links_content_unfetched(item) is True


def test_user_prompt_flags_unfetched_links():
    """When the linked content was never fetched, the prompt carries the guardrail
    note — the model must not reconstruct the linked content from the URL/domain."""
    item = _linked_item()
    prompt = _user_prompt(item, VOCAB)
    # Identity, not a substring: the prompt must carry the SHARED note verbatim, so the
    # judge can hold the generator to exactly what it was told.
    assert unfetched_links_note(item) in prompt


def test_user_prompt_does_not_flag_links_when_article_fetched():
    item = _linked_item(sources=[_source("external_article", "the fetched body")])
    prompt = _user_prompt(item, VOCAB)
    assert "NOT fetched" not in prompt
    assert "the fetched body" in prompt


def test_user_prompt_labels_thread_text_as_thread_not_article():
    """F1: a thread's own text is the item's own words, NOT a fetched linked page.
    It must reach the model under its own label, and it must not suppress the
    unfetched-links guardrail for a link nobody downloaded."""
    item = _linked_item(sources=[_source("thread", "1/ my thread\n\n2/ about agents")])
    prompt = _user_prompt(item, VOCAB)
    assert "1/ my thread" in prompt  # the thread text is NOT dropped — it is signal
    thread_idx = prompt.index("Thread (full text by the same author):")
    assert thread_idx < prompt.index("1/ my thread")
    assert "Linked article" not in prompt  # …but it is not served as a fetched article
    assert unfetched_links_note(item) in prompt  # …and the unfetched link is still flagged


def test_user_prompt_partial_fetch_flags_the_missing_link_with_counts():
    """F5: 2 links, 1 fetched — the fetched body is present AND the note says how
    many links were left unfetched, so a claim about the other one is checkable."""
    item = _linked_item(links=2, sources=[_source("external_article", "the fetched body")])
    prompt = _user_prompt(item, VOCAB)
    assert "the fetched body" in prompt
    note = unfetched_links_note(item)
    assert note is not None and "1 of 2" in note
    assert note in prompt


def test_user_prompt_marks_an_unfetched_quoted_post():
    """F3: the quoted post's content is never downloaded — say so, so the generator
    does not invent the content it was told to summarise."""
    item = _item("1", quoted_id="123")
    prompt = _user_prompt(item, VOCAB)
    assert "Quoted post" in prompt
    assert QUOTED_CONTENT_UNFETCHED_NOTE in prompt


def test_user_prompt_has_no_quoted_marker_without_a_quote():
    assert "Quoted post" not in _user_prompt(_item("1"), VOCAB)

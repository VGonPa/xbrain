# tests/test_executors_api.py
from datetime import datetime, timezone

from xbrain.executors.api import ApiExecutor, _user_prompt
from xbrain.models import Author, Item, Link, Topic

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

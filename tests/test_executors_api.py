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
    client = FakeAnthropic(
        [
            RuntimeError("503 service unavailable"),
            {"summary": "r", "primary_topic": "misc", "topics": ["misc"]},
        ]
    )
    ex = ApiExecutor(model="m", output_language="English", client=client)
    out = ex.enrich_items([_item("1"), _item("2")], VOCAB)
    assert {j.item_id for j in out} == {"2"}
    assert "enrichment failed for item 1" in capsys.readouterr().err

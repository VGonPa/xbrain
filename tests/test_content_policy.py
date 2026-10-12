from datetime import datetime, timezone

from xbrain.content_policy import is_article_candidate_url, prune_store, should_keep_item
from xbrain.models import (
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    Link,
    Media,
)

DT = datetime(2026, 7, 18, tzinfo=timezone.utc)


def _item(
    item_id: str,
    *,
    source: str = "bookmark",
    links: list[Link] | None = None,
    media: list | None = None,
    content: Content | None = None,
) -> Item:
    return Item(
        id=item_id,
        source=source,  # type: ignore[arg-type]
        url=f"https://x.com/alice/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text="Saved post",
        created_at=DT,
        captured_at=DT,
        links=links or [],
        media=media or [],
        content=content,
    )


def test_article_candidate_url_accepts_external_and_x_articles_only():
    assert is_article_candidate_url("https://example.com/article")
    assert is_article_candidate_url("https://x.com/i/article/123")
    assert not is_article_candidate_url("https://x.com/alice/status/123")


def test_policy_keeps_bookmarked_articles_and_videos():
    external = _item("1", links=[Link(url="https://example.com/a", domain="example.com")])
    x_article = _item("2", links=[Link(url="https://x.com/i/article/2", domain="x.com")])
    video = _item("3", media=[Media(type="video", url="https://video.twimg.com/v/3.mp4")])

    assert should_keep_item(external)
    assert should_keep_item(x_article)
    assert should_keep_item(video)


def test_policy_discards_own_tweets_and_plain_bookmarked_posts():
    assert not should_keep_item(
        _item(
            "1", source="own_tweet", links=[Link(url="https://example.com/a", domain="example.com")]
        )
    )
    assert not should_keep_item(_item("2"))
    assert not should_keep_item(_item("3", media=[Media(type="photo", url="https://img/3.jpg")]))
    assert not should_keep_item(
        _item("4", links=[Link(url="https://x.com/bob/status/4", domain="x.com")])
    )


def test_policy_keeps_article_sources_even_when_fetch_failed():
    success = _item(
        "1",
        content=Content(
            fetched_at=DT,
            sources=[
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.com/a",
                    title="A",
                    text="body",
                )
            ],
        ),
    )
    failed = _item(
        "2",
        content=Content(
            fetched_at=DT,
            sources=[
                ContentSourceFailure(
                    kind="x_article",
                    url="https://x.com/i/article/2",
                    failure_reason="js_required",
                )
            ],
        ),
    )

    assert should_keep_item(success)
    assert should_keep_item(failed)


def test_prune_store_removes_non_library_items():
    keep = _item("keep", links=[Link(url="https://example.com/a", domain="example.com")])
    discard = _item("discard")
    store = {"keep": keep, "discard": discard}

    assert prune_store(store) == ["discard"]
    assert store == {"keep": keep}

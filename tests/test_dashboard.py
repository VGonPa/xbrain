# tests/test_dashboard.py
import json
from datetime import datetime, timezone

from PIL import Image

from xbrain.dashboard import (
    _escape_for_script,
    collect_thumbnails,
    compute_dashboard_data,
    humanize_topic,
    render_dashboard_html,
)
from xbrain.models import (
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    MediaPhotoDownloaded,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoPending,
)

DT = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _item(
    item_id,
    source="bookmark",
    topic="ai-coding",
    handle="alice",
    name="Alice",
    links=None,
    media=None,
    content=None,
    created=DT,
    summary="resumen",
):
    return Item(
        id=item_id,
        source=source,
        url=f"https://x.com/{handle}/status/{item_id}",
        author=Author(handle=handle, name=name),
        text=f"text {item_id}",
        created_at=created,
        captured_at=DT,
        links=links or [],
        media=media or [],
        content=content,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary=summary,
            primary_topic=topic,
            topics=[topic],
        ),
    )


def test_humanize_topic_acronyms_and_ampersand():
    assert humanize_topic("ai-coding") == "AI Coding"
    assert humanize_topic("agentic-engineering") == "Agentic Engineering"
    assert humanize_topic("ai-and-jobs") == "AI & Jobs"
    assert humanize_topic("llm-foundations") == "LLM Foundations"


def test_compute_counts_topics_authors_and_deep_links():
    items = [
        _item("100", "bookmark", "ai-coding", "alice"),
        _item(
            "101",
            "bookmark",
            "ai-coding",
            "bob",
            "Bob",
            links=[Link(url="https://ex.com/a", domain="ex.com")],
        ),
        _item("102", "own_tweet", "claude-code", "vgonpa"),
    ]
    id2note = {"100": "/v/items/100.md", "101": "/v/items/101.md", "102": "/v/items/102.md"}
    data = compute_dashboard_data(items, {}, id2note, [], "JUN 1, 2026")

    m = data["meta"]
    assert (m["total"], m["bookmarks"], m["own"], m["enriched"], m["topics_count"]) == (
        3,
        2,
        1,
        3,
        2,
    )
    assert data["topics_sorted"][0] == {"slug": "ai-coding", "label": "AI Coding", "count": 2}
    # own_tweet authors are excluded from the "bookmarked authors" chart
    assert {a["handle"] for a in data["authors"]} == {"alice", "bob"}
    assert data["domains"][0]["domain"] == "ex.com"
    assert "2026-06" in data["months_data"]

    row = data["topic_data"]["ai-coding"]["samples"][0]
    assert row["url"].startswith("https://x.com/")
    assert row["note"].endswith(".md")


def test_long_form_and_media_counts():
    items = [
        _item(
            "1",
            content=Content(
                fetched_at=DT,
                sources=[
                    ContentSourceSuccess(
                        kind="external_article", url="https://ex.com/x", text="body", title="T"
                    )
                ],
            ),
        ),
        _item(
            "2",
            content=Content(
                fetched_at=DT,
                sources=[
                    ContentSourceFailure(
                        kind="external_article", url="https://ex.com/y", failure_reason="paywall"
                    )
                ],
            ),
        ),
        _item(
            "3",
            media=[
                MediaPhotoDownloaded(
                    url="https://p",
                    local_path="3/0.png",
                    width=10,
                    height=10,
                    bytes_size=99,
                    downloaded_at=DT,
                ),
                MediaVideoPending(url="https://v"),
            ],
        ),
        _item("4", media=[MediaPhotoPending(url="https://p2")]),
    ]
    data = compute_dashboard_data(items, {}, {}, [], "JUN 1, 2026")

    lf = data["meta"]["longform"]
    assert (lf["ext_saved"], lf["ext_failed"], lf["saved"], lf["total"]) == (1, 1, 1, 2)
    assert data["longform_full"]["items"][0]["title"] == "T"

    md = data["meta"]["media"]
    assert (md["photos_downloaded"], md["photos_pending"], md["videos"]) == (1, 1, 1)


def test_render_injects_data_and_library_and_leaves_no_placeholder():
    html = render_dashboard_html(
        {"meta": {"total": 7}}, template="A /*__DATA__*/ B /*__ECHARTS__*/ C", echarts="LIB"
    )
    assert '"total": 7' in html
    assert "LIB" in html
    assert "__DATA__" not in html and "__ECHARTS__" not in html


def test_render_uses_vendored_resources():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert '"total": 1' in html
    assert "/*__DATA__*/" not in html
    assert "echarts" in html.lower()


def test_render_escapes_script_breakout_in_user_text():
    """A `</script>` in post text must not close the inlined `<script>` block."""
    html = render_dashboard_html(
        {"s": "</script><img src=x onerror=alert(1)>"},
        template="<head><script>const DATA=/*__DATA__*/;</script></head>",
        echarts="",
    )
    assert html.count("</script>") == 1  # only the template's own closing tag
    assert "\\u003c/script" in html  # the payload's `<` was escaped


def test_escape_preserves_spaces_and_valid_content():
    assert _escape_for_script('{"a": "b c d"}') == '{"a": "b c d"}'


def test_render_injects_echarts_first_so_user_sentinel_is_not_spliced():
    """A field containing the `/*__ECHARTS__*/` sentinel must not splice the lib."""
    html = render_dashboard_html(
        {"s": "/*__ECHARTS__*/"}, template="X /*__ECHARTS__*/ Y /*__DATA__*/ Z", echarts="LIB"
    )
    assert html == 'X LIB Y {"s": "/*__ECHARTS__*/"} Z'


def test_render_survives_lone_surrogate():
    html = render_dashboard_html(
        {"s": "bad" + chr(0xD83D) + "x"}, template="/*__DATA__*/", echarts=""
    )
    html.encode("utf-8")  # a lone surrogate must not make the write crash


def test_growth_is_cumulative_across_months_with_month_slices():
    items = [
        _item(
            "1", "bookmark", "ai-coding", "alice", created=datetime(2026, 5, 3, tzinfo=timezone.utc)
        ),
        _item(
            "2",
            "bookmark",
            "ai-coding",
            "bob",
            "Bob",
            created=datetime(2026, 6, 4, tzinfo=timezone.utc),
        ),
        _item(
            "3",
            "own_tweet",
            "claude-code",
            "vgonpa",
            created=datetime(2026, 6, 5, tzinfo=timezone.utc),
        ),
    ]
    data = compute_dashboard_data(items, {}, {}, [], "x")
    assert data["months"] == ["2026-05", "2026-06"]
    assert data["new_total"] == [1, 2]
    assert data["cum_total"] == [1, 3]
    assert data["cum_bm"] == [1, 2]
    assert data["cum_own"] == [0, 1]
    june = data["months_data"]["2026-06"]
    assert (june["count"], june["bm"], june["own"]) == (2, 1, 1)
    assert june["top_topics"][0]["label"] == "AI Coding"
    assert "vgonpa" not in {a["handle"] for a in june["top_authors"]}


def test_domains_exclude_x_com():
    items = [
        _item("1", links=[Link(url="https://x.com/a", domain="x.com")]),
        _item("2", links=[Link(url="https://ex.com/b", domain="ex.com")]),
    ]
    data = compute_dashboard_data(items, {}, {}, [], "x")
    assert [d["domain"] for d in data["domains"]] == ["ex.com"]


def test_empty_store_does_not_crash():
    data = compute_dashboard_data([], {}, {}, [], "x")
    assert data["meta"]["total"] == 0
    assert data["topics_sorted"] == [] and data["authors"] == [] and data["months"] == []
    assert data["meta"]["longform"]["saved_pct"] == 0.0  # ZeroDivisionError guard
    render_dashboard_html(data)  # must not raise


def test_videos_row_content():
    items = [
        _item(
            "1",
            media=[
                MediaVideoDownloaded(
                    url="https://v",
                    thumbnail_url="https://poster",
                    duration_millis=95000,
                    local_path="1/0.mp4",
                    bytes_size=10,
                    downloaded_at=DT,
                )
            ],
        )
    ]
    data = compute_dashboard_data(items, {}, {}, [], "x")
    v = data["videos"]["items"][0]
    assert v["dur"] == 95 and v["poster"] == "https://poster"


def test_collect_thumbnails_none_missing_corrupt_and_real(tmp_path):
    photo_item = _item(
        "1",
        media=[
            MediaPhotoDownloaded(
                url="https://p",
                local_path="1/0.png",
                width=2,
                height=2,
                bytes_size=9,
                downloaded_at=DT,
            )
        ],
    )
    assert collect_thumbnails([photo_item], None, {}) == []  # no media root
    assert collect_thumbnails([photo_item], tmp_path, {}) == []  # file missing -> skipped
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "0.png").write_bytes(b"not an image")
    assert collect_thumbnails([photo_item], tmp_path, {}) == []  # corrupt -> skipped, no raise
    Image.new("RGB", (3, 3), "red").save(tmp_path / "1" / "0.png")
    thumbs = collect_thumbnails([photo_item], tmp_path, {"1": "/v/1.md"})
    assert len(thumbs) == 1
    assert thumbs[0]["thumb"].startswith("data:image/jpeg;base64,")
    assert thumbs[0]["handle"] == "alice" and thumbs[0]["note"] == "/v/1.md"


def test_generate_writes_dashboard_with_valid_blob_and_links_it(tmp_path):
    from xbrain.generate import generate

    store = {"1": _item("1"), "2": _item("2", "own_tweet", "claude-code", "vgonpa")}
    generate(store, tmp_path, output_language="Spanish")

    dashboard = tmp_path / "dashboard.html"
    assert dashboard.exists()
    assert "dashboard.html" in (tmp_path / "_index.md").read_text(encoding="utf-8")
    # The full store→compute→render→file path emits parseable JSON with right KPIs.
    text = dashboard.read_text(encoding="utf-8")
    blob = text.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    data = json.loads(blob)
    assert data["meta"]["total"] == 2
    assert data["meta"]["bookmarks"] == 1 and data["meta"]["own"] == 1

# tests/test_dashboard.py
import json
import re
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
    ArticleImageBlock,
    ArticleTextBlock,
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoPending,
    TopicPage,
)

DT = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _described_photo(
    local_path="1/0.png", *, description="Un gráfico de barras.", decorative=False
):
    """A `MediaPhotoDescribed` — the variant `xbrain describe` produces in place."""
    return MediaPhotoDescribed(
        url="https://p/" + local_path,
        local_path=local_path,
        width=3,
        height=3,
        bytes_size=9,
        downloaded_at=DT,
        is_decorative=decorative,
        description="" if decorative else description,
        description_lang="Spanish",
        description_version="v1",
        described_at=DT,
    )


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
    captured=DT,
    summary="resumen",
    confidence=None,
    suggested_topics=None,
    topics=None,
):
    assigned_topics = topics if topics is not None else [topic]
    return Item(
        id=item_id,
        source=source,
        url=f"https://x.com/{handle}/status/{item_id}",
        author=Author(handle=handle, name=name),
        text=f"text {item_id}",
        created_at=created,
        captured_at=captured,
        links=links or [],
        media=media or [],
        content=content,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary=summary,
            primary_topic=topic,
            topics=assigned_topics,
            topic_confidence=confidence,
            suggested_new_topics=suggested_topics or [],
        ),
    )


def test_humanize_topic_acronyms_and_ampersand():
    assert humanize_topic("ai-coding") == "AI Coding"
    assert humanize_topic("agentic-engineering") == "Agentic Engineering"
    assert humanize_topic("ai-and-jobs") == "AI & Jobs"
    assert humanize_topic("llm-foundations") == "LLM Foundations"


def test_compute_counts_topics_authors_and_deep_links():
    items = [
        _item("100", "bookmark", "ai-coding", "alice", confidence="high"),
        _item(
            "101",
            "bookmark",
            "ai-coding",
            "bob",
            "Bob",
            links=[Link(url="https://ex.com/a", domain="ex.com")],
            confidence="low",
            suggested_topics=["ai-systems"],
        ),
        _item("102", "own_tweet", "claude-code", "vgonpa", confidence="medium"),
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
    assert (m["processing_ready"], m["processing_ready_pct"]) == (2, 66.7)
    assert data["topics_sorted"][0] == {"slug": "ai-coding", "label": "AI Coding", "count": 2}
    # own_tweet authors are excluded from the "bookmarked authors" chart
    assert {a["handle"] for a in data["authors"]} == {"alice", "bob"}
    assert data["domains"][0]["domain"] == "ex.com"
    assert "2026-06" in data["months_data"]
    assert data["meta"]["taxonomy"]["confidence"]["high"] == 1
    assert data["meta"]["taxonomy"]["confidence"]["low"] == 1
    assert data["taxonomy"]["suggested"][0] == {"slug": "ai-systems", "count": 1}

    row = data["topic_data"]["ai-coding"]["samples"][0]
    assert row["id"] in {"100", "101"}
    assert row["url"].startswith("https://x.com/")
    assert row["note"].endswith(".md")
    assert row["confidence"] in {"high", "low"}


def test_graph_builds_topic_item_nodes_and_edges():
    items = [
        _item(
            "100",
            topic="ai-coding",
            confidence="high",
            topics=["ai-coding", "ai-agents"],
            summary="A" * 240,
            links=[Link(url="https://example.com/a", domain="example.com")],
            content=Content(
                fetched_at=DT,
                sources=[
                    ContentSourceSuccess(
                        kind="external_article",
                        url="https://example.com/a",
                        text="body",
                        title="A long-form title",
                    )
                ],
            ),
        ),
        _item(
            "101",
            topic="ai-agents",
            confidence="medium",
            topics=["ai-agents", "productivity"],
            content=Content(
                fetched_at=DT,
                sources=[ContentSourceSuccess(kind="x_video", url="https://x.com/v", text="v")],
            ),
        ),
        _item("102", topic="ai-coding", confidence="low", suggested_topics=["agent-memory"]),
    ]
    topic_pages = {
        "ai-coding": TopicPage(
            slug="ai-coding",
            overview="Coding overview",
            synthesized_at=DT,
            post_count_at_synth=2,
        )
    }

    data = compute_dashboard_data(items, topic_pages, {"100": "/v/items/100.md"}, [], "x")
    graph = data["graph"]

    assert graph["version"]
    assert {node["slug"] for node in graph["topics"]} == {
        "ai-agents",
        "ai-coding",
        "productivity",
    }
    coding = next(node for node in graph["topics"] if node["slug"] == "ai-coding")
    assert coding["count"] == 2
    assert coding["confidence"]["high"] == 1
    assert coding["confidence"]["low"] == 1
    assert coding["overview"] == "Coding overview"
    assert coding["note"] == "topics/ai-coding.md"
    assert {(edge["source"], edge["target"], edge["weight"]) for edge in graph["topic_edges"]} >= {
        ("topic:ai-agents", "topic:ai-coding", 1),
        ("topic:ai-agents", "topic:productivity", 1),
    }
    assert {
        (edge["source"], edge["target"], edge["kind"]) for edge in graph["membership_edges"]
    } >= {
        ("topic:ai-coding", "item:100", "primary"),
        ("topic:ai-agents", "item:100", "secondary"),
    }
    item = next(node for node in graph["items"] if node["item_id"] == "100")
    assert item["label"] == "A long-form title"
    assert len(item["summary"]) <= 180
    assert item["domain"] == "example.com"
    assert item["note"] == "/v/items/100.md"
    assert item["suggested_topics"] == []
    assert graph["emerging_topics"][0]["slug"] == "agent-memory"
    assert graph["facets"]["content_types"]["article"] == 1
    assert graph["facets"]["content_types"]["video"] == 1
    assert graph["insights"]["most_connected"]["slug"] == "ai-agents"


def test_graph_omits_unenriched_items_and_version_is_deterministic():
    enriched = _item("100", topic="ai-coding", confidence="high")
    unenriched = _item("101", topic="misc")
    unenriched.enriched = None

    first = compute_dashboard_data([enriched, unenriched], {}, {}, [], "x")["graph"]
    second = compute_dashboard_data([enriched, unenriched], {}, {}, [], "x")["graph"]

    assert [node["item_id"] for node in first["items"]] == ["100"]
    assert first["version"] == second["version"]


def test_taxonomy_triage_omits_manually_accepted_misc():
    accepted_misc = _item("100", topic="misc", confidence="high")
    weak_misc = _item("101", topic="misc", confidence="low")

    data = compute_dashboard_data([accepted_misc, weak_misc], {}, {}, [], "JUN 1, 2026")

    assert data["meta"]["taxonomy"]["misc"] == 2
    assert {row["id"] for row in data["taxonomy"]["review_items"]} == {"101"}


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
        _item(
            "5",
            content=Content(
                fetched_at=DT,
                sources=[
                    ContentSourceSuccess(
                        kind="x_article",
                        url="https://x.com/i/article/5",
                        text="body",
                        blocks=[
                            ArticleTextBlock(text="body"),
                            ArticleImageBlock(media=_described_photo("5/article/0.png")),
                            ArticleImageBlock(media=MediaPhotoPending(url="https://p3")),
                        ],
                    )
                ],
            ),
        ),
    ]
    data = compute_dashboard_data(items, {}, {}, [], "JUN 1, 2026")

    lf = data["meta"]["longform"]
    assert (lf["ext_saved"], lf["ext_failed"], lf["x_saved"], lf["saved"], lf["total"]) == (
        1,
        1,
        1,
        2,
        3,
    )
    assert data["longform_full"]["items"][0]["title"] == "T"
    assert data["meta"]["library"] == {
        "articles": 2,
        "videos": 1,
        "article_failed": 0,
        "source_failures": 1,
        "blocking_failures": 0,
        "post_only": 2,
    }

    md = data["meta"]["media"]
    assert (
        md["photos_downloaded"],
        md["photos_pending"],
        md["article_images_downloaded"],
        md["article_images_pending"],
        md["videos"],
    ) == (1, 1, 1, 1, 1)
    article_row = next(row for row in data["library_items"] if row["id"] == "5")
    assert article_row["article_images"] == [
        {
            "path": "_media/5/article/0.png",
            "description": "Un gráfico de barras.",
            "alt": "Article image 1",
            "width": 3,
            "height": 3,
            "decorative": False,
        }
    ]


def test_ops_section_counts_pending_work_and_article_failures():
    pending = _item("1", links=[Link(url="https://example.com/a", domain="example.com")])
    pending.enriched = None
    failed = _item(
        "2",
        content=Content(
            fetched_at=DT,
            sources=[
                ContentSourceFailure(
                    kind="x_article",
                    url="https://x.com/i/article/1",
                    failure_reason="js_required",
                )
            ],
        ),
    )
    failed.enriched = None
    processed_with_broken_link = _item(
        "4",
        media=[MediaVideoPending(url="https://v")],
        content=Content(
            fetched_at=DT,
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/alice/status/4",
                    text="video digest",
                ),
                ContentSourceFailure(
                    kind="external_article",
                    url="https://example.com/moved",
                    failure_reason="not_found",
                ),
            ],
        ),
    )
    media_item = _item(
        "3",
        media=[
            MediaPhotoPending(url="https://p/pending.jpg"),
            MediaPhotoDownloaded(
                url="https://p/done.jpg",
                local_path="3/0.jpg",
                width=10,
                height=10,
                bytes_size=100,
                downloaded_at=DT,
            ),
        ],
        content=Content(
            fetched_at=DT,
            sources=[
                ContentSourceSuccess(
                    kind="x_article",
                    url="https://x.com/i/article/3",
                    text="body",
                    blocks=[
                        ArticleTextBlock(text="body"),
                        ArticleImageBlock(media=MediaPhotoPending(url="https://p/article.jpg")),
                        ArticleImageBlock(
                            media=MediaPhotoDownloaded(
                                url="https://p/article-done.jpg",
                                local_path="3/article/0.jpg",
                                width=10,
                                height=10,
                                bytes_size=100,
                                downloaded_at=DT,
                            )
                        ),
                    ],
                )
            ],
        ),
    )

    data = compute_dashboard_data(
        [pending, failed, media_item, processed_with_broken_link],
        {},
        {"2": "/v/2.md", "4": "/v/4.md"},
        [],
        "x",
    )

    ops = data["ops"]
    assert ops["command"] == "uv run xbrain refresh-all --headless --video-max-size 4GB"
    assert ops["retry_failed_command"] == "uv run xbrain retry-failed --source bookmarks --headless"
    assert ops["retry_failed_bookmarks"] == 2
    assert ops["pending"] == {
        "fetch": 1,
        "media": 2,
        "describe": 2,
        "enrich": 2,
        "article_failures": 2,
        "blocking_article_failures": 1,
    }
    assert ops["article_failures"][0]["reason"] == "js_required"
    assert ops["article_failures"][0]["note"] == "/v/2.md"
    assert {row["id"] for row in ops["article_failures"]} == {"2", "4"}
    assert len(ops["recent_bookmarks"]) == 4


def test_render_dashboard_includes_ops_controls():
    html = render_dashboard_html({"meta": {"total": 1}, "ops": {"command": "cmd"}})
    assert 'id="ops-run"' in html
    assert 'id="ops-retry-failed"' in html
    assert "/api/refresh-all" in html
    assert "/api/retry-failed" in html
    assert "/api/reprocess-note" in html
    assert "/api/topic-action" in html
    assert "data-reprocess" not in html
    assert "Daily refresh" in html
    assert "Confirm topics" in html
    assert "Move to Misc" in html
    assert "Try suggestions" in html
    assert "Add suggestions" in html
    assert "Rebuild note" in html
    assert "Adding suggestions changes the shared vocabulary" in html
    assert ">Accept</button>" not in html
    assert ">Reject</button>" not in html


def test_render_dashboard_declares_escape_before_storage_strip_use():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert html.index("const esc=") < html.index("document.getElementById('storage-strip')")


def test_compute_dashboard_includes_storage_payload():
    item = _item("1", "bookmark", "ai-coding", "alice", confidence="high")
    data = compute_dashboard_data(
        [item],
        {},
        {"1": "/v/items/1.md"},
        [],
        "JUN 1, 2026",
        storage={"total_bytes": 1234, "total_label": "1.2 KB", "categories": []},
        item_storage={"1": {"record_bytes": 10, "record_label": "10 B"}},
    )

    assert data["meta"]["storage"]["total_label"] == "1.2 KB"
    assert data["topic_data"]["ai-coding"]["samples"][0]["storage"]["record_bytes"] == 10


def test_render_dashboard_includes_library_chat():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert 'id="chat-form"' in html
    assert 'id="nav-chat"' in html
    assert 'id="nav-atlas"' in html
    assert 'id="nav-ops"' in html
    assert 'id="dashboard-view"' in html
    assert 'id="chat-view"' in html
    assert "/api/chat" in html
    assert "Ask XBrain" in html
    assert 'id="library-browser"' in html
    assert 'id="filter-query"' in html
    assert 'id="filter-author"' in html
    assert 'id="filter-domain"' in html
    assert "/api/search" in html
    assert 'id="chat-scope"' in html
    assert "item_ids:itemIds" in html
    assert "'x-xbrain-action':'1'" in html


def test_render_dashboard_declares_workspace_navigation_sections():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert 'data-workspace="library"' in html
    assert 'data-workspace="atlas"' in html
    assert 'data-workspace="analytics"' in html
    assert 'data-workspace="ops"' in html
    assert 'id="signal-rail"' in html
    assert 'data-workspace-section="library atlas analytics"' in html
    assert "const WORKSPACES=" in html
    assert "viewFromHash" in html
    assert "ask:'library'" in html
    assert "signal-card" in html
    assert 'style="height:' not in html
    assert 'style="width:' not in html
    assert 'style="color:' not in html
    close_index = html.rfind("</html>")
    assert close_index != -1
    assert not html[close_index + len("</html>") :].strip()


def test_render_dashboard_includes_atlas_knowledge_graph_controls():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert "Atlas Knowledge Graph" in html
    assert 'id="atlas-tabs"' in html
    assert 'data-atlas-mode="map"' in html
    assert 'data-atlas-mode="timeline"' in html
    assert 'data-atlas-mode="list"' in html
    assert 'id="graph-search"' in html
    assert 'id="graph-date"' in html
    assert 'id="graph-date-from"' in html
    assert 'id="graph-date-to"' in html
    assert 'id="graph-content"' in html
    assert 'id="graph-confidence"' in html
    assert 'id="graph-docs"' in html
    assert 'id="graph-edge-mode"' in html
    assert 'id="graph-emerging"' in html
    assert 'id="graph-hop-1"' in html
    assert 'id="graph-hop-2"' in html
    assert 'id="graph-focusbar"' in html
    assert 'id="graph-clear-focus"' in html
    assert 'id="graph-clear-all"' in html
    assert 'id="graph-reset"' in html
    assert 'id="c-graph"' in html
    assert 'id="c-graph-timeline"' in html
    assert 'id="graph-list"' in html
    assert "xbrain.graph.layout.v1" in html
    assert "hiddenTopics" in html
    assert "focusSlug" in html
    assert "datePreset" in html
    assert "graphApplySearchFocus" in html
    assert "graphRoam" in html
    assert "toggleGraphPin" in html
    assert "data-graph-hide" in html
    assert "data-graph-promote" in html
    assert "promote_suggestions" in html
    assert "All time" in html
    assert "90 days" in html
    assert "Topic editing requires the served dashboard." in html
    assert 'role="tablist"' in html
    assert ">Documents</button>" in html


def test_render_dashboard_includes_accessible_chart_alternatives():
    html = render_dashboard_html({"meta": {"total": 1}})

    assert 'class="chart-fallback"' in html
    assert "Use the data table below for a text alternative." in html
    assert "element.setAttribute('role','img')" in html
    assert "element.setAttribute('tabindex','0')" in html


def test_render_dashboard_includes_forest_theme_controls_and_tokens():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert 'data-theme="dark"' in html
    assert 'id="theme-toggle"' in html
    assert "xbrain.theme" in html
    assert "--color-forest" in html
    assert "--color-bark" in html
    assert "--color-stone" in html
    assert "--shadow-panel" in html
    assert "--mauve" not in html
    assert "--sand" not in html


def test_dashboard_component_css_uses_tokens_after_theme_declarations():
    html = render_dashboard_html({"meta": {"total": 1}})
    component_css = html.split("*{box-sizing", 1)[1].split("</style>", 1)[0]
    assert not re.search(r"#[0-9A-Fa-f]{3,8}|rgba\(", component_css)


def test_dashboard_kpi_and_ops_layout_is_compact():
    html = render_dashboard_html({"meta": {"total": 1}})

    assert ".kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))" in html
    assert ".kpi{padding:13px 16px 12px" in html
    assert "min-height:92px" in html
    assert ".kpi .num{font-family:var(--display);font-weight:600;font-size:34px" in html
    assert (
        ".signal-rail{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:12px}"
        in html
    )
    assert (
        ".signal-card{border:1px solid var(--hairline);border-radius:var(--radius-4);background:var(--color-panel-soft);color:inherit;padding:9px 11px"
        in html
    )
    assert ".signal-card .value{font-family:var(--display);font-size:24px" in html
    assert ".ops{margin-top:12px;display:grid" in html
    assert ".ops-list{display:flex;flex-direction:column;gap:6px;max-height:238px" in html


def test_dashboard_copy_is_language_consistent():
    html = render_dashboard_html({"meta": {"total": 1}})
    for phrase in (
        "Pregunta sobre tu biblioteca",
        "Sin consulta",
        "Escribe una pregunta",
        "Buscando fuentes",
        "Sin respuesta",
        "Sin fuentes",
        "Abre este dashboard",
    ):
        assert phrase not in html


def test_render_dashboard_uses_accessible_controls_for_theme_topics_and_drawer():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html
    assert 'aria-labelledby="dw-title"' in html
    assert 'aria-hidden="true" inert' in html
    assert 'aria-label="Close detail drawer"' in html
    assert 'aria-live="polite"' in html
    assert 'aria-label="Question for XBrain"' in html
    assert 'id="tg-misc" type="button" aria-pressed="false"' in html
    assert 'class="topic-rank-row"' in html
    assert 'class="chart-action"' in html
    assert "prefers-reduced-motion" in html
    assert "REDUCED_MOTION" in html
    assert "SCROLL_OPTS" in html
    assert "All 45" not in html


def test_render_dashboard_uses_web_note_links_with_obsidian_fallback():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert "/notes?path=" in html
    assert "WEB_NOTES" in html
    assert "obsidian://open?path=" in html


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
            "1",
            "bookmark",
            "ai-coding",
            "alice",
            captured=datetime(2026, 5, 3, tzinfo=timezone.utc),
        ),
        _item(
            "2",
            "bookmark",
            "ai-coding",
            "bob",
            "Bob",
            captured=datetime(2026, 6, 4, tzinfo=timezone.utc),
        ),
        _item(
            "3",
            "own_tweet",
            "claude-code",
            "vgonpa",
            captured=datetime(2026, 6, 5, tzinfo=timezone.utc),
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


def test_growth_uses_acquisition_date_and_fills_empty_months():
    items = [
        _item(
            "1",
            created=datetime(2024, 1, 3, tzinfo=timezone.utc),
            captured=datetime(2026, 1, 5, tzinfo=timezone.utc),
        ),
        _item(
            "2",
            created=datetime(2025, 8, 4, tzinfo=timezone.utc),
            captured=datetime(2026, 3, 7, tzinfo=timezone.utc),
        ),
    ]

    data = compute_dashboard_data(items, {}, {}, [], "x")

    assert data["months"] == ["2026-01", "2026-02", "2026-03"]
    assert data["new_total"] == [1, 0, 1]
    assert data["cum_total"] == [1, 1, 2]
    assert data["months_data"]["2026-02"]["count"] == 0
    assert data["library_items"][0]["captured_date"] == "2026-03-07"
    assert data["library_items"][0]["published_date"] == "2025-08-04"


def test_meta_uses_all_assigned_topics_and_truthful_review_counts():
    items = [
        _item(
            "1",
            topic="primary",
            topics=["primary", "secondary"],
            confidence="medium",
            suggested_topics=["new-one", "new-two"],
        ),
        _item(
            "2",
            topic="primary",
            confidence="low",
            suggested_topics=["new-one"],
        ),
    ]

    data = compute_dashboard_data(items, {}, {}, [], "x")

    assert data["meta"]["topics_count"] == 2
    assert data["meta"]["primary_topics_count"] == 1
    assert data["taxonomy"]["review_count"] == 2
    assert data["taxonomy"]["suggestion_assignments"] == 3
    assert data["taxonomy"]["suggestion_unique"] == 2


def test_graph_layout_version_is_stable_when_only_documents_change():
    first = compute_dashboard_data([_item("1")], {}, {}, [], "x")["graph"]["version"]
    second = compute_dashboard_data([_item("1"), _item("2")], {}, {}, [], "x")["graph"]["version"]
    changed_taxonomy = compute_dashboard_data(
        [_item("1"), _item("2", topic="new-topic")], {}, {}, [], "x"
    )["graph"]["version"]

    assert first == second
    assert changed_taxonomy != first


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
    assert data["graph"]["topics"] == [] and data["graph"]["items"] == []
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
    assert thumbs[0]["desc"] == ""  # undescribed downloaded photo → present but empty caption


def test_described_photos_count_as_downloaded_and_populate_photo_posts():
    # Regression: `xbrain describe` transitions Downloaded -> Described in place.
    # The dashboard must keep counting described photos as downloaded photos and
    # keep their posts in the photo samples, or the whole described corpus vanishes.
    items = [
        _item("1", media=[_described_photo("1/0.png")]),
        _item("2", media=[_described_photo("2/0.png", decorative=True)]),
        _item("3", media=[MediaPhotoPending(url="https://p")]),
    ]
    data = compute_dashboard_data(items, {}, {}, [], "JUN 1, 2026")
    md = data["meta"]["media"]
    assert (md["photos_downloaded"], md["photos_pending"]) == (2, 1)  # both described counted
    assert data["photos"]["downloaded"] == 2
    # Posts 1 and 2 (described photos) appear as photo posts; post 3 (pending) does not.
    sample_notes = {s["url"] for s in data["photos"]["samples"]}
    assert "https://x.com/alice/status/1" in sample_notes
    assert "https://x.com/alice/status/2" in sample_notes


def test_collect_thumbnails_includes_described_and_carries_description(tmp_path):
    described = _item("1", media=[_described_photo("1/0.png", description="Diagrama de flujo.")])
    decorative = _item("2", media=[_described_photo("2/0.png", decorative=True)])
    for sub in ("1", "2"):
        (tmp_path / sub).mkdir()
        Image.new("RGB", (3, 3), "blue").save(tmp_path / sub / "0.png")
    thumbs = collect_thumbnails([described, decorative], tmp_path, {})
    assert len(thumbs) == 2  # described photos ARE thumbnailed (previously excluded)
    by_handle = {t["url"]: t for t in thumbs}
    assert by_handle["https://x.com/alice/status/1"]["desc"] == "Diagrama de flujo."
    assert by_handle["https://x.com/alice/status/2"]["desc"] == ""  # decorative → no caption


def test_collect_thumbnails_includes_article_images(tmp_path):
    article = _item(
        "1",
        content=Content(
            fetched_at=DT,
            sources=[
                ContentSourceSuccess(
                    kind="x_article",
                    url="https://x.com/i/article/1",
                    text="body",
                    blocks=[
                        ArticleTextBlock(text="body"),
                        ArticleImageBlock(
                            media=_described_photo(
                                "1/article/0.png", description="Diagrama inline."
                            )
                        ),
                    ],
                )
            ],
        ),
    )
    (tmp_path / "1" / "article").mkdir(parents=True)
    Image.new("RGB", (3, 3), "green").save(tmp_path / "1" / "article" / "0.png")

    thumbs = collect_thumbnails([article], tmp_path, {"1": "/v/1.md"})

    assert len(thumbs) == 1
    assert thumbs[0]["kind"] == "article"
    assert thumbs[0]["desc"] == "Diagrama inline."
    data = compute_dashboard_data([article], {}, {"1": "/v/1.md"}, thumbs, "x")
    assert data["photos"]["article_downloaded"] == 1
    assert data["photos"]["samples"][0]["url"] == "https://x.com/alice/status/1"


def test_generate_writes_dashboard_with_valid_blob_and_links_it(tmp_path):
    from xbrain.generate import generate

    store = {"1": _item("1"), "2": _item("2", "own_tweet", "claude-code", "vgonpa")}
    generate(store, tmp_path, output_language="Spanish")

    dashboard = tmp_path / "dashboard.html"
    assert dashboard.exists()
    # The index links the dashboard by absolute file:// URI so Obsidian opens it
    # in the browser (a relative .html link is unreliable there).
    index_text = (tmp_path / "_index.md").read_text(encoding="utf-8")
    assert f"]({dashboard.resolve().as_uri()})" in index_text
    assert "file://" in index_text and index_text.count("dashboard.html") >= 1
    # The full store→compute→render→file path emits parseable JSON with right KPIs.
    text = dashboard.read_text(encoding="utf-8")
    blob = text.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    data = json.loads(blob)
    assert data["meta"]["total"] == 2
    assert data["meta"]["bookmarks"] == 1 and data["meta"]["own"] == 1


def test_dashboard_runtime_workspace_drawer_and_chart_actions(tmp_path):
    from playwright.sync_api import sync_playwright

    items = [
        _item(
            "1",
            content=Content(
                fetched_at=DT,
                sources=[
                    ContentSourceSuccess(
                        kind="x_article",
                        url="https://x.com/i/article/1",
                        text="body",
                        blocks=[
                            ArticleTextBlock(text="body"),
                            ArticleImageBlock(
                                media=_described_photo(
                                    "1/article/0.png", description="Diagrama <clave> & seguro."
                                )
                            ),
                        ],
                    )
                ],
            ),
        ),
        _item("2", media=[MediaVideoPending(url="https://v")]),
    ]
    (tmp_path / "_media" / "1" / "article").mkdir(parents=True)
    Image.new("RGB", (3, 3), "green").save(tmp_path / "_media" / "1" / "article" / "0.png")
    data = compute_dashboard_data(items, {}, {"1": "/v/1.md", "2": "/v/2.md"}, [], "x")
    dashboard = tmp_path / "dashboard.html"
    dashboard.write_text(render_dashboard_html(data), encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 700})
        page.goto(dashboard.resolve().as_uri() + "#ask", wait_until="networkidle")
        assert page.locator("#chat-view").evaluate("el => el.classList.contains('open')")
        assert page.locator("#chat-view").get_attribute("aria-hidden") == "false"
        assert page.locator("#dashboard-view").evaluate("el => el.classList.contains('on')")
        assert page.locator("[aria-current='page']").evaluate("el => el.id") == "nav-library"
        assert page.locator("#drawer").evaluate("el => el.inert === true")

        page.keyboard.press("Escape")
        page.click("#nav-analytics")
        assert (
            page.locator(".chart-actions").first.evaluate("el => getComputedStyle(el).display")
            == "flex"
        )
        page.fill("#filter-date-from", "2026-01-01")
        page.fill("#filter-date-to", "2026-12-31")
        latest_month = page.locator("[data-action='latest-month']").first
        page.click("[data-action='latest-month']")
        page.wait_for_timeout(120)
        assert page.locator("#filter-date-from").input_value() == "2026-06-01"
        assert page.locator("#filter-date-to").input_value() == "2026-06-30"
        assert latest_month.get_attribute("aria-pressed") == "true"
        assert latest_month.text_content() == "Undo latest month"
        assert 32 <= latest_month.bounding_box()["height"] < 44
        latest_month.click()
        page.wait_for_timeout(120)
        assert page.locator("#filter-date-from").input_value() == "2026-01-01"
        assert page.locator("#filter-date-to").input_value() == "2026-12-31"
        assert latest_month.get_attribute("aria-pressed") == "false"
        assert latest_month.text_content() == "Latest month"

        page.click("#nav-library")
        page.click('[data-library-id="1"]')
        page.wait_for_timeout(120)
        assert page.locator("#drawer").evaluate("el => el.getAttribute('aria-hidden')") == "false"
        assert page.locator("#drawer").evaluate("el => el.inert === false")
        assert page.evaluate("document.activeElement.id") == "dwclose"
        assert page.locator("#drawer img").count() == 0
        assert page.locator("#drawer").get_by_text("Article images").count() == 0
        assert "Diagrama <clave> & seguro." not in page.locator("#drawer").inner_text()
        drawer_actions = page.locator("#drawer .drawer-actions .drawer-action")
        assert drawer_actions.count() == 4
        assert (
            max(
                drawer_actions.evaluate_all(
                    "actions => actions.map(action => action.getBoundingClientRect().height)"
                )
            )
            <= 32
        )
        original_action = page.locator("#drawer .drawer-action", has_text="Open original")
        assert original_action.get_attribute("href") == "https://x.com/alice/status/1"
        assert original_action.get_attribute("target") == "_blank"
        assert original_action.locator(".external-mark").text_content() == "↗"

        page.keyboard.press("Escape")
        page.wait_for_timeout(120)
        assert page.locator("#drawer").evaluate("el => el.getAttribute('aria-hidden')") == "true"
        assert page.locator("#drawer").evaluate("el => el.inert === true")

        page.click('[data-library-id="1"]')
        page.click('[data-item-atlas="1"]')
        page.wait_for_timeout(180)
        assert page.evaluate("location.hash") == "#atlas"
        assert page.locator("#graph-focusbar").is_visible()
        assert page.locator("#graph-focus-label").inner_text().startswith("Document ·")
        assert "Document ·" in page.locator("#graph-meta").text_content()
        page.keyboard.press("Escape")
        page.click("#graph-clear-focus")
        assert page.locator("#graph-focusbar").is_hidden()
        assert "Document ·" not in page.locator("#graph-meta").text_content()

        page.evaluate("openGraphTopic('ai-coding')")
        page.click('[data-graph-focus="ai-coding"]')
        assert page.locator("#graph-focusbar").is_visible()
        assert page.locator("#graph-focus-label").inner_text() == "Topic · AI Coding"
        page.select_option("#filter-content", "video")
        assert page.locator("#filter-content").input_value() == "video"
        page.click("#graph-clear-all")
        assert page.locator("#graph-focusbar").is_hidden()
        assert page.locator("#filter-content").input_value() == "all"

        page.click("#nav-analytics")
        page.click("[data-signal='readiness']")
        page.wait_for_timeout(120)
        assert page.evaluate("location.hash") == "#analytics"
        browser.close()


def test_dashboard_runtime_analytics_uses_selected_date_basis(tmp_path):
    from playwright.sync_api import sync_playwright

    items = [
        _item(
            "1",
            created=datetime(2026, 4, 12, tzinfo=timezone.utc),
            captured=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _item(
            "2",
            created=datetime(2026, 5, 20, tzinfo=timezone.utc),
            captured=datetime(2026, 6, 15, tzinfo=timezone.utc),
        ),
    ]
    data = compute_dashboard_data(items, {}, {}, [], "x")
    dashboard = tmp_path / "dashboard.html"
    dashboard.write_text(render_dashboard_html(data), encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 800})
        page.goto(dashboard.resolve().as_uri() + "#analytics", wait_until="networkidle")

        rows = page.locator("#table-activity tbody tr").evaluate_all(
            "rows => rows.map(row => [...row.cells].map(cell => cell.textContent))"
        )
        assert rows == [["June 2026", "2"]]
        assert page.locator("#analytics-activity-title").text_content() == ("Monthly acquisitions")

        page.select_option("#filter-date-field", "published_date")
        rows = page.locator("#table-activity tbody tr").evaluate_all(
            "rows => rows.map(row => [...row.cells].map(cell => cell.textContent))"
        )
        assert rows == [["April 2026", "1"], ["May 2026", "1"]]
        assert page.locator("#analytics-growth-title").text_content() == (
            "Published corpus over time"
        )
        assert page.locator("#analytics-activity-title").text_content() == ("Monthly publications")
        assert page.locator("#analytics-activity-table-label").text_content() == (
            "View monthly publications as a table"
        )
        assert (
            page.locator("#c-activity")
            .get_attribute("aria-label")
            .startswith("Monthly publications.")
        )
        assert page.locator("#kpis .kpi").nth(1).locator(".lab").text_content() == (
            "Published · 30 days"
        )

        latest_month = page.locator("[data-action='latest-month']").first
        assert "publication month" in latest_month.get_attribute("title")
        latest_month.click()
        assert page.locator("#filter-date-field").input_value() == "published_date"
        assert page.locator("#filter-date-from").input_value() == "2026-05-01"
        assert page.locator("#filter-date-to").input_value() == "2026-05-31"
        assert latest_month.get_attribute("aria-pressed") == "true"

        latest_month.click()
        assert page.locator("#filter-date-field").input_value() == "published_date"
        assert page.locator("#filter-date-from").input_value() == ""
        assert page.locator("#filter-date-to").input_value() == ""
        browser.close()


def test_dashboard_runtime_shared_filters_documents_and_mobile_accessibility(tmp_path):
    from playwright.sync_api import sync_playwright

    items = [
        _item("1", topic="ai-coding", confidence="high"),
        _item(
            "2",
            topic="learning",
            confidence="low",
            media=[MediaVideoPending(url="https://v")],
        ),
    ]
    data = compute_dashboard_data(items, {}, {"1": "/v/1.md", "2": "/v/2.md"}, [], "x")
    dashboard = tmp_path / "dashboard.html"
    dashboard.write_text(render_dashboard_html(data), encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(dashboard.resolve().as_uri() + "#library", wait_until="networkidle")

        assert page.locator(".library-table tbody tr").count() == 2
        page.click("#filter-more")
        page.select_option("#filter-content", "video")
        assert page.locator(".library-table tbody tr").count() == 1
        page.click("#nav-atlas")
        page.locator("#atlas-tab-map").focus()
        page.keyboard.press("ArrowRight")
        assert page.locator("#atlas-tab-timeline").get_attribute("aria-selected") == "true"
        page.keyboard.press("End")
        assert page.locator("#atlas-tab-list").get_attribute("aria-selected") == "true"
        page.locator("#graph-list [data-graph-item]").wait_for()
        assert page.locator("#graph-list [data-graph-item]").count() == 1
        page.click("#nav-analytics")
        assert page.locator("#kpis .kpi").first.locator(".num").inner_text() == "1"
        assert page.locator("#c-growth").get_attribute("role") == "img"
        assert page.locator("#table-growth").count() == 1
        assert page.evaluate("document.documentElement.scrollWidth") == 390
        assert (
            page.evaluate(
                "Math.min(...[...document.querySelectorAll('button')].filter(e=>e.offsetParent).map(e=>e.getBoundingClientRect().height))"
            )
            >= 44
        )
        assert errors == []
        browser.close()


def test_dashboard_runtime_reduced_motion_disables_topics_animation(tmp_path):
    from playwright.sync_api import sync_playwright

    items = [_item("1"), _item("2", topic="agentic-engineering")]
    data = compute_dashboard_data(items, {}, {}, [], "x")
    dashboard = tmp_path / "dashboard.html"
    dashboard.write_text(render_dashboard_html(data), encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 700})
        page.emulate_media(reduced_motion="reduce")
        page.goto(dashboard.resolve().as_uri() + "#atlas", wait_until="networkidle")
        assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
        assert (
            page.evaluate(
                "echarts.getInstanceByDom(document.getElementById('c-topics')).getOption().animation"
            )
            is False
        )
        browser.close()

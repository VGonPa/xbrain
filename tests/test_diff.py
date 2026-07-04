# tests/test_diff.py
"""Tests for `xbrain.diff` — pure module + CLI integration.

The module-level tests do not hit `xbrain.snapshot`: they pass plain data
directories. That mirrors the production contract — `diff_snapshots` does not
know what a snapshot is. The CLI tests round-trip through `snapshot_create`
to exercise the verb the user actually types.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from xbrain.cli import app
from xbrain.diff import (
    DiffReport,
    _tf_cosine,
    diff_snapshots,
    format_json,
    format_text,
)
from xbrain.models import Author, Enrichment, Item, Topic, TopicPage
from xbrain.rubrics import save_vocab
from xbrain.snapshot import snapshot_create
from xbrain.store import save_store, save_topic_pages

runner = CliRunner()


def _item(
    item_id: str,
    *,
    primary: str | None = None,
    extra_topics: list[str] | None = None,
) -> Item:
    """Build an Item, optionally with an enrichment record."""
    item = Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    if primary is not None:
        topics = [primary]
        if extra_topics:
            topics.extend(extra_topics)
        item.enriched = Enrichment(
            enriched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            executor="api",
            summary="s",
            primary_topic=primary,
            topics=topics,
        )
    return item


def _seed(
    data_dir: Path,
    *,
    items: dict[str, Item] | None = None,
    vocab_slugs: list[str] | None = None,
    pages: dict[str, TopicPage] | None = None,
) -> None:
    """Populate a data dir with whatever subset of artifacts a test needs."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if items is not None:
        save_store(items, data_dir / "items.json")
    if vocab_slugs is not None:
        save_vocab(
            [Topic(slug=slug, description=f"desc for {slug}") for slug in vocab_slugs],
            data_dir / "vocab.yaml",
        )
    if pages is not None:
        save_topic_pages(pages, data_dir / "topics.json")


def _page(slug: str, overview: str, count: int = 5) -> TopicPage:
    return TopicPage(
        slug=slug,
        overview=overview,
        notes=[],
        synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        post_count_at_synth=count,
    )


# --------------------------------------------------------------------- TF-IDF cosine


def test_tfidf_identical_strings_return_one() -> None:
    text = "the quick brown fox jumps"
    assert _tf_cosine(text, text) == pytest.approx(1.0)


def test_tfidf_empty_or_singleton_returns_zero() -> None:
    assert _tf_cosine("", "anything") == 0.0
    assert _tf_cosine("text", "") == 0.0
    # Single-character tokens get filtered (length >= 2 floor)
    assert _tf_cosine("a a a", "b b b") == 0.0


def test_tfidf_disjoint_vocabularies_return_zero() -> None:
    assert _tf_cosine("alpha bravo charlie", "xenon yankee zulu") == 0.0


def test_tfidf_partial_overlap_is_between_zero_and_one() -> None:
    text_a = "the agentic workflow runs locally with full traces"
    text_b = "the agentic workflow runs locally with full visibility"
    similarity = _tf_cosine(text_a, text_b)
    assert 0.5 < similarity < 1.0


def test_tfidf_accented_tokens_survive_lowercasing() -> None:
    # The tokenizer keeps a-zà-ÿ to handle Spanish/French overviews. An
    # accented text compared to itself must score 1.0 — proves we didn't
    # silently strip the accented chars to nothing.
    text = "el árbol creció con vigor"
    assert _tf_cosine(text, text) == pytest.approx(1.0)


# ----------------------------------------------------------------- diff_snapshots end-to-end


def test_identical_snapshots_produce_empty_diff(tmp_path: Path) -> None:
    data = tmp_path / "data"
    items = {"1": _item("1", primary="ai-coding"), "2": _item("2", primary="misc")}
    _seed(data, items=items, vocab_slugs=["ai-coding", "misc"])

    report = diff_snapshots(data, data)

    assert report.summary.reassigned == 0
    assert report.summary.reassigned_pct == 0.0
    assert report.items.top_transitions == []
    assert report.vocab.added == []
    assert report.vocab.removed == []
    for change in report.topics.per_slug.values():
        assert change.added == 0
        assert change.removed == 0


def test_reassigned_primary_topic_counts_correctly(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(
        a,
        items={"1": _item("1", primary="ai-coding"), "2": _item("2", primary="misc")},
        vocab_slugs=["ai-coding", "misc"],
    )
    _seed(
        b,
        items={
            # Item 1 reassigned ai-coding -> software-engineering
            "1": _item("1", primary="software-engineering"),
            # Item 2 unchanged
            "2": _item("2", primary="misc"),
        },
        vocab_slugs=["ai-coding", "misc", "software-engineering"],
    )

    report = diff_snapshots(a, b)

    assert report.summary.reassigned == 1
    # Only one transition row, with the right pair
    assert len(report.items.top_transitions) == 1
    transition = report.items.top_transitions[0]
    assert transition.from_topic == "ai-coding"
    assert transition.to_topic == "software-engineering"
    assert transition.count == 1


def test_items_only_in_one_snapshot_do_not_count_as_reassigned(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={"1": _item("1", primary="ai-coding")}, vocab_slugs=["ai-coding"])
    _seed(
        b,
        items={
            "1": _item("1", primary="ai-coding"),
            "2": _item("2", primary="misc"),  # new in B
        },
        vocab_slugs=["ai-coding", "misc"],
    )

    report = diff_snapshots(a, b)

    assert report.summary.reassigned == 0
    # Item 2's membership shows up in the misc topic
    assert report.topics.per_slug["misc"].members_b == 1
    assert report.topics.per_slug["misc"].added == 1


def test_unenriched_to_enriched_is_not_a_reassignment(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={"1": _item("1")}, vocab_slugs=["misc"])  # unenriched in A
    _seed(b, items={"1": _item("1", primary="misc")}, vocab_slugs=["misc"])

    report = diff_snapshots(a, b)

    assert report.summary.reassigned == 0
    # But the transition row IS surfaced so the user sees the move
    transitions = report.items.top_transitions
    assert any(t.from_topic is None and t.to_topic == "misc" for t in transitions)


def test_topic_growth_flag_triggers_above_threshold(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    # 10 members in A, 12 in B → growth = 20% → flagged (above 10% threshold, base >= 5)
    items_a = {str(i): _item(str(i), primary="ai-coding") for i in range(10)}
    items_b = {str(i): _item(str(i), primary="ai-coding") for i in range(12)}
    _seed(a, items=items_a, vocab_slugs=["ai-coding"])
    _seed(b, items=items_b, vocab_slugs=["ai-coding"])

    report = diff_snapshots(a, b)

    change = report.topics.per_slug["ai-coding"]
    assert change.flagged_growth is True
    assert change.growth_pct == pytest.approx(0.2)


def test_topic_growth_flag_respects_5_member_floor(tmp_path: Path) -> None:
    """A 3->5 jump is 67% growth but on a tiny base — should not flag."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    items_a = {str(i): _item(str(i), primary="misc") for i in range(3)}
    items_b = {str(i): _item(str(i), primary="misc") for i in range(5)}
    _seed(a, items=items_a, vocab_slugs=["misc"])
    _seed(b, items=items_b, vocab_slugs=["misc"])

    change = diff_snapshots(a, b).topics.per_slug["misc"]
    assert change.flagged_growth is False
    assert change.growth_pct == pytest.approx(2 / 3)


def test_growth_pct_is_none_when_members_a_is_zero(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={}, vocab_slugs=["ai-coding"])
    _seed(b, items={"1": _item("1", primary="ai-coding")}, vocab_slugs=["ai-coding"])

    change = diff_snapshots(a, b).topics.per_slug["ai-coding"]
    assert change.growth_pct is None
    assert change.flagged_growth is False


def test_overview_similarity_identical_flags_identical(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    overview = "The arc from autocomplete to agent orchestration across the year."
    _seed(
        a,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        pages={"ai-coding": _page("ai-coding", overview)},
    )
    _seed(
        b,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        pages={"ai-coding": _page("ai-coding", overview)},
    )

    change = diff_snapshots(a, b).topics.per_slug["ai-coding"]
    assert change.overview_flag == "identical"
    assert change.overview_similarity == pytest.approx(1.0)


def test_overview_similarity_missing_one_side_is_not_comparable(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(
        a,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        pages={"ai-coding": _page("ai-coding", "Overview present in A.")},
    )
    _seed(
        b,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        # no topics.json on B
    )

    change = diff_snapshots(a, b).topics.per_slug["ai-coding"]
    assert change.overview_flag == "not_comparable"
    assert change.overview_similarity is None


def test_overview_similarity_disjoint_flags_different(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(
        a,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        pages={"ai-coding": _page("ai-coding", "alpha bravo charlie delta echo")},
    )
    _seed(
        b,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        pages={"ai-coding": _page("ai-coding", "xenon yankee zulu omega tango")},
    )

    change = diff_snapshots(a, b).topics.per_slug["ai-coding"]
    assert change.overview_flag == "different"
    assert change.overview_similarity == pytest.approx(0.0)


def test_overview_similarity_similar_text_above_threshold(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(
        a,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        pages={
            "ai-coding": _page(
                "ai-coding",
                "the agentic workflow runs locally with full traces",
            )
        },
    )
    _seed(
        b,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        pages={
            "ai-coding": _page(
                "ai-coding",
                "the agentic workflow runs locally with full visibility",
            )
        },
    )

    change = diff_snapshots(a, b).topics.per_slug["ai-coding"]
    assert change.overview_flag in ("similar", "identical")
    assert change.overview_similarity is not None and change.overview_similarity > 0.5


def test_vocab_added_and_removed_set_correctly(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={}, vocab_slugs=["alpha", "bravo", "charlie"])
    _seed(b, items={}, vocab_slugs=["bravo", "charlie", "delta"])

    report = diff_snapshots(a, b)

    assert report.vocab.added == ["delta"]
    assert report.vocab.removed == ["alpha"]
    assert report.vocab.unchanged_count == 2


def test_missing_vocab_yaml_on_one_side_does_not_crash(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={}, vocab_slugs=["alpha", "bravo"])
    _seed(b, items={})  # no vocab.yaml on B at all

    report = diff_snapshots(a, b)

    assert report.vocab.added == []
    assert sorted(report.vocab.removed) == ["alpha", "bravo"]


def test_missing_topics_json_on_one_side_does_not_crash(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(
        a,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
        pages={"ai-coding": _page("ai-coding", "Overview text.")},
    )
    _seed(
        b,
        items={"1": _item("1", primary="ai-coding")},
        vocab_slugs=["ai-coding"],
    )

    change = diff_snapshots(a, b).topics.per_slug["ai-coding"]
    assert change.overview_flag == "not_comparable"


def test_top_transitions_sorted_by_count_then_pair(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    # 3 items: 2x ai-coding -> software, 1x misc -> software
    _seed(
        a,
        items={
            "1": _item("1", primary="ai-coding"),
            "2": _item("2", primary="ai-coding"),
            "3": _item("3", primary="misc"),
        },
        vocab_slugs=["ai-coding", "misc", "software"],
    )
    _seed(
        b,
        items={
            "1": _item("1", primary="software"),
            "2": _item("2", primary="software"),
            "3": _item("3", primary="software"),
        },
        vocab_slugs=["ai-coding", "misc", "software"],
    )

    transitions = diff_snapshots(a, b).items.top_transitions
    # The first one is the most frequent
    assert transitions[0].from_topic == "ai-coding"
    assert transitions[0].count == 2
    assert transitions[1].from_topic == "misc"
    assert transitions[1].count == 1


def test_format_text_renders_section_headers(tmp_path: Path) -> None:
    a = tmp_path / "a"
    _seed(a, items={"1": _item("1", primary="misc")}, vocab_slugs=["misc"])
    text = format_text(diff_snapshots(a, a))
    assert "ITEMS" in text
    assert "TOPICS" in text
    assert "OVERVIEWS" in text
    assert "VOCAB" in text


def test_format_json_round_trips_through_model(tmp_path: Path) -> None:
    a = tmp_path / "a"
    _seed(a, items={"1": _item("1", primary="misc")}, vocab_slugs=["misc"])
    text = format_json(diff_snapshots(a, a))
    parsed = json.loads(text)
    # Top-level keys match the model
    assert set(parsed.keys()) == {"summary", "items", "topics", "vocab", "media"}
    # And the model can re-validate its own JSON
    DiffReport.model_validate_json(text)


# ----------------------------------------------------------------------- media diff


def _item_with_media(item_id: str, media_entries: list) -> Item:
    """Construct an Item with the given media entries (no enrichment, no links)."""
    item = Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    item.media = media_entries
    return item


def test_diff_reports_media_state_counts(tmp_path: Path) -> None:
    """`MediaDiff` carries the four-variant counts on both sides + deltas.

    Scenario: A has 1 pending photo. B has 1 downloaded photo for the same
    item. The transition counts as `delta_downloaded=+1, delta_pending=-1`.
    """
    from xbrain.models import MediaPhotoDownloaded, MediaPhotoPending

    a = tmp_path / "a"
    b = tmp_path / "b"
    item_a = _item_with_media("1", [MediaPhotoPending(url="https://pbs.twimg.com/media/A.png")])
    item_b = _item_with_media(
        "1",
        [
            MediaPhotoDownloaded(
                url="https://pbs.twimg.com/media/A.png",
                local_path="1/0.png",
                width=10,
                height=8,
                bytes_size=100,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            )
        ],
    )
    _seed(a, items={"1": item_a}, vocab_slugs=[])
    _seed(b, items={"1": item_b}, vocab_slugs=[])

    report = diff_snapshots(a, b)
    assert report.media.a.pending == 1
    assert report.media.a.downloaded == 0
    assert report.media.b.pending == 0
    assert report.media.b.downloaded == 1
    assert report.media.delta_downloaded == 1
    assert report.media.delta_pending == -1
    # Surfaces in the summary line too — quick-glance counters.
    assert report.summary.media_delta_downloaded == 1


def test_diff_media_zero_when_no_media(tmp_path: Path) -> None:
    """A diff between two stores without media reports zero everywhere."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={"1": _item("1", primary="misc")}, vocab_slugs=["misc"])
    _seed(b, items={"1": _item("1", primary="misc")}, vocab_slugs=["misc"])

    report = diff_snapshots(a, b)
    assert report.media.a.downloaded == 0
    assert report.media.b.downloaded == 0
    assert report.media.delta_downloaded == 0


def test_diff_media_reports_delta_failed(tmp_path: Path) -> None:
    """The diff surfaces a `delta_failed` count between snapshots.

    Setup: snapshot A has 1 failed photo, snapshot B has 3 failed photos
    (e.g. a re-run hit more 4xx URLs). The delta is +2, surfaced both on
    `report.media.delta_failed` and on `summary.media_delta_failed`.
    """
    from xbrain.models import MediaPhotoFailed

    def _failed(url: str) -> MediaPhotoFailed:
        return MediaPhotoFailed(
            url=url,
            failure_reason="http_4xx",
            attempts=1,
            last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        )

    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(
        a,
        items={
            "1": _item_with_media("1", [_failed("https://pbs.twimg.com/media/A1.png")]),
        },
        vocab_slugs=[],
    )
    _seed(
        b,
        items={
            "1": _item_with_media(
                "1",
                [
                    _failed("https://pbs.twimg.com/media/A1.png"),
                    _failed("https://pbs.twimg.com/media/A2.png"),
                ],
            ),
            "2": _item_with_media("2", [_failed("https://pbs.twimg.com/media/B1.png")]),
        },
        vocab_slugs=[],
    )

    report = diff_snapshots(a, b)
    assert report.media.a.failed == 1
    assert report.media.b.failed == 3
    assert report.media.delta_failed == 2
    assert report.summary.media_delta_failed == 2


def test_diff_media_counts_video_pending_separately(tmp_path: Path) -> None:
    """Video-pending entries don't bleed into the photo counters."""
    from xbrain.models import MediaVideoPending

    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={"1": _item_with_media("1", [])}, vocab_slugs=[])
    _seed(
        b,
        items={
            "1": _item_with_media("1", [MediaVideoPending(url="https://video.twimg.com/x.mp4")])
        },
        vocab_slugs=[],
    )

    report = diff_snapshots(a, b)
    assert report.media.b.video_pending == 1
    assert report.media.b.downloaded == 0
    assert report.media.delta_video_pending == 1


def test_diff_media_counts_video_downloaded_and_failed(tmp_path: Path) -> None:
    """`xbrain download-videos` advances video_pending → video_downloaded /
    video_failed; the diff surfaces those as their own counters."""
    from xbrain.models import MediaVideoDownloaded, MediaVideoFailed, MediaVideoPending

    a = tmp_path / "a"
    b = tmp_path / "b"
    item_a = _item_with_media(
        "1",
        [
            MediaVideoPending(url="https://video.twimg.com/x.mp4"),
            MediaVideoPending(url="https://video.twimg.com/y.mp4"),
        ],
    )
    item_b = _item_with_media(
        "1",
        [
            MediaVideoDownloaded(
                url="https://video.twimg.com/x.mp4",
                local_path="1/0.mp4",
                bytes_size=100,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            ),
            MediaVideoFailed(
                url="https://video.twimg.com/y.mp4",
                failure_reason="http_4xx",
                attempts=1,
                last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            ),
        ],
    )
    _seed(a, items={"1": item_a}, vocab_slugs=[])
    _seed(b, items={"1": item_b}, vocab_slugs=[])

    report = diff_snapshots(a, b)
    assert report.media.a.video_pending == 2
    assert report.media.b.video_downloaded == 1
    assert report.media.b.video_failed == 1
    assert report.media.delta_video_pending == -2
    assert report.media.delta_video_downloaded == 1
    assert report.media.delta_video_failed == 1
    # JSON round-trips with the new counters.
    restored = type(report).model_validate(report.model_dump(mode="json"))
    assert restored.media.b.video_downloaded == 1
    # The text block renders the new rows.
    text = format_text(report)
    assert "video_downloaded:" in text
    assert "video_failed:" in text


def test_diff_media_reports_delta_described(tmp_path: Path) -> None:
    """`xbrain describe` advances downloaded → described; the diff surfaces +N described.

    Setup: snapshot A has 1 downloaded photo. Snapshot B has the same
    URL transitioned to described (same on-disk bytes, plus the new
    description payload). The transition shows up as
    `delta_downloaded=-1, delta_described=+1`.
    """
    from xbrain.models import MediaPhotoDescribed, MediaPhotoDownloaded

    a = tmp_path / "a"
    b = tmp_path / "b"
    item_a = _item_with_media(
        "1",
        [
            MediaPhotoDownloaded(
                url="https://pbs.twimg.com/media/A.png",
                local_path="1/0.png",
                width=10,
                height=8,
                bytes_size=100,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            )
        ],
    )
    item_b = _item_with_media(
        "1",
        [
            MediaPhotoDescribed(
                url="https://pbs.twimg.com/media/A.png",
                local_path="1/0.png",
                width=10,
                height=8,
                bytes_size=100,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
                is_decorative=False,
                description="A chart of accuracy by model.",
                description_lang="English",
                description_version="v1",
                described_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
            )
        ],
    )
    _seed(a, items={"1": item_a}, vocab_slugs=[])
    _seed(b, items={"1": item_b}, vocab_slugs=[])

    report = diff_snapshots(a, b)
    assert report.media.a.downloaded == 1
    assert report.media.a.described == 0
    assert report.media.b.downloaded == 0
    assert report.media.b.described == 1
    assert report.media.delta_downloaded == -1
    assert report.media.delta_described == 1
    assert report.summary.media_delta_described == 1


def test_diff_media_text_format_includes_described_row(tmp_path: Path) -> None:
    """The text renderer emits a `described:` row alongside the others."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={"1": _item_with_media("1", [])}, vocab_slugs=[])
    _seed(b, items={"1": _item_with_media("1", [])}, vocab_slugs=[])
    text = format_text(diff_snapshots(a, b))
    assert "described:" in text


def test_diff_media_text_format_includes_block(tmp_path: Path) -> None:
    """The text renderer includes a `MEDIA` block with the four counters."""
    from xbrain.models import MediaPhotoDownloaded

    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed(a, items={"1": _item_with_media("1", [])}, vocab_slugs=[])
    _seed(
        b,
        items={
            "1": _item_with_media(
                "1",
                [
                    MediaPhotoDownloaded(
                        url="u",
                        local_path="1/0.png",
                        width=4,
                        height=3,
                        bytes_size=10,
                        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
                    )
                ],
            )
        },
        vocab_slugs=[],
    )

    text = format_text(diff_snapshots(a, b))
    assert "MEDIA" in text
    assert "downloaded:" in text
    assert "+1" in text  # delta marker


# ------------------------------------------------------------------------- CLI


def _setup_repo(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        f'vault = "{tmp_path / "vault"}"\n'
        'output_subdir = "x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n',
        encoding="utf-8",
    )
    (tmp_path / "vault").mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    return data_dir


def test_cli_diff_compares_two_named_snapshots(tmp_path: Path, monkeypatch) -> None:
    data_dir = _setup_repo(tmp_path, monkeypatch)
    _seed(data_dir, items={"1": _item("1", primary="alpha")}, vocab_slugs=["alpha"])
    snap_a, _ = snapshot_create(data_dir, command="manual", dir_label="checkpoint-a")
    # Mutate then snapshot again — the reassignment is observable
    _seed(data_dir, items={"1": _item("1", primary="bravo")}, vocab_slugs=["alpha", "bravo"])
    snap_b, _ = snapshot_create(data_dir, command="manual", dir_label="checkpoint-b")

    result = runner.invoke(app, ["diff", snap_a.name, snap_b.name])

    assert result.exit_code == 0, result.output
    assert "ITEMS" in result.stdout
    assert "alpha" in result.stdout
    assert "bravo" in result.stdout


def test_cli_diff_defaults_b_to_live_data_dir(tmp_path: Path, monkeypatch) -> None:
    data_dir = _setup_repo(tmp_path, monkeypatch)
    _seed(data_dir, items={"1": _item("1", primary="alpha")}, vocab_slugs=["alpha"])
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="before")
    # Mutate the live data/ — that is the B side without an explicit name.
    _seed(data_dir, items={"1": _item("1", primary="bravo")}, vocab_slugs=["alpha", "bravo"])

    result = runner.invoke(app, ["diff", snap.name])

    assert result.exit_code == 0, result.output
    assert "live data/" in result.stdout
    # The reassignment shows up
    assert "reassigned: 1" in result.stdout


def test_cli_diff_format_json_parses_back(tmp_path: Path, monkeypatch) -> None:
    data_dir = _setup_repo(tmp_path, monkeypatch)
    _seed(data_dir, items={"1": _item("1", primary="alpha")}, vocab_slugs=["alpha"])
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="x")

    result = runner.invoke(app, ["diff", snap.name, "--format", "json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert set(parsed.keys()) == {"summary", "items", "topics", "vocab", "media"}


def test_cli_diff_unknown_snapshot_exits_1(tmp_path: Path, monkeypatch) -> None:
    _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["diff", "does-not-exist"])
    assert result.exit_code == 1
    assert "No snapshot named" in result.output


def test_cli_diff_unknown_format_exits_1(tmp_path: Path, monkeypatch) -> None:
    data_dir = _setup_repo(tmp_path, monkeypatch)
    _seed(data_dir, items={"1": _item("1", primary="alpha")}, vocab_slugs=["alpha"])
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="x")

    result = runner.invoke(app, ["diff", snap.name, "--format", "xml"])

    assert result.exit_code == 1
    assert "format" in result.output.lower()


# --- PR #28 review fixes: input validation ---


def test_diff_snapshots_raises_when_dir_does_not_exist(tmp_path):
    """Missing snapshot dir surfaces as clean FileNotFoundError, not silent empty diff."""
    import pytest

    from xbrain.diff import diff_snapshots

    real = tmp_path / "real"
    real.mkdir()
    (real / "items.json").write_text("{}", encoding="utf-8")
    ghost = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError, match="directory not found"):
        diff_snapshots(real, ghost)


def test_diff_snapshots_raises_when_both_sides_are_empty(tmp_path):
    """Both sides empty → clean FileNotFoundError naming the dirs.

    Guards against the silent-failure case where data/ was deleted out-of-band
    and diff would otherwise report a clean 'no changes' against a real snapshot.
    """
    import pytest

    from xbrain.diff import diff_snapshots

    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()

    with pytest.raises(FileNotFoundError, match="Both diff sides are empty"):
        diff_snapshots(a, b)


def test_diff_snapshots_wraps_corrupt_file_with_context(tmp_path):
    """A corrupt items.json surfaces with the path in the error message."""
    import pytest

    from xbrain.diff import diff_snapshots

    a = tmp_path / "a"
    a.mkdir()
    (a / "items.json").write_text("not json at all", encoding="utf-8")
    b = tmp_path / "b"
    b.mkdir()
    (b / "items.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to load.*items.json"):
        diff_snapshots(a, b)

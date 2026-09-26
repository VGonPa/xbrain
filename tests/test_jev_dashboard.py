# tests/test_jev_dashboard.py
"""What `jev.html` ships to the browser, and what the page says with it.

The page has ONE job: show, post by post, where enrich's topics and Jev disagree at the
configured threshold, and what Jev cost. It recomputes nothing — every number in the blob
comes from `jev/report.py` (`build_report` for the comparison, `run_history` for the cost),
so the tests below assert WHERE each value comes from, not merely that it appears.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import typer

from xbrain.dashboard import _resource
from xbrain.jev.assess import (
    build_topic_state,
    current_pairs,
    questions_digest,
    topic_contract,
)
from xbrain.evidence import SURFACE_KEYS
from xbrain.cli import app
from xbrain.jev.dashboard import (
    ASK_COMMAND,
    DOCS_URL,
    SURFACE_LABELS,
    compute_jev_dashboard_data,
    render_jev_dashboard_html,
)
from xbrain.jev.defaults import tokens_cost_usd
from xbrain.jev.models import JevRun, PrimaryChoice, TopicAssessment
from xbrain.jev.questions import STATE_KEY, build_topic_questions
from xbrain.jev.report import build_report, run_history
from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaVideoDownloaded,
    Topic,
    VideoFrame,
)

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
#: A fixed clock. The blob carries `summary["generated_at"]`, so a function that reads the
#: clock itself cannot be asserted against — and is not the pure function it claims to be.
NOW = datetime(2026, 9, 22, 18, 30, 5, tzinfo=timezone.utc)
VOCAB = [
    Topic(slug="ai-coding", description="IA."),
    Topic(slug="startups", description="Empresas."),
]
FALLBACK = "otro"
CHAR_LIMIT = 100_000
#: The one provider `jev.defaults.INPUT_USD_PER_MTOK` prices, so costs are real numbers.
PRICED_PROVIDER = "typesafe"


def _item(item_id: str = "1", text: str = "Claude Code hooks", topics=("ai-coding",)) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text,
        created_at=DT,
        captured_at=DT,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="s",
            primary_topic=topics[0],
            topics=list(topics),
        ),
    )


def _assessment(
    item: Item,
    contract: str | None = None,
    membership: dict[str, float] | None = None,
    provider: str = PRICED_PROVIDER,
    vocab: list[Topic] | None = None,
    choice: str = "ai-coding",
    input_tokens: int | None = 2000,
    char_limit: int = CHAR_LIMIT,
) -> TopicAssessment:
    """A record whose contract is the one TODAY'S ask would stamp, so it reads as current.

    Built the way `assess.assess_topics` builds it — the state as sent and the digest of the
    questions that went with it — so a change to either composition shows up here as a stale
    fixture instead of a test passing over a contract nobody computes any more.
    """
    state, state_chars = build_topic_state(item, char_limit)
    digest = questions_digest(build_topic_questions(vocab or VOCAB, FALLBACK))
    return TopicAssessment(
        item_id=item.id,
        provider=provider,
        model="jev-1.13.0",
        asked_at=DT,
        contract=contract or topic_contract(state[STATE_KEY], digest),
        state_chars=state_chars,
        membership=membership or {"startups": 0.123456, "ai-coding": 0.9},
        primary=PrimaryChoice(
            choice=choice,
            confidence=0.77,
            probabilities={"ai-coding": 0.7, "startups": 0.2, "otro": 0.1},
        ),
        input_tokens=input_tokens,
        truncated=state_chars > char_limit,
    )


def _run(started: datetime, tokens: int = 4000, requests: int = 2) -> JevRun:
    return JevRun(
        started_at=started,
        finished_at=started + timedelta(seconds=20),
        models=["jev-1.13.0"],
        requests=requests,
        ok=requests,
        failed=0,
        input_tokens_by_provider={PRICED_PROVIDER: tokens},
        input_tokens=tokens,
        input_tokens_unknown=0,
        interrupted=False,
    )


def _data(items: list[Item], assessments: dict[str, TopicAssessment], **kwargs) -> dict[str, Any]:
    options: dict[str, Any] = {
        "threshold": 0.85,
        "fallback": FALLBACK,
        "char_limit": CHAR_LIMIT,
        "id2note": {},
        "updated": "SEP 22, 2026",
        "now": NOW,
        "runs": [],
    }
    options.update(kwargs)
    return compute_jev_dashboard_data(items, assessments, VOCAB, **options)


def _post(data: dict[str, Any], item_id: str) -> dict[str, Any]:
    [post] = [post for post in data["posts"] if post["id"] == item_id]
    return post


# --------------------------------------------------------------------------- the numbers


def test_the_summary_is_build_reports_at_the_configured_threshold():
    """The three headline numbers must equal `xbrain jev report`'s. They do by construction
    only if the blob's summary IS `build_report`'s, at the same threshold, from the same
    current pairs — so that is what is asserted, whole."""
    fresh, doubtful = _item("1"), _item("2", topics=("startups",))
    assessments = {"1": _assessment(fresh), "2": _assessment(doubtful)}

    data = _data([fresh, doubtful], assessments)

    expected, _ = build_report(
        [(fresh, assessments["1"]), (doubtful, assessments["2"])], VOCAB, 0.85, now=NOW
    )
    assert data["summary"] == expected
    assert data["threshold"] == expected["threshold"] == 0.85


# --------------------------------------------------------------------------- the post cards


def test_each_card_has_one_row_per_topic_either_side_has_with_its_verdict():
    """Enrich's topics in enrich's order, then the ones only Jev backs — each with the
    probability and the verdict read off `build_report`'s comparison, never re-derived."""
    item = _item("1", topics=("ai-coding",))
    assessment = _assessment(item, membership={"ai-coding": 0.4, "startups": 0.97})

    jev = _post(_data([item], {"1": assessment}), "1")["jev"]

    assert jev["topics"] == [
        {"slug": "ai-coding", "enrich": True, "p": 0.4, "verdict": "solo_enrich"},
        {"slug": "startups", "enrich": False, "p": 0.97, "verdict": "solo_jev"},
    ]
    assert jev["primary"] == "ai-coding" and jev["jev_primary"] == "ai-coding"
    assert jev["jev_confidence"] == 0.77
    assert jev["primary_agrees"] is True and jev["jev_fallback"] is False
    # One enrich-only topic + one Jev-only topic; the primaries agree.
    assert (jev["enrich_only"], jev["jev_only"], jev["primary_differs"]) == (1, 1, False)
    assert jev["disagreements"] == 2


def test_a_topic_both_sides_hold_is_coinciden():
    item = _item("1", topics=("ai-coding",))

    jev = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]

    assert jev["topics"] == [
        {"slug": "ai-coding", "enrich": True, "p": 0.9, "verdict": "coinciden"}
    ]
    assert jev["disagreements"] == 0


def test_the_threshold_is_the_one_handed_in_and_nothing_on_the_page_moves_it():
    item = _item(topics=("ai-coding",))

    high = _data([item], {"1": _assessment(item)}, threshold=0.95)

    # ai-coding at 0.9 is backed at 0.85 and doubtful at 0.95: the page shows what it was given.
    assert high["summary"]["threshold"] == 0.95
    assert _post(high, "1")["jev"]["topics"][0]["verdict"] == "solo_enrich"


def test_a_topic_that_left_the_vocabulary_is_neither_confirmed_nor_denied():
    """An assigned slug Jev was never asked about has no probability and its own verdict —
    a fabricated 0.0 would show as the strongest disagreement in the corpus."""
    item = _item("1", topics=("ai-coding", "web3"))

    jev = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]

    assert jev["topics"] == [
        {"slug": "ai-coding", "enrich": True, "p": 0.9, "verdict": "coinciden"},
        {"slug": "web3", "enrich": True, "p": None, "verdict": "sin_juzgar"},
    ]
    assert jev["disagreements"] == 0


def test_a_primary_jev_does_not_share_is_a_disagreement_and_the_fallback_is_named():
    item = _item("1", topics=("ai-coding",))

    jev = _post(_data([item], {"1": _assessment(item, choice="otro")}), "1")["jev"]

    assert jev["primary_agrees"] is False and jev["primary_differs"] is True
    assert jev["jev_primary"] == "otro" and jev["jev_fallback"] is True
    assert jev["disagreements"] == 1


def test_posts_are_ordered_most_disagreement_first_then_the_unevaluated():
    agree = _item("1", topics=("ai-coding",))
    one = _item("2", topics=("ai-coding",))
    two = _item("3", topics=("ai-coding",))
    never = _item("0")
    assessments = {
        "1": _assessment(agree),
        "2": _assessment(one, membership={"ai-coding": 0.4, "startups": 0.1}),
        "3": _assessment(two, membership={"ai-coding": 0.4, "startups": 0.9}),
    }

    data = _data([never, agree, one, two], assessments)

    assert [post["id"] for post in data["posts"]] == ["3", "2", "1", "0"]
    assert [post["jev"]["disagreements"] for post in data["posts"][:3]] == [2, 1, 0]


def test_posts_that_disagree_equally_are_ordered_by_id():
    """Two renders of one side-car must list tied posts in the same order."""
    items = [_item(i, topics=("ai-coding",)) for i in ("b", "c", "a")]
    assessments = {
        i.id: _assessment(i, membership={"ai-coding": 0.4, "startups": 0.1}) for i in items
    }

    data = _data(items, assessments)

    assert [post["id"] for post in data["posts"]] == ["a", "b", "c"]


def test_every_post_is_a_card_and_says_whether_jev_evaluated_it():
    """The browser shows the whole corpus: an unevaluated post, a stale answer and a post
    with nothing to compare against are cards too, each with its status and no comparison."""
    compared, stale, never, plain = _item("1"), _item("2", text="old"), _item("3"), _item("4")
    plain.enriched = None
    assessments = {
        "1": _assessment(compared),
        "2": _assessment(stale, contract="e" * 64),
        "4": _assessment(plain),
    }

    data = _data([compared, stale, never, plain], assessments)

    status = {post["id"]: post["status"] for post in data["posts"]}
    assert status == {"1": "compared", "2": "stale", "3": "unevaluated", "4": "not_enriched"}
    assert all(post["jev"] is None for post in data["posts"] if post["id"] != "1")
    # What enrich said is on the card whether or not Jev was asked.
    assert _post(data, "3")["enrich"] == {"topics": ["ai-coding"], "primary": "ai-coding"}
    assert _post(data, "4")["enrich"] is None
    # The "Sin evaluar por Jev" count: posts with no CURRENT answer (none, or a stale one).
    assert data["totals"]["unevaluated"] == 2


def test_the_disagreement_counts_are_the_ones_the_report_computes():
    """The filter counts and `jev report`'s numbers are one number each."""
    agree, differ = _item("1", topics=("ai-coding",)), _item("2", topics=("startups",))

    data = _data([agree, differ], {"1": _assessment(agree), "2": _assessment(differ)})

    summary = data["summary"]
    assert summary["posts_with_disagreement"] == 1
    assert sum(1 for p in data["posts"] if p["jev"]["disagreements"]) == 1
    assert summary["posts_primary_differs"] == sum(
        1 for p in data["posts"] if p["jev"]["primary_differs"]
    )
    assert summary["posts_jev_only"] == sum(1 for p in data["posts"] if p["jev"]["jev_only"])


def test_a_truncated_assessment_is_flagged_on_its_card():
    """The page marks the post "recortado": Jev judged a CUT post, the one to distrust."""
    item = _item()
    assessment = _assessment(item).model_copy(update={"truncated": True})

    assert _post(_data([item], {"1": assessment}), "1")["jev"]["truncated"] is True


def test_the_card_says_who_answered_and_when():
    item = _item()

    jev = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]

    assert jev["model"] == "jev-1.13.0"
    assert jev["asked_at"] == DT.isoformat()


def test_each_post_carries_what_its_stored_assessment_cost():
    """Priced by `report.assessment_cost_usd` — and `None`, not zero, when the usage is
    unknown or nobody prices the provider. The mean is over the CURRENT posts it can price,
    and says how many: a stale record (with other tokens) never enters it."""
    counted, uncounted, foreign, stale = _item("1"), _item("2"), _item("3"), _item("4", text="old")
    assessments = {
        "1": _assessment(counted, input_tokens=2500),
        "2": _assessment(uncounted, input_tokens=None),
        "3": _assessment(foreign, input_tokens=9000, provider="otro-juez"),
        "4": _assessment(stale, contract="e" * 64, input_tokens=77_000),
    }

    data = _data([counted, uncounted, foreign, stale], assessments)

    jev = {post_id: _post(data, post_id)["jev"] for post_id in ("1", "2", "3")}
    assert jev["1"]["tokens"] == 2500
    assert jev["1"]["cost_usd"] == tokens_cost_usd(2500, PRICED_PROVIDER)
    assert jev["2"]["tokens"] is None and jev["2"]["cost_usd"] is None
    assert jev["3"]["cost_usd"] is None and jev["3"]["unpriced"] is True
    assert jev["1"]["unpriced"] is False
    per_post = data["cost"]["per_post"]
    assert per_post["mean_usd"] == tokens_cost_usd(2500, PRICED_PROVIDER)
    assert (per_post["n"], per_post["of"]) == (1, 3)
    assert per_post["unpriced_providers"] == ["otro-juez"]


def test_an_item_enrich_left_without_a_primary_is_still_compared():
    """No primary is not "nothing to compare": its assigned topics still have probabilities."""
    item = _item("1", topics=("ai-coding",))
    item.enriched.primary_topic = None

    post = _post(_data([item], {"1": _assessment(item)}), "1")

    assert post["jev"]["primary"] is None and post["jev"]["primary_agrees"] is False
    assert post["jev"]["topics"][0] == {
        "slug": "ai-coding",
        "enrich": True,
        "p": 0.9,
        "verdict": "coinciden",
    }


# --------------------------------------------------------------------------- the share card


def test_the_card_carries_the_whole_post_and_who_wrote_it():
    """The full text (the page collapses a long one), not a snippet — and the links to X and
    the note, against the SOURCE, not a constant."""
    long_post = "palabra " * 400
    item = _item("1", text=long_post)

    post = _post(_data([item], {}, id2note={"1": "/v/items/1.md"}), "1")

    assert post["text"] == long_post
    assert post["author"] == {"handle": "alice", "name": "Alice"}
    assert post["url"] == item.url and post["note"] == "/v/items/1.md"
    assert post["created"] == item.created_at.isoformat()


def _photo(item_id: str, n: int, description: str | None = None):
    common = {
        "url": f"https://pbs.twimg.com/{item_id}-{n}.jpg",
        "local_path": f"{item_id}/{n}.jpg",
        "width": 10,
        "height": 10,
        "bytes_size": 3,
        "downloaded_at": DT,
    }
    if description is None:
        return MediaPhotoDownloaded(**common)
    return MediaPhotoDescribed(
        **common,
        is_decorative=False,
        description=description,
        description_lang="Spanish",
        description_version="v1",
        described_at=DT,
    )


def test_photos_are_relative_paths_into_the_vaults_media_mirror_checked_per_file(tmp_path):
    """The page sits next to `_media/`, the folder the Obsidian notes embed from: a photo is
    `_media/<local_path>`, RELATIVE, and only when that file is really there — never base64."""
    item = _item("1")
    item.media = [_photo("1", 0, description="Un gráfico."), _photo("1", 1)]
    (tmp_path / "_media" / "1").mkdir(parents=True)
    (tmp_path / "_media" / "1" / "0.jpg").write_bytes(b"jpg")

    media = _post(_data([item], {}, page_dir=tmp_path), "1")["media"]

    assert media == [
        {"type": "photo", "src": "_media/1/0.jpg", "desc": "Un gráfico."},
        {"type": "photo", "src": None, "desc": ""},
    ]
    assert (tmp_path / media[0]["src"]).is_file()


def test_a_photo_caption_is_cut_for_the_tooltip():
    """The vision caption is the photo's alt text and tooltip; at ~400 characters each over
    ~1.3k photos, shipping it whole is a sixth of the page."""
    item = _item("1")
    item.media = [_photo("1", 0, description="d" * 500)]

    [photo] = _post(_data([item], {}), "1")["media"]

    assert photo["desc"] == "d" * 279 + "…"


def test_without_a_page_dir_no_photo_path_is_promised():
    item = _item("1")
    item.media = [_photo("1", 0)]

    assert _post(_data([item], {}), "1")["media"] == [{"type": "photo", "src": None, "desc": ""}]


def test_a_video_shows_its_first_local_frame_and_is_marked_as_a_video(tmp_path):
    item = _item("1")
    item.media = [
        MediaVideoDownloaded(
            url="https://video.twimg.com/1.mp4",
            thumbnail_url="https://pbs.twimg.com/poster.jpg",
            local_path="1/video.mp4",
            bytes_size=9,
            downloaded_at=DT,
        )
    ]
    item.content = Content(
        fetched_at=DT,
        sources=[
            ContentSourceSuccess(
                kind="x_video",
                url=item.url,
                text="transcript",
                frames=[VideoFrame(timestamp=1.0, local_path="1/frames/0.jpg")],
            )
        ],
    )
    (tmp_path / "_media" / "1" / "frames").mkdir(parents=True)
    (tmp_path / "_media" / "1" / "frames" / "0.jpg").write_bytes(b"jpg")

    media = _post(_data([item], {}, page_dir=tmp_path), "1")["media"]

    # A local still, never the remote poster: the page fetches nothing.
    assert media == [{"type": "video", "src": "_media/1/frames/0.jpg", "desc": ""}]


def test_at_most_four_media_per_card():
    item = _item("1")
    item.media = [_photo("1", n) for n in range(6)]

    assert len(_post(_data([item], {}), "1")["media"]) == 4


def test_the_quoted_post_is_the_one_jev_read(tmp_path):
    """The nested card is `quoted_source` — the same quoted post the evidence carries — cut
    for the page, with who wrote it."""
    item = _item("1")
    item.content = Content(
        fetched_at=DT,
        sources=[
            ContentSourceSuccess(
                kind="quoted_tweet",
                url="https://x.com/karpathy/status/9",
                text="q" * 700,
                author=Author(handle="karpathy", name="Andrej Karpathy"),
            )
        ],
    )

    quoted = _post(_data([item], {}), "1")["quoted"]

    assert quoted == {
        "handle": "karpathy",
        "name": "Andrej Karpathy",
        "url": "https://x.com/karpathy/status/9",
        "text": "q" * 600,
        "cut": True,
    }


def test_the_link_card_is_the_fetched_article_labelled_with_its_kind():
    """An `x_article` source may hold scraped replies rather than an article: the card
    names the kind so the page does not pass it off as one."""
    item = _item("1")
    item.links = [Link(url="https://blog.example.com/post", domain="blog.example.com")]
    item.content = Content(
        fetched_at=DT,
        sources=[
            ContentSourceSuccess(
                kind="x_article",
                url="https://x.com/i/article/5",
                title="Un artículo",
                text="cuerpo",
            )
        ],
    )

    assert _post(_data([item], {}), "1")["link"] == {
        "url": "https://x.com/i/article/5",
        "domain": "x.com",
        "title": "Un artículo",
        "kind": "x_article",
    }


def test_without_a_fetched_article_the_link_card_is_the_first_link():
    item = _item("1")
    item.links = [Link(url="https://www.example.com/a", domain="example.com")]

    assert _post(_data([item], {}), "1")["link"] == {
        "url": "https://www.example.com/a",
        "domain": "example.com",
        "title": None,
        "kind": None,
    }
    assert _post(_data([_item("2")], {}), "2")["link"] is None


# --------------------------------------------------------------------------- what Jev saw


def test_what_jev_saw_is_the_state_split_into_its_surfaces():
    """From `assess.state_surfaces` — the state `build_topic_state` sent — with Spanish labels,
    each surface's size, and how much of it the cut kept."""
    item = _item("1", text="Claude Code hooks")

    jev = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]

    assert jev["surfaces"] == [
        {"key": "tweet", "label": "Tweet", "chars": 17, "kept": 17, "text": "Claude Code hooks"},
        {"key": "author", "label": "Autor", "chars": 11, "kept": 11, "text": "alice\nAlice"},
    ]
    assert jev["state_chars"] == _assessment(item).state_chars == 17 + 1 + 11


def test_a_surface_ships_at_most_600_characters_but_says_its_full_size():
    item = _item("1", text="y" * 1500)

    [tweet, _author] = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]["surfaces"]

    assert tweet["text"] == "y" * 600
    assert (tweet["chars"], tweet["kept"]) == (1500, 1500)


def test_what_the_cut_left_out_is_on_the_surface():
    """Tweet (50) + author (11) under a 53-character limit: the tweet whole, 2 of the author's
    11 characters — the figures `state_surfaces` computes for the state that was sent."""
    item = _item("1", text="x" * 50)
    assessment = _assessment(item, char_limit=53)

    jev = _post(_data([item], {"1": assessment}, char_limit=53), "1")["jev"]

    assert assessment.truncated is True and jev["truncated"] is True
    assert [(s["key"], s["chars"], s["kept"]) for s in jev["surfaces"]] == [
        ("tweet", 50, 50),
        ("author", 11, 2),
    ]


def test_the_copy_command_is_the_jev_topics_id_option_the_cli_really_has():
    """The static page's "copiar comando" completes `ASK_COMMAND` with a post id; the line
    must be one `xbrain jev topics` accepts, not a remembered spelling."""
    topics = typer.main.get_command(app).commands["jev"].commands["topics"]  # type: ignore[attr-defined]

    assert ASK_COMMAND == "xbrain jev topics --id"
    assert any("--id" in param.opts for param in topics.params)
    assert _data([_item()], {})["ask_command"] == ASK_COMMAND


def test_every_surface_the_state_can_carry_has_a_spanish_label():
    assert set(SURFACE_KEYS["topics"]) <= set(SURFACE_LABELS)
    assert SURFACE_LABELS["quoted"] == "Post citado"
    assert SURFACE_LABELS["video_transcript"] == "Transcript del vídeo"


# --------------------------------------------------------------------------- cost & runs


def test_the_cost_block_is_report_run_history_over_the_whole_side_car():
    """Total and per-pass history come from `report.run_history` — the RAW side-car, stale
    records included, because they were paid for too."""
    fresh, stale = _item("1"), _item("2", text="old")
    assessments = {"1": _assessment(fresh), "2": _assessment(stale, contract="e" * 64)}
    runs = [_run(datetime(2026, 9, 20, tzinfo=timezone.utc))]

    data = _data([fresh, stale], assessments, runs=runs)

    history = run_history(runs, assessments)
    assert {key: data["cost"][key] for key in history} == history
    assert data["cost"]["total"]["requests"] == 2


def test_assessments_outside_the_log_are_announced_not_hidden():
    """The first real run predates the log: 20 records, an empty `runs.jsonl`. The page
    must say they exist instead of showing a total that silently omits them."""
    item = _item("1")

    data = _data([item], {"1": _assessment(item)}, runs=[])

    assert data["cost"]["out_of_log"]["assessments"] == 1
    assert data["cost"]["total"]["runs"] == 0


def test_the_stored_answers_card_counts_and_prices_the_current_ones_only():
    """With a stale record, the card's number and its cost describe the same set."""
    fresh, stale = _item("1"), _item("2", text="old")
    assessments = {
        "1": _assessment(fresh, input_tokens=2000),
        "2": _assessment(stale, contract="e" * 64, input_tokens=50_000),
    }

    current = _data([fresh, stale], assessments)["cost"]["current"]

    assert current["assessments"] == 1
    assert current["input_tokens"] == 2000
    assert current["cost_usd"] == pytest.approx(tokens_cost_usd(2000, PRICED_PROVIDER))


def test_a_run_log_that_cannot_be_read_replaces_the_cost_strip_not_the_page():
    """One torn line must not cost the operator the whole disagreement table."""
    item = _item()

    data = _data([item], {"1": _assessment(item)}, runs=[], runs_error="runs.jsonl: línea 3")

    assert data["cost"]["error"] == "runs.jsonl: línea 3"
    assert "total" not in data["cost"] and "runs" not in data["cost"]
    assert [post["id"] for post in data["posts"]] == ["1"]


# --------------------------------------------------------------------------- side-car accounting


def test_stale_and_orphaned_records_are_excluded_and_counted():
    fresh, stale, gone = _item("1"), _item("2", text="old"), _item("3")

    data = _data(
        [fresh, stale],
        {
            "1": _assessment(fresh),
            "2": _assessment(stale, contract="e" * 64),
            "3": _assessment(gone),
        },
    )

    # The stale one is still a card (marked, with no comparison); the orphan has no post.
    assert [(post["id"], post["status"]) for post in data["posts"]] == [
        ("1", "compared"),
        ("2", "stale"),
    ]
    totals = data["totals"]
    assert totals["models"] == data["summary"]["models"] == {"jev-1.13.0": 1}
    assert (totals["current"], totals["stale"], totals["orphans"]) == (1, 1, 1)
    assert totals["assessed"] == totals["current"] + totals["stale"] + totals["orphans"]


def test_the_summary_carries_the_clock_it_was_handed_and_never_reads_one():
    item = _item()

    assert _data([item], {"1": _assessment(item)})["summary"]["generated_at"] == NOW.isoformat()


def test_the_caller_may_hand_in_the_currency_it_already_computed():
    fresh, stale = _item("1"), _item("2", text="old")
    assessments = {"1": _assessment(fresh), "2": _assessment(stale, contract="e" * 64)}
    current = current_pairs(
        [fresh, stale], assessments, VOCAB, fallback=FALLBACK, char_limit=CHAR_LIMIT
    )

    handed_in = _data([fresh, stale], assessments, current=current)

    assert handed_in == _data([fresh, stale], assessments)


def test_a_handed_in_currency_computed_under_another_fallback_is_refused():
    item = _item()
    assessments = {"1": _assessment(item)}
    elsewhere = current_pairs(
        [item], assessments, VOCAB, fallback="otra-cosa", char_limit=CHAR_LIMIT
    )

    with pytest.raises(ValueError, match="otra-cosa"):
        _data([item], assessments, current=elsewhere)


def test_a_handed_in_currency_computed_under_another_char_limit_is_refused():
    item = _item()
    assessments = {"1": _assessment(item)}
    elsewhere = current_pairs([item], assessments, VOCAB, fallback=FALLBACK, char_limit=17)

    with pytest.raises(ValueError, match="17"):
        _data([item], assessments, current=elsewhere)


# --------------------------------------------------------------------------- the page


def _page() -> str:
    item = _item()
    return render_jev_dashboard_html(_data([item], {"1": _assessment(item)}))


def test_the_page_is_self_contained_and_leaves_no_sentinel():
    html = _page()

    assert "/*__DATA__*/" not in html and "/*__ECHARTS__*/" not in html
    assert "const DATA = " in html
    # No charting library any more: the page is a table. Fonts are the only fetch.
    assert "echarts" not in html.lower()
    fetched = re.findall(r'<(?:script|link|img|iframe)[^>]*\s(?:src|href)="(https?://[^"]*)"', html)
    assert fetched and all(url.startswith("https://fonts.g") for url in fetched)


def test_the_page_has_no_slider_and_no_threshold_input():
    """The threshold is FIXED (`[jev].threshold`). A control that moves it would show numbers
    `xbrain jev report` does not print."""
    template = _resource("jev.template.html")

    assert 'type="range"' not in template
    assert 'id="threshold"' not in template
    assert "derive(" not in template and "selfCheck" not in template
    # The only input is the search box; sort is a select; filters are buttons.
    inputs = re.findall(r"<input[^>]*>", template)
    assert len(inputs) == 1 and 'type="search"' in inputs[0]
    assert '<select id="sort"' in template


def test_the_word_noul_appears_nowhere_on_the_page():
    """Vendor jargon: the reader sees "probabilidad"."""
    template = _resource("jev.template.html")

    assert "noul" not in template.lower()
    assert "probabilidad" in template


def test_the_page_names_the_threshold_as_config_and_links_the_docs():
    template = _resource("jev.template.html")

    assert "(config)" in template
    assert "DATA.threshold" in template
    assert "DATA.docs_url" in template
    assert _data([_item()], {"1": _assessment(_item())})["docs_url"] == DOCS_URL
    assert DOCS_URL.startswith("https://github.com/VGonPa/xbrain/") and DOCS_URL.endswith(
        "docs/jev.md"
    )


def test_the_page_shows_the_three_numbers_from_the_summary_keys():
    """The headline reads `summary` fields by name — the ones `jev report` prints."""
    template = _resource("jev.template.html")

    for key in (
        "assigned_backed",
        "assigned_pairs",
        "enrich_backed_pct",
        "missing_pairs",
        "primary_agree_pct",
    ):
        assert f"S.{key}" in template, key


def test_the_page_says_what_it_is_for():
    template = _resource("jev.template.html")

    assert "Qué es esto" in template
    assert "Coste y peticiones" in template
    assert "Histórico por pasada" in template
    # The pre-log line: "N evaluaciones anteriores" + " al registro de pasadas".
    assert "fuera del registro de pasadas" in template


def test_scraped_text_cannot_close_the_script_tag_or_break_the_parse():
    vocab = [
        Topic(slug="ai-coding", description="IA. Fin.</script><img src=x>"),
        Topic(slug="startups", description="Empresas."),
    ]
    item = _item("1", text="cierra aquí </script><img src=x onerror=alert(1)>")
    data = compute_jev_dashboard_data(
        [item],
        {"1": _assessment(item, vocab=vocab)},
        vocab,
        threshold=0.85,
        fallback=FALLBACK,
        char_limit=CHAR_LIMIT,
        id2note={},
        updated="SEP 22, 2026",
        now=NOW,
        runs=[],
    )

    html = render_jev_dashboard_html(data)

    assert html.count("</script>") == 1  # the page's own script tag, and no other
    assert " " not in html
    assert "<img src=x" not in html
    assert json.loads(json.dumps(data))["posts"][0]["text"].startswith("cierra aquí </script>")


def test_the_boot_guard_is_registered_before_anything_that_can_throw():
    """A guard installed after the work it guards is not a guard: an uncaught throw at script
    evaluation would leave the header and an empty table, which reads as "nothing to fix"."""
    template = _resource("jev.template.html")
    guard = template.index("addEventListener('error'")

    for later in ("const DATA = ", "function boot("):
        assert guard < template.index(later), later
    assert template.count("boot();") == 1
    assert "<noscript>" in template


def _script_section(template: str, start: str, end: str) -> str:
    return template[template.index(start) : template.index(end)]


def test_the_page_has_four_hash_routed_tabs_and_fills_only_posts():
    """#posts (default), #topics, #compare, #config — the shell for PRs 9b–9d, whose tabs
    say they are coming instead of showing invented content."""
    template = _resource("jev.template.html")

    for tab in ("posts", "topics", "compare", "config"):
        assert f'href="#{tab}"' in template, tab
        assert f'id="tab-{tab}"' in template, tab
    assert template.count("llega en el siguiente PR") == 3
    assert "addEventListener('hashchange'" in template


def test_the_posts_view_lives_in_the_hash_so_it_can_be_bookmarked():
    template = _resource("jev.template.html")
    hash_code = _script_section(template, "function readHash(", "function writeHash(")
    write_code = _script_section(template, "function writeHash(", "/* end hash */")

    for key in ("f", "t", "q", "s"):
        assert f"get('{key}')" in hash_code, key
        assert f"'{key}'" in write_code, key
    # The default filter, when the hash names none, is "Con discrepancias".
    assert "|| 'disc'" in hash_code


def test_the_filter_rail_counts_come_from_the_report_not_from_the_page():
    """Each virtual filter's number is a `summary`/`totals` key; each topic's three numbers
    are its `per_topic` row. The page counts nothing it then presents as a report number."""
    template = _resource("jev.template.html")
    rail = _script_section(template, "function renderRail(", "/* end rail */")

    for key in (
        "T.items",
        "S.posts_with_disagreement",
        "T.unevaluated",
        "S.posts_jev_only",
        "S.posts_primary_differs",
        "S.primary_fallback",
        ".assigned",
        ".backed",
        ".disagreeing",
    ):
        assert key in rail, key
    for label in (
        "Todos",
        "Con discrepancias",
        "Sin evaluar por Jev",
        "Jev añadiría topic",
        "Primario distinto",
        "Jev eligió «",
    ):
        assert label in rail, label


def test_the_cards_are_built_from_text_nodes_never_from_html_strings():
    """Scraped post text, quoted posts, article titles and evidence reach the DOM as TEXT:
    the card code has no `innerHTML`, so no scraped string is ever parsed as markup."""
    template = _resource("jev.template.html")
    cards = _script_section(template, "/* cards */", "/* end cards */")

    assert "innerHTML" not in cards
    assert "textContent" in cards
    # Links only to http(s) — `Item.url` and a link's url are bare `str`s in the model.
    assert "httpUrl(" in cards


def test_the_card_shows_jev_vs_enrich_in_plain_words_and_what_jev_read():
    template = _resource("jev.template.html")
    cards = _script_section(template, "/* cards */", "/* end cards */")

    for words in (
        "coinciden",
        "solo enrich",
        "solo Jev",
        "probabilidad",
        "Lo que vio Jev",
        "recortado",
        "sin evaluar por Jev",
        "copiar comando",
        "DATA.ask_command",
        "DATA.surface_chars",
    ):
        assert words in cards, words


def test_the_keyboard_moves_by_card_and_by_disagreement():
    template = _resource("jev.template.html")
    keys = _script_section(template, "function onKey(", "/* end keys */")

    for key in ("'j'", "'k'", "'n'", "'p'"):
        assert key in keys, key
    assert "disagreements" in keys


def test_the_cards_render_incrementally():
    """~2.6k posts: fifty at a time, more when the reader reaches the end."""
    template = _resource("jev.template.html")

    assert "const PAGE = 50" in template
    assert "IntersectionObserver" in template


def test_the_topic_list_starts_folded_on_a_narrow_screen():
    """45 topics above the first card is a screen and a half on a phone."""
    rail = _script_section(_resource("jev.template.html"), "function renderRail(", "/* end rail */")

    assert "createElement('details')" in rail and "box.open = topicsOpen" in rail
    assert "let topicsOpen = !matchMedia('(max-width: 900px)').matches" in _resource(
        "jev.template.html"
    )


def test_the_page_follows_the_systems_light_or_dark_theme():
    assert "prefers-color-scheme: light" in _resource("jev.template.html")


def test_the_page_handles_the_cost_error_the_unpriced_row_and_an_empty_table():
    template = _resource("jev.template.html")

    assert "c.error" in template  # the run log could not be read: say so in the strip
    assert "sin tarifa" in template  # a card whose provider nobody prices
    assert "Ningún post coincide" in template
    assert "per_post.n" in template and "per_post.of" in template  # "media de K de las N"
    assert "posts_with_disagreement" in template  # the report's count, not the page's
    # One money format across the strip: the shared Python sentence is not shipped.
    assert "side_car_text" not in template

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
    TOPIC_MIN,
    MediaFiles,
    collect_jev_media,
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
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
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
    probability (unrounded) and the verdict read off `build_report`'s comparison."""
    item = _item("1", topics=("ai-coding",))
    assessment = _assessment(item, membership={"ai-coding": 0.4, "startups": 0.973456})

    jev = _post(_data([item], {"1": assessment}), "1")["jev"]

    assert jev["topics"] == [
        {"slug": "ai-coding", "enrich": True, "p": 0.4, "verdict": "solo_enrich"},
        {"slug": "startups", "enrich": False, "p": 0.973456, "verdict": "solo_jev"},
    ]
    assert jev["compared"] is True
    assert jev["primary"] == "ai-coding" and jev["jev_primary"] == "ai-coding"
    assert jev["jev_confidence"] == 0.77
    assert jev["primary_differs"] is False and jev["jev_fallback"] is False
    assert jev["disagreements"] == 2


def test_a_card_ships_only_the_fields_the_page_reads():
    """No `primary_agrees`, `enrich_only` or `jev_only` counts: the filters are the card's
    `in` list, and the primary line reads `primary_differs`."""
    item = _item("1")

    jev = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]

    assert not {"primary_agrees", "enrich_only", "jev_only"} & set(jev)


def test_a_topic_both_sides_hold_is_coinciden():
    item = _item("1", topics=("ai-coding",))

    jev = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]

    assert jev["topics"] == [
        {"slug": "ai-coding", "enrich": True, "p": 0.9, "verdict": "coinciden"}
    ]
    assert jev["disagreements"] == 0


def test_jevs_primary_is_a_row_when_no_other_row_names_it():
    """A topic filter must find the post whose primary Jev put on a topic it does not back
    at the threshold. The row is not a disagreement of its own: the primary line counts it."""
    item = _item("1", topics=("ai-coding",))
    assessment = _assessment(
        item, membership={"ai-coding": 0.9, "startups": 0.3}, choice="startups"
    )

    jev = _post(_data([item], {"1": assessment}), "1")["jev"]

    assert jev["topics"][-1] == {
        "slug": "startups",
        "enrich": False,
        "p": 0.3,
        "verdict": "primario_jev",
    }
    assert jev["disagreements"] == 1  # the primary, once


def test_the_fallback_is_not_a_topic_row():
    item = _item("1", topics=("ai-coding",))

    jev = _post(_data([item], {"1": _assessment(item, choice="otro")}), "1")["jev"]

    assert [row["slug"] for row in jev["topics"]] == ["ai-coding"]
    assert jev["jev_primary"] == "otro" and jev["jev_fallback"] is True
    assert jev["primary_differs"] is True and jev["disagreements"] == 1


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
    with nothing to compare against are cards too, each with its status."""
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
    assert _post(data, "2")["jev"] is None and _post(data, "3")["jev"] is None
    # What enrich said is on the card whether or not Jev was asked.
    assert _post(data, "3")["enrich"] == {"topics": ["ai-coding"], "primary": "ai-coding"}
    assert _post(data, "4")["enrich"] is None


def test_a_paid_answer_for_a_post_without_enrichment_still_shows_what_jev_sees():
    """Nothing to compare against is not nothing to show: Jev's own topics at the threshold
    and its primary, with no verdict against enrich and no disagreement."""
    plain = _item("1")
    plain.enriched = None
    assessment = _assessment(plain, membership={"ai-coding": 0.9, "startups": 0.95})

    jev = _post(_data([plain], {"1": assessment}), "1")["jev"]

    assert jev["compared"] is False
    assert jev["topics"] == [
        {"slug": "startups", "enrich": False, "p": 0.95, "verdict": "jev"},
        {"slug": "ai-coding", "enrich": False, "p": 0.9, "verdict": "jev"},
    ]
    assert jev["jev_primary"] == "ai-coding" and jev["primary"] is None
    assert jev["disagreements"] == 0 and jev["primary_differs"] is False
    assert jev["model"] == "jev-1.13.0" and jev["surfaces"]


def test_a_post_with_no_evidence_says_so_instead_of_offering_a_command():
    """`jev topics` skips a post with no evidence (`state_chars == 0`, `select_items`'s own
    test), so a copy-able command for it would do nothing."""
    empty, never = _item("1", text=" "), _item("2")
    empty.author = Author(handle="", name="")

    data = _data([empty, never], {})

    assert _post(data, "1")["no_evidence"] is True
    assert _post(data, "2")["no_evidence"] is False
    assert build_topic_state(empty, CHAR_LIMIT)[1] == 0


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

    jev = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]

    assert jev["primary"] is None and jev["primary_differs"] is True
    assert jev["topics"][0] == {
        "slug": "ai-coding",
        "enrich": True,
        "p": 0.9,
        "verdict": "coinciden",
    }


# --------------------------------------------------------------------------- filters = report counts


def _corpus() -> tuple[list[Item], dict[str, TopicAssessment]]:
    """One post per case the filters must tell apart."""
    agree = _item("agree", topics=("ai-coding",))
    enrich_only = _item("enrich-only", topics=("ai-coding", "startups"))
    jev_only = _item("jev-only", topics=("ai-coding",))
    fallback = _item("fallback", topics=("ai-coding",))
    other_primary = _item("other-primary", topics=("ai-coding",))
    stale = _item("stale", text="old")
    never = _item("never")
    plain = _item("plain")
    plain.enriched = None
    low = {"startups": 0.1}
    assessments = {
        "agree": _assessment(agree, membership={"ai-coding": 0.9, **low}),
        "enrich-only": _assessment(enrich_only, membership={"ai-coding": 0.9, **low}),
        "jev-only": _assessment(jev_only, membership={"ai-coding": 0.9, "startups": 0.9}),
        "fallback": _assessment(fallback, membership={"ai-coding": 0.9, **low}, choice="otro"),
        "other-primary": _assessment(
            other_primary, membership={"ai-coding": 0.9, **low}, choice="startups"
        ),
        "stale": _assessment(stale, contract="e" * 64),
        "plain": _assessment(plain),
    }
    return [agree, enrich_only, jev_only, fallback, other_primary, stale, never, plain], assessments


@pytest.mark.parametrize(
    ("key", "count"),
    [
        ("disc", ("summary", "posts_with_disagreement")),
        ("enrich_only", ("summary", "posts_enrich_only")),
        ("adds", ("summary", "posts_jev_only")),
        ("prim", ("summary", "posts_primary_differs")),
        ("fallback", ("summary", "primary_fallback")),
        ("uneval", ("summary", "items_unassessed")),
    ],
)
def test_each_filters_cards_are_exactly_the_posts_its_report_count_counts(key, count):
    """The page filters by `key in card["in"]` and prints the report's number beside it.
    Decided here, per card, from `ItemComparison` and the status — so the two cannot drift."""
    items, assessments = _corpus()
    data = _data(items, assessments)
    section, name = count

    members = [post["id"] for post in data["posts"] if key in post["in"]]

    assert len(members) == data[section][name]
    assert members  # every case is represented in the fixture


def test_the_filter_keys_name_the_right_posts():
    items, assessments = _corpus()
    data = _data(items, assessments)

    keys = {post["id"]: set(post["in"]) for post in data["posts"]}
    assert keys["agree"] == set()
    assert keys["enrich-only"] == {"disc", "enrich_only"}
    assert keys["jev-only"] == {"disc", "adds"}
    assert keys["fallback"] == {"disc", "prim", "fallback"}
    assert keys["other-primary"] == {"disc", "prim"}
    assert keys["stale"] == keys["never"] == {"uneval"}
    assert keys["plain"] == set()


def test_per_topic_numbers_are_the_cards_topic_rows():
    """For each topic, the rows the cards carry add up to its `per_topic` row: coinciden =
    backed, solo Jev = missing, solo enrich + solo Jev = disagreeing. A topic click under
    "Con discrepancias" lists exactly `disagreeing` posts."""
    items, assessments = _corpus()
    data = _data(items, assessments)

    for row in data["summary"]["per_topic"]:
        verdicts = [
            topic["verdict"]
            for post in data["posts"]
            if post["jev"] and post["jev"]["compared"]
            for topic in post["jev"]["topics"]
            if topic["slug"] == row["slug"]
        ]
        assert verdicts.count("coinciden") == row["backed"], row["slug"]
        assert verdicts.count("solo_jev") == row["missing"], row["slug"]
        assert verdicts.count("solo_enrich") == row["doubtful"], row["slug"]
        assert verdicts.count("solo_enrich") + verdicts.count("solo_jev") == row["disagreeing"]


def test_each_confusion_pairs_posts_show_that_pair_on_their_cards():
    """The Topics tab lists a pair's posts from `post_sets`: each of those cards must carry
    the pair — enrich's topic as solo enrich, Jev's as solo Jev; for a primary pair, the two
    primaries. The index ships in the page and nowhere in the summary."""
    items, assessments = _corpus()
    data = _data(items, assessments)
    cards = {post["id"]: post for post in data["posts"]}
    sets = data["post_sets"]

    assert sets["cx"] and sets["px"]
    assert all("ids" not in row for row in data["summary"]["topic_confusion"])
    for key, ids in sets["cx"].items():
        enrich, jev = (None if side == "-" else side for side in key.split("~"))
        for post_id in ids:
            verdicts = {r["slug"]: r["verdict"] for r in cards[post_id]["jev"]["topics"]}
            if enrich is not None:
                assert verdicts[enrich] == "solo_enrich", (key, post_id)
            if jev is not None:
                assert verdicts[jev] == "solo_jev", (key, post_id)
    for key, ids in sets["px"].items():
        enrich, jev = (None if side == "-" else side for side in key.split("~"))
        for post_id in ids:
            card = cards[post_id]["jev"]
            assert (card["primary"], card["jev_primary"]) == (enrich, jev)


def test_the_topic_pairs_are_what_the_cards_rows_say_recounted_from_the_cards():
    """An independent recount: from each card's rows (solo enrich × solo Jev, `-` for a side
    with none), rebuild the pair index and the counts — they must equal the report's."""
    items, assessments = _corpus()
    data = _data(items, assessments)

    recount: dict[str, set[str]] = {}
    for post in data["posts"]:
        if not (post["jev"] and post["jev"]["compared"]):
            continue
        rows = post["jev"]["topics"]
        only_enrich = [r["slug"] for r in rows if r["verdict"] == "solo_enrich"]
        only_jev = [r["slug"] for r in rows if r["verdict"] == "solo_jev"]
        if not (only_enrich or only_jev):
            continue
        for enrich in only_enrich or ["-"]:
            for jev in only_jev or ["-"]:
                recount.setdefault(f"{enrich}~{jev}", set()).add(post["id"])

    assert {k: set(v) for k, v in data["post_sets"]["cx"].items()} == recount
    counts = {
        f"{r['enrich'] or '-'}~{r['jev'] or '-'}": r["posts"]
        for r in data["summary"]["topic_confusion"]
    }
    assert counts == {k: len(v) for k, v in recount.items()}


def test_the_topic_floor_is_defined_once_and_shipped_to_the_page():
    assert TOPIC_MIN == 5
    assert _data([_item()], {})["topic_min"] == TOPIC_MIN


def test_a_cards_slugs_are_every_topic_enrich_or_jev_has_on_it():
    """What a topic filter matches outside the disagreement views."""
    items, assessments = _corpus()
    data = _data(items, assessments)

    assert _post(data, "other-primary")["slugs"] == ["ai-coding", "startups"]
    assert _post(data, "never")["slugs"] == ["ai-coding"]
    assert _post(data, "plain")["slugs"] == ["ai-coding"]


# --------------------------------------------------------------------------- the share card


def test_the_card_carries_the_whole_post_and_who_wrote_it():
    """The full text (the page collapses a long one), not a snippet — and the links to X and
    the note, against the SOURCE, not a constant."""
    long_post = "palabra " * 400
    item = _item("1", text=long_post)

    data = _data([item], {}, id2note={"1": "1.md"}, notes_dir="/v/items")
    post = _post(data, "1")

    assert post["text"] == long_post
    assert post["author"] == {"handle": "alice", "name": "Alice"}
    assert post["url"] == item.url and post["created"] == item.created_at.isoformat()
    # The directory once, the file per post: 2.6k copies of one long prefix were 435 KB.
    assert data["notes_dir"] == "/v/items" and post["note"] == "1.md"


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


def _files(mirrored=(), downloaded=()) -> MediaFiles:
    return MediaFiles(mirrored=frozenset(mirrored), downloaded=frozenset(downloaded))


def test_a_mirrored_photo_is_a_relative_path_into_the_vaults_media_folder():
    item = _item("1")
    item.media = [_photo("1", 0, description="Un gráfico.")]

    media = _post(_data([item], {}, media=_files(mirrored={"1/0.jpg"})), "1")["media"]

    assert media == [{"type": "photo", "src": "_media/1/0.jpg", "desc": "Un gráfico.", "why": None}]


def test_a_photo_without_a_file_on_the_page_says_why():
    """Downloaded but not mirrored yet (`xbrain generate` fixes it), never downloaded,
    failed, or gone from both places: four different next steps."""
    item = _item("1")
    pending = MediaPhotoPending(url="https://pbs.twimg.com/p.jpg")
    failed = MediaPhotoFailed(
        url="https://pbs.twimg.com/f.jpg", failure_reason="http_4xx", attempts=1, last_attempt_at=DT
    )
    item.media = [_photo("1", 0), _photo("1", 1), pending, failed]

    media = _post(_data([item], {}, media=_files(downloaded={"1/0.jpg"})), "1")["media"]

    assert [(m["src"], m["why"]) for m in media] == [
        (None, "not_mirrored"),
        (None, "missing"),
        (None, "not_downloaded"),
        (None, "failed"),
    ]


def test_a_photo_caption_is_cut_for_the_tooltip():
    """The vision caption is the photo's alt text and tooltip; at ~400 characters each over
    ~1.3k photos, shipping it whole is a sixth of the page."""
    item = _item("1")
    item.media = [_photo("1", 0, description="d" * 500)]

    [photo] = _post(_data([item], {}), "1")["media"]

    assert photo["desc"] == "d" * 279 + "…"


def _video(n: int) -> MediaVideoDownloaded:
    return MediaVideoDownloaded(
        url=f"https://video.twimg.com/{n}.mp4",
        thumbnail_url="https://pbs.twimg.com/poster.jpg",
        local_path=f"1/video-{n}.mp4",
        bytes_size=9,
        downloaded_at=DT,
    )


def test_each_video_shows_the_first_frame_of_its_own_source():
    """Videos pair with the post's `x_video` sources in order; a video with no extracted
    frame of its own is a placeholder, never the other video's still. A non-video source
    that carries frames is not a video's frame."""
    item = _item("1")
    item.media = [_video(0), _video(1)]
    item.content = Content(
        fetched_at=DT,
        sources=[
            ContentSourceSuccess(
                kind="external_article",
                url="https://e.com",
                text="art",
                frames=[VideoFrame(timestamp=0.0, local_path="1/frames/article.jpg")],
            ),
            ContentSourceSuccess(
                kind="x_video",
                url=item.url,
                text="transcript",
                frames=[
                    VideoFrame(timestamp=1.0, local_path="1/frames/0.jpg"),
                    VideoFrame(timestamp=2.0, local_path="1/frames/1.jpg"),
                ],
            ),
            ContentSourceSuccess(kind="x_video", url=item.url, text="second, no frames"),
        ],
    )

    media = _post(
        _data([item], {}, media=_files(mirrored={"1/frames/0.jpg", "1/frames/article.jpg"})), "1"
    )["media"]

    assert media == [
        {"type": "video", "src": "_media/1/frames/0.jpg", "desc": "", "why": None},
        {"type": "video", "src": None, "desc": "", "why": "no_frame"},
    ]


def test_at_most_four_media_per_card():
    item = _item("1")
    item.media = [_photo("1", n) for n in range(6)]

    assert len(_post(_data([item], {}), "1")["media"]) == 4


def _quoted_source(text: str) -> ContentSourceSuccess:
    return ContentSourceSuccess(
        kind="quoted_tweet",
        url="https://x.com/karpathy/status/9",
        text=text,
        author=Author(handle="karpathy", name="Andrej Karpathy"),
    )


def test_the_quoted_post_is_the_one_jev_read_cut_for_the_page():
    """`quoted_source` — the same quoted post the evidence carries — cut at 600 characters."""
    item = _item("1")
    item.content = Content(fetched_at=DT, sources=[_quoted_source("q" * 601)])

    assert _post(_data([item], {}), "1")["quoted"] == {
        "handle": "karpathy",
        "name": "Andrej Karpathy",
        "url": "https://x.com/karpathy/status/9",
        "text": "q" * 600,
        "cut": True,
        "missing": False,
        "why": None,
    }


def test_a_quoted_post_of_exactly_600_characters_is_not_cut():
    item = _item("1")
    item.content = Content(fetched_at=DT, sources=[_quoted_source("q" * 600)])

    quoted = _post(_data([item], {}), "1")["quoted"]

    assert quoted["text"] == "q" * 600 and quoted["cut"] is False


def test_a_quoted_post_that_could_not_be_read_is_still_shown_as_missing():
    """30 of 911 quote-tweets have no readable quoted source: the card says so and links to
    X, instead of looking like a post that quotes nothing."""
    failed = _item("1")
    failed.quoted_id = "77"
    failed.content = Content(
        fetched_at=DT,
        sources=[
            ContentSourceFailure(
                kind="quoted_tweet", url="https://x.com/b/status/77", failure_reason="not_found"
            )
        ],
    )
    bare = _item("2")
    bare.quoted_id = "88"

    data = _data([failed, bare], {})

    assert _post(data, "1")["quoted"] == {
        "handle": None,
        "name": None,
        "url": "https://x.com/b/status/77",
        "text": "",
        "cut": False,
        "missing": True,
        "why": "failed",
    }
    # Never fetched is a different next step from a fetch that failed.
    assert _post(data, "2")["quoted"]["url"] == "https://x.com/i/status/88"
    assert _post(data, "2")["quoted"]["missing"] is True
    assert _post(data, "2")["quoted"]["why"] == "not_fetched"


def test_the_link_card_is_the_fetched_article_labelled_with_its_kind():
    """An `x_article` source may hold scraped replies rather than an article: the card
    names the kind so the page does not pass it off as one."""
    item = _item("1")
    item.links = [Link(url="https://blog.example.com/post", domain="blog.example.com")]
    item.content = Content(
        fetched_at=DT,
        sources=[
            ContentSourceSuccess(
                kind="x_article", url="https://x.com/i/article/5", title="Un artículo", text="c"
            )
        ],
    )

    assert _post(_data([item], {}), "1")["link"] == {
        "url": "https://x.com/i/article/5",
        "domain": "x.com",
        "title": "Un artículo",
        "kind": "x_article",
        "failed": False,
    }


def test_a_link_whose_fetch_failed_says_it_could_not_be_read():
    item = _item("1")
    item.links = [Link(url="https://blog.example.com/post", domain="blog.example.com")]
    item.content = Content(
        fetched_at=DT,
        sources=[
            ContentSourceFailure(
                kind="external_article",
                url="https://blog.example.com/post",
                failure_reason="not_found",
            )
        ],
    )

    assert _post(_data([item], {}), "1")["link"] == {
        "url": "https://blog.example.com/post",
        "domain": "blog.example.com",
        "title": None,
        "kind": "external_article",
        "failed": True,
    }


def test_without_a_fetch_the_link_card_is_the_first_link():
    item = _item("1")
    item.links = [Link(url="https://www.example.com/a", domain="example.com")]

    assert _post(_data([item], {}), "1")["link"] == {
        "url": "https://www.example.com/a",
        "domain": "example.com",
        "title": None,
        "kind": None,
        "failed": False,
    }
    assert _post(_data([_item("2")], {}), "2")["link"] is None


# --------------------------------------------------------------------------- media on disk


def test_the_media_collector_is_the_only_place_that_looks_at_the_disk(tmp_path):
    """`collect_jev_media` stats each photo and frame in the page's `_media/` mirror and in
    `data/media/`; the blob builder only reads its answer, so it stays pure."""
    page_dir, media_root = tmp_path / "vault", tmp_path / "data-media"
    item = _item("1")
    item.media = [_photo("1", 0), _photo("1", 1), _photo("1", 2)]
    for root, name in (
        (page_dir / "_media", "0.jpg"),
        (media_root, "0.jpg"),
        (media_root, "1.jpg"),
    ):
        (root / "1").mkdir(parents=True, exist_ok=True)
        (root / "1" / name).write_bytes(b"jpg")

    files = collect_jev_media([item], page_dir, media_root)

    assert files == _files(mirrored={"1/0.jpg"}, downloaded={"1/0.jpg", "1/1.jpg"})
    media = _post(_data([item], {}, media=files), "1")["media"]
    assert [m["why"] for m in media] == [None, "not_mirrored", "missing"]
    assert (page_dir / media[0]["src"]).is_file()


# --------------------------------------------------------------------------- what Jev saw


def test_what_jev_saw_is_the_state_split_into_its_surfaces():
    """From `assess.state_surfaces`, with Spanish labels, sizes and what the cut kept. The
    tweet and the author are already on the card, so their text is not shipped twice."""
    item = _item("1", text="Claude Code hooks")

    jev = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]

    assert jev["surfaces"] == [
        {
            "key": "tweet",
            "label": "Tweet",
            "chars": 17,
            "kept": 17,
            "text": None,
            "same_as": "post",
        },
        {
            "key": "author",
            "label": "Autor",
            "chars": 11,
            "kept": 11,
            "text": None,
            "same_as": "author",
        },
    ]
    assert jev["state_chars"] == _assessment(item).state_chars == 17 + 1 + 11


def test_the_quoted_surface_points_at_the_quoted_card():
    item = _item("1")
    item.content = Content(fetched_at=DT, sources=[_quoted_source("citado")])

    surfaces = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]["surfaces"]

    [quoted] = [s for s in surfaces if s["key"] == "quoted"]
    assert quoted["text"] is None and quoted["same_as"] == "quoted"


def test_a_surface_ships_at_most_600_characters_but_says_its_full_size():
    item = _item("1")
    item.content = Content(
        fetched_at=DT,
        sources=[ContentSourceSuccess(kind="thread", url=item.url, text="y" * 1500)],
    )

    surfaces = _post(_data([item], {"1": _assessment(item)}), "1")["jev"]["surfaces"]

    [thread] = [s for s in surfaces if s["key"] == "thread"]
    assert thread["text"] == "y" * 600 and thread["same_as"] is None
    assert (thread["chars"], thread["kept"]) == (1500, 1500)


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


def test_an_evaluated_card_stays_within_its_byte_budget():
    """The page ships every post; an evaluated card is the heavy one. A 280-character tweet,
    a quoted post, a 2,000-character article and two topics: the tweet travels once (the
    surfaces point at it) and the card stays under 3 KB (2,852 bytes when this was written;
    most of it is the article surface's 600 characters)."""
    item = _item("1", text="t" * 280, topics=("ai-coding", "startups"))
    item.content = Content(
        fetched_at=DT,
        sources=[
            _quoted_source("q" * 300),
            ContentSourceSuccess(
                kind="external_article", url="https://e.com/a", title="A", text="a" * 2000
            ),
        ],
    )

    card = _post(_data([item], {"1": _assessment(item)}), "1")
    size = len(json.dumps(card, ensure_ascii=False).encode())

    assert json.dumps(card).count("t" * 280) == 1
    assert size <= 3000, size


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
    assert SURFACE_LABELS["video_transcript"] == "Transcripción del vídeo"


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


def test_hostile_strings_in_every_scraped_field_survive_the_page_parse_intact():
    """Author, quoted post, link title, photo caption and an evidence surface each carry a
    script-closer, markup and a JavaScript line separator (U+2028/U+2029): the page's DATA,
    parsed back out of the HTML, equals the blob — nothing broke out, nothing was mangled."""
    hostile = "</script><img src=x onerror=alert(1)>\u2028\u2029'\"&"
    item = _item("1", text="post " + hostile)
    item.author = Author(handle="h" + hostile, name="N" + hostile)
    item.media = [_photo("1", 0, description="foto " + hostile)]
    item.content = Content(
        fetched_at=DT,
        sources=[
            _quoted_source("citado " + hostile),
            ContentSourceSuccess(
                kind="external_article", url="https://e.com", title="T" + hostile, text="a"
            ),
            ContentSourceSuccess(kind="thread", url=item.url, text="hilo " + hostile),
        ],
    )
    data = _data([item], {"1": _assessment(item)})

    html = render_jev_dashboard_html(data)

    assert html.count("</script>") == 1
    assert "\u2028" not in html and "\u2029" not in html
    payload = html.rsplit("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(payload) == json.loads(json.dumps(data))
    card = json.loads(payload)["posts"][0]
    assert card["quoted"]["text"].endswith(hostile)
    assert card["link"]["title"].endswith(hostile)
    assert card["media"][0]["desc"].endswith(hostile)
    assert any(s["text"] and s["text"].endswith(hostile) for s in card["jev"]["surfaces"])


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
    # Topics is built (9b); Comparar comes next (9c), Configuración after it (9d).
    assert template.count("llega en el siguiente PR") == 1
    assert "llegan en un PR posterior" in template
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


def test_each_filter_is_a_card_key_with_the_reports_count_beside_it():
    """The rail's views are `(key, name, report count)` triples: the page tests membership
    with `p.in.includes(key)` — decided in Python — and prints the report's number for it.
    Every key the blob can put on a card has exactly one view, with its own name."""
    template = _resource("jev.template.html")
    rail = _script_section(template, "const FILTERS = [", "function railButton(")

    views = re.findall(r"\{key: '(\w+)', name: '([^']*)'[^}]*count: \(\) => (DATA\.[\w.]+)\}", rail)
    assert {key: (name, count) for key, name, count in views} == {
        "all": ("Todos", "DATA.totals.items"),
        "disc": ("Con discrepancias", "DATA.summary.posts_with_disagreement"),
        "enrich_only": ("Enrich asigna y Jev no", "DATA.summary.posts_enrich_only"),
        "adds": ("Jev añadiría topic", "DATA.summary.posts_jev_only"),
        "prim": ("Primario distinto", "DATA.summary.posts_primary_differs"),
        "fallback": ("Jev eligió «", "DATA.summary.primary_fallback"),
        "uneval": ("Sin evaluar por Jev", "DATA.summary.items_unassessed"),
    }
    matches = _script_section(template, "function matches(", "function sorted(")
    assert "p.in.includes(view.f)" in matches


def test_a_topic_under_a_disagreement_view_lists_the_posts_its_number_counts():
    """Under "Con discrepancias" a topic lists the cards whose row for it is solo enrich or
    solo Jev — its `per_topic.disagreeing`; under the one-direction views, that direction."""
    template = _resource("jev.template.html")

    assert (
        "const TOPIC_VERDICTS = {disc: ['solo_enrich', 'solo_jev'], "
        "enrich_only: ['solo_enrich'], adds: ['solo_jev']};" in template
    )
    assert (
        "const TOPIC_COUNT = {disc: 'disagreeing', enrich_only: 'doubtful', adds: 'missing'};"
        in template
    )


def test_the_verdict_chips_name_each_verdict_the_blob_can_send():
    template = _resource("jev.template.html")
    chips = _script_section(template, "const VERDICT = {", "function probBar(")

    assert dict(re.findall(r"(\w+): \['\w+', '([^']+)'", chips)) == {
        "coinciden": "coinciden",
        "solo_enrich": "solo enrich",
        "solo_jev": "solo Jev",
        "sin_juzgar": "sin juzgar",
        "primario_jev": "primario de Jev",
        "jev": "Jev lo ve",
    }


def test_the_cards_are_built_from_text_nodes_never_from_html_strings():
    """Scraped post text, quoted posts, article titles and evidence reach the DOM as TEXT:
    the card code has no HTML-parsing sink at all, and every href comes from `httpUrl`, the
    http(s) regex in `linkified`, or the note link built from DATA."""
    template = _resource("jev.template.html")
    cards = _script_section(template, "/* cards */", "/* end cards */")

    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "DOMParser"):
        assert sink not in cards, sink
    assert "textContent" in cards
    assert cards.count(".href = ") == 1 and "a.href = href;" in cards
    hrefs = re.findall(r"\blink\(([^,]+),", cards)
    assert set(hrefs) == {
        "x",
        "url",
        "href",
        "'obsidian://open?path=' + encodeURIComponent(DATA.notes_dir + '/' + p.note)",
        "topicHref(r.slug)",
    }
    assert "const x = httpUrl(p.url);" in cards and cards.count("const href = httpUrl(") == 2
    assert "text.replace(/https?:\\/\\/[^\\s]+/g" in cards


def test_an_unread_quoted_post_names_the_command_that_reads_it():
    """`not_fetched`: nothing has tried to read it yet, and `xbrain refresh-quoted` is the
    command that fills quoted posts in (`xbrain fetch` fetches linked articles, not these)."""
    cards = _script_section(_resource("jev.template.html"), "/* cards */", "/* end cards */")

    assert "q.why === 'not_fetched'" in cards
    assert "Sin leer todavía: corre xbrain refresh-quoted" in cards


def test_the_topic_index_reads_each_column_from_the_per_topic_row():
    """Every number in the index is a `per_topic` field; the page computes none of them."""
    topics = _script_section(_resource("jev.template.html"), "/* topics */", "/* end topics */")
    columns = _script_section(topics, "const TOPIC_COLUMNS = [", "];")

    assert re.findall(r"key: '(\w+)'", columns) == [
        "label",
        "assigned",
        "backed",
        "backed_pct",
        "missing",
        "disagreeing",
        "enrich_primary",
        "jev_primary",
    ]
    assert "noul" not in topics.lower()


def test_the_topic_index_opens_on_the_worst_agreement_among_topics_with_enough_posts():
    """The default order is `per_topic`'s own (worst backing first, by the exact ratio),
    with topics enrich put on fewer than `TOPIC_MIN` posts moved to the end — and the page
    says what N is."""
    topics = _script_section(_resource("jev.template.html"), "/* topics */", "/* end topics */")

    assert "const TOPIC_MIN = DATA.topic_min;" in topics
    assert "DATA.summary.per_topic.map(r => r.slug)" in topics
    assert "nf(TOPIC_MIN)" in topics


def test_the_topic_detail_reads_the_confusion_the_report_computed():
    topics = _script_section(_resource("jev.template.html"), "/* topics */", "/* end topics */")

    assert "DATA.summary.topic_confusion" in topics
    assert "DATA.summary.primary_confusion" in topics
    for words in ("Coinciden", "Solo enrich", "Solo Jev", "ver en Posts", "Con qué se confunde"):
        assert words in topics, words


def test_the_topics_code_builds_text_nodes_and_only_links_inside_the_page():
    topics = _script_section(_resource("jev.template.html"), "/* topics */", "/* end topics */")

    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "DOMParser"):
        assert sink not in topics, sink
    assert ".href = " not in topics
    # Every link's href is `topicHref(slug)` or a variable built by `hashHref` — both
    # `#<tab>?<URLSearchParams>`, so nothing scraped ever becomes a URL here.
    for arg in set(re.findall(r"\blink\(([^,]+),", topics)):
        assert (
            arg == "topicHref(slug)"
            or f"const {arg} = hashHref(" in topics
            or f"const {arg} = indexHref();" in topics
        ), arg
    assert "return '#' + tab + (qs ? '?' + qs : '');" in topics


def test_the_topics_words_that_carry_meaning_are_pinned():
    """A pair's other side: nothing instead, no primary, Jev's «none of these», and an
    enrich primary that has since left the vocabulary — four different facts, four labels."""
    topics = _script_section(_resource("jev.template.html"), "/* topics */", "/* end topics */")

    for words in (
        "'sin principal'",
        "'nada en su lugar'",
        "' (ninguno del vocabulario)'",
        "' (ya no está en el vocabulario)'",
        "'Enrich no lo pone en ningún post comparado.'",
        "'Enrich nunca lo elige como principal.'",
        "'Jev nunca lo elige como principal.'",
        "'Ese cruce ya no existe en estos datos; abajo, todos los posts del topic.'",
        "no está en el vocabulario actual",
        "'La pestaña Topics no pudo dibujarse: '",
        "no suman en la columna «Principal según Jev».",
        "'pocos datos'",
    ):
        assert words in topics, words


def test_the_posts_rail_names_what_the_topics_tab_names():
    """The rail says which Topics-tab number its list matches — by that tab's own names."""
    template = _resource("jev.template.html")
    rail = _script_section(template, "function renderRail(", "/* end rail */")
    topics = _script_section(template, "/* topics */", "/* end topics */")

    for name in ("Discrepancias", "Jev lo añadiría"):
        assert f"«{name}»" in rail and f"name: '{name}'" in topics, name
    assert "«Solo enrich»" in rail and "name: 'Solo enrich'" in topics
    assert "ficha del topic →" in rail


def test_the_topics_view_lives_in_the_hash():
    template = _resource("jev.template.html")
    reader = _script_section(template, "function readTopicsHash(", "function writeTopicsSort(")

    for key in ("t", "cx", "px", "o", "d"):
        assert f"get('{key}')" in reader, key


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
        "sin evidencia",
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
    assert "const disagrees = (p) => p.in.includes('disc');" in keys


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

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

from xbrain.dashboard import _resource
from xbrain.jev.assess import (
    build_topic_state,
    current_pairs,
    questions_digest,
    topic_contract,
)
from xbrain.jev.dashboard import DOCS_URL, compute_jev_dashboard_data, render_jev_dashboard_html
from xbrain.jev.defaults import tokens_cost_usd
from xbrain.jev.models import JevRun, PrimaryChoice, TopicAssessment
from xbrain.jev.questions import STATE_KEY, build_topic_questions
from xbrain.jev.report import build_report, run_history
from xbrain.models import Author, Enrichment, Item, Topic

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
) -> TopicAssessment:
    """A record whose contract is the one TODAY'S ask would stamp, so it reads as current.

    Built the way `assess.assess_topics` builds it — the state as sent and the digest of the
    questions that went with it — so a change to either composition shows up here as a stale
    fixture instead of a test passing over a contract nobody computes any more.
    """
    state, state_chars = build_topic_state(item, CHAR_LIMIT)
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


def test_the_threshold_is_the_one_handed_in_and_nothing_on_the_page_moves_it():
    item = _item(topics=("ai-coding",))

    low = _data([item], {"1": _assessment(item)}, threshold=0.95)

    # ai-coding at 0.9 is backed at 0.85 and doubtful at 0.95: the page shows what it was given.
    assert low["summary"]["threshold"] == 0.95
    assert _post(low, "1")["enrich"] == [{"slug": "ai-coding", "p": 0.9, "ok": False}]


# --------------------------------------------------------------------------- the post table


def test_each_post_marks_enrichs_topics_and_names_what_jev_would_add():
    """✓ / ✗ per enrich topic, from `build_report`'s comparison, never re-derived."""
    item = _item("1", topics=("ai-coding",))
    assessment = _assessment(item, membership={"ai-coding": 0.4, "startups": 0.97})

    post = _post(_data([item], {"1": assessment}), "1")

    assert post["enrich"] == [{"slug": "ai-coding", "p": 0.4, "ok": False}]
    assert post["adds"] == [{"slug": "startups", "p": 0.97}]
    assert post["primary"] == "ai-coding" and post["jev_primary"] == "ai-coding"
    assert post["primary_agrees"] is True and post["jev_fallback"] is False
    # One doubtful topic + one missing topic; the primaries agree.
    assert post["disagreements"] == 2


def test_a_topic_that_left_the_vocabulary_is_neither_confirmed_nor_denied():
    """An assigned slug Jev was never asked about is `ok: None`, with no probability — a
    fabricated 0.0 would show as the strongest disagreement in the corpus."""
    item = _item("1", topics=("ai-coding", "web3"))

    post = _post(_data([item], {"1": _assessment(item)}), "1")

    assert post["enrich"] == [
        {"slug": "ai-coding", "p": 0.9, "ok": True},
        {"slug": "web3", "p": None, "ok": None},
    ]
    assert post["disagreements"] == 0


def test_a_primary_jev_does_not_share_is_a_disagreement_and_the_fallback_is_named():
    item = _item("1", topics=("ai-coding",))

    post = _post(_data([item], {"1": _assessment(item, choice="otro")}), "1")

    assert post["primary_agrees"] is False
    assert post["jev_primary"] == "otro" and post["jev_fallback"] is True
    assert post["disagreements"] == 1


def test_posts_are_ordered_most_disagreement_first():
    agree = _item("1", topics=("ai-coding",))
    one = _item("2", topics=("ai-coding",))
    two = _item("3", topics=("ai-coding",))
    assessments = {
        "1": _assessment(agree),
        "2": _assessment(one, membership={"ai-coding": 0.4, "startups": 0.1}),
        "3": _assessment(two, membership={"ai-coding": 0.4, "startups": 0.9}),
    }

    data = _data([agree, one, two], assessments)

    assert [post["id"] for post in data["posts"]] == ["3", "2", "1"]
    assert [post["disagreements"] for post in data["posts"]] == [2, 1, 0]


def test_posts_that_disagree_equally_are_ordered_by_id():
    """Two renders of one side-car must list tied posts in the same order."""
    items = [_item(i, topics=("ai-coding",)) for i in ("b", "c", "a")]
    assessments = {
        i.id: _assessment(i, membership={"ai-coding": 0.4, "startups": 0.1}) for i in items
    }

    data = _data(items, assessments)

    assert [post["id"] for post in data["posts"]] == ["a", "b", "c"]


def test_the_disagreement_count_is_the_one_the_report_computes():
    """The page's filter count and `jev report`'s "posts con desacuerdo" are one number."""
    agree, differ = _item("1", topics=("ai-coding",)), _item("2", topics=("startups",))

    data = _data([agree, differ], {"1": _assessment(agree), "2": _assessment(differ)})

    assert data["summary"]["posts_with_disagreement"] == 1
    assert sum(1 for post in data["posts"] if post["disagreements"]) == 1


def test_a_truncated_assessment_is_flagged_on_its_row():
    """The page marks the post "recortado": Jev judged a CUT post, the one to distrust."""
    item = _item()
    assessment = _assessment(item).model_copy(update={"truncated": True})

    assert _post(_data([item], {"1": assessment}), "1")["truncated"] is True


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

    assert _post(data, "1")["tokens"] == 2500
    assert _post(data, "1")["cost_usd"] == tokens_cost_usd(2500, PRICED_PROVIDER)
    assert _post(data, "2")["tokens"] is None and _post(data, "2")["cost_usd"] is None
    assert _post(data, "3")["cost_usd"] is None and _post(data, "3")["unpriced"] is True
    assert _post(data, "1")["unpriced"] is False
    per_post = data["cost"]["per_post"]
    assert per_post["mean_usd"] == tokens_cost_usd(2500, PRICED_PROVIDER)
    assert (per_post["n"], per_post["of"]) == (1, 3)
    assert per_post["unpriced_providers"] == ["otro-juez"]


def test_an_item_without_enrichment_is_not_a_row_but_is_counted():
    """`compare_item` returns None for it: there is nothing to disagree with."""
    item, plain = _item("1"), _item("2")
    plain.enriched = None

    data = _data([item, plain], {"1": _assessment(item), "2": _assessment(plain)})

    assert [post["id"] for post in data["posts"]] == ["1"]
    assert (data["totals"]["compared"], data["totals"]["not_compared"]) == (1, 1)


def test_an_item_enrich_left_without_a_primary_is_still_a_row():
    """No primary is not "nothing to compare": its assigned topics still have probabilities."""
    item = _item("1", topics=("ai-coding",))
    item.enriched.primary_topic = None

    post = _post(_data([item], {"1": _assessment(item)}), "1")

    assert post["primary"] is None and post["primary_agrees"] is False
    assert post["enrich"] == [{"slug": "ai-coding", "p": 0.9, "ok": True}]


def test_the_row_links_the_post_and_its_note():
    item = _item("1")

    post = _post(_data([item], {"1": _assessment(item)}, id2note={"1": "/v/items/1.md"}), "1")

    # Against the SOURCE, not a constant.
    assert post["url"] == item.url and post["note"] == "/v/items/1.md"
    assert post["handle"] == "alice"


def test_a_post_past_the_budget_is_cut_with_an_ellipsis():
    long_post = "palabra " * 80
    items = [_item("1", text=long_post), _item("2", text="Claude Code hooks")]
    data = _data(items, {i.id: _assessment(i) for i in items})

    cut, whole = _post(data, "1")["text"], _post(data, "2")["text"]
    assert len(cut) == 240 and cut.endswith("…")
    assert cut[:239] == " ".join(long_post.split())[:239]
    assert whole == "Claude Code hooks"


@pytest.mark.parametrize(
    ("length", "expected"),
    [(239, "x" * 239), (240, "x" * 240), (241, "x" * 239 + "…")],
)
def test_the_cut_is_tested_at_the_boundary(length, expected):
    item = _item("1", text="x" * length)

    assert _post(_data([item], {"1": _assessment(item)}), "1")["text"] == expected


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

    assert [post["id"] for post in data["posts"]] == ["1"]
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
    # The only inputs are the search box and the "only disagreements" switch.
    inputs = re.findall(r"<input[^>]*>", template)
    assert len(inputs) == 2
    assert any('type="search"' in tag for tag in inputs)
    assert any('type="checkbox"' in tag and "checked" in tag for tag in inputs)


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


def test_the_page_reads_the_form_state_the_browser_restored():
    """A reload restores the checkbox and the search box; the table must follow them, not
    the defaults the script started with."""
    template = _resource("jev.template.html")
    boot = template[template.index("function boot(") :]

    assert "onlyDisagreements = $('only-disagreements').checked" in boot
    assert "$('search').value" in boot


def test_the_page_handles_the_cost_error_the_unpriced_row_and_an_empty_table():
    template = _resource("jev.template.html")

    assert "c.error" in template  # the run log could not be read: say so in the strip
    assert "sin tarifa" in template  # a row whose provider nobody prices
    assert "No hay posts comparados" in template  # distinct from "nothing matches"
    assert "per_post.n" in template and "per_post.of" in template  # "media de K de las N"
    assert "posts_with_disagreement" in template  # the report's count, not the page's
    # One money format across the strip: the shared Python sentence is not shipped.
    assert "side_car_text" not in template

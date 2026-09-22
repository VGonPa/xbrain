# tests/test_jev_report.py
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.jev.assess import build_topic_state, questions_digest, topic_contract
from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.questions import STATE_KEY, build_topic_questions
from xbrain.jev.report import (
    THRESHOLD_DEPENDENT_KEYS,
    THRESHOLD_FREE_KEYS,
    Pair,
    build_report,
    compare_item,
    current_assessments,
    render_report_markdown,
    summarize,
    write_reports,
)
from xbrain.models import Author, Enrichment, Item, Topic

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
VOCAB = [
    Topic(slug="ai-coding", description="Construir software con IA."),
    Topic(slug="startups", description="Fundar empresas."),
    Topic(slug="misc", description="Lo demás."),
]
FALLBACK = "otro"
CHAR_LIMIT = 100_000
#: The price `jev.defaults.INPUT_USD_PER_MTOK` knows. Assessments are built with it by
#: default so `cost_usd` is a real number; `provider="fake"` is the unpriced counterpart.
PRICED_PROVIDER = "typesafe"


def _cells(row: str) -> list[str]:
    r"""The cells a markdown renderer sees in one row.

    `\\` is a LITERAL backslash, not an escape, so it is removed BEFORE the split: a bare
    `(?<!\\)\|` lookbehind reads `\\|` as an escaped pipe, which is exactly the mistake the
    implementation used to make. An oracle that shares the bug certifies it.
    """
    return re.split(r"(?<!\\)\|", re.sub(r"\\\\", "", row))


def _rows_under(text: str, heading: str) -> list[str]:
    """The DATA rows of the table under `heading` — header and separator dropped.

    Binds a row to the section it is printed under. An unanchored `"| 1 | misc |" in text`
    passes just as happily when two section titles have been swapped.
    """
    body = text.split(heading, 1)[1].split("\n## ", 1)[0].splitlines()
    return [line for line in body if line.startswith("| ") and not line.startswith("| item")]


def _headline_line(text: str, prefix: str) -> str:
    """The single headline line starting with `prefix` — assert on the line, not the file."""
    return next(line for line in text.splitlines() if line.startswith(prefix))


def _item(
    item_id="1", text="Claude Code hooks", topics=("ai-coding", "misc"), enriched=True
) -> Item:
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
        )
        if enriched
        else None,
    )


def _cheap_assessment(membership: dict[str, float], choice: str = "a-perfect") -> TopicAssessment:
    """An assessment with a PLACEHOLDER contract, for tests where currency is not the subject.

    `_assessment` computes the real contract, which costs a question-set hash per call — fine
    for a handful of fixtures, not for the 2,500 a rounding-order test needs.
    """
    return TopicAssessment(
        item_id="x",
        provider=PRICED_PROVIDER,
        model="jev-1.13.0",
        asked_at=DT,
        contract="a" * 64,
        state_chars=3,
        membership=membership,
        primary=PrimaryChoice(choice=choice, confidence=0.7, probabilities={choice: 1.0}),
        input_tokens=0,
    )


def _assessment(
    item: Item,
    membership,
    choice="ai-coding",
    probabilities=None,
    tokens=1000,
    provider=PRICED_PROVIDER,
) -> TopicAssessment:
    """A record whose `contract` is the one TODAY'S ask would stamp, so it reads as current.

    The contract is built the way `assess.assess_topics` builds it — the state as sent and
    the digest of the questions that went with it — rather than being hand-written, so a
    change to either composition shows up here as a stale fixture rather than as a test
    that keeps passing over a contract nobody computes any more.
    """
    state, state_chars = build_topic_state(item, CHAR_LIMIT)
    digest = questions_digest(build_topic_questions(VOCAB, FALLBACK))
    return TopicAssessment(
        item_id=item.id,
        provider=provider,
        model="jev-1.13.0",
        asked_at=DT,
        contract=topic_contract(state[STATE_KEY], digest),
        state_chars=state_chars,
        membership=membership,
        primary=PrimaryChoice(
            choice=choice,
            confidence=0.7,
            probabilities=probabilities
            or {"ai-coding": 0.6, "startups": 0.3, "misc": 0.1, "otro": 0.0},
        ),
        input_tokens=tokens,
    )


def test_compare_item_splits_doubtful_missing_and_ranks_the_primary():
    item = _item()
    assessment = _assessment(
        item,
        {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2},
        choice="startups",
        probabilities={"startups": 0.5, "ai-coding": 0.4, "misc": 0.1, "otro": 0.0},
    )
    comparison = compare_item(item, assessment, 0.85)
    assert comparison is not None
    assert comparison.assigned == ("ai-coding", "misc")
    assert comparison.doubtful == (Pair("1", "misc", 0.2),)
    assert comparison.missing == (Pair("1", "startups", 0.9),)
    assert comparison.jev_assigned == ("ai-coding", "startups")
    assert comparison.jev_primary == "startups"
    assert comparison.primary_agrees is False
    assert comparison.primary_rank == 2  # ai-coding is second in the choice distribution


def test_compare_item_reports_no_rank_and_no_agreement_when_enrich_has_no_primary():
    """An item enrich left without a primary topic has nothing to agree WITH.

    `primary_agrees` must be False and `primary_rank` None. Reporting agreement — or a rank
    for a topic that does not exist — would let a corpus with no primaries at all show a
    perfect agreement rate, which is the one number the report is read for.
    """
    item = _item()
    item.enriched = Enrichment(
        enriched_at=DT,
        executor="claude-code",
        summary="s",
        primary_topic=None,
        topics=["ai-coding", "misc"],
    )
    assessment = _assessment(item, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1})

    comparison = compare_item(item, assessment, 0.85)

    assert comparison is not None
    assert comparison.primary_topic is None
    assert comparison.primary_rank is None
    assert comparison.primary_agrees is False
    assert summarize([(item, assessment)], VOCAB, 0.85)["primary_agree"] == 0


def test_compare_item_is_none_without_enrichment():
    item = _item(enriched=False)
    assert compare_item(item, _assessment(item, {"ai-coding": 0.9, "startups": 0.1}), 0.85) is None


def test_current_assessments_drops_stale_ones():
    item = _item()
    fresh = _assessment(item, {"ai-coding": 0.9, "startups": 0.1, "misc": 0.1})
    stale = fresh.model_copy(update={"contract": "f" * 64})
    kw = {"fallback": FALLBACK, "char_limit": CHAR_LIMIT}
    assert current_assessments([item], {"1": fresh}, VOCAB, **kw) == [(item, fresh)]
    assert current_assessments([item], {"1": stale}, VOCAB, **kw) == []


def test_summarize_counts_pairs_topics_primary_and_cost():
    a = _item("1")
    b = _item("2", text="Seed round", topics=("startups",))
    pairs = [
        (a, _assessment(a, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2}, choice="ai-coding")),
        (
            b,
            _assessment(
                b,
                {"ai-coding": 0.1, "startups": 0.99, "misc": 0.05},
                choice="otro",
                probabilities={"otro": 0.6, "startups": 0.4, "ai-coding": 0.0, "misc": 0.0},
                tokens=500,
            ),
        ),
    ]
    summary = summarize(pairs, VOCAB, 0.85)
    assert summary["items_assessed"] == 2 and summary["items_compared"] == 2
    assert summary["assigned_pairs"] == 3 and summary["assigned_backed"] == 2
    assert summary["enrich_backed_pct"] == 66.7
    assert summary["jev_pairs"] == 3 and summary["jev_backed"] == 2
    assert summary["jev_backed_pct"] == 66.7
    assert summary["doubtful_pairs"] == 1 and summary["missing_pairs"] == 1
    assert summary["primary_agree"] == 1 and summary["primary_agree_pct"] == 50.0
    assert summary["primary_fallback"] == 1
    assert summary["input_tokens"] == 1500 and summary["cost_usd"] == 0.0001
    assert summary["models"] == {"jev-1.13.0": 2}
    by_slug = {row["slug"]: row for row in summary["per_topic"]}
    # `doubtful` and `unjudged` ride WITH the row, not only in the corpus total: they are
    # already counted by `_slug_counts`, the dashboard's chart-01 tooltip displays them per
    # topic, and a number displayed per topic that only exists as a corpus sum is a number
    # nothing can check. `assigned = backed + doubtful + unjudged`, row by row.
    assert by_slug["misc"] == {
        "slug": "misc",
        "assigned": 1,
        "backed": 0,
        "doubtful": 1,
        "unjudged": 0,
        "backed_pct": 0.0,
        "missing": 0,
    }
    assert by_slug["startups"] == {
        "slug": "startups",
        "assigned": 1,
        "backed": 1,
        "doubtful": 0,
        "unjudged": 0,
        "backed_pct": 100.0,
        "missing": 1,
    }
    assert [row["slug"] for row in summary["per_topic"]] == ["misc", "ai-coding", "startups"]


def test_summarize_prices_each_record_by_its_own_provider_and_names_the_unpriced_one():
    """An unpriced judge contributes 0.0 to the bill, and the summary says WHICH zero it is.

    `INPUT_USD_PER_MTOK` prices `typesafe` and nothing else, so a panel that mixes judges
    must not borrow one vendor's rate for another's records — an invented rate reads as a
    bill. A bare `0.0` cannot say "nobody prices this judge", which is why the provider is
    NAMED, exactly as `cli._jev_cost_line` names it for a run.
    """
    a = _item("1")
    b = _item("2", text="Seed round", topics=("startups",))
    million = 1_000_000
    pairs = [
        (a, _assessment(a, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1}, tokens=million)),
        (
            b,
            _assessment(
                b, {"ai-coding": 0.1, "startups": 0.9, "misc": 0.1}, provider="fake", tokens=million
            ),
        ),
    ]
    summary = summarize(pairs, VOCAB, 0.85)
    # Two million tokens counted, only the typesafe million billable: 1 Mtok × 0.042 $/Mtok.
    # A literal, not the code's own formula — and one that separates the two mutations this
    # test exists for: flat-rating both records gives 0.084, pricing neither gives 0.0.
    assert summary["input_tokens"] == 2 * million
    assert summary["cost_usd"] == 0.042
    assert summary["providers"] == {PRICED_PROVIDER: 1, "fake": 1}
    assert summary["unpriced_providers"] == ["fake"]
    # Both judges answered with the same model name, so `models` counts two.
    assert summary["models"] == {"jev-1.13.0": 2}


def test_summarize_counts_records_that_reported_no_token_usage_separately():
    """`input_tokens=None` is a documented provider behaviour; folding it into 0 would
    report a run that WAS paid for as free. The count is what makes the total read as
    "at least this much"."""
    item = _item()
    pairs = [
        (item, _assessment(item, {"ai-coding": 0.9, "startups": 0.1, "misc": 0.1}, tokens=None))
    ]
    summary, comparisons = build_report(pairs, VOCAB, 0.85)
    assert summary["input_tokens"] == 0
    assert summary["input_tokens_unknown"] == 1
    assert summary["cost_usd"] == 0.0
    # And the marker reaches the reader: `Coste: 0 tokens de entrada` on its own describes
    # a run that was paid for as free.
    assert "(+1 sin recuento)" in render_report_markdown(summary, comparisons, {"1": item})


def test_summarize_does_not_count_a_topic_jev_was_never_asked_about_as_backed():
    """An assigned topic that has left the vocabulary is UNJUDGED, never "backed".

    `enrich` validates topics against the vocabulary AT WRITE TIME, so an item enriched
    under an older `vocab.yaml` can carry a slug today's questions no longer ask about.
    Such a pair is in neither `membership` nor `doubtful`, so counting `backed` as
    `assigned - doubtful` would silently report Jev as endorsing an assignment Jev was
    never shown. The three buckets PARTITION `assigned_pairs`.
    """
    item = _item(topics=("ai-coding", "retired-topic"))
    pairs = [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1}))]
    summary = summarize(pairs, VOCAB, 0.85)
    assert summary["assigned_pairs"] == 2
    assert summary["assigned_backed"] == 1
    assert summary["doubtful_pairs"] == 0
    assert summary["assigned_unjudged"] == 1
    assert (
        summary["assigned_backed"] + summary["doubtful_pairs"] + summary["assigned_unjudged"]
        == summary["assigned_pairs"]
    )
    assert summary["enrich_backed_pct"] == 50.0


def test_markdown_and_files(tmp_path: Path):
    item = _item()
    pairs = [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2}))]
    summary = summarize(pairs, VOCAB, 0.85)
    comparisons = [c for i, a in pairs if (c := compare_item(i, a, 0.85))]
    text = render_report_markdown(summary, comparisons, {"1": item})
    assert text.startswith("# Jev · topics")
    # On the line they belong to, not anywhere in the document.
    assert (
        _headline_line(text, "Umbral ")
        == "Umbral 0.850 · modelo: jev-1.13.0 (1) · proveedor: typesafe (1)"
    )
    assert "| 1 | misc | 0.200 |" in text  # doubtful row
    assert "| 1 | startups | 0.900 |" in text  # missing row
    json_path, md_path = write_reports(summary, comparisons, {"1": item}, tmp_path / "jev")
    assert json_path.name == "topics-report.json" and md_path.name == "topics-report.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["doubtful_pairs"] == 1
    assert payload["items"][0]["doubtful"] == [{"slug": "misc", "noul": 0.2}]


def test_markdown_names_the_unpriced_provider_rather_than_printing_a_bare_zero():
    item = _item()
    pairs = [
        (item, _assessment(item, {"ai-coding": 0.9, "startups": 0.1, "misc": 0.1}, provider="fake"))
    ]
    summary, comparisons = build_report(pairs, VOCAB, 0.85)
    text = render_report_markdown(summary, comparisons, {"1": item})
    assert "proveedor sin tarifa: fake" in text


def test_a_boundary_noul_reads_like_the_page_and_not_like_its_own_contradiction():
    """`0.8496` under a heading that means "below 0.850" must not render as `0.85`.

    ONE side-car, two artifacts, and this was the one place they printed different numbers
    for the same value: `jev.html` renders every noul compared against the umbral at three
    decimals — it was changed to, because two decimals made 331 rows on the real corpus
    contradict their own heading — while `topics-report.md` still rendered two. A pair at
    0.8496 is doubtful, and under `## Dudosas` it printed `0.85` beside a headline reading
    `Umbral 0.85`: a row that reads as a bug in the tool.

    The umbral is formatted the same way for the same reason — a reader compares the column
    with the headline, and a like-for-like comparison needs both at one precision.
    """
    item = _item()
    # `choice="startups"` against enrich's `ai-coding` primary, so the real-disagreement
    # section has a row and its `conf.` column can be pinned in the same assertion.
    pairs = [
        (
            item,
            _assessment(item, {"ai-coding": 0.8496, "startups": 0.9, "misc": 0.2}, "startups"),
        )
    ]
    summary, comparisons = build_report(pairs, VOCAB, 0.85)

    text = render_report_markdown(summary, comparisons, {"1": item})

    assert "| 1 | ai-coding | 0.850 |" in text, text
    assert _headline_line(text, "Umbral ").startswith("Umbral 0.850 ·")
    # The confidence column is deliberately NOT moved: it is never compared against the
    # umbral, and the page prints it at two decimals too.
    assert "| 1 | ai-coding | startups | 0.70 |" in text, text


# ------------------------------------------------------------------- markdown rendering


def test_markdown_escapes_a_pipe_in_the_post_text_instead_of_splitting_the_row():
    """A post containing `|` must not grow extra cells in the table it lands in.

    This corpus is AI/dev posts from X, where shell pipelines (`cat x | grep y`) and `A | B`
    phrasing are ordinary, so an unescaped pipe is not a corner case — it is a ragged row in
    the one file a person actually reads.
    """
    item = _item(text="cost | benefit | ratio")
    pairs = [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2}))]
    summary = summarize(pairs, VOCAB, 0.85)
    comparisons = [c for i, a in pairs if (c := compare_item(i, a, 0.85))]

    text = render_report_markdown(summary, comparisons, {"1": item})

    row = next(line for line in text.splitlines() if line.startswith("| 1 | misc |"))
    assert "cost \\| benefit \\| ratio" in row
    # `| item | topic | noul | texto |` is four cells between five unescaped pipes, so a split
    # on them yields six fields: the four cells plus the empty ones outside the leading and
    # trailing bars. Splitting on UNESCAPED pipes is what a markdown renderer does, so it is
    # what the count must use.
    assert len(_cells(row)) == 6


def test_summarize_reports_cost_as_a_float_even_when_nothing_was_priced():
    """`sum()` over an empty run returns `int 0`, and `round(0, 4)` keeps it an int.

    The dashboard consumes this key; a type that changes with the contents of the side-car
    is drift, and `~0 $` instead of `~0.0 $` is the same drift reaching the reader.
    """
    summary = summarize([], VOCAB, 0.85)
    assert summary["cost_usd"] == 0.0
    assert isinstance(summary["cost_usd"], float)
    assert "~0.0000 $" in render_report_markdown(summary, [], {})


def test_per_topic_puts_never_assigned_topics_after_the_ones_with_a_real_backing_rate():
    """ "Peor primero" means worst BACKING first, and a topic nobody assigned has no backing
    rate to be worst at.

    `_pct` returns 0.0 for a zero whole, so a 30-topic vocabulary with a long tail of unused
    topics would bury the answer this table exists to give under rows about nothing.
    """
    item = _item(topics=("ai-coding",))
    pairs = [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1}))]

    rows = summarize(pairs, VOCAB, 0.85)["per_topic"]

    assert [row["slug"] for row in rows] == ["ai-coding", "misc", "startups"]
    assert rows[0]["assigned"] == 1 and rows[0]["backed_pct"] == 100.0
    assert all(row["assigned"] == 0 for row in rows[1:])


def test_summarize_counts_a_primary_jev_was_never_asked_about_separately():
    """A primary that left the vocabulary is "Jev was never asked", not "Jev disagrees".

    The membership side already gives that fact its own bucket (`assigned_unjudged`);
    counting the same fact as a primary disagreement is the asymmetry that bucket exists to
    remove. The denominator still counts the item — exactly as `assigned_pairs` still counts
    an unjudged pair — so the bucket makes the reason visible without hiding the item.
    """
    item = _item(topics=("retired-topic", "ai-coding"))
    assessment = _assessment(item, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1})

    comparison = compare_item(item, assessment, 0.85)
    summary = summarize([(item, assessment)], VOCAB, 0.85)

    assert comparison is not None
    assert comparison.primary_topic == "retired-topic"
    assert comparison.primary_unjudged is True
    assert summary["primary_unjudged"] == 1
    assert summary["primary_agree"] == 0
    assert summary["primary_agree_pct"] == 0.0
    # An in-vocabulary primary is NOT unjudged, whether or not Jev agreed with it.
    agreed = _item(topics=("ai-coding", "misc"))
    other = _assessment(agreed, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1})
    assert summarize([(agreed, other)], VOCAB, 0.85)["primary_unjudged"] == 0


def test_build_report_returns_the_comparisons_the_summary_was_computed_from():
    """One comparison pass, not two.

    The CLI needs both halves; re-running `compare_item` for the second is twice the work on
    a ~3,000-item corpus and a second call site that has to be handed the same threshold.
    """
    a = _item("1")
    b = _item("2", text="Seed round", topics=("startups",), enriched=False)
    pairs = [
        (a, _assessment(a, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2})),
        (b, _assessment(b, {"ai-coding": 0.1, "startups": 0.99, "misc": 0.05})),
    ]
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    summary, comparisons = build_report(pairs, VOCAB, 0.85, now=now)

    assert summary == summarize(pairs, VOCAB, 0.85, now=now)
    # `b` has no enrichment, so it is assessed but not comparable — the list is the FILTERED
    # one the summary counted, not one comparison per pair.
    assert summary["items_assessed"] == 2 and summary["items_compared"] == 1
    assert comparisons == [compare_item(a, pairs[0][1], 0.85)]


def test_summarize_stamps_when_it_ran_and_the_markdown_reads_that_stamp():
    """The JSON carries the stamp too, and the markdown reads it instead of the clock.

    Task 5's dashboard reads the JSON; without a stamp it cannot say how old the report it is
    rendering is. Taking the header date from the summary also makes `render_report_markdown`
    a pure function of its inputs — it was the one clock call in the module.
    """
    item = _item()
    pairs = [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1}))]
    stamped = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    summary, comparisons = build_report(pairs, VOCAB, 0.85, now=stamped)

    assert summary["generated_at"] == "2026-01-02T03:04:05+00:00"
    text = render_report_markdown(summary, comparisons, {"1": item})
    assert text.startswith("# Jev · topics — 2026-01-02")


def test_snippet_truncates_a_long_post_and_still_escapes_every_pipe_it_keeps():
    """The cut runs first and the escape second, so the two cannot interfere.

    Escaping first and cutting after could land the cut between a backslash and its pipe,
    leaving a half-written escape in the cell. Whichever way the post is cut, the row it
    lands in still has exactly the cells its header declares.
    """
    item = _item(text="a | b " * 40)  # 240 characters, pipes throughout, well past the cut
    pairs = [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2}))]
    summary, comparisons = build_report(pairs, VOCAB, 0.85)

    text = render_report_markdown(summary, comparisons, {"1": item})

    row = next(line for line in text.splitlines() if line.startswith("| 1 | misc |"))
    assert "…" in row  # the post really was cut
    assert "a \\| b" in row  # and what survived is still escaped
    assert len(_cells(row)) == 6


# ------------------------------------------------------------------- the threshold boundary


def test_a_noul_exactly_at_the_threshold_counts_as_backed_not_doubtful():
    """`defaults.DEFAULT_THRESHOLD` defines backing as AT OR ABOVE `t`, and `[jev].threshold`
    defaults to exactly 0.85 while providers routinely answer rounded values.

    One flipped comparison silently moves an item between "respaldada" and "dudosa" in every
    number both files quote, and no fixture with 0.95/0.20 values can see it.
    """
    item = _item(topics=("ai-coding", "misc"))
    assessment = _assessment(item, {"ai-coding": 0.85, "startups": 0.85, "misc": 0.84})

    comparison = compare_item(item, assessment, 0.85)

    assert comparison is not None
    assert comparison.doubtful == (Pair("1", "misc", 0.84),)  # only what is BELOW t
    assert comparison.missing == (Pair("1", "startups", 0.85),)  # at t, unassigned -> candidate
    assert comparison.jev_assigned == ("ai-coding", "startups")  # at t -> Jev backs it


def test_threshold_zero_backs_everything_and_threshold_one_backs_only_certainty():
    """Both ends of the closed interval are legal and meaningful, and they are opposites."""
    item = _item(topics=("ai-coding", "misc"))
    assessment = _assessment(item, {"ai-coding": 1.0, "startups": 0.85, "misc": 0.0})

    at_zero = compare_item(item, assessment, 0.0)
    at_one = compare_item(item, assessment, 1.0)

    assert at_zero is not None and at_one is not None
    assert at_zero.doubtful == ()
    assert at_zero.jev_assigned == ("ai-coding", "startups", "misc")
    assert [pair.slug for pair in at_one.doubtful] == ["misc"]
    assert at_one.jev_assigned == ("ai-coding",)
    assert at_one.missing == ()


def test_doubtful_and_missing_are_ordered_within_one_item():
    """Per-item ordering, with TWO pairs — one pair cannot show a direction."""
    item = _item(topics=("ai-coding", "misc", "startups"))
    assessment = _assessment(item, {"ai-coding": 0.95, "startups": 0.60, "misc": 0.20})

    comparison = compare_item(item, assessment, 0.85)

    assert comparison is not None
    assert comparison.doubtful == (Pair("1", "misc", 0.2), Pair("1", "startups", 0.6))


def test_missing_is_ordered_strongest_first_within_one_item():
    item = _item(topics=("misc",))
    assessment = _assessment(item, {"ai-coding": 0.90, "startups": 0.99, "misc": 0.95})

    comparison = compare_item(item, assessment, 0.85)

    assert comparison is not None
    assert comparison.missing == (Pair("1", "startups", 0.99), Pair("1", "ai-coding", 0.9))


def test_primary_rank_breaks_a_tie_by_option_name():
    """The tie-break is the property `primary_rank`'s docstring sells: "two runs of the same
    distribution can never report different ranks".

    Distributions are stored and re-read as JSON, so insertion order is whatever the provider
    sent — a stable sort with no tie-break really does rank the same numbers differently.
    """
    item = _item(topics=("ai-coding", "misc"))
    assessment = _assessment(
        item,
        {"ai-coding": 0.9, "startups": 0.1, "misc": 0.1},
        choice="misc",
        # Insertion order deliberately NOT alphabetical: a stable sort with no tie-break ranks
        # `ai-coding` 2nd, the tie-break ranks it 1st.
        probabilities={"misc": 0.4, "ai-coding": 0.4, "startups": 0.2, "otro": 0.0},
    )

    comparison = compare_item(item, assessment, 0.85)

    assert comparison is not None
    assert comparison.primary_rank == 1


# ------------------------------------------------------------- what the side-car dropped


def test_the_summary_counts_the_side_car_records_the_filter_dropped():
    """A retired side-car must never render identically to one nobody ever wrote.

    The three numbers PARTITION the side-car, so a record cannot leave the comparison without
    a counter saying where it went — the difference between "run `jev topics`" and "you have
    just retired 2,400 paid records" is the price of the whole corpus.
    """
    item = _item()
    pairs = [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1}))]

    summary, comparisons = build_report(pairs, VOCAB, 0.85, stale=4, orphans=2)

    assert summary["assessments_stored"] == 7
    assert summary["assessments_stale"] == 4
    assert summary["assessments_orphaned"] == 2
    assert summary["items_assessed"] == 1
    text = render_report_markdown(summary, comparisons, {"1": item})
    assert _headline_line(text, "Evaluaciones:") == (
        "Evaluaciones: 1 vigentes de 7 guardadas · 4 caducadas · 2 huérfanas"
    )


def test_a_lone_stale_record_agrees_in_number_and_grammar():
    summary = summarize([], VOCAB, 0.85, stale=1)
    assert "· 1 caducada" in render_report_markdown(summary, [], {})
    # Nothing orphaned: the segment has nothing to say and is not printed.
    assert "huérfana" not in render_report_markdown(summary, [], {})


# ------------------------------------------------------------------ the threshold key sets


def test_every_summary_key_is_declared_threshold_dependent_or_threshold_free():
    """Task 5's slider recomputes the dependent keys client-side and keeps the rest.

    A key in neither set is one the dashboard has to guess about, and a wrong guess is a
    number on screen the report would not print. A key in BOTH is a contradiction.
    """
    item = _item()
    summary = summarize(
        [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1}))],
        VOCAB,
        0.85,
    )

    assert THRESHOLD_DEPENDENT_KEYS & THRESHOLD_FREE_KEYS == frozenset()
    assert THRESHOLD_DEPENDENT_KEYS | THRESHOLD_FREE_KEYS == set(summary)


def test_the_threshold_dependent_keys_are_the_ones_that_actually_move():
    """Declared, then VERIFIED against two real thresholds: a key that moves while declared
    free is exactly the drift the two sets exist to prevent."""
    item = _item(topics=("ai-coding", "misc"))
    pairs = [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.90, "misc": 0.20}))]
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    low = summarize(pairs, VOCAB, 0.10, now=now)
    high = summarize(pairs, VOCAB, 0.99, now=now)

    moved = {key for key in low if low[key] != high[key]}
    assert moved <= THRESHOLD_DEPENDENT_KEYS
    # And it is not a vacuous subset: the headline numbers really do move.
    assert {"doubtful_pairs", "jev_pairs", "enrich_backed_pct"} <= moved


# --------------------------------------------------------------------- per-topic ordering


def test_per_topic_orders_on_the_exact_ratio_not_the_rounded_percentage():
    """2499 of 2500 rounds to 100.0 and must still sort BELOW a perfect 2500 of 2500.

    `_pct` keeps one decimal for the reader. Sorting on it makes the one topic with a real
    disagreement tie with every perfect topic and then fall to ALPHABETICAL order — in a
    30-topic vocabulary the answer lands below rows about nothing, under "peor primero".
    """
    vocab = [
        Topic(slug="a-perfect", description="Respaldado siempre."),
        Topic(slug="b-almost", description="Respaldado casi siempre."),
    ]
    backed = _cheap_assessment({"a-perfect": 0.99, "b-almost": 0.99})
    doubted = _cheap_assessment({"a-perfect": 0.99, "b-almost": 0.10})
    pairs = [
        (_item(str(i), topics=("a-perfect", "b-almost")), backed if i else doubted)
        for i in range(2500)
    ]

    rows = summarize(pairs, vocab, 0.85)["per_topic"]

    # Both render 100.0 to the reader; only the exact ratio can order them.
    assert [row["backed_pct"] for row in rows] == [100.0, 100.0]
    assert [row["slug"] for row in rows] == ["b-almost", "a-perfect"]
    assert rows[0]["backed"] == 2499 and rows[1]["backed"] == 2500


def test_a_per_topic_row_partitions_its_own_assigned_pairs():
    """`backed + doubtful + unjudged == assigned`, per row, the same arithmetic as the
    corpus-wide total. The shelter that makes `unjudged` zero per row depends on the caller
    having filtered for currency, and `summarize` is public."""
    a = _item("1", topics=("ai-coding", "misc"))
    b = _item("2", text="Seed round", topics=("startups",))
    pairs = [
        (a, _assessment(a, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.20})),
        (b, _assessment(b, {"ai-coding": 0.1, "startups": 0.99, "misc": 0.1})),
    ]

    summary = summarize(pairs, VOCAB, 0.85)

    by_slug = {row["slug"]: row for row in summary["per_topic"]}
    assert by_slug["misc"]["assigned"] == 1 and by_slug["misc"]["backed"] == 0
    assert by_slug["ai-coding"]["backed"] == 1
    assert by_slug["startups"]["backed"] == 1
    # The rows account for every judged assigned pair the corpus-wide total counts.
    assert sum(row["assigned"] for row in summary["per_topic"]) == (
        summary["assigned_pairs"] - summary["assigned_unjudged"]
    )
    # And each row's own three buckets partition it, which is what lets a consumer show
    # `dudosas` / `sin juzgar` per topic against a number the report also emits.
    for row in summary["per_topic"]:
        assert row["backed"] + row["doubtful"] + row["unjudged"] == row["assigned"]
    assert sum(row["doubtful"] for row in summary["per_topic"]) == summary["doubtful_pairs"]


# ------------------------------------------------------- markdown: sections, order, cuts


def test_the_tables_are_worst_first_under_their_own_heading_and_cut_at_top():
    """Binds three things one substring assertion cannot: which rows go under which heading,
    which direction each table sorts, and that `top` really cuts.

    Under a swapped-title mutant the header still says "peor primero" while the table shows
    the twenty LEAST doubtful pairs, and the reader has no way to tell.
    """
    items = [_item(str(i), topics=("misc",)) for i in (1, 2, 3)]
    pairs = [
        (item, _assessment(item, {"ai-coding": 0.99, "startups": 0.86 + i / 100, "misc": noul}))
        for i, (item, noul) in enumerate(zip(items, [0.10, 0.40, 0.80]))
    ]
    summary, comparisons = build_report(pairs, VOCAB, 0.85)

    text = render_report_markdown(summary, comparisons, {i.id: i for i in items}, top=2)

    # Ascending noul, and only `top` of them.
    doubtful = [_cells(row)[3].strip() for row in _rows_under(text, "## Dudosas")]
    assert doubtful == ["0.100", "0.400"]
    # Descending noul, so the STRONGEST candidates are the ones that survive the cut.
    missing = [_cells(row)[3].strip() for row in _rows_under(text, "## Candidatas que faltan")]
    assert missing == ["0.990", "0.990"]


def test_a_cut_table_says_how_many_rows_it_dropped():
    """`top 20` reads the same whether there were 7 rows or 700. A silent drop in the artifact
    that exists to be read is the same failure as a silent drop in a number."""
    items = [_item(str(i), topics=("misc",)) for i in (1, 2, 3)]
    pairs = [
        (item, _assessment(item, {"ai-coding": 0.1, "startups": 0.1, "misc": noul}))
        for item, noul in zip(items, [0.10, 0.40, 0.80])
    ]
    summary, comparisons = build_report(pairs, VOCAB, 0.85)

    cut = render_report_markdown(summary, comparisons, {i.id: i for i in items}, top=2)
    whole = render_report_markdown(summary, comparisons, {i.id: i for i in items}, top=20)

    assert "_… y 1 fila más (el JSON las lleva todas)._" in cut
    assert "fila más" not in whole and "filas más" not in whole


def test_markdown_escapes_a_backslash_before_a_pipe_instead_of_re_opening_the_cell():
    r"""A post containing `\|` must not re-open the cell.

    Escaping only the pipe turns `grep 'foo\|bar'` into `foo\\|bar`, and cmark-gfm's row
    scanner consumes `\` plus ONE character: it eats both backslashes and the `|` becomes a
    LIVE delimiter. A BRE alternation, a shell escape and a Windows path all hit this.
    """
    item = _item(text=r"grep 'foo\|bar' file")
    summary, comparisons = build_report(
        [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2}))],
        VOCAB,
        0.85,
    )

    text = render_report_markdown(summary, comparisons, {"1": item})

    row = _rows_under(text, "## Dudosas")[0]
    assert len(_cells(row)) == 6
    assert r"grep 'foo\\\|bar' file" in row


def test_a_multiline_post_stays_on_one_row():
    """The other half of `_snippet`'s contract: a newline ENDS the row, and multi-line posts
    are the common case on X."""
    item = _item(text="línea uno\nlínea dos")
    summary, comparisons = build_report(
        [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2}))],
        VOCAB,
        0.85,
    )

    text = render_report_markdown(summary, comparisons, {"1": item})

    assert _rows_under(text, "## Dudosas") == ["| 1 | misc | 0.200 | línea uno línea dos |"]


def test_a_row_for_an_item_missing_from_the_store_renders_an_empty_text_cell():
    """A report built from a side-car whose item was deleted still renders a well-formed row."""
    item = _item()
    summary, comparisons = build_report(
        [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2}))],
        VOCAB,
        0.85,
    )

    text = render_report_markdown(summary, comparisons, {})

    row = _rows_under(text, "## Dudosas")[0]
    assert row == "| 1 | misc | 0.200 |  |"
    assert len(_cells(row)) == 6


def test_the_markdown_names_the_unjudged_slugs_not_just_their_count():
    """`**sin juzgar:** 4` cannot tell a reader WHICH topic to put back in `vocab.yaml`.

    The JSON record carried the slugs and the file a person reads did not, which made the one
    bucket that says "your vocabulary moved" the one bucket nobody could act on.
    """
    item = _item(topics=("ai-coding", "retired-topic"))
    summary, comparisons = build_report(
        [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1}))],
        VOCAB,
        0.85,
    )

    text = render_report_markdown(summary, comparisons, {"1": item})

    assert summary["assigned_unjudged"] == 1
    assert _rows_under(text, "## Asignaciones sin juzgar") == [
        "| 1 | retired-topic | Claude Code hooks |"
    ]


def test_the_primary_tables_are_split_by_reason_and_ranked_worst_first():
    """One table headed "en desacuerdo" says the opposite of what the module establishes: an
    item with no primary, and one whose primary left the vocabulary, are not disagreements.

    Within a section the worst conflict comes first — enrich's primary furthest DOWN Jev's own
    ranking — because the file shows only the first `top` rows.
    """
    no_primary = _item("1")
    no_primary.enriched = Enrichment(
        enriched_at=DT, executor="claude-code", summary="s", primary_topic=None, topics=["misc"]
    )
    retired = _item("2", text="Seed round", topics=("retired-topic", "misc"))
    close_call = _item("3", text="Otro post", topics=("startups", "misc"))
    far_conflict = _item("4", text="Un cuarto", topics=("misc", "startups"))
    membership = {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1}
    # `close_call`'s primary is 2nd in Jev's ranking; `far_conflict`'s is 3rd — worse.
    ranked = {"ai-coding": 0.6, "startups": 0.3, "misc": 0.1, "otro": 0.0}
    pairs = [
        (no_primary, _assessment(no_primary, membership, probabilities=ranked)),
        (retired, _assessment(retired, membership, probabilities=ranked)),
        (close_call, _assessment(close_call, membership, probabilities=ranked)),
        (far_conflict, _assessment(far_conflict, membership, probabilities=ranked)),
    ]
    summary, comparisons = build_report(pairs, VOCAB, 0.85)

    text = render_report_markdown(summary, comparisons, {item.id: item for item, _ in pairs})

    assert "## Primario que no coincide (top 20 por motivo)" in text
    assert "### Sin primario en enrich (1)" in text
    assert "### Primario sin juzgar (salió del vocabulario) (1)" in text
    assert "### Desacuerdo real (2)" in text
    # Worst first inside the section: rank 3 (`misc`) before rank 2 (`startups`).
    real = text.split("### Desacuerdo real (2)", 1)[1].splitlines()
    rows = [line for line in real if line.startswith("| 4 |") or line.startswith("| 3 |")]
    assert [_cells(row)[1].strip() for row in rows] == ["4", "3"]


def test_a_primary_the_provider_left_out_of_its_distribution_gets_its_own_bucket():
    """`primary_rank is None` used to mean three different things and count as an ordinary
    disagreement.

    `PrimaryChoice` does not enforce full coverage, so a provider may score a slug in
    `membership` and omit it from the Choice. That is the one case where Jev DEMONSTRABLY
    never considered the topic, and it was the one rendered as a bare `—`.
    """
    item = _item(topics=("misc", "ai-coding"))
    assessment = _assessment(
        item,
        {"ai-coding": 0.95, "startups": 0.1, "misc": 0.90},
        choice="ai-coding",
        probabilities={"ai-coding": 0.9, "startups": 0.1, "otro": 0.0},  # no `misc`
    )

    comparison = compare_item(item, assessment, 0.85)
    summary = summarize([(item, assessment)], VOCAB, 0.85)

    assert comparison is not None
    assert comparison.primary_rank is None
    assert comparison.primary_unjudged is False  # `misc` IS in membership — Jev was asked
    assert comparison.primary_unranked is True
    assert summary["primary_unranked"] == 1
    assert summary["primary_unjudged"] == 0
    text = render_report_markdown(summary, [comparison], {"1": item})
    assert "### Primario ausente de la distribución de Jev (1)" in text


def test_summarize_counts_the_records_whose_evidence_was_cut():
    """Truncation says a judgement was made on PARTIAL evidence, so a silent zero is the
    reading that matters."""
    item = _item()
    assessment = _assessment(item, {"ai-coding": 0.9, "startups": 0.1, "misc": 0.1}).model_copy(
        update={"truncated": True}
    )

    summary, comparisons = build_report([(item, assessment)], VOCAB, 0.85)

    assert summary["truncated"] == 1
    assert "truncados: 1" in render_report_markdown(summary, comparisons, {"1": item})


def test_a_naive_now_is_refused_rather_than_stamped_as_utc():
    """`TopicAssessment.asked_at` refuses exactly this, and says why: we do not coerce,
    because that masks the bug. A consumer ageing the report would read local time as UTC."""
    item = _item()
    pairs = [(item, _assessment(item, {"ai-coding": 0.9, "startups": 0.1, "misc": 0.1}))]

    with pytest.raises(ValueError, match="now"):
        summarize(pairs, VOCAB, 0.85, now=datetime(2026, 1, 2, 3, 4, 5))


# --------------------------------------------------------------- the JSON reader's contract


def test_the_json_record_carries_every_field_a_reader_of_it_reads(tmp_path: Path):
    """The record IS the contract for whoever reads the file. Asserting one key lets the two
    fields this work added vanish from the JSON with nothing going red."""
    item = _item(topics=("retired-topic", "ai-coding"))
    summary, comparisons = build_report(
        [(item, _assessment(item, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2}))],
        VOCAB,
        0.85,
    )

    json_path, _ = write_reports(summary, comparisons, {"1": item}, tmp_path / "jev")

    record = json.loads(json_path.read_text(encoding="utf-8"))["items"][0]
    assert set(record) == {
        "item_id",
        "assigned",
        "primary_topic",
        "jev_primary",
        "jev_confidence",
        "primary_rank",
        "doubtful",
        "missing",
        "jev_assigned",
        "unjudged",
        "primary_unjudged",
        "primary_unranked",
        "primary_agrees",
    }
    assert record["unjudged"] == ["retired-topic"]
    assert record["primary_unjudged"] is True
    assert record["missing"] == [{"slug": "startups", "noul": 0.9}]
    assert record["jev_assigned"] == ["ai-coding", "startups"]
    assert record["primary_agrees"] is False


def test_write_reports_leaves_the_last_good_report_whole_when_a_rename_dies(tmp_path, monkeypatch):
    """The atomicity the docstring claims, pinned the way `test_jev_store` pins the side-car's.

    It also pins the WEAKER joint guarantee the docstring now states: the two files are atomic
    individually, so a death between them leaves the JSON new and the markdown previous.
    """
    item = _item()
    assessment = _assessment(item, {"ai-coding": 0.95, "startups": 0.9, "misc": 0.2})
    jev_dir = tmp_path / "jev"
    first, comparisons = build_report([(item, assessment)], VOCAB, 0.85)
    _, md_path = write_reports(first, comparisons, {"1": item}, jev_dir)
    before = md_path.read_bytes()
    real_replace = os.replace

    def _boom(src, dst):
        if str(dst).endswith(".md"):
            raise OSError("disco lleno")
        return real_replace(src, dst)

    monkeypatch.setattr("xbrain.store.os.replace", _boom)
    other, other_comparisons = build_report([(item, assessment)], VOCAB, 0.10)

    with pytest.raises(OSError, match="disco lleno"):
        write_reports(other, other_comparisons, {"1": item}, jev_dir)

    assert md_path.read_bytes() == before
    # No half-written temp file survives the failure.
    assert sorted(p.name for p in jev_dir.iterdir()) == ["topics-report.json", "topics-report.md"]


def test_a_fallback_primary_gets_its_own_section_not_the_disagreement_one():
    """Jev answering "none of these" says the VOCABULARY is missing a topic, not that enrich
    is wrong. Different diagnosis, different remedy, so a different section."""
    item = _item(topics=("ai-coding", "misc"))
    assessment = _assessment(
        item,
        {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1},
        choice="otro",
        probabilities={"otro": 0.6, "ai-coding": 0.4, "startups": 0.0, "misc": 0.0},
    )
    summary, comparisons = build_report([(item, assessment)], VOCAB, 0.85)

    text = render_report_markdown(summary, comparisons, {"1": item})

    assert summary["primary_fallback"] == 1
    assert summary["primary_agree"] == 0
    assert "### Jev eligió el fallback (1)" in text
    assert "### Desacuerdo real (0)" in text


def test_the_markdown_explains_why_a_section_can_count_less_than_the_headline():
    """The headline counters are INDEPENDENT; the sections PARTITION by priority.

    An item whose primary left the vocabulary AND that Jev answered with the fallback is in
    both headline counters but appears once, under the first reason that applies. So a reader
    can see `**primario = fallback:** 1` two lines above `### Jev eligió el fallback (0)` —
    and the file has to say why, not just the code.
    """
    item = _item("2", text="Seed round", topics=("retired-topic", "startups"))
    assessment = _assessment(
        item,
        {"ai-coding": 0.1, "startups": 0.99, "misc": 0.05},
        choice="otro",
        probabilities={"otro": 0.6, "startups": 0.4, "ai-coding": 0.0, "misc": 0.0},
    )
    summary, comparisons = build_report([(item, assessment)], VOCAB, 0.85)

    text = render_report_markdown(summary, comparisons, {"2": item})

    # The overlap really is on the page: counted twice above, filed once below.
    assert summary["primary_fallback"] == 1 and summary["primary_unjudged"] == 1
    assert "**primario = fallback:** 1" in text
    assert "### Jev eligió el fallback (0)" in text
    assert "### Primario sin juzgar (salió del vocabulario) (1)" in text
    assert "estas cifras pueden sumar menos que las de arriba" in text

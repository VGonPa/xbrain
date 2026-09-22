# tests/test_jev_report.py
import json
from datetime import datetime, timezone
from pathlib import Path

from xbrain.jev.assess import build_topic_state, questions_digest, topic_contract
from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.questions import STATE_KEY, build_topic_questions
from xbrain.jev.report import (
    Pair,
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
    assert by_slug["misc"] == {
        "slug": "misc",
        "assigned": 1,
        "backed": 0,
        "backed_pct": 0.0,
        "missing": 0,
    }
    assert by_slug["startups"] == {
        "slug": "startups",
        "assigned": 1,
        "backed": 1,
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
    pairs = [
        (a, _assessment(a, {"ai-coding": 0.95, "startups": 0.1, "misc": 0.1})),
        (b, _assessment(b, {"ai-coding": 0.1, "startups": 0.9, "misc": 0.1}, provider="fake")),
    ]
    summary = summarize(pairs, VOCAB, 0.85)
    # 2000 tokens counted, but only the 1000 typesafe ones are billable.
    assert summary["input_tokens"] == 2000
    assert summary["cost_usd"] == round(1000 / 1e6 * 0.042, 4)
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
    summary = summarize(pairs, VOCAB, 0.85)
    assert summary["input_tokens"] == 0
    assert summary["input_tokens_unknown"] == 1
    assert summary["cost_usd"] == 0.0
    # And the marker reaches the reader: `Coste: 0 tokens de entrada` on its own describes
    # a run that was paid for as free.
    assert "(+1 sin recuento)" in render_report_markdown(summary, [], {"1": item})


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
    assert "Umbral 0.85" in text and "jev-1.13.0" in text
    assert "| 1 | misc | 0.20 |" in text  # doubtful row
    assert "| 1 | startups | 0.90 |" in text  # missing row
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
    summary = summarize(pairs, VOCAB, 0.85)
    text = render_report_markdown(summary, [], {"1": item})
    assert "sin tarifa: fake" in text

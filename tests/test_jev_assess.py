# tests/test_jev_assess.py
from datetime import datetime, timezone

import pytest

from tests.jev_fakes import FakeJevClient
from xbrain.evidence import evidence_text
from xbrain.jev.assess import (
    RunResult,
    assess_topics,
    assessment_is_current,
    build_topic_state,
    run_assessments,
    select_items,
    topic_contract,
)
from xbrain.jev.client import JevError
from xbrain.jev.questions import build_topic_questions
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.verification import fingerprint_output

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _vocab():
    return [
        Topic(slug="ai-coding", description="Construir software con IA."),
        Topic(slug="startups", description="Fundar y financiar empresas."),
    ]


def _item(
    item_id: str = "1",
    text: str = "Claude Code ships hooks",
    topics=("ai-coding",),
    author: Author | None = None,
) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=author if author is not None else Author(handle="alice", name="Alice"),
        text=text,
        created_at=DT,
        captured_at=DT,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="Resumen.",
            primary_topic=topics[0],
            topics=list(topics),
        ),
    )


def test_assess_topics_asks_on_the_evidence_and_stamps_the_contract():
    item, vocab = _item(), _vocab()
    client = FakeJevClient(nouls={"ai-coding": 0.97}, primary="ai-coding")
    assessment = assess_topics(item, vocab, client, fallback="otro", char_limit=100_000, now=DT)
    state, questions = client.calls[0]
    assert state == {"post": evidence_text(item, "topics")}
    assert len(questions) == 3
    # What was SENT is what the builder produces — the contract below hashes exactly this.
    assert questions == build_topic_questions(vocab, "otro")
    assert assessment.membership == {"ai-coding": 0.97, "startups": 0.05}
    assert assessment.primary.choice == "ai-coding"
    assert assessment.primary.confidence == 0.9
    assert assessment.primary.probabilities == {"ai-coding": 1.0, "startups": 0.0, "otro": 0.0}
    # Provenance travels on the result, never assumed by the record.
    assert assessment.provider == "fake"
    assert assessment.model == "jev-1.13.0"
    assert assessment.output_fingerprint == fingerprint_output(item, "topics")
    assert assessment.contract == topic_contract(state["post"], questions)
    assert assessment.truncated is False
    assert assessment.state_chars == len(state["post"])
    assert (assessment.input_tokens, assessment.output_tokens) == (100, 10)
    assert assessment.asked_at == DT


def test_state_is_cut_at_char_limit_and_state_chars_is_the_pre_cut_length():
    item = _item(text="x" * 50)
    # The author surface is evidence too: "alice\nAlice\n" + the 50 x's.
    assert len(evidence_text(item, "topics")) == 62
    state, state_chars = build_topic_state(item, char_limit=10)
    assert len(state["post"]) == 10
    assert state_chars == 62
    assessment = assess_topics(item, _vocab(), FakeJevClient(), fallback="otro", char_limit=10)
    assert assessment.truncated is True
    # PRE-cut, so a report can say how much was dropped; the post-cut length is char_limit.
    assert assessment.state_chars == 62


def test_parse_rejects_a_missing_noul_and_an_unknown_choice():
    class _MissingNoul(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            del result.answers["topic__startups"]
            return result

    with pytest.raises(JevError, match="startups"):
        assess_topics(_item(), _vocab(), _MissingNoul(), fallback="otro", char_limit=100)
    with pytest.raises(JevError, match="no está entre las opciones"):
        assess_topics(
            _item(), _vocab(), FakeJevClient(primary="banana"), fallback="otro", char_limit=100
        )


def test_currency_tracks_vocab_and_evidence_but_not_re_enrichment():
    item, vocab = _item(), _vocab()
    assessment = assess_topics(item, vocab, FakeJevClient(), fallback="otro", char_limit=100)
    kw = {"fallback": "otro", "char_limit": 100}
    assert assessment_is_current(assessment, item, vocab, **kw)
    re_enriched = _item(topics=("startups",))
    assert assessment_is_current(assessment, re_enriched, vocab, **kw)
    changed_vocab = [Topic(slug="ai-coding", description="Otra descripción."), vocab[1]]
    assert not assessment_is_current(assessment, item, changed_vocab, **kw)
    assert not assessment_is_current(assessment, _item(text="otro texto"), vocab, **kw)
    assert not assessment_is_current(assessment, item, vocab, fallback="ninguno", char_limit=100)


def test_select_items_skips_current_unless_forced_and_respects_ids_limit_and_evidence():
    vocab = _vocab()
    store = {
        "1": _item("1"),
        "2": _item("2", text="Fundraising tips"),
        # No handle and blank text: the author surface is evidence too, so an item with
        # only a handle would NOT be evidence-free.
        "3": _item("3", text="   ", author=Author(handle="", name="")),
    }
    assert evidence_text(store["3"], "topics") == ""
    kw = {"fallback": "otro", "char_limit": 100}
    current = assess_topics(store["1"], vocab, FakeJevClient(), **kw)
    assessments = {"1": current}
    picked = select_items(store, assessments, vocab, ids=None, limit=None, force=False, **kw)
    assert [item.id for item in picked] == ["2"]  # 1 is current, 3 has no evidence
    forced = select_items(store, assessments, vocab, ids=None, limit=None, force=True, **kw)
    assert [item.id for item in forced] == ["1", "2"]
    limited = select_items(store, assessments, vocab, ids=None, limit=1, force=True, **kw)
    assert [item.id for item in limited] == ["1"]
    by_id = select_items(store, assessments, vocab, ids=["2"], limit=None, force=False, **kw)
    assert [item.id for item in by_id] == ["2"]
    with pytest.raises(JevError, match="ids desconocidos: 9"):
        select_items(store, assessments, vocab, ids=["9"], limit=None, force=False, **kw)


def test_run_assessments_records_failures_and_keeps_going():
    items = [_item("1"), _item("2", text="BOOM"), _item("3")]
    client = FakeJevClient(fail_when=lambda state: "BOOM" in state["post"])
    result = run_assessments(
        items, _vocab(), client, fallback="otro", char_limit=100, concurrency=2
    )
    assert isinstance(result, RunResult)
    assert [a.item_id for a in result.assessed] == ["1", "3"]
    assert result.failed == [("2", "fake failure")]


def test_run_assessments_raises_when_every_item_fails():
    client = FakeJevClient(fail_when=lambda state: True)
    with pytest.raises(JevError, match="ninguna de las 2 evaluaciones"):
        run_assessments(
            [_item("1"), _item("2")],
            _vocab(),
            client,
            fallback="otro",
            char_limit=100,
            concurrency=2,
        )


def test_run_assessments_validates_questions_before_any_call():
    client = FakeJevClient()
    with pytest.raises(ValueError, match="choca"):
        run_assessments(
            [_item()], _vocab(), client, fallback="startups", char_limit=100, concurrency=1
        )
    assert client.calls == []


def test_run_assessments_reports_progress():
    seen: list[tuple[int, int]] = []
    run_assessments(
        [_item("1"), _item("2")],
        _vocab(),
        FakeJevClient(),
        fallback="otro",
        char_limit=100,
        concurrency=1,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]

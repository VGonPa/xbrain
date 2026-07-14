# tests/test_contract_fingerprint.py
"""A verdict binds to the CONTRACT it was judged under — not just to the output text.

A verdict is not a property of the output alone. It is the result of judging THAT output
against THAT source under THAT rubric. `fingerprint_output` hashed only the output, so
#86 could change what the judge reads (author metadata, thread, quoted-post markers,
images, titles) and rewrite `rubric-verify.md` + `rubric-summary.md` — **without touching
a single output character** — and every stored verdict still matched, still looked
current, and still painted its badge. Including the verdicts issued under the contract
that was measured letting a false attribution through 8 times out of 8.

`fingerprint_contract` binds all three arms. Each arm gets a test here: change the
output → stale; change the source → stale; change the rubric → stale; change nothing →
fresh, and the badge paints.
"""

from datetime import datetime, timezone


from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Enrichment,
    Item,
    VerificationVerdict,
    VideoFrame,
)
from xbrain.verification import (
    contract_fingerprint,
    count_invalidated_verdicts,
    fingerprint_output,
    rubric_digest,
)


def _item(*, summary: str = "A crisp summary.", frames: tuple[str, ...] = ("A chart.",)) -> Item:
    return Item(
        id="7",
        source="bookmark",
        url="https://x.com/a/status/7",
        author=Author(handle="a", name="A"),
        text="watch this",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        enriched=Enrichment(
            enriched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            executor="claude-code",
            summary=summary,
            primary_topic="ai-coding",
            topics=["ai-coding"],
        ),
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/v",
                    title="A talk",
                    text="the transcript body",
                    has_speech=True,
                    frames=[
                        VideoFrame(
                            timestamp=float(i), local_path=f"7/frames/{i}.png", description=d
                        )
                        for i, d in enumerate(frames)
                    ],
                    digest="A long digest.",
                )
            ],
        ),
    )


# ------------------------------------------------------------------ the three arms


def test_the_fingerprint_changes_when_the_OUTPUT_changes():
    before = contract_fingerprint(_item(), "summary", "English")
    after = contract_fingerprint(_item(summary="A different summary."), "summary", "English")
    assert before is not None and before != after


def test_the_fingerprint_changes_when_the_SOURCE_changes():
    """The judge read a source. Enrich adds a frame description, `describe` adds an image,
    a fetch lands the article — the output is untouched, but what supports it is not the
    same evidence any more. The verdict must not survive that."""
    before = contract_fingerprint(_item(), "summary", "English")
    after = contract_fingerprint(_item(frames=("A chart.", "A new slide.")), "summary", "English")
    assert before is not None and before != after


def test_the_fingerprint_changes_when_the_RUBRIC_changes(tmp_path, monkeypatch):
    """The rules the judge applied are part of the verdict. #86 rewrote them and every
    stored verdict still matched — this is the arm that closes that."""
    from xbrain import rubrics as rubrics_mod

    before = contract_fingerprint(_item(), "summary", "English")

    original = rubrics_mod._RUBRICS_DIR
    shadow = tmp_path / "rubrics"
    shadow.mkdir()
    for rubric in original.glob("*.md"):
        shadow.joinpath(rubric.name).write_text(
            rubric.read_text(encoding="utf-8"), encoding="utf-8"
        )
    verify = shadow / "rubric-verify.md"
    verify.write_text(verify.read_text(encoding="utf-8") + "\nA NEW RULE.\n", encoding="utf-8")
    monkeypatch.setattr(rubrics_mod, "_RUBRICS_DIR", shadow)
    rubric_digest.cache_clear()  # the digest is cached per run; a rubric edit must invalidate

    after = contract_fingerprint(_item(), "summary", "English")
    rubric_digest.cache_clear()
    assert before is not None and before != after


def test_the_fingerprint_is_stable_when_NOTHING_changes():
    """The negative arm: a hash that changed on every call would invalidate everything and
    teach the reader to ignore the badge."""
    assert contract_fingerprint(_item(), "summary", "English") == contract_fingerprint(
        _item(), "summary", "English"
    )


def test_each_target_gets_its_own_contract():
    """A digest is judged against a different source AND a different generation rubric than
    a summary, so their contracts cannot collide."""
    item = _item()
    assert contract_fingerprint(item, "summary", "English") != contract_fingerprint(
        item, "digest", "English"
    )


def test_the_language_is_part_of_the_contract():
    """The rubric is language-substituted; a judge given the Spanish rubric applied a
    different text."""
    assert contract_fingerprint(_item(), "summary", "English") != contract_fingerprint(
        _item(), "summary", "Spanish"
    )


def test_no_output_no_fingerprint():
    item = _item()
    item.enriched = None
    assert contract_fingerprint(item, "summary", "English") is None


def test_the_contract_hash_is_not_the_output_hash():
    """Guard against a refactor collapsing the two: the whole defect was a fingerprint that
    saw only the output."""
    item = _item()
    assert contract_fingerprint(item, "summary", "English") != fingerprint_output(item, "summary")


# ------------------------------------------------------------------ migration


def _verdict(contract: str | None) -> VerificationVerdict:
    return VerificationVerdict(
        verdict="FAIL",
        faithfulness="FAIL",
        output_fingerprint="a" * 64,
        contract_fingerprint=contract,
        verified_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        flags=["unsupported"],
    )


def test_a_legacy_verdict_loads_and_is_stale():
    """Every verdict stored before this change was judged under the OLD contract. It must
    load (never crash on the old shape), and it must evaluate STALE — not be grandfathered
    in, because we cannot honestly say what it was judged against."""
    legacy = VerificationVerdict.model_validate(
        {
            "verdict": "FAIL",
            "faithfulness": "FAIL",
            "adherence": "PASS",
            "output_fingerprint": "b" * 64,
            "verified_at": "2026-06-01T00:00:00Z",
            "flags": ["unsupported"],
        }
    )
    assert legacy.contract_fingerprint is None  # the old shape carries none


def test_count_invalidated_verdicts_counts_every_stale_stored_verdict():
    """The CLI reports the number, because the number IS the point: it says how much of the
    stored verification the contract change just retired."""
    fresh_item = _item()
    fresh = _verdict(contract_fingerprint(fresh_item, "summary", "English"))
    fresh_item.verification["summary"] = fresh

    legacy_item = _item()
    legacy_item.id = "8"
    legacy_item.verification["summary"] = _verdict(None)  # judged under the old contract

    changed_item = _item(summary="regenerated since it was judged")
    changed_item.id = "9"
    changed_item.verification["summary"] = _verdict(
        contract_fingerprint(_item(), "summary", "English")
    )

    store = {"7": fresh_item, "8": legacy_item, "9": changed_item}
    invalidated, total = count_invalidated_verdicts(store, "English")
    assert (invalidated, total) == (2, 3)

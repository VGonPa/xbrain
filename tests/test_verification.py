# tests/test_verification.py
import json
from datetime import datetime, timezone

import pytest

from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Enrichment,
    Item,
    VideoFrame,
)
from xbrain.verification import (
    ALL_TARGETS,
    aggregate_verify_judgments,
    apply_verdicts_to_store,
    export_verify_worksheet,
    fingerprint_output,
    import_verify_fingerprints,
    import_verify_judgments,
    items_for_verification,
    parse_targets,
    render_verify_report,
)


def test_parse_targets_resolves_and_validates():
    assert parse_targets("all") == ALL_TARGETS
    assert parse_targets("digest") == ("digest",)
    with pytest.raises(ValueError, match=r"summary\|digest"):
        parse_targets("bogus")


def _item(
    item_id: str = "7",
    *,
    summary: str = "A crisp summary.",
    topics: tuple[str, ...] = ("ai-coding",),
    transcript: str = "the full transcript body",
    frame_descs: tuple[str, ...] = ("A chart.",),
    digest: str = "A long digest.",
) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="watch this",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        enriched=Enrichment(
            enriched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            executor="claude-code",
            summary=summary,
            primary_topic=topics[0],
            topics=list(topics),
        ),
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/v",
                    title="A talk",
                    text=transcript,
                    has_speech=True,
                    frames=[
                        VideoFrame(
                            timestamp=float(i),
                            local_path=f"{item_id}/frames/{i}.png",
                            description=d,
                        )
                        for i, d in enumerate(frame_descs)
                    ],
                    digest=digest,
                )
            ],
        ),
    )


# ---------------------------------------------------------------- selection


def test_items_for_verification_pairs_each_target_with_output():
    store = {"7": _item()}
    pairs = items_for_verification(store, ("summary", "digest", "topics"))
    assert {t for _, t in pairs} == {"summary", "digest", "topics"}


def test_items_for_verification_skips_absent_output():
    """An item with a summary but no digest yields only the summary pair."""
    store = {"7": _item(digest="")}
    pairs = items_for_verification(store, ("summary", "digest"))
    assert [t for _, t in pairs] == ["summary"]


# ---------------------------------------------------------------- export


def test_export_worksheet_carries_source_output_and_rubrics(tmp_path):
    path = tmp_path / "ws.json"
    store = {"7": _item(summary="S.", transcript="deep talk", frame_descs=("A chart.",))}
    export_verify_worksheet(
        items_for_verification(store, ("summary",)), path, "claude-code", "English"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["items"][0]
    assert entry["target"] == "summary"
    assert entry["output"] == "S."
    assert "deep talk" in entry["source"]  # transcript in the source
    assert "A chart." in entry["source"]  # frames in the source
    assert "{language}" not in entry["generation_rubric"]  # summary rubric, substituted
    assert "{language}" not in data["verify_rubric"]
    assert "English" in data["verify_rubric"]
    assert data["judgments"] == []


def test_export_topics_output_shows_primary_and_topics(tmp_path):
    path = tmp_path / "ws.json"
    store = {"7": _item(topics=("ai-coding", "ai-industry"))}
    export_verify_worksheet(items_for_verification(store, ("topics",)), path, "manual", "English")
    entry = json.loads(path.read_text(encoding="utf-8"))["items"][0]
    assert "ai-coding" in entry["output"]
    assert "ai-industry" in entry["output"]


# ---------------------------------------------------------------- import


def test_import_reads_judgments(tmp_path):
    path = tmp_path / "ws.json"
    export_verify_worksheet(
        items_for_verification({"7": _item()}, ("summary",)), path, "claude-code", "English"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["judgments"] = [{"item_id": "7", "target": "summary", "verdict": "PASS"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert import_verify_judgments(path)[0]["verdict"] == "PASS"


def test_import_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_verify_judgments(tmp_path / "nope.json")


def test_import_rejects_non_list(tmp_path):
    path = tmp_path / "ws.json"
    path.write_text(json.dumps({"judgments": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        import_verify_judgments(path)


# ---------------------------------------------------------------- aggregate


def _j(verdict, faithfulness="PASS", adherence="PASS", flags=None, item_id="7", target="summary"):
    return {
        "item_id": item_id,
        "target": target,
        "verdict": verdict,
        "faithfulness": faithfulness,
        "adherence": adherence,
        "flags": flags or [],
    }


def test_aggregate_unanimous_pass():
    out = aggregate_verify_judgments([[_j("PASS")], [_j("PASS")], [_j("PASS")]])
    assert len(out) == 1
    assert out[0]["verdict"] == "PASS"
    assert out[0]["divergent"] is False
    assert out[0]["n_judges"] == 3


def test_aggregate_one_faithfulness_fail_forces_fail():
    """One judge's faithfulness FAIL sinks the group, even if two say PASS."""
    out = aggregate_verify_judgments(
        [
            [_j("PASS")],
            [_j("FAIL", faithfulness="FAIL", flags=[{"claim": "€150M", "issue": "unsupported"}])],
            [_j("PASS")],
        ]
    )
    assert out[0]["verdict"] == "FAIL"
    assert out[0]["faithfulness"] == "FAIL"
    assert out[0]["divergent"] is True  # verdicts disagreed
    assert {f["claim"] for f in out[0]["flags"]} == {"€150M"}


def test_aggregate_adherence_review_yields_review():
    out = aggregate_verify_judgments([[_j("PASS")], [_j("REVIEW", adherence="REVIEW")]])
    assert out[0]["verdict"] == "REVIEW"
    assert out[0]["adherence"] == "REVIEW"
    assert out[0]["divergent"] is True


def test_aggregate_dedupes_flags_across_judges():
    same = {"claim": "X", "issue": "unsupported"}
    out = aggregate_verify_judgments(
        [
            [_j("FAIL", faithfulness="FAIL", flags=[same])],
            [_j("FAIL", faithfulness="FAIL", flags=[same])],
        ]
    )
    assert len(out[0]["flags"]) == 1


def test_aggregate_tolerates_malformed_judgments_and_flags():
    """A non-dict judgment or a non-dict flag from a broken judge is skipped, not fatal."""
    out = aggregate_verify_judgments(
        [
            [
                "not a dict",  # non-dict judgment
                _j(
                    "FAIL", faithfulness="FAIL", flags=["bare string", {"claim": "X", "issue": "y"}]
                ),
            ]
        ]
    )
    assert len(out) == 1
    assert out[0]["verdict"] == "FAIL"
    assert [f["claim"] for f in out[0]["flags"]] == ["X"]  # the bare-string flag was dropped


def test_aggregate_raw_fail_verdict_forces_fail_even_without_axes():
    """A judge's verdict=FAIL with the axis fields omitted must still yield group FAIL —
    a FAIL must never be swallowed to PASS (the worst failure mode for a verifier)."""
    out = aggregate_verify_judgments(
        [
            [
                {
                    "item_id": "7",
                    "target": "summary",
                    "verdict": "FAIL",
                    "flags": [{"claim": "€150M", "issue": "unsupported"}],
                }
            ]
        ]
    )
    assert out[0]["verdict"] == "FAIL"


def test_aggregate_verdict_casing_is_normalised():
    """Lowercase verdicts still count (`fail` == `FAIL`)."""
    out = aggregate_verify_judgments(
        [[{"item_id": "7", "target": "summary", "verdict": "fail", "faithfulness": "fail"}]]
    )
    assert out[0]["verdict"] == "FAIL"
    assert out[0]["faithfulness"] == "FAIL"


def test_aggregate_groups_by_item_and_target():
    sets = [[_j("PASS", target="summary"), _j("FAIL", faithfulness="FAIL", target="digest")]]
    out = aggregate_verify_judgments(sets)
    by_target = {r["target"]: r["verdict"] for r in out}
    assert by_target == {"summary": "PASS", "digest": "FAIL"}


# ---------------------------------------------------------------- report


def test_render_report_counts_and_leads_with_fail():
    aggregated = aggregate_verify_judgments(
        [
            [_j("PASS", item_id="1")],
            [
                _j(
                    "FAIL",
                    faithfulness="FAIL",
                    item_id="2",
                    flags=[{"claim": "bad", "issue": "unsupported"}],
                )
            ],
        ]
    )
    json_report, md = render_verify_report(aggregated)
    data = json.loads(json_report)
    assert data["counts"] == {"PASS": 1, "REVIEW": 0, "FAIL": 1}
    assert data["total"] == 2
    # The FAIL is surfaced with its flag; the clean PASS is not cluttering the md.
    assert "FAIL" in md
    assert "unsupported" in md
    assert "[1]" not in md  # clean pass omitted from the human report


def test_render_report_all_pass_says_so():
    aggregated = aggregate_verify_judgments([[_j("PASS", item_id="1")]])
    _, md = render_verify_report(aggregated)
    assert "passed" in md.lower()


# ------------------------------------------------- fingerprint + write-verdicts (#79 badge)


def test_fingerprint_output_is_deterministic_and_none_when_absent():
    item = _item(summary="A crisp summary.")
    fp1 = fingerprint_output(item, "summary")
    fp2 = fingerprint_output(item, "summary")
    assert fp1 == fp2 and len(fp1) == 64  # sha256 hex
    # A different summary yields a different fingerprint.
    assert fingerprint_output(_item(summary="Different."), "summary") != fp1
    # Absent output → None.
    assert fingerprint_output(_item(summary=""), "summary") is None


def test_export_verify_worksheet_stamps_judged_output_fingerprint(tmp_path):
    """The worksheet stamps each entry with the fingerprint of the JUDGED output (#79), so
    the writer can bind a verdict to it later without a live recompute."""
    item = _item(summary="A crisp summary.")
    ws = tmp_path / "ws.json"
    export_verify_worksheet(
        items_for_verification({"7": item}, ("summary",)), ws, "claude-code", "English"
    )
    entry = next(e for e in json.loads(ws.read_text())["items"] if e["target"] == "summary")
    assert entry["output_fingerprint"] == fingerprint_output(item, "summary")


def test_import_verify_fingerprints_round_trips_and_rejects_garbage(tmp_path):
    item = _item(summary="A crisp summary.")
    ws = tmp_path / "ws.json"
    export_verify_worksheet(
        items_for_verification({"7": item}, ("summary", "digest")), ws, "claude-code", "English"
    )
    fingerprints = import_verify_fingerprints([ws])
    assert fingerprints[("7", "summary")] == fingerprint_output(item, "summary")
    assert fingerprints[("7", "digest")] == fingerprint_output(item, "digest")

    # A hand-edited/garbage hash is rejected — the key is dropped, not trusted.
    garbage = tmp_path / "garbage.json"
    garbage.write_text(
        json.dumps({"items": [{"item_id": "9", "target": "summary", "output_fingerprint": "nope"}]})
    )
    assert import_verify_fingerprints([garbage]) == {}


def test_import_verify_fingerprints_drops_conflicting_stamps(tmp_path):
    """Off-workflow: two worksheets stamp DIFFERENT fingerprints for the same (item, target).
    The conflicting key is DROPPED (fail-safe → fingerprint-missing → not written), while a
    key both agree on still resolves and writes (#79 divergent-stamp co-mingling)."""

    def _ws(path, summary_fp: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "items": [
                        {"item_id": "7", "target": "summary", "output_fingerprint": summary_fp},
                        {"item_id": "7", "target": "digest", "output_fingerprint": "c" * 64},
                    ]
                }
            )
        )

    ws_a, ws_b = tmp_path / "a.json", tmp_path / "b.json"
    _ws(ws_a, "a" * 64)
    _ws(ws_b, "b" * 64)  # summary stamp conflicts across worksheets; digest agrees

    fingerprints = import_verify_fingerprints([ws_a, ws_b])
    assert ("7", "summary") not in fingerprints  # conflict → dropped
    assert fingerprints[("7", "digest")] == "c" * 64  # agreement → kept

    # Consequence: the dropped key is not written (fingerprint-missing); the agreed key is.
    store = {"7": _item()}
    aggregated = [_j("FAIL", target="summary"), _j("FAIL", target="digest")]
    result = apply_verdicts_to_store(store, aggregated, fingerprints)
    assert "summary" not in store["7"].verification  # dropped → no verdict, no badge
    assert store["7"].verification["digest"].output_fingerprint == "c" * 64
    assert ("7", "summary", "fingerprint-missing") in result.skipped


def test_apply_verdicts_writes_verdict_with_the_judged_fingerprint():
    store = {"7": _item(summary="Judged summary.")}
    judged_fp = fingerprint_output(store["7"], "summary")
    aggregated = aggregate_verify_judgments(
        [[_j("FAIL", faithfulness="FAIL", flags=[{"claim": "€150M", "issue": "unsupported"}])]]
    )
    result = apply_verdicts_to_store(store, aggregated, {("7", "summary"): judged_fp})
    assert result.written == 1 and result.skipped == []
    verdict = store["7"].verification["summary"]
    assert verdict.verdict == "FAIL"
    assert verdict.output_fingerprint == judged_fp
    assert verdict.flags == ["unsupported"]  # the top flag issue is stored for the badge


def test_apply_verdicts_stores_judged_fingerprint_not_live_recompute():
    """THE headline fix (#79): the summary is regenerated to "B" AFTER export/judge but
    BEFORE write — the stored fingerprint must be the judged one (of "A"), NEVER a recompute
    of the live "B", so `generate` on "B" later finds a mismatch and shows no badge."""
    store = {"7": _item(summary="A — the judged summary.")}
    judged_fp = fingerprint_output(store["7"], "summary")
    aggregated = aggregate_verify_judgments([[_j("FAIL", faithfulness="FAIL")]])

    store["7"].enriched.summary = "B — regenerated before the write."  # output changed
    live_fp = fingerprint_output(store["7"], "summary")
    assert live_fp != judged_fp

    result = apply_verdicts_to_store(store, aggregated, {("7", "summary"): judged_fp})
    assert result.written == 1
    stored = store["7"].verification["summary"].output_fingerprint
    assert stored == judged_fp  # the JUDGED output's fingerprint
    assert stored != live_fp  # never the live recompute


def test_apply_verdicts_tallies_skipped_records_with_reasons():
    """A dropped verdict is never silent — item-gone, unknown record, and a missing judged
    fingerprint are each tallied on the result."""
    store = {"7": _item(summary="S.")}
    judged_fp = fingerprint_output(store["7"], "summary")
    aggregated = [
        _j("FAIL", item_id="7", target="summary"),  # writes (has a judged fingerprint)
        _j("FAIL", item_id="ghost", target="summary"),  # item-gone
        _j("FAIL", item_id="7", target="digest"),  # fingerprint-missing (not in the map)
        "not a dict",  # malformed-record
    ]
    result = apply_verdicts_to_store(store, aggregated, {("7", "summary"): judged_fp})
    assert result.written == 1
    reasons = {reason for _, _, reason in result.skipped}
    assert reasons == {"item-gone", "fingerprint-missing", "malformed-record"}
    assert result.attempted == 4
    assert "1 de 4" in result.summary() and "item-gone" in result.summary()


# ---------------------------------------------------------------- source text


def test_source_text_includes_author():
    """The judge's source carries the item's author metadata — attributing a post
    to its own author must be verifiable from the source, not world knowledge."""
    from xbrain.verification import _source_text

    text = _source_text(_item())
    assert "[Author]" in text
    assert "@a (A)" in text


def test_source_text_marks_unfetched_links():
    """A link whose content was never fetched is explicitly marked, so a judge
    treats any claim about the linked content as unsupported."""
    from xbrain.models import Link
    from xbrain.verification import _source_text

    item = _item()
    item.links = [Link(url="https://t.co/x", domain="time.com")]
    # the only content source is the x_video transcript — no fetched article
    text = _source_text(item)
    assert "NOT fetched" in text
    assert "time.com" in text


def test_source_text_no_unfetched_marker_when_article_fetched():
    from xbrain.models import Link
    from xbrain.verification import _source_text

    item = _item()
    item.links = [Link(url="https://example.com/a", domain="example.com")]
    item.content.sources.append(
        ContentSourceSuccess(
            kind="external_article",
            url="https://example.com/a",
            title="The piece",
            text="the fetched article body",
        )
    )
    text = _source_text(item)
    assert "NOT fetched" not in text
    assert "[Linked article" in text
    assert "the fetched article body" in text


def _label_of(text: str, needle: str) -> str:
    """The `[Label]` block heading the line `needle` appears under, in `_source_text`."""
    label = "<none>"
    for line in text.splitlines():
        if line.startswith("["):
            label = line
        if needle in line:
            return label
    raise AssertionError(f"{needle!r} not found in source:\n{text}")


def test_source_text_labels_thread_text_as_thread_not_a_fetched_article():
    """F1 (the regression this PR must not ship): a THREAD's own text is the
    poster's own words. Served under `[Linked article]` it tells the skeptical judge
    an article was fetched and hands it the thread body as that article's content —
    so a claim about the linked piece would be judged SUPPORTED against text that is
    not the piece. It must keep its own label AND still flag the unfetched link."""
    from xbrain.models import Link
    from xbrain.verification import _source_text

    item = _item()
    item.links = [Link(url="https://t.co/x", domain="time.com")]
    item.content.sources.append(
        ContentSourceSuccess(
            kind="thread",
            url="https://x.com/a/status/7",
            text="1/ my thread about agents",
        )
    )
    text = _source_text(item)
    assert "1/ my thread about agents" in text  # not dropped — a thread is real evidence
    assert _label_of(text, "1/ my thread about agents") == "[Thread — full text, same author]"
    assert "[Linked article" not in text
    assert "[Links — content NOT fetched]" in text


def test_source_text_partial_fetch_marks_the_unfetched_link_with_counts():
    """F5: 2 links, 1 fetched — the article block is present, but the judge is told
    only 1 of 2 links was fetched, so a claim about the other one stays checkable."""
    from xbrain.models import Link
    from xbrain.verification import _source_text

    item = _item()
    item.links = [
        Link(url="https://example.com/a", domain="example.com"),
        Link(url="https://t.co/x", domain="time.com"),
    ]
    item.content.sources.append(
        ContentSourceSuccess(
            kind="external_article",
            url="https://example.com/a",
            title="The piece",
            text="the fetched article body",
        )
    )
    text = _source_text(item)
    assert "the fetched article body" in text
    assert "[Links — content NOT fetched]" in text
    assert "1 of 2" in text


def test_source_text_marks_an_unfetched_quoted_post():
    """F3: `quoted_id` is captured but the quoted post is never fetched — the judge
    must be told, so a summary describing the quoted content is unsupported."""
    from xbrain.verification import _source_text

    item = _item()
    item.quoted_id = "999"
    text = _source_text(item)
    assert "[Quoted post — content NOT fetched]" in text


def test_source_text_carries_images_and_titles_the_generators_ship():
    """F4: the judge must not see LESS than the generator. The image descriptions,
    the article title and the video title all reach the generators — a summary
    grounded in them would otherwise be judged unsupported (a false FAIL)."""
    from xbrain.models import Link, MediaPhotoDescribed
    from xbrain.verification import _source_text

    item = _item()
    item.links = [Link(url="https://example.com/a", domain="example.com")]
    item.media = [
        MediaPhotoDescribed(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="7/0.jpg",
            width=4,
            height=3,
            bytes_size=512,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            is_decorative=False,
            description="A bar chart of GPU prices.",
            description_lang="English",
            description_version="v1",
            described_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
        )
    ]
    item.content.sources.append(
        ContentSourceSuccess(
            kind="external_article",
            url="https://example.com/a",
            title="The TIME piece",
            text="the fetched article body",
        )
    )
    text = _source_text(item)
    assert _label_of(text, "A bar chart of GPU prices.") == "[Images in the post]"
    assert "The TIME piece" in text  # the article title the api prompt already ships
    assert _label_of(text, "A talk") == "[Video title]"  # the video title the digest ships


def test_source_text_omits_decorative_image_descriptions():
    """A decorative photo carries no description — it must not add an empty bullet."""
    from xbrain.verification import _source_text

    assert "[Images in the post]" not in _source_text(_item())

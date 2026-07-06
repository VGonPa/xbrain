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
    export_verify_worksheet,
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

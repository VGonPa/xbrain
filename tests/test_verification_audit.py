# tests/test_verification_audit.py
"""Unit tests for the verifier-audit stage (PR-2, judge≠party).

The pure seams: consequential selection, the audit-worksheet round-trip, and the
deterministic merge/re-verdict (a confirmed flag keeps the FAIL, an all-revoked
record drops to REVIEW/PASS, a divergent record is resolved by the auditor).
"""

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
from xbrain.verification import render_verify_report
from xbrain.verification_audit import (
    consequential_records,
    export_audit_worksheet,
    import_audit_judgments,
    load_report_records,
    merge_audit,
)


def _item(item_id: str = "7", *, summary: str = "A crisp summary.") -> Item:
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
                    text="the full transcript body about agents",
                    has_speech=True,
                    frames=[
                        VideoFrame(
                            timestamp=0.0, local_path="7/frames/0.png", description="A chart."
                        )
                    ],
                    digest="A long digest.",
                )
            ],
        ),
    )


def _record(
    *,
    item_id: str = "7",
    target: str = "summary",
    verdict: str = "FAIL",
    faithfulness: str = "FAIL",
    adherence: str = "PASS",
    divergent: bool = False,
    flags: list | None = None,
) -> dict:
    return {
        "item_id": item_id,
        "target": target,
        "verdict": verdict,
        "faithfulness": faithfulness,
        "adherence": adherence,
        "divergent": divergent,
        "n_judges": 3,
        "flags": flags if flags is not None else [{"claim": "€150M", "issue": "unsupported"}],
    }


# ---------------------------------------------------------------- selection


def test_consequential_selects_fail_and_divergent_only():
    records = [
        _record(item_id="1", verdict="PASS", faithfulness="PASS", divergent=False, flags=[]),
        _record(item_id="2", verdict="FAIL"),
        _record(item_id="3", verdict="REVIEW", faithfulness="PASS", divergent=True, flags=[]),
    ]
    picked = {r["item_id"] for r in consequential_records(records)}
    assert picked == {"2", "3"}  # the clean unanimous PASS is not audited


def test_consequential_normalises_verdict_casing_and_skips_non_dicts():
    records = ["not a dict", _record(item_id="9", verdict="fail")]
    assert [r["item_id"] for r in consequential_records(records)] == ["9"]


# ---------------------------------------------------------------- export


def test_export_audit_worksheet_carries_source_output_flags_and_rubric(tmp_path):
    path = tmp_path / "audit-ws.json"
    store = {"7": _item(summary="S.")}
    export_audit_worksheet(
        consequential_records([_record()]), store, path, "claude-code", "English"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["items"][0]
    assert entry["item_id"] == "7"
    assert entry["output"] == "S."
    assert "transcript body about agents" in entry["source"]
    assert entry["current_verdict"] == "FAIL"
    assert entry["flags"][0]["claim"] == "€150M"
    assert "{language}" not in data["audit_rubric"]
    assert "English" in data["audit_rubric"]
    assert data["audits"] == []


def test_export_audit_worksheet_skips_records_with_no_item_in_store(tmp_path):
    path = tmp_path / "audit-ws.json"
    export_audit_worksheet([_record(item_id="ghost")], {}, path, "manual", "English")
    assert json.loads(path.read_text(encoding="utf-8"))["items"] == []


# ---------------------------------------------------------------- import


def test_import_audit_reads_audits(tmp_path):
    path = tmp_path / "audit-ws.json"
    path.write_text(
        json.dumps({"audits": [{"item_id": "7", "target": "summary"}]}), encoding="utf-8"
    )
    assert import_audit_judgments(path)[0]["item_id"] == "7"


def test_import_audit_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_audit_judgments(tmp_path / "nope.json")


def test_import_audit_rejects_non_list(tmp_path):
    path = tmp_path / "audit-ws.json"
    path.write_text(json.dumps({"audits": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        import_audit_judgments(path)


def test_load_report_records_reads_records(tmp_path):
    path = tmp_path / "verify-report.json"
    path.write_text(json.dumps({"total": 1, "records": [_record()]}), encoding="utf-8")
    assert load_report_records(path)[0]["item_id"] == "7"


def test_load_report_records_rejects_non_list(tmp_path):
    path = tmp_path / "verify-report.json"
    path.write_text(json.dumps({"records": 5}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        load_report_records(path)


# ---------------------------------------------------------------- merge


def _audit(item_id="7", target="summary", flags=None, reverdict="PASS"):
    return {
        "item_id": item_id,
        "target": target,
        "flags": flags if flags is not None else [],
        "reverdict": reverdict,
    }


def test_merge_confirmed_flag_keeps_fail():
    """A CONFIRMED faithfulness flag keeps the record FAIL — even if the auditor's
    overall reverdict tries to wash it to PASS (a confirmed flag can never be washed)."""
    record = _record()
    audit = _audit(
        flags=[
            {
                "claim": "€150M",
                "issue": "unsupported",
                "audit": "CONFIRM",
                "reason": "not in source",
            }
        ],
        reverdict="PASS",
    )
    out = merge_audit([record], [audit])[0]
    assert out["verdict"] == "FAIL"
    assert out["faithfulness"] == "FAIL"
    assert out["audited"] is True
    assert out["flags"][0]["claim"] == "€150M"  # the confirmed flag survives


def test_merge_all_flags_revoked_flips_to_pass():
    """Every faithfulness flag revoked and no adherence issue → the FAIL drops to PASS."""
    record = _record(adherence="PASS")
    audit = _audit(
        flags=[
            {"claim": "€150M", "issue": "unsupported", "audit": "REVOKE", "reason": "in transcript"}
        ],
        reverdict="PASS",
    )
    out = merge_audit([record], [audit])[0]
    assert out["verdict"] == "PASS"
    assert out["faithfulness"] == "PASS"
    assert out["flags"] == []  # revoked flag removed from the report


def test_merge_all_flags_revoked_keeps_review_when_adherence_issue_remains():
    """Revoking the faithfulness flag drops the FAIL to REVIEW, not PASS, because a
    soft adherence issue still stands."""
    record = _record(adherence="REVIEW")
    audit = _audit(
        flags=[
            {"claim": "€150M", "issue": "unsupported", "audit": "REVOKE", "reason": "in source"}
        ],
        reverdict="PASS",
    )
    out = merge_audit([record], [audit])[0]
    assert out["verdict"] == "REVIEW"
    assert out["faithfulness"] == "PASS"


def test_merge_divergent_no_flags_resolved_by_auditor():
    """A divergent record with no flags is a tie the auditor breaks: reverdict PASS → PASS."""
    record = _record(verdict="REVIEW", faithfulness="PASS", divergent=True, flags=[])
    out = merge_audit([record], [_audit(reverdict="PASS")])[0]
    assert out["verdict"] == "PASS"
    assert out["audited"] is True


def test_merge_divergent_auditor_can_escalate_to_fail():
    """The blind-spot catch: on a divergent record the auditor may escalate to FAIL,
    adding the flag the N judges missed."""
    record = _record(verdict="REVIEW", faithfulness="PASS", divergent=True, flags=[])
    audit = _audit(
        flags=[{"claim": "invented benchmark", "issue": "unsupported", "audit": "CONFIRM"}],
        reverdict="FAIL",
    )
    out = merge_audit([record], [audit])[0]
    assert out["verdict"] == "FAIL"
    assert any(f["claim"] == "invented benchmark" for f in out["flags"])


def test_merge_default_keeps_unaudited_flag():
    """A record flag with no matching audit decision defaults to CONFIRM (fail-safe):
    an unaudited flag is never silently dropped."""
    record = _record(flags=[{"claim": "€150M", "issue": "unsupported"}])
    out = merge_audit([record], [_audit(flags=[], reverdict="PASS")])[0]
    assert out["verdict"] == "FAIL"  # the un-revoked flag still stands
    assert out["flags"][0]["claim"] == "€150M"


def test_merge_passes_through_records_without_audit():
    """A record with no audit is returned untouched (and NOT marked audited)."""
    record = _record(item_id="7")
    other = _record(item_id="8", verdict="PASS", faithfulness="PASS", flags=[])
    out = {
        r["item_id"]: r
        for r in merge_audit(
            [record, other],
            [
                _audit(
                    item_id="7",
                    flags=[{"claim": "€150M", "issue": "unsupported", "audit": "REVOKE"}],
                )
            ],
        )
    }
    assert out["8"].get("audited") is not True
    assert out["8"]["verdict"] == "PASS"
    assert out["7"]["audited"] is True


def test_merge_normalises_audit_casing():
    """`revoke`/`confirm` in any casing are normalised (mirror the aggregate guards)."""
    record = _record()
    audit = _audit(
        flags=[{"claim": "€150M", "issue": "unsupported", "audit": "revoke"}], reverdict="pass"
    )
    out = merge_audit([record], [audit])[0]
    assert out["verdict"] == "PASS"


def test_merge_skips_malformed_audits_and_flag_decisions():
    """A non-dict audit, or a non-dict flag decision, is skipped — never fatal."""
    record = _record()
    out = merge_audit(
        [record],
        [
            "not a dict",
            _audit(flags=["bare", {"claim": "€150M", "issue": "unsupported", "audit": "REVOKE"}]),
        ],
    )[0]
    assert out["verdict"] == "PASS"  # the valid REVOKE still applied; the bare string ignored


def test_merged_report_marks_audited_records():
    """A merged FAIL renders through the shared report with an `· audited` marker."""
    record = _record()
    audit = _audit(
        flags=[{"claim": "€150M", "issue": "unsupported", "audit": "CONFIRM"}], reverdict="FAIL"
    )
    merged = merge_audit([record], [audit])
    _, md = render_verify_report(merged)
    assert "audited" in md
    assert "FAIL" in md


def test_merge_resorts_worst_first():
    """After the merge, records re-sort worst-verdict-first (like the aggregate)."""
    a = _record(item_id="1", verdict="FAIL")  # will be revoked → PASS
    b = _record(item_id="2", verdict="FAIL")  # confirmed → stays FAIL
    audits = [
        _audit(
            item_id="1",
            flags=[{"claim": "€150M", "issue": "unsupported", "audit": "REVOKE"}],
            reverdict="PASS",
        ),
        _audit(
            item_id="2",
            flags=[{"claim": "€150M", "issue": "unsupported", "audit": "CONFIRM"}],
            reverdict="FAIL",
        ),
    ]
    out = merge_audit([a, b], audits)
    assert out[0]["item_id"] == "2"  # the surviving FAIL leads
    assert out[0]["verdict"] == "FAIL"
    assert out[1]["verdict"] == "PASS"

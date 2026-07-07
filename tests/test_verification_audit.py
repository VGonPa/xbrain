# tests/test_verification_audit.py
"""Unit tests for the verifier-audit stage (PR-2, judge≠party).

The redesigned merge core enforces one invariant: a verdict lowers ONLY when the
specific cited evidence that produced it is explicitly revoked (faithfulness-axis,
confidence-gated); guards only escalate. Every washing path a reviewer reproduced is
pinned as a regression test below.
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


def _flag(claim="€150M", issue="unsupported", axis="faithfulness"):
    return {"claim": claim, "issue": issue, "axis": axis}


def _record(
    *,
    item_id="7",
    target="summary",
    verdict="FAIL",
    faithfulness="FAIL",
    adherence="PASS",
    divergent=False,
    flags=None,
):
    return {
        "item_id": item_id,
        "target": target,
        "verdict": verdict,
        "faithfulness": faithfulness,
        "adherence": adherence,
        "divergent": divergent,
        "n_judges": 3,
        "flags": flags if flags is not None else [_flag()],
    }


def _audit(item_id="7", target="summary", flags=None, reverdict="PASS"):
    audit = {"item_id": item_id, "target": target, "flags": flags if flags is not None else []}
    if reverdict is not None:
        audit["reverdict"] = reverdict
    return audit


def _decision(
    claim="€150M",
    issue="unsupported",
    axis="faithfulness",
    audit="REVOKE",
    confidence=0.9,
    reason=None,
):
    return {
        "claim": claim,
        "issue": issue,
        "axis": axis,
        "audit": audit,
        "confidence": confidence,
        "reason": reason,
    }


def _one(aggregated, audits, **kw):
    records, log = merge_audit(aggregated, audits, **kw)
    return records[0], log


# ---------------------------------------------------------------- selection


def test_consequential_selects_fail_and_divergent_only():
    records = [
        _record(item_id="1", verdict="PASS", faithfulness="PASS", flags=[]),
        _record(item_id="2", verdict="FAIL"),
        _record(item_id="3", verdict="REVIEW", faithfulness="PASS", divergent=True, flags=[]),
    ]
    assert {r["item_id"] for r in consequential_records(records)} == {"2", "3"}


def test_consequential_excludes_plain_non_divergent_review():
    """A plain REVIEW (not divergent) is NOT consequential — the auditor never sees it."""
    records = [
        _record(item_id="5", verdict="REVIEW", faithfulness="PASS", divergent=False, flags=[])
    ]
    assert consequential_records(records) == []


def test_consequential_normalises_casing_and_skips_non_dicts():
    records = ["not a dict", _record(item_id="9", verdict="fail")]
    assert [r["item_id"] for r in consequential_records(records)] == ["9"]


# ---------------------------------------------------------------- export


def test_export_audit_worksheet_carries_source_output_flags_and_rubric(tmp_path):
    path = tmp_path / "audit-ws.json"
    exported, skipped = export_audit_worksheet(
        consequential_records([_record()]),
        {"7": _item(summary="S.")},
        path,
        "claude-code",
        "English",
    )
    assert (exported, skipped) == (1, [])
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["items"][0]
    assert entry["output"] == "S."
    assert "transcript body about agents" in entry["source"]
    assert entry["current_verdict"] == "FAIL"
    assert entry["flags"][0]["claim"] == "€150M"
    rubric = data["audit_rubric"]
    assert "{language}" not in rubric and "English" in rubric and "confidence" in rubric.lower()


def test_export_audit_worksheet_reports_skipped_missing_items(tmp_path):
    """A consequential record whose item left the store is skipped AND reported (not a
    bare `continue` with a now-wrong count)."""
    path = tmp_path / "audit-ws.json"
    exported, skipped = export_audit_worksheet(
        [_record(item_id="ghost")], {}, path, "manual", "English"
    )
    assert exported == 0
    assert skipped == ["ghost"]
    assert json.loads(path.read_text(encoding="utf-8"))["items"] == []


# ---------------------------------------------------------------- import / load


def test_import_audit_reads_audits(tmp_path):
    path = tmp_path / "audit-ws.json"
    path.write_text(json.dumps({"audits": [_audit()]}), encoding="utf-8")
    assert import_audit_judgments(path)[0]["item_id"] == "7"


def test_import_audit_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_audit_judgments(tmp_path / "nope.json")


def test_import_audit_rejects_non_list(tmp_path):
    path = tmp_path / "audit-ws.json"
    path.write_text(json.dumps({"audits": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        import_audit_judgments(path)


def test_import_audit_rejects_invalid_flag_decision_loudly(tmp_path):
    """A mistyped `audit` value is rejected, not silently coerced to a no-op."""
    path = tmp_path / "audit-ws.json"
    path.write_text(
        json.dumps({"audits": [_audit(flags=[_decision(audit="MAYBE")])]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="CONFIRM.REVOKE"):
        import_audit_judgments(path)


def test_import_audit_rejects_invalid_reverdict_loudly(tmp_path):
    path = tmp_path / "audit-ws.json"
    path.write_text(json.dumps({"audits": [_audit(reverdict="MEH")]}), encoding="utf-8")
    with pytest.raises(ValueError, match="PASS.REVIEW.FAIL"):
        import_audit_judgments(path)


def test_import_audit_rejects_non_numeric_confidence(tmp_path):
    """NIT 4: a garbage `confidence` like "high" is rejected, not silently gated to 0.0."""
    path = tmp_path / "audit-ws.json"
    path.write_text(
        json.dumps({"audits": [_audit(flags=[_decision(audit="REVOKE", confidence="high")])]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="confidence"):
        import_audit_judgments(path)


def test_import_audit_rejects_out_of_range_confidence(tmp_path):
    path = tmp_path / "audit-ws.json"
    path.write_text(
        json.dumps({"audits": [_audit(flags=[_decision(audit="REVOKE", confidence=1.5)])]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="confidence"):
        import_audit_judgments(path)


def test_import_audit_allows_omitted_confidence(tmp_path):
    """An omitted confidence is valid (means 0.0 → a REVOKE gates)."""
    flag = {"claim": "€150M", "issue": "unsupported", "axis": "faithfulness", "audit": "REVOKE"}
    path = tmp_path / "audit-ws.json"
    path.write_text(json.dumps({"audits": [_audit(flags=[flag])]}), encoding="utf-8")
    assert import_audit_judgments(path)[0]["flags"][0]["audit"] == "REVOKE"


def test_load_report_records_reads_records(tmp_path):
    path = tmp_path / "verify-report.json"
    path.write_text(json.dumps({"total": 1, "records": [_record()]}), encoding="utf-8")
    assert load_report_records(path)[0]["item_id"] == "7"


def test_load_report_records_rejects_non_list(tmp_path):
    path = tmp_path / "verify-report.json"
    path.write_text(json.dumps({"records": 5}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        load_report_records(path)


# ------------------------------------------------- merge: the washing paths


def test_merge_confirmed_flag_keeps_fail():
    """A CONFIRMED faithfulness flag keeps FAIL — even if the reverdict says PASS."""
    out, _ = _one([_record()], [_audit(flags=[_decision(audit="CONFIRM")], reverdict="PASS")])
    assert out["verdict"] == "FAIL"
    assert out["faithfulness"] == "FAIL"
    assert out["audited"] is True


def test_merge_raw_verdict_fail_with_no_revocable_evidence_stays_fail():
    """WASH PATH #1: a raw verdict=FAIL with faithfulness=PASS, adherence=PASS has no
    revocable evidence — the auditor's PASS cannot wash it."""
    record = _record(verdict="FAIL", faithfulness="PASS", adherence="PASS", flags=[])
    out, _ = _one([record], [_audit(flags=[], reverdict="PASS")])
    assert out["verdict"] == "FAIL"


def test_merge_flagless_faithfulness_fail_stays_fail():
    """A faithfulness=FAIL with NO cited flag cannot be cleared (nothing to revoke)."""
    record = _record(verdict="FAIL", faithfulness="FAIL", flags=[])
    out, _ = _one([record], [_audit(flags=[], reverdict="PASS")])
    assert out["verdict"] == "FAIL"


def test_merge_confirm_wins_over_revoke_on_same_key():
    """WASH PATH #2: a CONFIRM and a REVOKE on the SAME (claim, issue) key never wash —
    CONFIRM wins regardless of ordering."""
    audit = _audit(
        flags=[
            _decision(audit="CONFIRM", confidence=0.9),
            _decision(audit="REVOKE", confidence=0.9),
        ]
    )
    out, _ = _one([_record()], [audit])
    assert out["verdict"] == "FAIL"
    assert out["flags"][0]["claim"] == "€150M"


def test_merge_added_confirmed_faith_flag_reestablishes_fail_after_original_revoked():
    """BUG 1: revoking the original faithfulness flag while the auditor CONFIRMS a NEW
    faithfulness hallucination must NOT wash to PASS — the added confirmed flag keeps the
    FAIL (safety must not fall back to the optional free-text reverdict)."""
    record = _record(faithfulness="FAIL", flags=[_flag(claim="F1")])
    audit = _audit(
        flags=[
            _decision(claim="F1", audit="REVOKE", confidence=0.99),
            _decision(
                claim="F2-NEW",
                issue="hallucination",
                axis="faithfulness",
                audit="CONFIRM",
                confidence=0.99,
            ),
        ],
        reverdict=None,
    )
    out, log = _one([record], [audit])
    assert out["verdict"] == "FAIL"
    assert out["faithfulness"] == "FAIL"
    assert any(f["claim"] == "F2-NEW" for f in out["flags"])
    assert log["anomalies"] == []


def test_pass_review_never_carries_surviving_faithfulness_flag():
    """INVARIANT: a standing (confirmed/unaudited) faithfulness flag forces FAIL — even
    when the aggregate's faithfulness axis was PASS (inconsistent judge) and the auditor
    resolves to PASS. A PASS/REVIEW record can NEVER carry a confirmed faithfulness flag."""
    record = _record(
        verdict="REVIEW", faithfulness="PASS", divergent=True, flags=[_flag(claim="X")]
    )
    out, log = merge_audit([record], [_audit(reverdict="PASS")])
    r = out[0]
    assert r["verdict"] == "FAIL"
    assert not (
        r["verdict"] in ("PASS", "REVIEW") and any(f["axis"] == "faithfulness" for f in r["flags"])
    )
    assert log["anomalies"] == []


def test_merge_revoke_then_confirm_same_key_confirm_still_wins():
    """NIT 3: a REVOKE listed BEFORE a CONFIRM on the same key still keeps FAIL (kills a
    'first-seen wins' mutation — CONFIRM wins regardless of order)."""
    audit = _audit(
        flags=[
            _decision(audit="REVOKE", confidence=0.9),
            _decision(audit="CONFIRM", confidence=0.9),
        ]
    )
    out, _ = _one([_record()], [audit])
    assert out["verdict"] == "FAIL"


def test_merge_same_claim_different_issue_are_distinct_flags():
    """Keying on the full (claim, issue) pair: revoking one issue keeps the other."""
    record = _record(flags=[_flag(issue="unsupported"), _flag(issue="wrong figure")])
    audit = _audit(flags=[_decision(issue="unsupported", audit="REVOKE", confidence=0.9)])
    out, _ = _one([record], [audit])
    assert out["verdict"] == "FAIL"  # the "wrong figure" faithfulness flag still stands
    assert {f["issue"] for f in out["flags"]} == {"wrong figure"}


def test_merge_adherence_revoke_does_not_clear_faithfulness_fail():
    """WASH PATH #3 (axis conflation): revoking an ADHERENCE flag must not clear a
    faithfulness FAIL driven by a separate faithfulness flag."""
    record = _record(
        faithfulness="FAIL",
        flags=[
            _flag(claim="€150M", issue="unsupported", axis="faithfulness"),
            _flag(claim="too wordy", issue="too long", axis="adherence"),
        ],
    )
    audit = _audit(
        flags=[
            _decision(
                claim="too wordy",
                issue="too long",
                axis="adherence",
                audit="REVOKE",
                confidence=0.9,
            )
        ]
    )
    out, _ = _one([record], [audit])
    assert out["verdict"] == "FAIL"
    assert out["faithfulness"] == "FAIL"


def test_merge_all_faithfulness_flags_revoked_flips_to_pass():
    """Every faithfulness flag validly revoked and no adherence issue → PASS."""
    out, _ = _one(
        [_record(adherence="PASS")], [_audit(flags=[_decision(audit="REVOKE", confidence=0.9)])]
    )
    assert out["verdict"] == "PASS"
    assert out["faithfulness"] == "PASS"
    assert out["flags"] == []


def test_merge_all_revoked_keeps_review_when_adherence_issue_remains():
    out, _ = _one(
        [_record(adherence="REVIEW")], [_audit(flags=[_decision(audit="REVOKE", confidence=0.9)])]
    )
    assert out["verdict"] == "REVIEW"
    assert out["faithfulness"] == "PASS"


def test_merge_adherence_fail_floor_survives_faithfulness_revoke():
    """WASH PATH: an adherence=FAIL floor is never audited away by a faithfulness revoke."""
    out, _ = _one(
        [_record(adherence="FAIL")], [_audit(flags=[_decision(audit="REVOKE", confidence=0.9)])]
    )
    assert out["verdict"] == "FAIL"


# ------------------------------------------------- merge: confidence gate


def test_merge_low_confidence_revoke_is_gated_and_keeps_fail():
    """WASH PATH: a REVOKE below the confidence threshold is gated (kept), not applied."""
    out, log = _one([_record()], [_audit(flags=[_decision(audit="REVOKE", confidence=0.5)])])
    assert out["verdict"] == "FAIL"
    assert log["gated"] and log["gated"][0]["item_id"] == "7"


def test_merge_missing_confidence_treated_as_zero_and_gated():
    flag = {"claim": "€150M", "issue": "unsupported", "axis": "faithfulness", "audit": "REVOKE"}
    out, _ = _one([_record()], [_audit(flags=[flag])])
    assert out["verdict"] == "FAIL"  # no confidence → 0.0 → gated


def test_merge_high_confidence_revoke_applies():
    out, _ = _one([_record()], [_audit(flags=[_decision(audit="REVOKE", confidence=0.7)])])
    assert out["verdict"] == "PASS"


# ------------------------------------------------- merge: mass-revocation guard


def test_merge_mass_revocation_guard_keeps_fails():
    """WASH PATH #4: when a run would clear a suspiciously high fraction of FAILs, ALL
    such revocations are suppressed and the records stay FAIL."""
    records = [_record(item_id="1"), _record(item_id="2")]
    audits = [
        _audit(item_id="1", flags=[_decision(audit="REVOKE", confidence=0.9)]),
        _audit(item_id="2", flags=[_decision(audit="REVOKE", confidence=0.9)]),
    ]
    out, log = merge_audit(records, audits)
    assert [r["verdict"] for r in out] == ["FAIL", "FAIL"]
    assert log["mass_revocation_guard"] is True


def test_merge_single_legit_revoke_not_tripped_by_mass_guard():
    """One revoke among several FAILs (< threshold) applies normally."""
    records = [_record(item_id="1"), _record(item_id="2"), _record(item_id="3")]
    audits = [_audit(item_id="1", flags=[_decision(audit="REVOKE", confidence=0.9)])]
    out, log = merge_audit(records, audits)
    by_id = {r["item_id"]: r["verdict"] for r in out}
    assert by_id["1"] == "PASS"
    assert log["mass_revocation_guard"] is False


# ------------------------------------------------- merge: divergence + escalation


def test_merge_divergent_no_flags_resolved_by_auditor():
    record = _record(verdict="REVIEW", faithfulness="PASS", divergent=True, flags=[])
    out, _ = _one([record], [_audit(reverdict="PASS")])
    assert out["verdict"] == "PASS"


def test_merge_divergent_absent_reverdict_stays_review():
    """WASH PATH: a divergent tie with no (valid) reverdict is NOT resolved to PASS."""
    record = _record(verdict="REVIEW", faithfulness="PASS", divergent=True, flags=[])
    out, _ = _one([record], [_audit(reverdict=None)])
    assert out["verdict"] == "REVIEW"


def test_merge_divergent_auditor_escalates_to_fail_blind_spot():
    record = _record(verdict="REVIEW", faithfulness="PASS", divergent=True, flags=[])
    audit = _audit(
        flags=[
            _decision(
                claim="invented benchmark",
                issue="unsupported",
                axis="faithfulness",
                audit="CONFIRM",
            )
        ],
        reverdict="FAIL",
    )
    out, _ = _one([record], [audit])
    assert out["verdict"] == "FAIL"
    assert any(f["claim"] == "invented benchmark" for f in out["flags"])


# ------------------------------------------------- merge: bookkeeping / defensiveness


def test_merge_default_keeps_unaudited_flag():
    out, _ = _one([_record()], [_audit(flags=[], reverdict="PASS")])
    assert out["verdict"] == "FAIL"
    assert out["flags"][0]["claim"] == "€150M"


def test_merge_passes_through_records_without_audit():
    record, other = (
        _record(item_id="7"),
        _record(item_id="8", verdict="PASS", faithfulness="PASS", flags=[]),
    )
    audits = [_audit(item_id="7", flags=[_decision(audit="REVOKE", confidence=0.9)])]
    out = {r["item_id"]: r for r in merge_audit([record, other], audits)[0]}
    assert out["8"].get("audited") is not True
    assert out["7"]["audited"] is True


def test_merge_unmatched_audit_is_surfaced_not_dropped():
    """WASH PATH #6: an audit that matches no record is reported, never silently lost."""
    _, log = merge_audit([_record(item_id="7")], [_audit(item_id="404", flags=[])])
    assert log["matched"] == 0
    assert log["unmatched"] == [{"item_id": "404", "target": "summary"}]


def test_merge_normalises_casing():
    audit = _audit(flags=[_decision(audit="revoke", confidence=0.9)], reverdict="pass")
    out, _ = _one([_record()], [audit])
    assert out["verdict"] == "PASS"


def test_merge_skips_malformed_audits_and_flag_decisions():
    audit = _audit(flags=["bare", _decision(audit="REVOKE", confidence=0.9)])
    out, _ = _one([_record()], ["not a dict", audit])
    assert out["verdict"] == "PASS"


def test_merge_skips_non_dict_report_record():
    """A non-dict report record is skipped, not crashed on (`record.get`)."""
    records, _ = merge_audit(
        ["not a dict", _record()], [_audit(flags=[_decision(audit="CONFIRM")])]
    )
    assert [r["verdict"] for r in records] == ["FAIL"]


def test_merge_resorts_worst_first():
    a = _record(item_id="1")
    b = _record(item_id="2")
    audits = [
        _audit(item_id="1", flags=[_decision(audit="REVOKE", confidence=0.9)], reverdict="PASS"),
        _audit(item_id="2", flags=[_decision(audit="CONFIRM")], reverdict="FAIL"),
    ]
    records, _ = merge_audit([a, b], audits)
    assert records[0]["item_id"] == "2" and records[0]["verdict"] == "FAIL"
    assert records[1]["verdict"] == "PASS"


# ------------------------------------------------- render: reason + audit section


def test_merged_report_marks_audited_and_renders_reason():
    """WASH PATH #5: the auditor's cited `reason` is threaded through and rendered."""
    audit = _audit(
        flags=[_decision(audit="CONFIRM", reason="not in transcript at any timestamp")],
        reverdict="FAIL",
    )
    records, log = merge_audit([_record()], [audit])
    _, md = render_verify_report(records, log)
    assert "audited" in md
    assert "not in transcript at any timestamp" in md


def test_report_audit_section_surfaces_washed_and_guard():
    records = [_record(item_id="1"), _record(item_id="2")]
    audits = [
        _audit(item_id="1", flags=[_decision(audit="REVOKE", confidence=0.9)]),
        _audit(item_id="2", flags=[_decision(audit="REVOKE", confidence=0.9)]),
    ]
    merged, log = merge_audit(records, audits)
    _, md = render_verify_report(merged, log)
    assert "## Audit" in md
    assert "mass-revocation guard" in md.lower()


def test_report_audit_section_lists_washed_record():
    records, log = merge_audit(
        [_record()], [_audit(flags=[_decision(audit="REVOKE", confidence=0.9)])]
    )
    _, md = render_verify_report(records, log)
    assert "washed" in md
    assert "FAIL → PASS" in md


def test_report_audit_section_renders_gated_and_unmatched_lines():
    """NIT 5: the gated (low-confidence) and unmatched audit LINES actually render."""
    records = [_record(item_id="7")]
    audits = [
        _audit(item_id="7", flags=[_decision(audit="REVOKE", confidence=0.4)]),  # gated
        _audit(item_id="404", flags=[]),  # unmatched
    ]
    merged, log = merge_audit(records, audits)
    _, md = render_verify_report(merged, log)
    assert "gated" in md
    assert "unmatched audit" in md
    assert "[404]" in md

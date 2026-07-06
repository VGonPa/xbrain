"""The verifier-audit stage: a judge≠party re-check of the consequential verdicts.

PR-1 (`verification.py`) reduces N judge passes to one aggregated verdict per
`(item, target)`. This stage takes ONLY the consequential subset — the records the
ensemble marked **FAIL** or **divergent** — and hands them to a second, independent
auditor who re-checks every flag against the source and CONFIRMS or REVOKES it. The
merge then re-verdicts **deterministically**, enforcing one central invariant:

    A verdict can only be LOWERED when the SPECIFIC cited evidence that produced it
    was explicitly revoked (a faithfulness FAIL clears only when EVERY faithfulness
    flag is revoked with confidence ≥ threshold). Guards only ever ESCALATE; they
    never de-escalate a pre-existing FAIL. Safety lives in code, not in the rubric.

Three deterministic backstops, mirroring `cv-guardrail.apply_verifier_audits`:
- **Confidence gate** — a REVOKE applies only at `confidence ≥ min_confidence`;
  below that the flag is kept (gated) and surfaced, never lowering the verdict.
- **Axis scoping** — a faithfulness FAIL clears only from revoked *faithfulness*
  flags; revoking an adherence note can never wash a faithfulness FAIL.
- **Mass-revocation guard** — if one run would clear a suspiciously high fraction of
  the FAIL verdicts, ALL such revocations are suppressed and those records stay FAIL.

Report-only: the merge returns new record dicts + an `audit_log` and never mutates
the store. Keyless worksheet+agents engine, like `enrich`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from xbrain.models import Item
from xbrain.rubrics import load_rubric
from xbrain.verification import (
    _ORDER_VERDICT,
    _VERDICT_ORDER,
    _output_for,
    _source_text,
    derive_verdict,
    flag_axis,
    parse_targets,
)

_VERDICTS = ("PASS", "REVIEW", "FAIL")
_AUDIT_DECISIONS = ("CONFIRM", "REVOKE")

# A REVOKE below this confidence is gated (kept, surfaced) — never lowers a verdict.
DEFAULT_MIN_CONFIDENCE = 0.7
# If ≥2 FAIL records would be cleared AND their fraction of all FAILs exceeds this,
# the whole run's FAIL-clearing revocations are suppressed (degenerate wash guard).
DEFAULT_MASS_REVOCATION_MAX = 0.5


def _norm(value: object) -> str:
    """Upper-case + trim a verdict/decision field so `revoke` == `REVOKE`."""
    return str(value).strip().upper()


def _flag_key(flag: dict) -> tuple[str, str]:
    """Match key for a flag — the FULL normalised `(claim, issue)` pair.

    Keying on the pair (not the claim alone) mirrors `verification._union_flags`'s
    dedup, so an auditor decision joins the exact flag it rules on and a CONFIRM on
    one span cannot be confused with a REVOKE on a same-claim-different-issue span.
    """
    claim = " ".join(str(flag.get("claim", "")).split()).lower()
    issue = " ".join(str(flag.get("issue", "")).split()).lower()
    return (claim, issue)


def _confidence(value: object) -> float:
    """Parse a `confidence` field to a float; a missing/non-numeric value is 0.0."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _more_severe(a: str, b: str) -> str:
    """The higher-severity of two verdicts (PASS < REVIEW < FAIL)."""
    return _ORDER_VERDICT[max(_VERDICT_ORDER.get(a, 0), _VERDICT_ORDER.get(b, 0))]


def consequential_records(aggregated: list[dict]) -> list[dict]:
    """The subset an independent auditor must re-check: every FAIL or divergent record.

    A unanimous clean PASS (and a plain non-divergent REVIEW) is left alone; a
    non-dict entry from a malformed report is skipped rather than crashing.
    """
    return [
        record
        for record in aggregated
        if isinstance(record, dict)
        and (_norm(record.get("verdict")) == "FAIL" or bool(record.get("divergent")))
    ]


def export_audit_worksheet(
    records: list[dict],
    store: dict[str, Item],
    path: Path,
    executor: str,
    output_language: str,
) -> tuple[int, list[str]]:
    """Write the audit worksheet: per consequential record, the source + output + the
    ensemble's current verdict + its flags, for the auditor to CONFIRM/REVOKE.

    A record whose item is no longer in the store is skipped and its id reported.
    Returns `(exported_count, skipped_item_ids)` so the caller can print an honest
    count instead of the number of records passed in.
    """
    items: list[dict] = []
    skipped: list[str] = []
    for record in records:
        item_id = str(record.get("item_id"))
        item = store.get(item_id)
        if item is None:
            skipped.append(item_id)
            continue
        target = parse_targets(str(record.get("target")))[0]
        items.append(
            {
                "item_id": item.id,
                "target": target,
                "author": item.author.handle,
                "output": _output_for(item, target),
                "source": _source_text(item),
                "current_verdict": record.get("verdict"),
                "faithfulness": record.get("faithfulness"),
                "adherence": record.get("adherence"),
                "divergent": record.get("divergent"),
                "flags": record.get("flags", []),
            }
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "executor": executor,
        "instructions": (
            "You are an INDEPENDENT auditor (judge ≠ party). For each entry in "
            "`items`, re-check every flag against its `source` following "
            "`audit_rubric`, and append one object to `audits` with keys "
            "{item_id, target, reverdict, flags:[{claim, issue, axis, audit, "
            "confidence, reason}]} where `audit` is CONFIRM or REVOKE and a REVOKE "
            "needs confidence ≥ 0.7 to apply. Then run: "
            "xbrain verify --audit --apply <this file>."
        ),
        "audit_rubric": load_rubric("verify-audit", language=output_language),
        "items": items,
        "audits": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(items), skipped


def _validate_audit(audit: dict) -> None:
    """Reject an invalid `audit`/`reverdict` enum LOUDLY (a mistyped escalation must
    not be silently coerced to a lenient default).
    """
    reverdict = audit.get("reverdict")
    if reverdict is not None and _norm(reverdict) not in _VERDICTS:
        raise ValueError(f"audit reverdict must be PASS|REVIEW|FAIL, got {reverdict!r}")
    for flag in audit.get("flags") or []:
        if not isinstance(flag, dict):
            continue  # structural noise (a bare string) is skipped by the merge
        if _norm(flag.get("audit")) not in _AUDIT_DECISIONS:
            raise ValueError(
                f"audit flag decision must be CONFIRM|REVOKE, got {flag.get('audit')!r}"
            )


def import_audit_judgments(path: Path) -> list[dict]:
    """Read + validate the `audits` list from one filled audit worksheet.

    Non-dict entries are tolerated (skipped later), but a dict with an invalid
    `audit`/`reverdict` enum is rejected loudly — a mistyped decision that would
    otherwise be coerced into a no-op could hide a real escalation or revocation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audit worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("audit worksheet must be a JSON object")
    audits = data.get("audits", [])
    if not isinstance(audits, list):
        raise ValueError("audit worksheet `audits` must be a list")
    for audit in audits:
        if isinstance(audit, dict):
            _validate_audit(audit)
    return audits


def load_report_records(path: Path) -> list[dict]:
    """Read the aggregated `records` back from a `verify-report.json` written by PR-1."""
    if not path.exists():
        raise FileNotFoundError(f"Verify report not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("verify report must be a JSON object")
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError("verify report `records` must be a list")
    return records


def _decisions_by_key(audit_flags: list) -> dict[tuple[str, str], dict]:
    """Fold the auditor's per-flag decisions into one ruling per `(claim, issue)`.

    A CONFIRM always wins over a REVOKE on the same key (never wash on a contradictory
    pair of decisions); two REVOKEs keep the higher confidence.
    """
    decisions: dict[tuple[str, str], dict] = {}
    for flag in audit_flags or []:
        if not isinstance(flag, dict):
            continue
        disposition = _norm(flag.get("audit"))
        if disposition not in _AUDIT_DECISIONS:
            continue
        key = _flag_key(flag)
        confidence = _confidence(flag.get("confidence"))
        reason = flag.get("reason")
        prev = decisions.get(key)
        if prev is None:
            decisions[key] = {
                "disposition": disposition,
                "confidence": confidence,
                "reason": reason,
            }
            continue
        winner = "CONFIRM" if "CONFIRM" in (prev["disposition"], disposition) else "REVOKE"
        decisions[key] = {
            "disposition": winner,
            "confidence": max(prev["confidence"], confidence),
            "reason": reason or prev["reason"],
        }
    return decisions


def _partition_flags(
    flags: list[dict], decisions: dict[tuple[str, str], dict], min_confidence: float, suppress: bool
) -> tuple[list[dict], int, int, list[dict]]:
    """Split a record's flags into survivors vs applied revocations.

    Returns `(surviving, n_faith_flags, n_faith_revoked, gated)`. A flag is revoked
    only by an explicit REVOKE at `confidence ≥ min_confidence` while not `suppress`ed
    (the mass-revocation guard); an unaudited flag defaults to CONFIRM (fail-safe).
    """
    surviving: list[dict] = []
    gated: list[dict] = []
    n_faith_flags = 0
    n_faith_revoked = 0
    for flag in flags:
        axis = flag_axis(flag)
        if axis == "faithfulness":
            n_faith_flags += 1
        decision = decisions.get(_flag_key(flag))
        if decision is not None and decision["disposition"] == "REVOKE":
            if decision["confidence"] >= min_confidence and not suppress:
                n_faith_revoked += int(axis == "faithfulness")
                continue  # dropped from the report — the cited evidence was revoked
            gated.append(  # a revoke that did NOT apply (low confidence or mass-suppressed)
                {
                    "claim": flag.get("claim"),
                    "issue": flag.get("issue"),
                    "confidence": decision["confidence"],
                }
            )
        surviving.append(
            {
                "claim": flag.get("claim"),
                "issue": flag.get("issue"),
                "axis": axis,
                "audit": "CONFIRM",
                "reason": decision["reason"] if decision else None,
            }
        )
    return surviving, n_faith_flags, n_faith_revoked, gated


def _added_flags(audit_flags: list, record_keys: set[tuple[str, str]]) -> list[dict]:
    """Blind-spot flags the auditor added that are NOT already on the record."""
    added: list[dict] = []
    for flag in audit_flags or []:
        if not isinstance(flag, dict) or _flag_key(flag) in record_keys:
            continue
        if _norm(flag.get("audit")) == "REVOKE":
            continue
        added.append(
            {
                "claim": flag.get("claim"),
                "issue": flag.get("issue"),
                "axis": flag_axis(flag),
                "audit": "CONFIRM",
                "reason": flag.get("reason"),
            }
        )
    return added


def _final_verdict(
    prior_verdict: str,
    faith_after: str,
    adherence: str,
    divergent: bool,
    revocable_fail_cleared: bool,
    reverdict: str,
) -> str:
    """The deterministic re-verdict enforcing the can't-de-escalate invariant.

    The `floor` is what the audit is NOT allowed to lower: the adherence axis (never
    audited), an unrevocable prior FAIL (a FAIL whose cited faithfulness evidence was
    not fully revoked — including a raw verdict=FAIL with no revocable flag at all),
    and a prior REVIEW that is not a resolvable divergence tie. From the floor the
    verdict may only ESCALATE (a confirmed/added faithfulness flag, or the auditor's
    reverdict); it can drop to PASS only when the floor itself is PASS.
    """
    floor = derive_verdict("PASS", adherence)  # adherence axis is never audited
    if prior_verdict == "FAIL" and not revocable_fail_cleared:
        floor = "FAIL"
    divergence_resolvable = (
        divergent and adherence == "PASS" and faith_after == "PASS" and reverdict == "PASS"
    )
    if prior_verdict == "REVIEW" and not divergence_resolvable:
        floor = _more_severe(floor, "REVIEW")

    final = _more_severe(floor, derive_verdict(faith_after, adherence))
    if reverdict in _VERDICTS:
        final = _more_severe(final, reverdict)  # the auditor may only escalate the floor
    return final


def _faith_after(prior_faith: str, all_faith_revoked: bool, added: list[dict]) -> str:
    """The faithfulness axis after the audit.

    A prior FAIL clears to PASS ONLY when every faithfulness flag was validly revoked;
    a prior PASS escalates to FAIL only when the auditor ADDED a faithfulness flag (a
    shared blind spot the N judges missed).
    """
    if prior_faith == "FAIL":
        return "PASS" if all_faith_revoked else "FAIL"
    return "FAIL" if any(f["axis"] == "faithfulness" for f in added) else "PASS"


def _apply_audit(record: dict, audit: dict, *, min_confidence: float, suppress: bool) -> dict:
    """Fold one auditor decision into one aggregated record, re-verdicting it.

    Faithfulness clears to PASS only when it was FAIL and EVERY faithfulness flag was
    validly revoked; a single confirmed (or unaudited) faithfulness flag keeps it
    FAIL, and a raw verdict=FAIL with no revocable evidence stays FAIL. `suppress`
    (the mass-revocation guard) disables revocations for this record.
    """
    prior_verdict = _norm(record.get("verdict"))
    prior_faith = _norm(record.get("faithfulness", "PASS"))
    adherence = _norm(record.get("adherence", "PASS"))
    audit_flags = audit.get("flags") or []
    flags = [f for f in (record.get("flags") or []) if isinstance(f, dict)]

    decisions = _decisions_by_key(audit_flags)
    surviving, n_faith_flags, n_faith_revoked, gated = _partition_flags(
        flags, decisions, min_confidence, suppress
    )
    added = _added_flags(audit_flags, {_flag_key(f) for f in flags})

    all_faith_revoked = n_faith_flags > 0 and n_faith_revoked == n_faith_flags
    faith_after = _faith_after(prior_faith, all_faith_revoked, added)
    revocable_fail_cleared = (
        prior_verdict == "FAIL"
        and prior_faith == "FAIL"
        and all_faith_revoked
        and adherence != "FAIL"
    )
    verdict = _final_verdict(
        prior_verdict,
        faith_after,
        adherence,
        bool(record.get("divergent")),
        revocable_fail_cleared,
        _norm(audit.get("reverdict")),
    )
    return {
        **record,
        "faithfulness": faith_after,
        "adherence": adherence,
        "flags": surviving + added,
        "verdict": verdict,
        "audited": True,
        "_gated": gated,
    }


def _index_audits(audits: list[dict]) -> dict[tuple[str, str], dict]:
    """Index audits by `(item_id, target)`, skipping non-dicts (defensive)."""
    by_key: dict[tuple[str, str], dict] = {}
    for audit in audits:
        if not isinstance(audit, dict):
            continue
        by_key[(str(audit.get("item_id")), str(audit.get("target")))] = audit
    return by_key


def _mass_revocation_tripped(
    aggregated: list[dict], by_key: dict[tuple[str, str], dict], min_confidence: float
) -> bool:
    """Would this run clear a suspiciously high fraction of the FAIL verdicts?

    Computes each record's would-be verdict with revocations applied (no suppression)
    and trips when ≥2 FAILs drop AND their fraction of all FAIL records exceeds the
    threshold — a degenerate run suspected of washing FAILs in bulk.
    """
    severe = [r for r in aggregated if isinstance(r, dict) and _norm(r.get("verdict")) == "FAIL"]
    if len(severe) < 2:
        return False
    dropped = 0
    for record in severe:
        audit = by_key.get((str(record.get("item_id")), str(record.get("target"))))
        if audit is None:
            continue
        merged = _apply_audit(record, audit, min_confidence=min_confidence, suppress=False)
        if _norm(merged["verdict"]) != "FAIL":
            dropped += 1
    return dropped >= 2 and dropped / len(severe) > DEFAULT_MASS_REVOCATION_MAX


def merge_audit(
    aggregated: list[dict],
    audits: list[dict],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[list[dict], dict]:
    """Merge the auditor's decisions onto the aggregated records and re-verdict.

    Returns `(records, audit_log)`. Each audit joins its record by `(item_id, target)`;
    a matched record is re-verdicted by `_apply_audit`, an unmatched record passes
    through untouched, and an unmatched AUDIT is reported (never silently dropped). The
    mass-revocation guard suppresses FAIL-clearing revocations for the whole run when it
    trips. Records re-sort worst-verdict-first (then divergent), like the aggregate.
    """
    by_key = _index_audits(audits)
    supplied = len(by_key)
    suppress = _mass_revocation_tripped(aggregated, by_key, min_confidence)

    records: list[dict] = []
    matched: set[tuple[str, str]] = set()
    washed: list[dict] = []
    gated: list[dict] = []
    for record in aggregated:
        if not isinstance(record, dict):
            continue  # a malformed report record is skipped, like consequential_records
        key = (str(record.get("item_id")), str(record.get("target")))
        audit = by_key.get(key)
        if audit is None:
            records.append(record)
            continue
        matched.add(key)
        record_suppress = suppress and _norm(record.get("verdict")) == "FAIL"
        merged = _apply_audit(
            record, audit, min_confidence=min_confidence, suppress=record_suppress
        )
        prior, now = _norm(record.get("verdict")), _norm(merged["verdict"])
        if _VERDICT_ORDER.get(now, 0) < _VERDICT_ORDER.get(prior, 0):
            washed.append({"item_id": key[0], "target": key[1], "from": prior, "to": now})
        for gate in merged.pop("_gated", []):
            gated.append({"item_id": key[0], "target": key[1], **gate})
        records.append(merged)

    records.sort(
        key=lambda r: (-_VERDICT_ORDER.get(_norm(r["verdict"]), 0), not r.get("divergent"))
    )
    unmatched = [{"item_id": k[0], "target": k[1]} for k in by_key if k not in matched]
    audit_log = {
        "supplied": supplied,
        "matched": len(matched),
        "unmatched": unmatched,
        "washed": washed,
        "gated": gated,
        "mass_revocation_guard": suppress,
    }
    return records, audit_log

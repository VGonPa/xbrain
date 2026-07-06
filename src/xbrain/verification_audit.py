"""The verifier-audit stage: a judge≠party re-check of the consequential verdicts.

PR-1 (`verification.py`) reduces N judge passes to one aggregated verdict per
`(item, target)`. This stage takes ONLY the consequential subset — the records the
ensemble marked **FAIL** or **divergent** — and hands them to a second, independent
auditor who re-checks every flag against the source and CONFIRMS or REVOKES it.
The merge then recomputes the verdict **deterministically**: a confirmed
faithfulness flag keeps the FAIL, a record whose flags are ALL revoked drops to
REVIEW (or PASS if no adherence issue remains), and a divergence-only tie is
resolved by the auditor. This catches both a lone hallucinated flag and a shared
blind spot of the N judges — exactly `cv-guardrail`'s `cv-fact-verifier`.

Report-only, like the rest of the layer: the merge returns new record dicts and
never mutates the store. Keyless worksheet+agents engine, like `enrich`.
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
    parse_targets,
)

_VERDICTS = ("PASS", "REVIEW", "FAIL")


def _norm(value: object) -> str:
    """Upper-case + trim a verdict/decision field so `revoke` == `REVOKE`."""
    return str(value).strip().upper()


def _flag_key(flag: dict) -> str:
    """Match key for a flag — its `claim` span, normalised (case/whitespace-insensitive).

    The auditor rules on a claim it copies verbatim; issue wording may drift
    between a judge and the auditor, so the claim span alone joins them.
    """
    return " ".join(str(flag.get("claim", "")).split()).lower()


def consequential_records(aggregated: list[dict]) -> list[dict]:
    """The subset an independent auditor must re-check: every FAIL or divergent record.

    A unanimous clean PASS is left alone (nothing to audit); a non-dict entry from a
    malformed report is skipped rather than crashing the selection.
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
) -> None:
    """Write the audit worksheet: per consequential record, the source + output + the
    ensemble's current verdict + its flags, for the auditor to CONFIRM/REVOKE.

    A record whose item is no longer in the store is skipped (the audit needs the
    live source + output to re-check against).
    """
    items = []
    for record in records:
        item = store.get(str(record.get("item_id")))
        if item is None:
            continue
        targets = parse_targets(str(record.get("target")))
        target = targets[0]
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
            "{item_id, target, reverdict, flags:[{claim, issue, audit, reason}]} "
            "where `audit` is CONFIRM or REVOKE. Then run: "
            "xbrain verify --audit --apply <this file>."
        ),
        "audit_rubric": load_rubric("verify-audit", language=output_language),
        "items": items,
        "audits": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_audit_judgments(path: Path) -> list[dict]:
    """Read the `audits` list from one filled audit worksheet."""
    if not path.exists():
        raise FileNotFoundError(f"Audit worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("audit worksheet must be a JSON object")
    audits = data.get("audits", [])
    if not isinstance(audits, list):
        raise ValueError("audit worksheet `audits` must be a list")
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


def _more_severe(a: str, b: str) -> str:
    """The higher-severity of two verdicts (PASS < REVIEW < FAIL)."""
    return _ORDER_VERDICT[max(_VERDICT_ORDER.get(a, 0), _VERDICT_ORDER.get(b, 0))]


def _reverdict(faithfulness: str, adherence: str, divergent: bool, auditor: str) -> str:
    """Recompute a record's verdict after the audit, deterministically.

    The hard floor is faithfulness/adherence: any FAIL axis is FAIL, an adherence
    REVIEW is REVIEW. A confirmed faithfulness flag therefore keeps the FAIL no
    matter what the auditor's overall `reverdict` says — a verifier never washes a
    flag it left confirmed. When the ONLY remaining signal is divergence (the floor
    is PASS), the auditor breaks the tie in either direction; otherwise the auditor
    may only ESCALATE the floor (the shared-blind-spot catch), never lower it.
    """
    if faithfulness == "FAIL" or adherence == "FAIL":
        floor = "FAIL"
    elif adherence == "REVIEW":
        floor = "REVIEW"
    else:
        floor = "PASS"
    if floor == "PASS" and divergent:
        return auditor if auditor in _VERDICTS else "REVIEW"
    return _more_severe(floor, auditor) if auditor in _VERDICTS else floor


def _standing_flags(record_flags: list, decisions: dict[str, str]) -> list[dict]:
    """The record's flags that survive the audit, each tagged with its disposition.

    A flag with no matching audit decision defaults to CONFIRM (fail-safe: an
    unaudited flag is never silently dropped); a REVOKE decision removes it.
    """
    standing: list[dict] = []
    for flag in record_flags:
        if not isinstance(flag, dict):
            continue
        if decisions.get(_flag_key(flag)) == "REVOKE":
            continue
        standing.append(
            {"claim": flag.get("claim"), "issue": flag.get("issue"), "audit": "CONFIRM"}
        )
    return standing


def _added_flags(audit_flags: list, record_keys: set[str]) -> list[dict]:
    """Blind-spot flags the auditor added that are NOT already on the record."""
    added: list[dict] = []
    for flag in audit_flags:
        if not isinstance(flag, dict) or _flag_key(flag) in record_keys:
            continue
        if _norm(flag.get("audit")) == "REVOKE":
            continue
        added.append({"claim": flag.get("claim"), "issue": flag.get("issue"), "audit": "CONFIRM"})
    return added


def _apply_audit(record: dict, audit: dict) -> dict:
    """Fold one auditor decision into one aggregated record, re-verdicting it.

    Faithfulness clears to PASS only when it was FAIL and EVERY original flag was
    revoked; a single confirmed (or unaudited) flag keeps it FAIL. Auditor-added
    flags surface a blind spot but escalate only via the deterministic re-verdict.
    """
    original_flags = record.get("flags") or []
    audit_flags = audit.get("flags") or []
    decisions = {_flag_key(f): _norm(f.get("audit")) for f in audit_flags if isinstance(f, dict)}
    record_keys = {_flag_key(f) for f in original_flags if isinstance(f, dict)}
    surviving = _standing_flags(original_flags, decisions)
    added = _added_flags(audit_flags, record_keys)

    all_revoked = bool(record_keys) and not surviving
    faithfulness = _norm(record.get("faithfulness", "PASS"))
    if faithfulness == "FAIL" and all_revoked:
        faithfulness = "PASS"
    adherence = _norm(record.get("adherence", "PASS"))
    verdict = _reverdict(
        faithfulness, adherence, bool(record.get("divergent")), _norm(audit.get("reverdict"))
    )
    return {
        **record,
        "faithfulness": faithfulness,
        "flags": surviving + added,
        "verdict": verdict,
        "audited": True,
    }


def merge_audit(aggregated: list[dict], audits: list[dict]) -> list[dict]:
    """Merge the auditor's decisions onto the aggregated records and re-verdict.

    Each audit joins its record by `(item_id, target)`. A record with a matching
    audit is re-verdicted deterministically (`_apply_audit`); a record without one
    passes through untouched (and unmarked). Malformed audits (non-dict, or missing
    the join key) are skipped, mirroring the aggregate's defensive guards. The
    result re-sorts worst-verdict-first (then divergent), like the aggregate.
    """
    by_key: dict[tuple[str, str], dict] = {}
    for audit in audits:
        if not isinstance(audit, dict):
            continue
        by_key[(str(audit.get("item_id")), str(audit.get("target")))] = audit
    merged: list[dict] = []
    for record in aggregated:
        match = by_key.get((str(record.get("item_id")), str(record.get("target"))))
        merged.append(_apply_audit(record, match) if match is not None else record)
    merged.sort(key=lambda r: (-_VERDICT_ORDER.get(_norm(r["verdict"]), 0), not r.get("divergent")))
    return merged

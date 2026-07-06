"""The `verify` step: semantic verification of generated enrichment.

An LLM-as-judge ensemble scores each generated output (a short `summary`, a
long-form video `digest`, or a `topics` assignment) for **faithfulness** (are its
claims supported by the source?) and **rubric-adherence**, producing a per-item
verdict PASS / REVIEW / FAIL + cited flags. This module is the deterministic
plumbing (select → export worksheet → import → aggregate → render report); the
judging itself is done by agents filling the worksheet, and the consequential
verdicts are audited by a judge≠party pass (`verification_audit`, PR-2).

Report-only: it never mutates the store. Mirrors `cv-guardrail` (judges →
aggregate → verifier). Keyless worksheet+agents engine, like `enrich`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from xbrain.executors.api import _video_frame_descriptions
from xbrain.models import Item
from xbrain.rubrics import load_rubric
from xbrain.video_digest import _video_source
from xbrain.worksheet import _article_text, _video_transcript

VerifyTarget = Literal["summary", "digest", "topics"]
ALL_TARGETS: tuple[VerifyTarget, ...] = ("summary", "digest", "topics")

# The generation rubric each target is judged against (a valid-but-wrong topic is
# an adherence failure, so the topics judge reads the topics rubric).
_TARGET_RUBRIC: dict[VerifyTarget, str] = {
    "summary": "summary",
    "digest": "video-digest",
    "topics": "topics",
}

# Verdict ordering for "worst wins" aggregation.
_VERDICT_ORDER: dict[str, int] = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
_ORDER_VERDICT: dict[int, str] = {0: "PASS", 1: "REVIEW", 2: "FAIL"}


def parse_targets(target: str) -> tuple[VerifyTarget, ...]:
    """Resolve the `--target` flag to a typed tuple, or raise on an unknown value."""
    if target == "all":
        return ALL_TARGETS
    if target in ALL_TARGETS:
        return (cast(VerifyTarget, target),)
    raise ValueError(f"--target must be summary|digest|topics|all, got {target!r}")


def _output_for(item: Item, target: VerifyTarget) -> str | None:
    """The generated output for `target` on `item`, or None when absent.

    `summary`/`topics` live on `item.enriched`; `digest` lives on the `x_video`
    source. None means "nothing generated for this target" → skip it.
    """
    if target == "summary":
        return item.enriched.summary if item.enriched and item.enriched.summary else None
    if target == "topics":
        if not item.enriched or not item.enriched.topics:
            return None
        return f"primary_topic: {item.enriched.primary_topic}\ntopics: {', '.join(item.enriched.topics)}"
    source = _video_source(item)
    return source.digest if (source and source.digest) else None


def _source_text(item: Item) -> str:
    """The ground-truth source the judge checks the output against.

    Concatenates the labelled evidence present on the item — the FULL video
    transcript (what it says), the frame descriptions (what it shows), the article
    body, and the tweet text — so a claim in the output can be traced to it.
    """
    parts: list[str] = []
    transcript = _video_transcript(item)
    if transcript:
        parts += ["[Video transcript]", transcript]
    frames = _video_frame_descriptions(item)
    if frames:
        parts += ["[Video frames shown]", *(f"- {description}" for description in frames)]
    article = _article_text(item)
    if article:
        parts += ["[Linked article]", article]
    if item.text:
        parts += ["[Tweet]", item.text]
    return "\n".join(parts)


def items_for_verification(
    store: dict[str, Item], targets: tuple[VerifyTarget, ...]
) -> list[tuple[Item, VerifyTarget]]:
    """Every `(item, target)` pair that has a generated output to verify."""
    pairs: list[tuple[Item, VerifyTarget]] = []
    for item in store.values():
        for target in targets:
            if _output_for(item, target) is not None:
                pairs.append((item, target))
    return pairs


def export_verify_worksheet(
    pairs: list[tuple[Item, VerifyTarget]],
    path: Path,
    executor: str,
    output_language: str,
) -> None:
    """Write a worksheet the judge fills: per `(item, target)`, the source + the
    generated output + the target's generation rubric + the verify rubric.
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "executor": executor,
        "instructions": (
            "You are an independent judge. For each entry in `items`, judge its "
            "`output` against its `source` and `generation_rubric` following "
            "`verify_rubric`, and append one object to `judgments` with keys "
            "{item_id, target, verdict, faithfulness, adherence, flags}. Then run: "
            "xbrain verify --apply <this file>."
        ),
        "verify_rubric": load_rubric("verify", language=output_language),
        "items": [
            {
                "item_id": item.id,
                "target": target,
                "author": item.author.handle,
                "output": _output_for(item, target),
                "source": _source_text(item),
                "generation_rubric": load_rubric(_TARGET_RUBRIC[target], language=output_language),
            }
            for item, target in pairs
        ],
        "judgments": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_verify_judgments(path: Path) -> list[dict]:
    """Read the `judgments` list from one filled worksheet (one judge's pass)."""
    if not path.exists():
        raise FileNotFoundError(f"Worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("worksheet must be a JSON object")
    judgments = data.get("judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("worksheet `judgments` must be a list")
    return judgments


def _worst(verdicts: list[str]) -> str:
    """The most severe verdict (FAIL > REVIEW > PASS); PASS for an empty list."""
    return _ORDER_VERDICT[max((_VERDICT_ORDER.get(v, 0) for v in verdicts), default=0)]


def flag_axis(flag: dict) -> str:
    """Which axis a flag belongs to: `"adherence"` iff explicitly tagged so, else
    `"faithfulness"`.

    Defaulting an untagged flag to `faithfulness` is the SAFE choice: the audit
    stage clears a faithfulness FAIL only when EVERY faithfulness flag is revoked,
    so an untagged flag counts as faithfulness evidence and cannot be washed away
    as if it were a soft adherence note.
    """
    return (
        "adherence" if str(flag.get("axis", "")).strip().lower() == "adherence" else "faithfulness"
    )


def _union_flags(judgments: list[dict]) -> list[dict]:
    """Every judge's flags, de-duplicated by `(claim, issue)`, in first-seen order.

    Each flag carries its `axis` (`faithfulness`|`adherence`) so the audit stage can
    scope a revocation to the right axis (revoking an adherence note must never clear
    a faithfulness FAIL).
    """
    seen: set[tuple[str, str]] = set()
    flags: list[dict] = []
    for judgment in judgments:
        for flag in judgment.get("flags") or []:
            if not isinstance(flag, dict):
                continue  # a malformed judge may emit a bare string flag — skip, don't crash
            pair = (str(flag.get("claim")), str(flag.get("issue")))
            if pair not in seen:
                seen.add(pair)
                flags.append(
                    {
                        "claim": flag.get("claim"),
                        "issue": flag.get("issue"),
                        "axis": flag_axis(flag),
                    }
                )
    return flags


def derive_verdict(faithfulness: str, adherence: str) -> str:
    """The verdict implied by the two axes ALONE: FAIL if either axis fails, REVIEW on
    a soft adherence issue, else PASS.

    The shared deterministic core (mirrors `cv-guardrail._score_clusters`): both the
    aggregate and the verifier-audit re-verdict derive from THIS function, so the
    scoring is identical before and after an audit and cannot drift.
    """
    if faithfulness == "FAIL" or adherence == "FAIL":
        return "FAIL"
    if adherence == "REVIEW":
        return "REVIEW"
    return "PASS"


def _group_verdict(faithfulness: str, adherence: str, verdicts: list[str]) -> str:
    """FAIL if any axis OR any judge's own verdict failed, else REVIEW, else PASS.

    A raw `verdict == "FAIL"` must sink the group even when the judge left the
    `faithfulness`/`adherence` axes at their lenient defaults — otherwise a
    FAIL-but-under-populated judgment would silently render as PASS, the worst
    failure mode for a verification layer. Built on the shared `derive_verdict` core
    (axes) widened by the judges' own raw verdicts, so both stages agree.
    """
    return _worst([derive_verdict(faithfulness, adherence), *verdicts])


def _verdict_of(judgment: dict, key: str, default: str = "PASS") -> str:
    """Read a verdict-like field, upper-cased + trimmed, so `fail`/`Fail` == `FAIL`."""
    return str(judgment.get(key, default)).strip().upper()


def _aggregate_group(item_id: str, target: str, judgments: list[dict]) -> dict:
    """Combine one `(item_id, target)` group's N judgments into a verdict record."""
    faithfulness = (
        "FAIL" if any(_verdict_of(j, "faithfulness") == "FAIL" for j in judgments) else "PASS"
    )
    adherence = _worst([_verdict_of(j, "adherence") for j in judgments])
    verdicts = [_verdict_of(j, "verdict") for j in judgments]
    return {
        "item_id": item_id,
        "target": target,
        "verdict": _group_verdict(faithfulness, adherence, verdicts),
        "faithfulness": faithfulness,
        "adherence": adherence,
        "divergent": len(set(verdicts)) > 1,
        "n_judges": len(judgments),
        "flags": _union_flags(judgments),
    }


def aggregate_verify_judgments(judgment_sets: list[list[dict]]) -> list[dict]:
    """Combine N judges' passes into one verdict per `(item_id, target)`.

    Faithfulness is unforgiving — one judge's `faithfulness=FAIL` makes the group
    FAIL. Adherence takes the worst. `divergent` is True when the judges' own
    verdicts were not unanimous (a signal for the judge≠party audit). Flags are
    unioned + de-duplicated. Worst verdicts (then divergent) sort first.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for judgments in judgment_sets:
        for judgment in judgments:
            if not isinstance(judgment, dict):
                continue  # a malformed judge worksheet may hold a non-object — skip it
            key = (str(judgment.get("item_id")), str(judgment.get("target")))
            groups.setdefault(key, []).append(judgment)
    aggregated = [_aggregate_group(iid, target, js) for (iid, target), js in groups.items()]
    aggregated.sort(key=lambda r: (-_VERDICT_ORDER[r["verdict"]], not r["divergent"]))
    return aggregated


def _render_flag(flag: dict) -> str:
    """One flag line, appending the auditor's cited `reason` when present."""
    reason = f" — {flag['reason']}" if flag.get("reason") else ""
    return f"    - ⚑ {flag['issue']}: “{flag['claim']}”{reason}"


def _render_record_lines(record: dict) -> list[str]:
    """The badge line for one consequential record plus its (reason-annotated) flags."""
    badge = {"FAIL": "❌", "REVIEW": "⚠️", "PASS": "✅"}[record["verdict"]]
    divergent = " · divergent" if record["divergent"] else ""
    audited = " · audited" if record.get("audited") else ""
    header = (
        f"- {badge} **{record['verdict']}** `{record['target']}` "
        f"[{record['item_id']}] "
        f"(faithfulness {record['faithfulness']}, adherence {record['adherence']}, "
        f"{record['n_judges']} judges{divergent}{audited})"
    )
    return [header, *(_render_flag(flag) for flag in record["flags"])]


def _render_audit_section(audit_log: dict) -> list[str]:
    """The `## Audit` block: match counts, the mass-revocation guard, unmatched audits
    and every audit-WASHED record (a FAIL/REVIEW the auditor lowered) so a wash never
    hides in the clean bucket.
    """
    unmatched = audit_log.get("unmatched", [])
    lines = [
        "",
        "## Audit",
        f"{audit_log.get('matched', 0)}/{audit_log.get('supplied', 0)} audits matched"
        + (f" · {len(unmatched)} unmatched" if unmatched else ""),
    ]
    if audit_log.get("mass_revocation_guard"):
        lines.append(
            "- 🛑 mass-revocation guard TRIPPED — revocations on FAIL records were "
            "suppressed (kept FAIL); a human must review."
        )
    for washed in audit_log.get("washed", []):
        lines.append(
            f"- ✅ washed `{washed['target']}` [{washed['item_id']}] "
            f"{washed['from']} → {washed['to']}"
        )
    for gate in audit_log.get("gated", []):
        lines.append(
            f"- ⏸ gated (low-confidence revoke, kept) `{gate['target']}` [{gate['item_id']}] "
            f"“{gate['claim']}” (confidence {gate['confidence']})"
        )
    for miss in unmatched:
        lines.append(f"- ⚠️ unmatched audit [{miss['item_id']}] `{miss['target']}` — not applied")
    for anomaly in audit_log.get("anomalies", []):
        lines.append(
            f"- ❗ ANOMALY [{anomaly['item_id']}] `{anomaly['target']}` — verdict "
            f"{anomaly['verdict']} still carries a confirmed faithfulness flag; investigate."
        )
    return lines


def render_verify_report(aggregated: list[dict], audit_log: dict | None = None) -> tuple[str, str]:
    """Render `(json_report, markdown_report)` from the aggregated verdicts.

    When `audit_log` is supplied (the verifier-audit stage), an `## Audit` section is
    appended surfacing match counts, the mass-revocation guard, unmatched audits and
    every audit-washed record — so a lowered verdict is always visible.
    """
    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    for record in aggregated:
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
    total = len(aggregated)
    report = {"total": total, "counts": counts, "records": aggregated}
    if audit_log is not None:
        report["audit_log"] = audit_log

    lines = [
        "# Verify report",
        "",
        f"**{total}** outputs judged — "
        f"✅ {counts['PASS']} PASS · ⚠️ {counts['REVIEW']} REVIEW · ❌ {counts['FAIL']} FAIL",
        "",
    ]
    for record in aggregated:
        if record["verdict"] == "PASS" and not record["divergent"] and not record.get("audited"):
            continue  # the report leads with what needs a human; clean passes are in the JSON
        lines.extend(_render_record_lines(record))
    if counts["FAIL"] == 0 and counts["REVIEW"] == 0:
        lines.append("_No FAIL/REVIEW verdicts — the corpus passed._")
    if audit_log is not None:
        lines.extend(_render_audit_section(audit_log))
    return json.dumps(report, indent=2, ensure_ascii=False), "\n".join(lines) + "\n"

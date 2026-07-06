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


def _union_flags(judgments: list[dict]) -> list[dict]:
    """Every judge's flags, de-duplicated by `(claim, issue)`, in first-seen order."""
    seen: set[tuple[str, str]] = set()
    flags: list[dict] = []
    for judgment in judgments:
        for flag in judgment.get("flags") or []:
            if not isinstance(flag, dict):
                continue  # a malformed judge may emit a bare string flag — skip, don't crash
            pair = (str(flag.get("claim")), str(flag.get("issue")))
            if pair not in seen:
                seen.add(pair)
                flags.append({"claim": flag.get("claim"), "issue": flag.get("issue")})
    return flags


def _group_verdict(faithfulness: str, adherence: str, verdicts: list[str]) -> str:
    """FAIL if faithfulness/adherence failed, else REVIEW if any reviewed, else PASS."""
    if faithfulness == "FAIL" or adherence == "FAIL":
        return "FAIL"
    if adherence == "REVIEW" or "REVIEW" in verdicts:
        return "REVIEW"
    return "PASS"


def _aggregate_group(item_id: str, target: str, judgments: list[dict]) -> dict:
    """Combine one `(item_id, target)` group's N judgments into a verdict record."""
    faithfulness = "FAIL" if any(j.get("faithfulness") == "FAIL" for j in judgments) else "PASS"
    adherence = _worst([str(j.get("adherence", "PASS")) for j in judgments])
    verdicts = [str(j.get("verdict", "PASS")) for j in judgments]
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


def render_verify_report(aggregated: list[dict]) -> tuple[str, str]:
    """Render `(json_report, markdown_report)` from the aggregated verdicts."""
    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    for record in aggregated:
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
    total = len(aggregated)
    report = {"total": total, "counts": counts, "records": aggregated}

    lines = [
        "# Verify report",
        "",
        f"**{total}** outputs judged — "
        f"✅ {counts['PASS']} PASS · ⚠️ {counts['REVIEW']} REVIEW · ❌ {counts['FAIL']} FAIL",
        "",
    ]
    for record in aggregated:
        if record["verdict"] == "PASS" and not record["divergent"]:
            continue  # the report leads with what needs a human; clean passes are in the JSON
        badge = {"FAIL": "❌", "REVIEW": "⚠️", "PASS": "✅"}[record["verdict"]]
        divergent = " · divergent" if record["divergent"] else ""
        lines.append(
            f"- {badge} **{record['verdict']}** `{record['target']}` "
            f"[{record['item_id']}] "
            f"(faithfulness {record['faithfulness']}, adherence {record['adherence']}, "
            f"{record['n_judges']} judges{divergent})"
        )
        for flag in record["flags"]:
            lines.append(f"    - ⚑ {flag['issue']}: “{flag['claim']}”")
    if counts["FAIL"] == 0 and counts["REVIEW"] == 0:
        lines.append("_No FAIL/REVIEW verdicts — the corpus passed._")
    return json.dumps(report, indent=2, ensure_ascii=False), "\n".join(lines) + "\n"

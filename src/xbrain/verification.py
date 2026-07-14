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

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from xbrain.executors.api import (
    QUOTED_CONTENT_UNFETCHED_NOTE,
    _content_image_descriptions,
    _video_frame_descriptions,
    quoted_content_unfetched,
    thread_text,
    unfetched_links_note,
)
from xbrain.models import Item, Verdict, VerificationVerdict
from xbrain.rubrics import ARTICLE_CHAR_LIMIT, load_rubric
from xbrain.video_digest import _video_source
from xbrain.worksheet import _link_content_source, _video_transcript

logger = logging.getLogger(__name__)

VerifyTarget = Literal["summary", "digest", "topics"]
ALL_TARGETS: tuple[VerifyTarget, ...] = ("summary", "digest", "topics")

# A stored/exported output fingerprint is a lowercase-hex sha256 (see `fingerprint_output`);
# anything else in a filled worksheet is hand-edited garbage and is rejected, not trusted.
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")

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


def fingerprint_output(item: Item, target: VerifyTarget) -> str | None:
    """The sha256 hex of the item's output text for `target` right now, or None when the
    output is absent.

    This is the SINGLE canonicalization shared by TWO callers: `export_verify_worksheet`,
    which stamps each entry's fingerprint at export time (the fingerprint of the output the
    judge actually sees — later threaded through the filled worksheet to the writer), and
    `generate._verdict_badge`, which recomputes it against the CURRENT output. A badge shows
    iff the judged text is byte-identical to what a reader sees now. The write path
    (`apply_verdicts_to_store`) does NOT call this — it stores the export-time fingerprint
    verbatim, so a regeneration between export and write can never bind a verdict to output
    it never judged. Hashing (not storing the text) keeps `items.json` small and makes the
    staleness comparison exact — the moment the summary/digest/topics output is re-generated,
    its fingerprint diverges and the stored verdict is treated as stale. Not a security
    primitive (sha256 is a stable content hash); it just needs to be collision-resistant
    enough that two different outputs never share a fingerprint.
    """
    output = _output_for(item, target)
    if output is None:
        return None
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def _video_parts(item: Item) -> list[str]:
    """The video evidence: its title, the FULL transcript (what it says) and the frame
    descriptions (what it shows). The title reaches the digest generator, so the judge
    must see it too — a digest naming the talk is otherwise unsupported."""
    source = _video_source(item)
    parts: list[str] = []
    if source and source.title:
        parts += ["[Video title]", source.title]
    transcript = _video_transcript(item)
    if transcript:
        parts += ["[Video transcript]", transcript]
    frames = _video_frame_descriptions(item)
    if frames:
        parts += ["[Video frames shown]", *(f"- {description}" for description in frames)]
    return parts


def _article_parts(item: Item) -> list[str]:
    """The fetched LINKED article, with its title (which the api prompt already ships).

    Only `LINK_CONTENT_KINDS` reach this label. A thread or a transcript under it would
    tell the judge a page was downloaded when none was, and hand it the wrong text as
    that page's content.
    """
    source = _link_content_source(item)
    if source is None:
        return []
    label = f"[Linked article — {source.title}]" if source.title else "[Linked article]"
    return [label, source.text[:ARTICLE_CHAR_LIMIT]]


def _unfetched_parts(item: Item) -> list[str]:
    """The markers for content the pipeline never downloaded: links whose body is
    missing (all of them, or the remainder of a PARTIAL fetch) and a quoted post, which
    no fetcher retrieves. An output describing either is then checkable as unsupported
    instead of being waved through against evidence that is not there."""
    parts: list[str] = []
    links_note = unfetched_links_note(item)
    if links_note:
        parts += [
            "[Links — content NOT fetched]",
            *(f"- {link.url}  (domain: {link.domain})" for link in item.links),
            links_note,
        ]
    if quoted_content_unfetched(item):
        parts += ["[Quoted post — content NOT fetched]", QUOTED_CONTENT_UNFETCHED_NOTE]
    return parts


def _source_text(item: Item) -> str:
    """The ground-truth source the judge checks the output against.

    Concatenates the labelled evidence present on the item — the author metadata (WHO
    POSTED it, not who wrote its content), the video title + FULL transcript (what it
    says) + frame descriptions (what it shows), the image descriptions, the fetched
    article body, the poster's own thread text, and the tweet text — so a claim in the
    output can be traced to it. The judge must see everything the GENERATORS see: an
    output grounded in a photo description or an article title would otherwise be
    judged against a source that never held it, and flagged unsupported (a false FAIL).

    Every kind of evidence keeps its OWN label. A thread is the poster's own words, not
    a fetched page. Serving one under `[Linked article]` would affirmatively tell the
    skeptical judge that a link was downloaded and hand it text that is not that link's
    content — turning an unsupported claim about the linked piece into a PASS. Content
    the pipeline never downloaded (a link's body, a quoted post) is marked as such.
    """
    parts: list[str] = ["[Author]", f"@{item.author.handle} ({item.author.name})"]
    parts += _video_parts(item)
    images = _content_image_descriptions(item)
    if images:
        parts += ["[Images in the post]", *(f"- {description}" for description in images)]
    parts += _article_parts(item)
    thread = thread_text(item)
    if thread:
        parts += ["[Thread — full text, same author]", thread]
    parts += _unfetched_parts(item)
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
                # The fingerprint of the EXACT output shown to the judge, captured here so the
                # verdict binds to the JUDGED text, not to whatever the store holds at write
                # time (#79). Threaded through the filled worksheet to `apply_verdicts_to_store`.
                "output_fingerprint": fingerprint_output(item, target),
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


def _fold_fingerprints(stamps: Iterable[tuple[tuple[str, str], str]]) -> dict[tuple[str, str], str]:
    """Fold `((item_id, target), fingerprint)` stamps into one map — **conflict is fail-safe,
    never silently first-seen**.

    If two stamps for the same `(item, target)` disagree, the key is DROPPED entirely rather
    than resolved to whichever came first. A dropped key becomes `fingerprint-missing` at
    write, so no verdict is persisted and nothing is badged: we refuse to guess which stamp
    was the judged one. The single conflict policy for every fingerprint source (judge
    worksheets, report records, the audit worksheet).
    """
    fingerprints: dict[tuple[str, str], str] = {}
    conflicting: set[tuple[str, str]] = set()
    for key, fingerprint in stamps:
        existing = fingerprints.get(key)
        if existing is not None and existing != fingerprint:
            logger.debug("conflicting output_fingerprint stamps for %s — dropping key", key)
            conflicting.add(key)
        fingerprints.setdefault(key, fingerprint)
    for key in conflicting:
        del fingerprints[key]
    return fingerprints


def _entry_stamps(entries: Iterable[object]) -> Iterator[tuple[tuple[str, str], str]]:
    """Every valid judged-fingerprint stamp carried by `entries` (worksheet items or report
    records — both key it the same way), skipping the missing/garbage ones."""
    for entry in entries:
        resolved = _entry_fingerprint(entry)
        if resolved is not None:
            yield resolved


def import_verify_fingerprints(paths: list[Path]) -> dict[tuple[str, str], str]:
    """Map every `(item_id, target)` to the fingerprint of the output that was JUDGED, read
    from the worksheet(s) `items` block (stamped by `export_verify_worksheet`, and carried
    through to the audit worksheet by `export_audit_worksheet`).

    This is the plumbing that lets the writer store the JUDGED fingerprint instead of a
    write-time recompute (#79): the N judge copies all derive from one export, so their
    `items` blocks carry IDENTICAL fingerprints for each `(item, target)`. A missing/garbage
    fingerprint (an old worksheet without the field, or a hand-edited hash) is skipped, so
    the writer treats that `(item, target)` as unfingerprintable and does not badge it.
    Conflicting stamps drop the key (see `_fold_fingerprints`).
    """
    return _fold_fingerprints(
        stamp for path in paths for stamp in _entry_stamps(_worksheet_items(path))
    )


def record_fingerprints(records: Iterable[object]) -> dict[tuple[str, str], str]:
    """The judged fingerprints carried BY the aggregated records themselves (stamped into
    `verify-report.json` by `stamp_record_fingerprints`).

    The audit stage reads the REPORT back — not the judge worksheets — so this is how the
    judged fingerprint reaches the post-audit write for every record in the merged report,
    including the ones the auditor never saw (the clean passes outside the consequential set).
    A record with no stamp, or a hand-edited one, resolves to nothing → `fingerprint-missing`.
    """
    return _fold_fingerprints(_entry_stamps(records))


def combine_fingerprints(*sources: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    """Union of several judged-fingerprint maps under the one fail-safe conflict policy.

    The post-audit write reads the fingerprint from TWO sources that agree by construction —
    the report records and the applied audit worksheet, both descending from the same
    judge-export stamp. If they DISAGREE (one of the two artifacts was hand-edited), the key
    is dropped and the record is skipped, never resolved by precedence.
    """
    return _fold_fingerprints(stamp for source in sources for stamp in source.items())


def stamp_record_fingerprints(
    aggregated: list[dict], fingerprints: dict[tuple[str, str], str]
) -> None:
    """Stamp each aggregated record with the JUDGED fingerprint of its output, in place.

    Carries the judge-export fingerprint into `verify-report.json`, which is what the audit
    stage reads back — so the audited (authoritative) verdict can be persisted bound to the
    output the judges actually saw, exactly like the plain apply path. A record with no judged
    fingerprint is left UNSTAMPED (never a guessed value): the writer then skips it.
    """
    for record in aggregated:
        fingerprint = fingerprints.get((str(record.get("item_id")), str(record.get("target"))))
        if fingerprint is not None:
            record["output_fingerprint"] = fingerprint


def _worksheet_items(path: Path) -> list:
    """The `items` block of one worksheet, or `[]` when it is absent/misshapen."""
    if not path.exists():
        raise FileNotFoundError(f"Worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", []) if isinstance(data, dict) else []
    return items if isinstance(items, list) else []


def _entry_fingerprint(entry: object) -> tuple[tuple[str, str], str] | None:
    """`((item_id, target), fingerprint)` for a worksheet entry carrying a valid stamped
    fingerprint, else None (a non-dict entry or a missing/garbage hash)."""
    if not isinstance(entry, dict):
        return None
    fingerprint = entry.get("output_fingerprint")
    key = (str(entry.get("item_id")), str(entry.get("target")))
    if isinstance(fingerprint, str) and _FINGERPRINT_RE.match(fingerprint):
        return key, fingerprint
    logger.debug("worksheet entry %s carries no valid output_fingerprint", key)
    return None


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
                logger.debug("skipping non-dict flag in judgment: %r", flag)
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
                logger.debug("skipping non-dict judgment in worksheet: %r", judgment)
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


def _flag_issues(flags: list) -> list[str]:
    """The concise issue labels from an aggregated record's flags, in order — what the
    note badge surfaces as the "top flag issue". A non-dict or issue-less flag is skipped
    (a malformed judge/auditor flag must not crash the write path)."""
    issues: list[str] = []
    for flag in flags:
        if isinstance(flag, dict) and flag.get("issue"):
            issues.append(str(flag["issue"]))
        else:
            logger.debug("skipping flag with no issue label: %r", flag)
    return issues


def _axis(value: object) -> Verdict:
    """Coerce an aggregated axis value to a valid `Verdict`, defaulting to PASS on anything
    unexpected — so a malformed record downgrades gracefully instead of crashing the typed
    `VerificationVerdict` construction."""
    normalized = str(value).strip().upper()
    return cast(Verdict, normalized) if normalized in _VERDICT_ORDER else "PASS"


@dataclass(frozen=True)
class VerdictWriteResult:
    """The outcome of an opt-in `--write-verdicts` pass: how many verdicts were persisted,
    and every record that was SKIPPED with its reason — so a dropped FAIL is never silent."""

    written: int
    skipped: list[tuple[str, str, str]] = field(default_factory=list)  # (item_id, target, reason)

    @property
    def attempted(self) -> int:
        return self.written + len(self.skipped)

    def summary(self) -> str:
        """A one-line human summary: written / attempted, and the skip reasons tallied."""
        if not self.skipped:
            return f"{self.written} verdicts escritos"
        tally = Counter(reason for _, _, reason in self.skipped)
        detail = ", ".join(f"{count} {reason}" for reason, count in sorted(tally.items()))
        return (
            f"{self.written} de {self.attempted} verdicts escritos "
            f"({len(self.skipped)} omitidos: {detail})"
        )


def _verdict_skip_reason(
    record: object, store: dict[str, Item], fingerprints: dict[tuple[str, str], str]
) -> str | None:
    """Why this aggregated record cannot be written as a verdict, or None if it can.

    The judged fingerprint MUST come from `fingerprints` (stamped at export) — a record with
    no such fingerprint is `fingerprint-missing`, never silently re-fingerprinted against the
    live store (that is the exact bug this plumbing closes, #79).
    """
    if not isinstance(record, dict):
        return "malformed-record"
    item_id = str(record.get("item_id"))
    target = str(record.get("target"))
    if item_id not in store:
        return "item-gone"
    if target not in ALL_TARGETS:
        return "bad-target"
    if _verdict_of(record, "verdict", "") not in _VERDICT_ORDER:
        return "bad-verdict"
    fingerprint = fingerprints.get((item_id, target))
    if fingerprint is None:
        return "fingerprint-missing"
    if not _FINGERPRINT_RE.match(fingerprint):
        return "fingerprint-invalid"
    return None


def _build_verdict(record: dict, output_fingerprint: str) -> VerificationVerdict:
    """Assemble one `VerificationVerdict` from an aggregated record + its JUDGED fingerprint."""
    return VerificationVerdict(
        verdict=cast(Verdict, _verdict_of(record, "verdict")),
        faithfulness=_axis(record.get("faithfulness")),
        adherence=_axis(record.get("adherence")),
        output_fingerprint=output_fingerprint,
        verified_at=datetime.now(timezone.utc),
        flags=_flag_issues(record.get("flags") or []),
    )


def apply_verdicts_to_store(
    store: dict[str, Item],
    aggregated: list[dict],
    fingerprints: dict[tuple[str, str], str],
) -> VerdictWriteResult:
    """Persist each aggregated verdict onto its item as a `VerificationVerdict`, keyed by
    target, storing the JUDGED output fingerprint from `fingerprints` (opt-in
    `verify --write-verdicts`).

    The stored `output_fingerprint` is the one stamped at worksheet export (threaded here via
    `import_verify_fingerprints`), NOT a write-time recompute against the live store — so a
    regeneration between export and write cannot bind a verdict to output it never judged
    (#79). Backward-compatible and defensive: a record is SKIPPED (never crashes the write,
    never silently dropped) when it is not a dict, its item is gone from the store, its target
    is unknown, its verdict is not PASS/REVIEW/FAIL, or it has no valid judged fingerprint —
    each reason is tallied on the returned `VerdictWriteResult`. Mutates `store` in place; the
    caller snapshots + saves.
    """
    written = 0
    skipped: list[tuple[str, str, str]] = []
    for record in aggregated:
        reason = _verdict_skip_reason(record, store, fingerprints)
        if reason is not None:
            as_dict = record if isinstance(record, dict) else {}
            skipped.append((str(as_dict.get("item_id")), str(as_dict.get("target")), reason))
            continue
        item_id, target = str(record["item_id"]), str(record["target"])
        store[item_id].verification[target] = _build_verdict(
            record, fingerprints[(item_id, target)]
        )
        written += 1
    return VerdictWriteResult(written=written, skipped=skipped)

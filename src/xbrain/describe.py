"""Describe downloaded photos with a vision LLM; feed descriptions into enrich.

The `describe_all` orchestrator walks every photo entry the downloader
has produced, batches the bytes into vision-API calls (default: 5 images
per call to Claude Sonnet), parses the per-image JSON judgments, and
transitions matched entries to `MediaPhotoDescribed`. The persisted
description is consumed by the enrich-time prompt in `executors.api`
and the topic-synth prompt in `topic_synth` — decorative photos are
filtered out at that consumption seam so they introduce no topic noise.

The structure mirrors `xbrain.executors.api` and `xbrain.topic_synth`:
a recoverable-errors tuple, per-batch failure isolation,
`logger.warning` on every failure, `RuntimeError` on total failure, and
a `SUMMARY: described: N, failed: M, skipped: K` stderr line for the
CLI. Programmer bugs (`AttributeError`) and `KeyboardInterrupt`
propagate — narrow `except` clauses only.

I/O dependencies (the Anthropic client, the media root path) are
keyword-injectable so tests run offline. The store mutation is in
place; the caller is expected to wrap each transition with a
store-write (the `on_progress` callback fires after every batch).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from xbrain.models import (
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
)
from xbrain.rubrics import load_rubric

logger = logging.getLogger(__name__)

# Default per-call image count. The vision rubric is tuned for batches in
# the 1-10 range; 5 is the sweet spot the spec settled on (12-15 % token
# saving vs per-image, modest added complexity).
_DEFAULT_BATCH_SIZE = 5

# Token ceiling for the JSON list response. Per-image average is ~3
# sentences ≈ 80 tokens of prose + 20 of JSON scaffolding = ~100 tokens.
# A batch of 5 fits comfortably under 600; the cap is set high enough to
# survive an over-eager model that emits long descriptions.
_MAX_TOKENS = 1200

# Map file extensions to the Anthropic vision media-type strings. The
# downloader writes one of these three (see `xbrain.media._FORMAT_EXTENSIONS`).
_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _utcnow() -> datetime:
    """Default `now` clock — UTC-aware `datetime.now()`.

    Module-level (not a lambda inside `describe_all`) so tests can
    monkeypatch a deterministic clock without going through the `now=`
    keyword argument on every call site.
    """
    return datetime.now(timezone.utc)


@dataclass
class DescribeReport:
    """Counts emitted by `describe_all` for the CLI's SUMMARY line.

    `photos_skipped_already_described` is the idempotency proof — a
    no-op re-run with no version bump must report every previously
    described photo here. `batches_attempted` counts API calls actually
    issued; `batches_failed` counts the ones the recoverable-errors
    tuple swallowed. A run with `photos_attempted > 0 and
    photos_described == 0` raises before this report leaves the
    orchestrator.
    """

    items_processed: int = 0
    photos_attempted: int = 0
    photos_described: int = 0
    photos_failed: int = 0
    photos_skipped_already_described: int = 0
    batches_attempted: int = 0
    batches_failed: int = 0
    elapsed_seconds: float = 0.0
    # Per-item failures keyed by item id → list of (url, error) tuples.
    # Surfaces in the verbose CLI output without re-walking the store.
    per_item_failures: dict[str, list[tuple[str, str]]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Candidate:
    """One photo eligible for description on this run.

    Holds back-references to `Item` and the media-list `index` so the
    orchestrator can swap the transitioned variant back into place
    without re-scanning the store. `bytes_data` is loaded lazily by
    `_load_bytes` — failing to read the file is a per-photo failure,
    not a total-run abort.
    """

    item_id: str
    item: Item
    index: int
    entry: MediaPhotoDownloaded | MediaPhotoDescribed


def _recoverable_errors() -> tuple[type[Exception], ...]:
    """Exception classes a per-batch failure should swallow + log + continue on.

    Mirrors `xbrain.executors.api._recoverable_errors`. `anthropic.APIError`
    covers auth, rate-limit, server-side and network errors the SDK
    normalises. `ValueError` covers validator rejections (and
    `pydantic.ValidationError`, a `ValueError` subclass in pydantic v2).
    `json.JSONDecodeError` covers malformed LLM responses. `KeyError`
    covers responses missing expected fields. `OSError` covers
    file-read failures when streaming photo bytes off disk (a missing
    file under `data/media/` is per-photo recoverable, not a total
    abort).

    Lazy-imported because `anthropic` is optional in the test
    environment (the client is faked via `tests.conftest.FakeAnthropic`).
    """
    try:
        from anthropic import APIError

        return (APIError, ValueError, json.JSONDecodeError, KeyError, OSError)
    except ImportError:
        return (ValueError, json.JSONDecodeError, KeyError, OSError)


def _is_stale(entry: MediaPhotoDescribed, *, current_version: str) -> bool:
    """A described entry is stale when its version no longer matches the config.

    Bumping `describe_version` in `config.toml` is the manual trigger to
    re-describe the whole corpus against a new rubric — no `--force`
    needed. The version check is exact-string: there is no ordering
    relation between versions, only equality, so a deliberate downgrade
    is also a "describe again" signal.
    """
    return entry.description_version != current_version


def _eligible(
    entry: object,
    *,
    force: bool,
    current_version: str,
) -> bool:
    """Decide whether `describe_all` should attempt this entry on THIS run.

    `MediaPhotoDownloaded` entries are always eligible — they have not
    been described yet by definition. `MediaPhotoDescribed` entries are
    eligible only when `--force` is set or the persisted version is
    stale vs the current `describe_version` config. Every other
    variant (`MediaPhotoPending`, `MediaPhotoFailed`, `MediaVideoPending`)
    is out of scope — describing only runs over photos whose bytes are
    already on disk.
    """
    if isinstance(entry, MediaPhotoDownloaded):
        return True
    if isinstance(entry, MediaPhotoDescribed):
        if force:
            return True
        return _is_stale(entry, current_version=current_version)
    return False


def _tally_skipped(
    entry: object,
    *,
    current_version: str,
    report: DescribeReport,
) -> None:
    """Bump `photos_skipped_already_described` when this entry is a no-op skip.

    Pulled out of the candidate iterator so the loop body in
    `_iter_candidates` keeps a low complexity grade. Only described
    entries on the current version count as skips — pending/failed/
    video entries are silently out of scope for `xbrain describe`.
    """
    if isinstance(entry, MediaPhotoDescribed) and not _is_stale(
        entry, current_version=current_version
    ):
        report.photos_skipped_already_described += 1


def _iter_item_candidates(
    item_id: str,
    item: Item,
    *,
    force: bool,
    current_version: str,
    report: DescribeReport,
) -> Iterator[_Candidate]:
    """Yield every eligible candidate inside one item's media list.

    Pulled out of the cross-item iterator so the outer scan stays
    flat. Per-entry tallies (skips) go on the report; the caller
    decides whether the item itself counts as processed by checking
    if this iterator yielded anything.
    """
    for index, entry in enumerate(item.media):
        if not _eligible(entry, force=force, current_version=current_version):
            _tally_skipped(entry, current_version=current_version, report=report)
            continue
        assert isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed))
        yield _Candidate(item_id=item_id, item=item, index=index, entry=entry)


def _take_with_limit(candidates: Iterator[_Candidate], limit: int | None) -> Iterator[_Candidate]:
    """Yield from `candidates`, stopping after `limit` items.

    Pulled out so `_iter_candidates` does not interleave the
    limit-countdown with the per-item bookkeeping. `None` means "no
    limit" — yields everything.
    """
    if limit is None:
        yield from candidates
        return
    remaining = limit
    for candidate in candidates:
        if remaining <= 0:
            return
        remaining -= 1
        yield candidate


def _iter_candidates(
    items: dict[str, Item],
    *,
    force: bool,
    limit: int | None,
    items_filter: set[str] | None,
    current_version: str,
    report: DescribeReport,
) -> Iterator[_Candidate]:
    """Yield each candidate eligible for description on THIS run.

    Side effects on `report`: bumps `items_processed` once per item
    that contributes at least one yielded candidate, and bumps
    `photos_skipped_already_described` for each `MediaPhotoDescribed`
    entry passed over (via `_tally_skipped`). Stops yielding once
    `limit` is exhausted.

    `items_processed` is bumped on the FIRST yielded candidate of an
    item, not at the end of its scan — that way a `limit` that
    truncates mid-item still counts the item as processed (work
    happened on it). Items whose every photo was skipped do NOT count
    as processed.
    """
    seen_item_ids: set[str] = set()

    def _all_eligible() -> Iterator[_Candidate]:
        for item_id, item in items.items():
            if items_filter is not None and item_id not in items_filter:
                continue
            if not item.media:
                continue
            yield from _iter_item_candidates(
                item_id,
                item,
                force=force,
                current_version=current_version,
                report=report,
            )

    for candidate in _take_with_limit(_all_eligible(), limit):
        if candidate.item_id not in seen_item_ids:
            seen_item_ids.add(candidate.item_id)
            report.items_processed += 1
        yield candidate


def _chunked(candidates: list[_Candidate], size: int) -> Iterator[list[_Candidate]]:
    """Split a candidate list into batches of at most `size` entries.

    `size` is guaranteed positive by the CLI layer (`Typer` validates
    integer ranges); this helper does not re-validate. An empty input
    yields nothing — the orchestrator never issues a no-op API call.
    """
    for start in range(0, len(candidates), size):
        yield candidates[start : start + size]


def _media_type(local_path: str) -> str:
    """Map an on-disk path's extension to its Anthropic media-type string.

    The downloader writes one of `.jpg` / `.png` / `.webp` (see
    `xbrain.media._FORMAT_EXTENSIONS`). Anything else came from a hand
    edit of `items.json` or a future format we have not registered —
    fall back to `image/jpeg` and let Anthropic reject it if the bytes
    do not match. The fall-back keeps a per-photo failure out of the
    total-failure raise path.
    """
    suffix = Path(local_path).suffix.lower()
    return _MEDIA_TYPES.get(suffix, "image/jpeg")


def _load_bytes(media_root: Path, local_path: str) -> bytes:
    """Read the photo bytes from `data/media/<local_path>`.

    Raises `OSError` (a `FileNotFoundError` subclass when the file is
    missing). The orchestrator's `_recoverable_errors` tuple catches it
    so a missing file is a per-photo failure (the operator can re-run
    `xbrain media` to repopulate), never a whole-batch abort.
    """
    return (media_root / local_path).read_bytes()


def _build_image_block(data: bytes, media_type: str) -> dict:
    """Build one Anthropic vision content block from raw photo bytes.

    The wire shape is `{type: image, source: {type: base64, media_type, data}}`.
    Tests bypass this by using a `FakeAnthropic` that does not inspect
    `messages`; production uses the real SDK which validates the shape.
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def _system_prompt(language: str) -> str:
    """Build the system prompt — the declarative rubric with `{language}` substituted."""
    return load_rubric("describe-image", language=language)


def _user_directive(batch_size: int) -> str:
    """Plain-text directive that follows the image blocks in the user turn.

    Tells the model the index range so the JSON list it emits is
    contractually one-to-one with the input order. Indices are
    zero-based to match Python and the rubric's example.
    """
    last = batch_size - 1
    return (
        f"Describe images 0 through {last}. "
        "Return a JSON list with one entry per image, in the order received."
    )


def _extract_response_text(response: object) -> str:
    """Pull the JSON-bearing text out of a vision response, stripping fences.

    The Anthropic SDK packs the model's reply in `.content` as a list
    of typed blocks; only `text` blocks carry JSON. Some models wrap
    the JSON in a ```json ... ``` Markdown fence despite the rubric
    explicitly forbidding it — strip a single leading/trailing fence
    pair so the downstream `json.loads` does not trip on a Markdown
    artefact.
    """
    blocks = [b for b in getattr(response, "content", []) if getattr(b, "type", None) == "text"]
    if not blocks:
        raise ValueError("vision response has no text block")
    text = "".join(b.text for b in blocks).strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()
    return text


def _validate_judgment_entry(entry: object, *, batch_size: int) -> dict:
    """Validate one per-image judgment dict and return it on success.

    Pulled out of `_parse_batch_response` so the loop body in the
    parser stays a one-liner — keeps the complexity grade off C.
    Enforces the wire contract: required keys + int (not bool) index
    in `[0, batch_size)` + bool decorative flag + str description.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"vision response entry is not a JSON object: {entry!r}")
    if not {"index", "is_decorative", "description"} <= entry.keys():
        raise ValueError(
            f"vision response entry missing required keys "
            f"(got {sorted(entry)!r}, need index/is_decorative/description)"
        )
    index = entry["index"]
    # `bool` is a subclass of `int` in Python — exclude it explicitly so a
    # `True`/`False` index does not silently become 1/0.
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(f"vision response `index` must be int, got {index!r}")
    if not (0 <= index < batch_size):
        raise ValueError(f"vision response `index` {index} out of batch range [0, {batch_size})")
    if not isinstance(entry["is_decorative"], bool):
        raise ValueError(
            f"vision response `is_decorative` must be bool, got {entry['is_decorative']!r}"
        )
    if not isinstance(entry["description"], str):
        raise ValueError(f"vision response `description` must be str, got {entry['description']!r}")
    return entry


def _parse_batch_response(response: object, batch_size: int) -> list[dict]:
    """Extract and validate the JSON list of judgments from a vision response.

    Splits responsibility cleanly: `_extract_response_text` deals with
    transport (text blocks + Markdown fence noise), this function deals
    with shape (list of judgments with unique indices), and
    `_validate_judgment_entry` deals with per-entry typing. Each step
    raises `ValueError` so the orchestrator's `_recoverable_errors`
    tuple catches every shape violation as a per-batch failure.
    """
    text = _extract_response_text(response)
    decoded = json.loads(text)
    if not isinstance(decoded, list):
        raise ValueError(f"vision response is not a JSON list (got {type(decoded).__name__})")
    seen_indices: set[int] = set()
    judgments: list[dict] = []
    for entry in decoded:
        validated = _validate_judgment_entry(entry, batch_size=batch_size)
        index = validated["index"]
        if index in seen_indices:
            raise ValueError(f"vision response has duplicate index {index}")
        seen_indices.add(index)
        judgments.append(validated)
    return judgments


def _apply_judgment(
    candidate: _Candidate,
    judgment: dict,
    *,
    language: str,
    version: str,
    described_at: datetime,
) -> MediaPhotoDescribed:
    """Build the post-transition `MediaPhotoDescribed` variant for one candidate.

    Decorative entries are written with an empty description regardless
    of what the model returned for the `description` field — the rubric
    contract is "empty for decorative" and we enforce it at the
    boundary so downstream code can rely on `is_decorative => not
    description`. The pre-existing `local_path` / `width` / `height` /
    `bytes_size` / `downloaded_at` fields are carried over verbatim
    from the prior variant (Downloaded or Described).
    """
    is_decorative = bool(judgment["is_decorative"])
    description = "" if is_decorative else str(judgment["description"])
    return MediaPhotoDescribed(
        url=candidate.entry.url,
        local_path=candidate.entry.local_path,
        width=candidate.entry.width,
        height=candidate.entry.height,
        bytes_size=candidate.entry.bytes_size,
        downloaded_at=candidate.entry.downloaded_at,
        is_decorative=is_decorative,
        description=description,
        description_lang=language,
        description_version=version,
        described_at=described_at,
    )


def _record_batch_failure(
    report: DescribeReport,
    batch: list[_Candidate],
    error: str,
) -> None:
    """Mark every candidate in a failed batch as a per-photo failure.

    A per-batch API error means we have no judgments for any of the
    images in that batch — each photo stays in its current variant
    (`MediaPhotoDownloaded` or stale `MediaPhotoDescribed`) and is
    eligible again on the next `xbrain describe` run. The counts and
    `per_item_failures` map reflect the per-photo unit so a partial-batch
    success on a follow-up run can still net positive without confusing
    the totals.
    """
    report.batches_failed += 1
    for candidate in batch:
        report.photos_failed += 1
        report.per_item_failures.setdefault(candidate.item_id, []).append(
            (candidate.entry.url, error)
        )


def describe_all(
    items: dict[str, Item],
    media_root: Path,
    *,
    model: str,
    output_language: str,
    description_version: str,
    force: bool = False,
    limit: int | None = None,
    items_filter: list[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    client: object | None = None,
    on_progress: Callable[[], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> DescribeReport:
    """Describe every eligible photo across the store; return a structured report.

    Eligibility (without `--force`):
    - `MediaPhotoDownloaded` — always.
    - `MediaPhotoDescribed` whose `description_version` ≠ `description_version`
      (the per-call argument, sourced from `config.describe_version`).

    With `--force`:
    - Every `MediaPhotoDownloaded` AND every `MediaPhotoDescribed` is
      re-described. The persisted description is overwritten.

    Out of scope (every run):
    - `MediaPhotoPending` — describing only runs over photos with bytes
      on disk; the operator must call `xbrain media` first.
    - `MediaPhotoFailed` — same reason.
    - `MediaVideoPending` — photos only; vision-API support for video
      is a separate phase.

    The function mutates `items` in place; the caller is expected to
    wrap each batch transition with a store write (the `on_progress`
    callback fires after every batch, success or failure). The
    Ctrl-C-coherent invariant lives there: the store is written
    between batches, never mid-API-call.

    `media_root` is the directory under which `<item_id>/<index>.<ext>`
    photo files live (typically `data/media/`).

    Raises:
        RuntimeError: when EVERY batch attempted in the run fails. A
            total failure (a revoked API key, an exhausted quota, a
            corrupted media tree) must surface as a non-zero exit. The
            CLI's `_handle_cli_errors` converts this into a clean
            operator message + exit code 1.
    """
    if client is None:
        from anthropic import Anthropic  # lazy: tests inject FakeAnthropic

        client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    clock: Callable[[], datetime] = now if now is not None else _utcnow

    started = time.monotonic()
    report = DescribeReport()
    filter_set = set(items_filter) if items_filter else None
    candidates = list(
        _iter_candidates(
            items,
            force=force,
            limit=limit,
            items_filter=filter_set,
            current_version=description_version,
            report=report,
        )
    )
    if not candidates:
        report.elapsed_seconds = time.monotonic() - started
        return report

    system = _system_prompt(output_language)
    recoverable = _recoverable_errors()

    for batch in _chunked(candidates, batch_size):
        _run_one_batch(
            batch=batch,
            client=client,
            model=model,
            system=system,
            output_language=output_language,
            description_version=description_version,
            clock=clock,
            media_root=media_root,
            recoverable=recoverable,
            report=report,
        )
        if on_progress is not None:
            on_progress()

    report.elapsed_seconds = time.monotonic() - started
    _raise_on_total_failure(report)
    _print_partial_failure_summary(report)
    return report


def _run_one_batch(
    *,
    batch: list[_Candidate],
    client: object,
    model: str,
    system: str,
    output_language: str,
    description_version: str,
    clock: Callable[[], datetime],
    media_root: Path,
    recoverable: tuple[type[Exception], ...],
    report: DescribeReport,
) -> None:
    """Execute one batch end-to-end: build, call, parse, apply, catch.

    Pulled out of `describe_all` to keep the outer loop's complexity
    under grade C while preserving the per-batch isolation contract.
    Programmer bugs (`AttributeError`, ...) and `KeyboardInterrupt`
    fall outside `recoverable` and propagate.
    """
    report.batches_attempted += 1
    report.photos_attempted += len(batch)
    try:
        content_blocks = _build_user_blocks(batch, media_root)
        response = _messages_create(
            client=client,
            model=model,
            max_tokens=_MAX_TOKENS,
            system=system,
            content_blocks=content_blocks,
        )
        judgments = _parse_batch_response(response, batch_size=len(batch))
        _apply_batch_judgments(
            batch=batch,
            judgments=judgments,
            language=output_language,
            version=description_version,
            described_at=clock(),
            report=report,
        )
    except recoverable as exc:
        message = str(exc)
        logger.warning("describe: batch failed (%d photos): %s", len(batch), message)
        _record_batch_failure(report, batch, message)


def _raise_on_total_failure(report: DescribeReport) -> None:
    """Raise `RuntimeError` when every attempted batch failed.

    A total-failure run is a non-zero-exit signal — the CLI's
    `_handle_cli_errors` converts the raise into a clean operator
    message + exit code 1. Pulled out so `describe_all` keeps its
    complexity grade under C.
    """
    if report.photos_attempted > 0 and report.photos_described == 0:
        raise RuntimeError(
            f"All {report.batches_attempted} describe batches failed "
            f"({report.photos_attempted} photos); see warnings above for details."
        )


def _print_partial_failure_summary(report: DescribeReport) -> None:
    """Print the SUMMARY line on partial failure — mirrors `executors.api`.

    Stays silent on a clean run (no failures = no noise) and on a
    total-failure run (the raise above is the signal). The line shape
    matches `xbrain.media.emit_summary_line` so log parsers can rely
    on a single `SUMMARY:` prefix across all stages.
    """
    if report.photos_failed > 0:
        print(
            f"SUMMARY: described: {report.photos_described}, "
            f"failed: {report.photos_failed}, "
            f"skipped: {report.photos_skipped_already_described}",
            file=sys.stderr,
        )


def _build_user_blocks(batch: list[_Candidate], media_root: Path) -> list[dict]:
    """Build the user-turn content blocks for one batch: images + directive.

    Loads each photo's bytes from disk (raises `OSError` on a missing
    file — caught by the orchestrator's `_recoverable_errors` so the
    whole batch is marked failed and the next run re-attempts). Each
    image is encoded once; the directive text block follows last so the
    rubric's "Describe images 0 through N" framing pairs with the
    visual context above it.
    """
    blocks: list[dict] = []
    for candidate in batch:
        data = _load_bytes(media_root, candidate.entry.local_path)
        blocks.append(_build_image_block(data, _media_type(candidate.entry.local_path)))
    blocks.append({"type": "text", "text": _user_directive(len(batch))})
    return blocks


def _messages_create(
    *,
    client: object,
    model: str,
    max_tokens: int,
    system: str,
    content_blocks: list[dict],
) -> object:
    """Thin wrapper around the Anthropic SDK's `messages.create`.

    Pulled out so the orchestrator stays readable and so tests can
    inject a `FakeAnthropic` whose `messages.create` records kwargs.
    Returns the raw response — the parser inspects `.content` blocks.
    """
    return client.messages.create(  # type: ignore[attr-defined]
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content_blocks}],
    )


def _apply_batch_judgments(
    *,
    batch: list[_Candidate],
    judgments: list[dict],
    language: str,
    version: str,
    described_at: datetime,
    report: DescribeReport,
) -> None:
    """Apply per-image judgments to the batch, transitioning each entry.

    Pairs each judgment with the candidate at its `index`. A judgment
    count that does not match the batch size is a contract violation
    (parser already enforces the index range) — if the list is short,
    the missing candidates are tallied as per-photo failures so they
    re-enter the candidate pool on the next run; if it is long, the
    excess is a programmer bug (parser-side dup check rules out
    duplicate indices already).
    """
    by_index = {entry["index"]: entry for entry in judgments}
    if len(by_index) != len(judgments):
        # Defence in depth — `_parse_batch_response` already rejects
        # duplicate indices, so reaching here means the parser was
        # bypassed. Raise so the developer sees it.
        raise ValueError("internal: duplicate judgment indices survived parser check")
    for position, candidate in enumerate(batch):
        judgment = by_index.get(position)
        if judgment is None:
            report.photos_failed += 1
            report.per_item_failures.setdefault(candidate.item_id, []).append(
                (candidate.entry.url, f"vision response omitted index {position}")
            )
            continue
        new_entry = _apply_judgment(
            candidate,
            judgment,
            language=language,
            version=version,
            described_at=described_at,
        )
        candidate.item.media[candidate.index] = new_entry
        report.photos_described += 1


def emit_summary_line(report: DescribeReport, *, out: "io.IOBase | None" = None) -> None:
    """Print the SUMMARY line on stderr (mirrors `media.emit_summary_line`).

    Stays silent on a fully no-op run (no attempts, no skips) — a
    `--limit 0` or an `--items` filter that matched nothing produces no
    noise. `out` is injectable for tests; defaults to `sys.stderr`.
    """
    if report.photos_attempted == 0 and report.photos_skipped_already_described == 0:
        return
    target = out if out is not None else sys.stderr
    print(
        f"SUMMARY: described: {report.photos_described}, "
        f"failed: {report.photos_failed}, "
        f"skipped: {report.photos_skipped_already_described}",
        file=target,
    )

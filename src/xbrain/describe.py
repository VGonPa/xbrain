"""Describe downloaded photos with a vision LLM; feed descriptions into enrich.

The `describe_all` orchestrator walks every photo entry the downloader
has produced, batches the bytes into vision-API calls (default: 5 images
per call to Claude Sonnet), parses the per-image JSON judgments, and
transitions matched entries to `MediaPhotoDescribed`. Decorative photos
(avatars, reaction memes) are filtered at the downstream consumption
seam so they introduce no topic noise.

The structure mirrors `xbrain.executors.api` and `xbrain.media`:
recoverable-errors tuple, per-batch failure isolation, `logger.warning`
on every failure, `RuntimeError` on total failure. Programmer bugs and
`KeyboardInterrupt` propagate. The `SUMMARY: described: N, failed: M,
skipped: K` stderr line is emitted exclusively by the CLI via
`emit_summary_line` (single source of truth).
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import logging
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from xbrain.models import (
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    SupportedLanguage,
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
# downloader writes one of these four (`.jpg` is mapped twice — once for
# `.jpg`, once for `.jpeg` — so the cardinality of distinct media types
# is three: image/jpeg, image/png, image/webp). See
# `xbrain.media._FORMAT_EXTENSIONS` for the producer side.
_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class MessagesClient(Protocol):
    """Minimal structural type for the Anthropic SDK `client.messages` seam.

    Defined locally so the orchestrator does not need to type-ignore an
    untyped `client.messages.create(...)` call. The real
    `anthropic.Anthropic().messages` satisfies this Protocol; the test
    fakes (`tests.conftest.FakeAnthropic`, `_FakeVisionClient`) do too.
    """

    def create(self, **kwargs: object) -> object:
        """Send one Anthropic `messages.create` call; return the raw response."""
        ...


class VisionClient(Protocol):
    """Minimal structural type for the Anthropic SDK client itself.

    Drops the `# type: ignore[attr-defined]` that would otherwise be
    needed on `client.messages.create(...)`. The protocol is local to
    this module because describe is the only consumer; `executors.api`
    still uses a duck-typed client for now (out of scope for this PR).
    """

    messages: MessagesClient


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
    tuple swallowed PLUS batches where the SDK refused or the model
    returned fewer judgments than the batch size. A run with
    `photos_attempted > 0 and photos_described == 0` raises before
    this report leaves the orchestrator.
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
    without re-scanning the store. Bytes are loaded lazily by
    `_load_bytes` — failing to read the file is a per-batch failure,
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


def _is_stale(
    entry: MediaPhotoDescribed,
    *,
    current_version: str,
    current_language: str,
) -> bool:
    """A described entry is stale when its version OR language drifted.

    Two triggers, both no-`--force`:

    1. `description_version != current_version` — bumping
       `[describe].version` in `config.toml` invalidates the corpus
       against a new rubric.
    2. `description_lang != current_language` — switching
       `[paths].output_language` from Spanish to English (or back)
       leaves stale-language prose in place that would otherwise
       drift into the enrich prompt as a mixed-language vault.

    Equality is exact-string for both: there is no ordering relation
    between versions or languages, so a deliberate downgrade also
    triggers re-describe.
    """
    if entry.description_version != current_version:
        return True
    return entry.description_lang != current_language


def _is_eligible(
    entry: object, *, force: bool, current_version: str, current_language: str
) -> bool:
    """Decide whether `describe_all` should attempt this entry on THIS run.

    `MediaPhotoDownloaded` entries are always eligible — they have not
    been described yet by definition. `MediaPhotoDescribed` entries are
    eligible only when `--force` is set or the persisted version/language
    is stale vs the current config. Every other variant
    (`MediaPhotoPending`, `MediaPhotoFailed`, `MediaVideoPending`) is
    out of scope — describing only runs over photos whose bytes are
    already on disk.
    """
    if isinstance(entry, MediaPhotoDownloaded):
        return True
    if isinstance(entry, MediaPhotoDescribed):
        if force:
            return True
        return _is_stale(
            entry,
            current_version=current_version,
            current_language=current_language,
        )
    return False


def _tally_idempotency_skip(
    entry: object,
    *,
    current_version: str,
    current_language: str,
    report: DescribeReport,
) -> None:
    """Bump `photos_skipped_already_described` for same-version+same-language describes.

    Only fresh `MediaPhotoDescribed` entries count as idempotency skips.
    Pending/failed/video entries are silently out of scope for
    `xbrain describe`. Pulled out of the candidate iterator so the loop
    body stays under radon grade C.
    """
    if isinstance(entry, MediaPhotoDescribed) and not _is_stale(
        entry,
        current_version=current_version,
        current_language=current_language,
    ):
        report.photos_skipped_already_described += 1


def _filtered_items(
    items: dict[str, Item],
    items_filter: set[str] | None,
) -> Iterator[tuple[str, Item]]:
    """Yield `(item_id, item)` pairs the run should scan, skipping empty media.

    Pulled out so `_iter_eligible_candidates` stays under radon grade C.
    An `items_filter` of `None` is "every item"; an empty media list
    is silently skipped (no photos to consider).
    """
    for item_id, item in items.items():
        if items_filter is not None and item_id not in items_filter:
            continue
        if not item.media:
            continue
        yield item_id, item


def _iter_eligible_candidates(
    items: dict[str, Item],
    *,
    force: bool,
    limit: int | None,
    items_filter: set[str] | None,
    current_version: str,
    current_language: str,
    report: DescribeReport,
) -> Iterator[_Candidate]:
    """Yield each candidate eligible for description, with all bookkeeping inline.

    Side effects on `report` (mirrors `media._iter_eligible_attempts`):

    - bumps `items_processed` once per item that contributes at least
      one yielded candidate (first yielded candidate of an item — a
      `limit` that truncates mid-item still counts the item, items
      whose every photo was skipped do NOT count);
    - bumps `photos_skipped_already_described` for each
      `MediaPhotoDescribed` entry on the current version+language that
      gets passed over (via `_tally_idempotency_skip`).

    Stops yielding once `limit` is exhausted. Replaces the four-helper
    chain that was here before (`_tally_skipped` + `_iter_item_candidates`
    + `_take_with_limit` + outer `_iter_candidates`).
    """
    remaining = limit
    seen_item_ids: set[str] = set()
    for item_id, item in _filtered_items(items, items_filter):
        for index, entry in enumerate(item.media):
            if remaining is not None and remaining <= 0:
                return
            if not _is_eligible(
                entry,
                force=force,
                current_version=current_version,
                current_language=current_language,
            ):
                _tally_idempotency_skip(
                    entry,
                    current_version=current_version,
                    current_language=current_language,
                    report=report,
                )
                continue
            assert isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed))
            if item_id not in seen_item_ids:
                seen_item_ids.add(item_id)
                report.items_processed += 1
            if remaining is not None:
                remaining -= 1
            yield _Candidate(item_id=item_id, item=item, index=index, entry=entry)


def _media_type(local_path: str) -> str:
    """Map an on-disk path's extension to its Anthropic media-type string.

    The downloader writes one of `.jpg` / `.jpeg` / `.png` / `.webp`
    (see `xbrain.media._FORMAT_EXTENSIONS`). Anything else came from a
    hand edit of `items.json` or a future format we have not registered
    — we emit a `logger.warning` (so the operator can see the wrong
    MIME type was sent) and fall back to `image/jpeg`. Anthropic will
    reject the request if the bytes do not match; that surfaces as a
    per-batch failure rather than a silent total-failure raise.
    """
    suffix = Path(local_path).suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix)
    if media_type is None:
        logger.warning(
            "describe: unknown extension %r for local_path %s; "
            "sending as image/jpeg (Anthropic may reject)",
            suffix,
            local_path,
        )
        return "image/jpeg"
    return media_type


def _load_bytes(media_root: Path, local_path: str) -> bytes:
    """Read the photo bytes from `data/media/<local_path>`.

    Raises `OSError` (a `FileNotFoundError` subclass when the file is
    missing). The orchestrator's `_recoverable_errors` tuple catches it
    so a missing file is a per-batch failure (the operator can re-run
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


def _extract_response_text(response: object) -> str:
    """Pull the JSON-bearing text out of a vision response, stripping fences.

    The Anthropic SDK packs the model's reply in `.content` as a list
    of typed blocks; only `text` blocks carry JSON. Some models wrap
    the JSON in a ```json ... ``` Markdown fence despite the rubric
    explicitly forbidding it — strip a single leading/trailing fence
    pair (with or without a language tag) so the downstream
    `json.loads` does not trip on a Markdown artefact.
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
    tuple catches every shape violation as a per-batch failure. The
    caller logs `response.stop_reason` alongside any `JSONDecodeError`
    so a `max_tokens` truncation surfaces as a diagnosable cause.
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
    language: SupportedLanguage,
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


def _apply_batch_as_decorative_empty(
    *,
    batch: list[_Candidate],
    language: SupportedLanguage,
    version: str,
    described_at: datetime,
    report: DescribeReport,
) -> None:
    """Mark every photo in a refused batch as decorative + empty description.

    The Anthropic SDK returns `stop_reason="refusal"` on safety-policy
    refusals (identifiable faces, NSFW, ...). The rubric already says
    "if undescribable, mark decorative with empty description" — we
    apply that at the SDK level too, so the batch makes progress
    instead of churning forever. Each photo transitions to
    `MediaPhotoDescribed(is_decorative=True, description="")` and
    counts toward `photos_described`, not `photos_failed`.
    """
    for candidate in batch:
        new_entry = MediaPhotoDescribed(
            url=candidate.entry.url,
            local_path=candidate.entry.local_path,
            width=candidate.entry.width,
            height=candidate.entry.height,
            bytes_size=candidate.entry.bytes_size,
            downloaded_at=candidate.entry.downloaded_at,
            is_decorative=True,
            description="",
            description_lang=language,
            description_version=version,
            described_at=described_at,
        )
        candidate.item.media[candidate.index] = new_entry
        report.photos_described += 1


def describe_all(
    items: dict[str, Item],
    media_root: Path,
    *,
    model: str,
    output_language: SupportedLanguage,
    description_version: str,
    force: bool = False,
    limit: int | None = None,
    items_filter: list[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    client: VisionClient | None = None,
    on_progress: Callable[[], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> DescribeReport:
    """Describe every eligible photo across the store; return a structured report.

    Eligibility (without `--force`):

    - `MediaPhotoDownloaded` — always.
    - `MediaPhotoDescribed` whose `description_version` ≠ the configured
      version, OR whose `description_lang` ≠ the configured language.
      The language check guards against mixed-language vaults that
      would otherwise leak Spanish prose into an English run (or
      vice-versa) via the enrich prompt.

    With `--force`:

    - Every `MediaPhotoDownloaded` AND every `MediaPhotoDescribed` is
      re-described. The persisted description is overwritten.

    Out of scope (every run):

    - `MediaPhotoPending`, `MediaPhotoFailed`, `MediaVideoPending` —
      describing only runs over photos with bytes on disk.

    The function mutates `items` in place; the caller is expected to
    wrap each batch transition with a store write (the `on_progress`
    callback fires after every batch). The Ctrl-C-coherent invariant
    lives there: the store is written between batches.

    Raises:
        RuntimeError: when EVERY batch attempted in the run fails. The
            CLI's `_handle_cli_errors` converts this into a clean
            operator message + exit code 1.
    """
    if client is None:
        from anthropic import Anthropic  # lazy: tests inject FakeAnthropic

        # reads ANTHROPIC_API_KEY from the environment; the real `Anthropic`
        # class is structurally compatible with `VisionClient` so the cast
        # is a documentation tool, not a runtime conversion.
        active_client: VisionClient = Anthropic()  # type: ignore[assignment]
    else:
        active_client = client
    clock: Callable[[], datetime] = now if now is not None else _utcnow

    started = time.monotonic()
    report = DescribeReport()
    filter_set = set(items_filter) if items_filter else None
    candidates = list(
        _iter_eligible_candidates(
            items,
            force=force,
            limit=limit,
            items_filter=filter_set,
            current_version=description_version,
            current_language=output_language,
            report=report,
        )
    )
    if not candidates:
        report.elapsed_seconds = time.monotonic() - started
        return report

    system = _system_prompt(output_language)
    recoverable = _recoverable_errors()

    # `itertools.batched` requires Python 3.12 (declared in `pyproject.toml`).
    # It returns tuples; the orchestrator works in lists so it can use `len`
    # and slice freely.
    for batch_tuple in itertools.batched(candidates, batch_size):
        batch = list(batch_tuple)
        _run_one_batch(
            batch=batch,
            client=active_client,
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
    return report


def _run_one_batch(
    *,
    batch: list[_Candidate],
    client: VisionClient,
    model: str,
    system: str,
    output_language: SupportedLanguage,
    description_version: str,
    clock: Callable[[], datetime],
    media_root: Path,
    recoverable: tuple[type[Exception], ...],
    report: DescribeReport,
) -> None:
    """Execute one batch end-to-end: build, call, parse, apply, catch.

    Pulled out of `describe_all` to keep the outer loop's complexity
    under grade C while preserving the per-batch isolation contract.
    Refusal responses (`stop_reason="refusal"`) are converted to a
    decorative-empty transition for each photo in the batch — that
    satisfies the spec's "if undescribable, mark decorative" rule at
    the SDK level so the run makes progress instead of churning.

    Programmer bugs (`AttributeError`, ...) and `KeyboardInterrupt`
    fall outside `recoverable` and propagate.
    """
    report.batches_attempted += 1
    report.photos_attempted += len(batch)
    try:
        content_blocks = _build_user_blocks(batch, media_root)
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": content_blocks}],
        )
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            logger.warning(
                "describe: batch of %d photos refused by SDK; "
                "marking each as decorative with empty description",
                len(batch),
            )
            _apply_batch_as_decorative_empty(
                batch=batch,
                language=output_language,
                version=description_version,
                described_at=clock(),
                report=report,
            )
            return
        try:
            judgments = _parse_batch_response(response, batch_size=len(batch))
        except json.JSONDecodeError as exc:
            # Surface stop_reason so a `max_tokens` truncation is
            # diagnosable from the warning line alone — otherwise the
            # operator gets a generic "Expecting value: line N column M".
            logger.warning(
                "describe: malformed JSON from vision response (stop_reason=%r): %s",
                stop_reason,
                exc,
            )
            raise
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
    # Trailing text directive: tell the model the index range so its JSON
    # list is contractually one-to-one with the input order. Indices are
    # zero-based to match Python and the rubric's example.
    blocks.append(
        {
            "type": "text",
            "text": (
                f"Describe images 0 through {len(batch) - 1}. "
                "Return a JSON list with one entry per image, in the order received."
            ),
        }
    )
    return blocks


def _apply_batch_judgments(
    *,
    batch: list[_Candidate],
    judgments: list[dict],
    language: SupportedLanguage,
    version: str,
    described_at: datetime,
    report: DescribeReport,
) -> None:
    """Apply per-image judgments to the batch, transitioning each entry.

    Pairs each judgment with the candidate at its `index`. The parser
    already enforces the index range and rejects duplicates, so the
    `by_index` dict is built directly without a paranoid re-check.

    A judgment list SHORTER than the batch is a partial-data contract
    violation: the missing candidates are tallied as per-photo failures
    (so they re-enter the candidate pool on the next run) AND the
    batch counts as `batches_failed += 1` even though some judgments
    landed — the batch did not return complete data, so the operator's
    view of "did this API call do its job" is "no, partially".
    """
    by_index = {entry["index"]: entry for entry in judgments}
    batch_had_omission = False
    for position, candidate in enumerate(batch):
        judgment = by_index.get(position)
        if judgment is None:
            batch_had_omission = True
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
    if batch_had_omission:
        # A batch with missing judgments did not return complete data
        # for the API call we made — count it as a batch failure so
        # the "did this run net positive?" check is honest.
        report.batches_failed += 1


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

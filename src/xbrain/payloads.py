"""Persist the raw X payload so `extract` is a re-runnable transformation over data we own.

THE FAILURE THIS REMOVES. `fetch` captured X's GraphQL response in-flight, extracted an
`Item`, and threw the original away. So when a parse bug surfaced months later — we read
`legacy.full_text` (capped at 280 chars) and never read `note_tweet`, which was present in
EVERY payload — the fix was not a re-parse. It was a network round-trip to X: a logged-in
browser session, rate limits, and tweets that may since have been deleted or protected. For
432 items that truncated text is the ONLY evidence there is.

Disk is free. Going back to the source is not.

With the payload on disk, a future parse bug — a field we misread, a field nobody read — is
fixed by re-running the parser offline: zero network, zero rate limits, no dependency on the
tweet still existing. `reextract_from_payloads` shows the diff BEFORE applying it, so a
parse fix is validated against the whole corpus before it touches the store.

WHAT THIS DOES NOT DO. It does not repair the existing items. That data is gone from our
side and needs `xbrain refetch-truncated` (#95). This makes sure we are never here again.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xbrain.extract.graphql import parse_tweets
from xbrain.models import Item

# Substrings that mark a key as credential-bearing. These payloads come from an
# AUTHENTICATED session: a sweep of a real captured response
# (`tests/fixtures/art-OpenWiki.json`, 16,746 bytes) found NO auth material — X carries auth
# in request headers, not response bodies. We scrub anyway. The cost of scrubbing is nothing;
# the cost of being wrong once is a credential on disk, in a directory we now write to on
# every sync.
# WHOLE key names, never substrings. The first version matched substrings, and "auth" ate
# "author" / "author_id" / "authors" / "authorship" — deleting an author block on write,
# silently and permanently, with the original already discarded. Today's tweet subtree
# survived only by luck (it uses `core.user_results`), and luck is precisely what this module
# exists to eliminate: THE PAYLOAD IS KEPT FOR THE PARSER WE HAVE NOT WRITTEN. A substring
# deny-list cannot be reasoned about; an explicit one can.
_SECRET_KEYS = frozenset(
    """
    auth_token authorization access_token refresh_token bearer_token id_token
    csrf_token x_csrf_token ct0 cookie cookies set_cookie session session_id sessionid
    guest_id guest_token secret client_secret password credential credentials api_key apikey
    """.split()
)


def _is_secret_key(key: str) -> bool:
    """True only for an EXACT credential key. A key that merely CONTAINS one of these
    substrings (`author`, `authored_at`, `note_tweet`) is item data and must survive."""
    return key.lower() in _SECRET_KEYS


def scrub(obj: Any) -> Any:
    """Recursively drop every credential-bearing key, keeping all item data.

    Applied at the SEAM (inside `save_payload`), never left to the caller — a caller that
    forgets is exactly how a token reaches disk.
    """
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if not _is_secret_key(k)}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def payload_path(payload_dir: Path, item_id: str) -> Path:
    """One gzipped file per item, sharded by the id's last two chars.

    Per-item files, not an append-only log: our access pattern is "re-parse item X" and
    "re-parse everything", both of which a log makes O(n) and needs compaction for. A
    per-item file is idempotent on re-sync (the same tweet overwrites itself), trivially
    lookup-able, and lets a single item be repaired in isolation. The shard keeps any one
    directory from holding 100k entries.
    """
    return payload_dir / item_id[-2:] / f"{item_id}.json.gz"


def save_payload(payload_dir: Path, item_id: str, subtree: dict) -> None:
    """Persist one tweet's FULL result subtree, scrubbed and gzipped.

    We store the tweet's whole `tweet_results.result` subtree — legacy, note_tweet, the user
    object, entities, extended_entities, quoted_status_result, card — and drop only the
    timeline envelope (cursors, pagination, instructions), which carries no item data and
    would otherwise be duplicated ~20× per response. Keeping too little is how we got here,
    so the bias is towards keeping everything that could ever describe the item.
    """
    path = payload_path(payload_dir, item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(scrub(subtree), ensure_ascii=False).encode("utf-8")
    # Atomic: truncate-and-write in a tight loop over thousands of tweets means one Ctrl-C
    # leaves a truncated gzip that `load_payload` cannot read and `reextract` dies on. A
    # reader must see the old bytes or the new bytes, never half of either.
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(body))
    os.replace(tmp, path)


def load_payload(payload_dir: Path, item_id: str) -> dict | None:
    """The stored subtree; None when absent; raises `CorruptPayload` when unreadable.

    A truncated gzip (an interrupted write) EXISTS, so it is not `missing` and is never
    skipped — unhandled, it killed the whole command and left the operator bisecting by hand.
    """
    path = payload_path(payload_dir, item_id)
    if not path.exists():
        return None
    try:
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorruptPayload(f"unreadable payload for {item_id}: {exc}") from exc


class CorruptPayload(RuntimeError):
    """A stored payload exists but cannot be read (a truncated write, a damaged file)."""


def stored_ids(payload_dir: Path) -> set[str]:
    """Every item id we can re-extract offline."""
    if not payload_dir.exists():
        return set()
    return {path.name.removesuffix(".json.gz") for path in payload_dir.rglob("*.json.gz")}


def payload_stats(payload_dir: Path) -> dict[str, int]:
    """Measure what is ACTUALLY on disk: count, raw bytes, gzipped bytes, mean per item.

    The disk figures in the original PR body were the gzip of `tests/fixtures/art-OpenWiki
    .json` — an X *Article* body, which contains no tweets at all. A tweet's real subtree
    (`note_tweet` + `quoted_status_result` + `card` + `extended_entities` + a full user
    object) is a different thing entirely, and its size was never measured. Nor was the
    secrets sweep, which ran over that same non-authenticated file — the one claim that had
    to be measured on the real thing.

    Both were presented as measured. They were not. This function is how they get measured:
    run one `xbrain sync`, then `xbrain payload-stats`. A number I have not taken is worth
    more than one I have taken from the wrong file.
    """
    raw = gzipped = 0
    count = 0
    for path in payload_dir.rglob("*.json.gz") if payload_dir.exists() else []:
        blob = path.read_bytes()
        gzipped += len(blob)
        raw += len(gzip.decompress(blob))
        count += 1
    return {
        "count": count,
        "raw_bytes": raw,
        "gzipped_bytes": gzipped,
        "mean_gzipped_bytes": gzipped // count if count else 0,
    }


@dataclass
class ReextractReport:
    """What a re-parse WOULD change (dry run) or DID change (`--apply`)."""

    total: int = 0
    covered: int = 0
    changed: list[tuple[str, str, Any, Any]] = field(default_factory=list)
    # A payload that is PRESENT but unreadable or unparseable. It is neither coverage nor
    # `missing`, and reporting it as either is the failure this whole module exists to
    # prevent: "cannot be re-extracted" must never look like "re-extracted cleanly".
    unparseable: list[str] = field(default_factory=list)
    # Items with no stored payload. Surfaced explicitly because "no payload" must never be
    # indistinguishable from "re-extracted cleanly": one means fine, the other means
    # UNCHECKABLE, and silence would read as the first.
    missing: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.covered}/{self.total} items re-extracted from a stored payload; "
            f"{len(self.changed)} field(s) would change; "
            f"{len(self.missing)} item(s) have no payload; "
            f"{len(self.unparseable)} item(s) could not be parsed"
        )


# The fields a re-parse may legitimately correct.
#
# `media` is NOT among them, and this is CRITICAL. The store holds ENRICHED media — photos
# with a vision-LLM description, videos with a downloaded `local_path` — while a fresh parse
# emits PENDING states. Overwriting would destroy the description, which is an EVIDENCE
# SURFACE for both the generator and the judge: the summary written from it survives
# verbatim, nothing bumps `fetched_at`, so it is never re-enriched, and the next `verify`
# hands the judge a source with no description and a summary asserting the slide's contents.
# Unsupported → FAIL. The module built to make the corpus repairable would MANUFACTURE
# defects in it, one per described photo. Adding genuinely-new media is `refresh-media`'s
# job (`refresh._rebuild_media`), which preserves enriched states by design.
#
# `captured_at` is when WE saw the tweet, so a re-parse must never touch it; the id joins.
_REPARSED_FIELDS = ("text", "links", "quoted_id", "thread", "author")


def _diff_item(old: Item, new: Item) -> list[tuple[str, Any, Any]]:
    """Every re-parsed field where the stored item and a fresh parse disagree."""
    return [
        (name, getattr(old, name), getattr(new, name))
        for name in _REPARSED_FIELDS
        if getattr(old, name) != getattr(new, name)
    ]


def reextract_from_payloads(
    store: dict[str, Item], payload_dir: Path, *, apply: bool = False
) -> ReextractReport:
    """Re-run the parser over every stored payload and report what changes.

    Dry by default: this is the instrument that lets a parse fix be validated against the
    whole corpus BEFORE it touches the store. Had it existed, `note_tweet` would have been
    an afternoon's diff instead of a re-fetch we still cannot fully perform.
    """
    report = ReextractReport(total=len(store))
    for item_id, item in store.items():
        parsed = _reparse(payload_dir, item, report)
        if parsed is None:
            continue
        report.covered += 1
        changes = _diff_item(item, parsed)
        report.changed += [(item_id, name, old, new) for name, old, new in changes]
        if apply:
            _apply_changes(item, changes)
    return report


def _reparse(payload_dir: Path, item: Item, report: ReextractReport) -> Item | None:
    """Re-parse one item's stored payload, bucketing every way it can fail to yield one."""
    try:
        subtree = load_payload(payload_dir, item.id)
    except CorruptPayload:
        report.unparseable.append(item.id)
        return None
    if subtree is None:
        report.missing.append(item.id)
        return None
    parsed = parse_tweets({"tweet_results": {"result": subtree}}, item.source)
    if not parsed:
        # Present, readable, and the parser makes nothing of it — X schema drift, or a
        # payload an earlier scrub damaged. NOT coverage. Counting it as clean was the
        # missing-PARSE case walking straight through the missing-FILE guard.
        report.unparseable.append(item.id)
        return None
    return parsed[0]


def _apply_changes(item: Item, changes: list[tuple[str, Any, Any]]) -> None:
    """Write the re-parsed fields, and INVALIDATE anything derived from what changed.

    Fixing the text (#95: 432 items truncated mid-word) exists to regenerate the summaries
    written from half a sentence. Writing the corrected text and marking nothing leaves
    `generate` rendering the FULL tweet beside a summary built from the TRUNCATED one —
    silently, and looking perfectly fine.
    """
    for name, _old, new in changes:
        setattr(item, name, new)
    if any(name == "text" for name, _o, _n in changes):
        item.enriched = None

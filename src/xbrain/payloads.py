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
_SECRET_MARKERS = (
    "token",
    "cookie",
    "session",
    "auth",
    "bearer",
    "csrf",
    "secret",
    "password",
    "credential",
    "guest_id",
    "api_key",
)

# The extractor's own field. It must survive scrubbing: `note_tweet` contains "tweet", not a
# secret marker — but be explicit rather than lucky.
_KEEP = frozenset({"note_tweet", "note_tweet_results"})


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return key not in _KEEP and any(marker in lowered for marker in _SECRET_MARKERS)


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
    path.write_bytes(gzip.compress(body))


def load_payload(payload_dir: Path, item_id: str) -> dict | None:
    """The stored subtree, or None when this item predates payload persistence."""
    path = payload_path(payload_dir, item_id)
    if not path.exists():
        return None
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def stored_ids(payload_dir: Path) -> set[str]:
    """Every item id we can re-extract offline."""
    if not payload_dir.exists():
        return set()
    return {path.name.removesuffix(".json.gz") for path in payload_dir.rglob("*.json.gz")}


@dataclass
class ReextractReport:
    """What a re-parse WOULD change (dry run) or DID change (`--apply`)."""

    total: int = 0
    covered: int = 0
    changed: list[tuple[str, str, Any, Any]] = field(default_factory=list)
    # Items with no stored payload. Surfaced explicitly because "no payload" must never be
    # indistinguishable from "re-extracted cleanly": one means fine, the other means
    # UNCHECKABLE, and silence would read as the first.
    missing: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.covered}/{self.total} items have a stored payload; "
            f"{len(self.changed)} field(s) would change; "
            f"{len(self.missing)} item(s) cannot be re-extracted (no payload)"
        )


# The fields a re-parse may legitimately correct. `captured_at` is when WE saw the tweet, so
# a re-parse must never touch it, and the id is the join key.
_REPARSED_FIELDS = ("text", "links", "media", "quoted_id", "thread", "author")


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
        subtree = load_payload(payload_dir, item_id)
        if subtree is None:
            report.missing.append(item_id)
            continue
        report.covered += 1
        parsed = parse_tweets({"tweet_results": {"result": subtree}}, item.source)
        if not parsed:
            continue
        for name, old, new in _diff_item(item, parsed[0]):
            report.changed.append((item_id, name, old, new))
            if apply:
                setattr(item, name, new)
    return report

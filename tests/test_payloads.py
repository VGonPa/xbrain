# tests/test_payloads.py
"""Persisting the raw X payload so `extract` becomes re-runnable offline.

We threw the payload away. `note_tweet` was in EVERY one of them — we simply never read
the field — and by the time we noticed, repairing 432 items' only evidence needed a
network round-trip to X, through a logged-in browser, against tweets that may since have
been deleted. Disk is free; going back to the source is not.
"""

import gzip
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest

from xbrain.models import Author, Item
from xbrain.payloads import (
    load_payload,
    reextract_from_payloads,
    save_payload,
    scrub,
    stored_ids,
)


def _item(item_id: str = "1", text: str = "old text") -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text=text,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )


def _subtree(item_id: str = "1", *, full_text: str = "old text", note: str | None = None) -> dict:
    tweet = {
        "__typename": "Tweet",
        "rest_id": item_id,
        "core": {"user_results": {"result": {"legacy": {"screen_name": "a", "name": "A"}}}},
        "legacy": {"full_text": full_text, "created_at": "Wed Jan 01 10:00:00 +0000 2025"},
    }
    if note is not None:
        tweet["note_tweet"] = {"note_tweet_results": {"result": {"text": note}}}
    return tweet


# ---------------------------------------------------------------- round-trip


def test_payload_round_trips_byte_for_byte(tmp_path: Path):
    """The stored payload must be the SAME OBJECT back — not a lossy summary of it.
    Keeping too little is exactly how we got here."""
    subtree = _subtree("77", note="the long-form body nobody read")
    save_payload(tmp_path, "77", subtree)
    assert load_payload(tmp_path, "77") == subtree


def test_payloads_are_gzipped_on_disk(tmp_path: Path):
    """JSON compresses ~4x. On a real captured response (16,746 bytes) gzip gives 3,781 —
    23%. At 100k items that is 380 MB instead of 1.7 GB, so the compression is what keeps
    'just keep everything' an honest answer rather than a hand-wave."""
    save_payload(tmp_path, "77", _subtree("77"))
    path = next(tmp_path.rglob("*.json.gz"))
    assert gzip.decompress(path.read_bytes())  # it really is gzip
    with pytest.raises(UnicodeDecodeError):
        path.read_text(encoding="utf-8")  # ...and not plain JSON on disk


def test_load_payload_is_none_when_absent(tmp_path: Path):
    assert load_payload(tmp_path, "nope") is None


def test_stored_ids_lists_what_we_can_re_extract(tmp_path: Path):
    save_payload(tmp_path, "1", _subtree("1"))
    save_payload(tmp_path, "2", _subtree("2"))
    assert stored_ids(tmp_path) == {"1", "2"}


# ---------------------------------------------------------------- secrets


def test_scrub_removes_anything_that_smells_like_a_credential():
    """These payloads come from an AUTHENTICATED session. A real captured response carries
    no auth material (auth rides in request headers, not response bodies — swept and
    confirmed on `tests/fixtures/art-OpenWiki.json`), but we scrub anyway: the cost of
    being wrong once is a credential on disk, and the cost of scrubbing is nothing."""
    dirty = {
        "rest_id": "1",
        "auth_token": "SECRET",
        "legacy": {"full_text": "keep me", "session_id": "SECRET", "guest_id": "SECRET"},
        "nested": [{"csrf_token": "SECRET", "keep": "yes"}],
    }
    clean = scrub(dirty)
    flat = json.dumps(clean)
    assert "SECRET" not in flat
    assert clean["rest_id"] == "1"
    assert clean["legacy"]["full_text"] == "keep me"  # item data survives untouched
    assert clean["nested"][0]["keep"] == "yes"


def test_saved_payloads_are_scrubbed_on_the_way_to_disk(tmp_path: Path):
    """Scrubbing at the seam, not at the caller — a caller that forgets is how a token
    reaches disk."""
    save_payload(tmp_path, "1", {"rest_id": "1", "auth_token": "SECRET"})
    assert "SECRET" not in json.dumps(load_payload(tmp_path, "1"))


# ---------------------------------------------------------------- re-extract


def test_reextract_reports_the_diff_without_touching_the_store(tmp_path: Path):
    """THE POINT OF THE WHOLE PR: a parse fix is validated BEFORE it is applied.

    The stored payload holds the truth; the store holds what an older parser made of it. A
    re-parse shows exactly what would change — and changes nothing. Had this existed, the
    `note_tweet` bug (#95) would have been an afternoon's diff over data we already owned,
    instead of a re-fetch we can no longer fully perform.
    """
    store = {"1": _item("1", text="what the OLD parser stored")}
    save_payload(tmp_path, "1", _subtree("1", full_text="THE WHOLE POST."))

    report = reextract_from_payloads(store, tmp_path, apply=False)

    assert report.changed == [("1", "text", "what the OLD parser stored", "THE WHOLE POST.")]
    assert store["1"].text == "what the OLD parser stored", "a dry run must not mutate the store"


def test_reextract_applies_the_fix_when_asked(tmp_path: Path):
    store = {"1": _item("1", text="what the OLD parser stored")}
    save_payload(tmp_path, "1", _subtree("1", full_text="THE WHOLE POST."))

    report = reextract_from_payloads(store, tmp_path, apply=True)

    assert store["1"].text == "THE WHOLE POST."
    assert report.changed


def test_the_stored_payload_carries_the_field_the_old_parser_never_read(tmp_path: Path):
    """The `note_tweet` body — the field that cost us 432 items — survives the round-trip
    intact, so a parser taught to read it (#95) can recover the full post OFFLINE."""
    save_payload(tmp_path, "1", _subtree("1", full_text="cut off", note="the whole thing"))
    stored = load_payload(tmp_path, "1")
    assert stored["note_tweet"]["note_tweet_results"]["result"]["text"] == "the whole thing"


def test_reextract_is_a_no_op_when_the_parser_agrees_with_the_store(tmp_path: Path):
    store = {"1": _item("1", text="same text")}
    save_payload(tmp_path, "1", _subtree("1", full_text="same text"))
    report = reextract_from_payloads(store, tmp_path, apply=True)
    assert report.changed == []
    assert store["1"].text == "same text"


def test_items_without_a_payload_are_VISIBLE_not_silently_clean(tmp_path: Path):
    """Backwards compatibility's dangerous edge: an item with no stored payload must never
    be indistinguishable from one that re-extracted cleanly. Silence would read as 'this
    item is fine' when the truth is 'this item cannot be checked at all'."""
    store = {"1": _item("1"), "2": _item("2")}
    save_payload(tmp_path, "1", _subtree("1"))

    report = reextract_from_payloads(store, tmp_path, apply=False)

    assert report.missing == ["2"]
    assert report.covered == 1
    assert report.total == 2


def test_recapture_history_persists_payloads(monkeypatch, tmp_path: Path):
    """P8. `_recapture_history` is the shared harness behind EVERY backfill
    (`refresh-media`, `refresh-quoted`), and it called `extract_source` without a
    `payload_dir` — so a full-history scroll walked past every item and stored nothing.

    That scroll is the single cheapest payload backfill available for the existing 2,168
    items: it has NO skip-known, so it re-sees the whole timeline. Wasting it would mean the
    only way to get payloads for the existing corpus is another full scroll later.
    """
    from xbrain import cli

    seen: dict[str, Path | None] = {}

    def fake_extract_source(context, src, url, known_ids, *args, **kwargs):
        seen["payload_dir"] = kwargs.get("payload_dir")
        return []

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "extract_source", fake_extract_source)
    monkeypatch.setattr(cli, "x_context", lambda *a, **k: _Ctx())

    cfg = SimpleNamespace(
        x_handle="v", storage_state_path=tmp_path / "s.json", payload_dir=tmp_path / "payloads"
    )
    cli._recapture_history(cfg, "bookmarks", label="refresh-media")

    assert seen["payload_dir"] == tmp_path / "payloads", (
        "the full-history scroll must persist payloads — it is the cheapest backfill we have"
    )

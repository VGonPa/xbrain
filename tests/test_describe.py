"""Tests for `xbrain.describe` — the vision-describe orchestrator.

The Anthropic client is faked via `tests.conftest.FakeAnthropic`; no
real API calls. Photo bytes are written to a tmp `data/media/` tree so
the orchestrator can read them through its normal `_load_bytes` path.

Coverage targets every contract the spec calls out:
- variant transitions (Downloaded → Described; Described stale → Described)
- idempotency (no-op re-runs skip already-described entries)
- batching (5-at-a-time by default; partial batch at the end)
- per-batch error isolation (one failing batch does not abort the run)
- total-failure raise (every batch errored)
- refusal handling (decorative + empty description, never crashes)
- language plumbing (the rubric ships with `{language}` substituted)
- programmer-bug propagation (`AttributeError` is NOT swallowed)
- Ctrl-C propagation (`KeyboardInterrupt` is NOT swallowed)
- stale-version logic + `--force` semantics
- summary line on stderr
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.describe import (
    DescribeReport,
    _is_eligible,
    _parse_batch_response,
    _validate_judgment_entry,
    describe_all,
    emit_summary_line,
)
from xbrain.models import (
    Author,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
    MediaVideoPending,
)

from tests.conftest import FakeAnthropic, FakeBlock, FakeResponse

# --------------------------------------------------------------------- fixtures + helpers


def _photo_bytes_jpg() -> bytes:
    """A minimal 4x3 JPEG so the loader's `read_bytes` returns something realistic.

    We do NOT exercise Pillow in describe-tests — the orchestrator only
    reads the file and base64-encodes it. Any non-empty bytes work; we
    use real JPEG bytes so the media-type mapping has something
    plausible to round-trip.
    """
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (4, 3), color=(1, 2, 3)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_photo(media_root: Path, item_id: str, index: int, ext: str = ".jpg") -> str:
    """Write a fake photo to `data/media/<item_id>/<index><ext>`; return the rel path."""
    rel = f"{item_id}/{index}{ext}"
    dst = media_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(_photo_bytes_jpg())
    return rel


def _downloaded(
    *,
    item_id: str = "1",
    index: int = 0,
    url: str | None = None,
    media_root: Path | None = None,
) -> MediaPhotoDownloaded:
    """Build a `MediaPhotoDownloaded` whose bytes are on disk (when `media_root`)."""
    rel = f"{item_id}/{index}.jpg"
    if media_root is not None:
        _write_photo(media_root, item_id, index)
    return MediaPhotoDownloaded(
        url=url or f"https://pbs.twimg.com/media/{item_id}-{index}.jpg",
        local_path=rel,
        width=4,
        height=3,
        bytes_size=512,
        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )


def _described(
    *,
    item_id: str = "1",
    index: int = 0,
    version: str = "v1",
    media_root: Path | None = None,
) -> MediaPhotoDescribed:
    """Build a `MediaPhotoDescribed` with optional on-disk bytes."""
    rel = f"{item_id}/{index}.jpg"
    if media_root is not None:
        _write_photo(media_root, item_id, index)
    return MediaPhotoDescribed(
        url=f"https://pbs.twimg.com/media/{item_id}-{index}.jpg",
        local_path=rel,
        width=4,
        height=3,
        bytes_size=512,
        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        is_decorative=False,
        description="a previously-described image",
        description_lang="English",
        description_version=version,
        described_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
    )


def _item(item_id: str, media: list) -> Item:
    """Build an `Item` populated with the given media entries."""
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="text",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=media,
    )


def _judgment(index: int, *, decorative: bool = False, description: str = "ok") -> dict:
    """Build one per-image judgment dict matching the wire contract."""
    return {
        "index": index,
        "is_decorative": decorative,
        "description": "" if decorative else description,
    }


def _payload(judgments: list[dict]) -> list[dict]:
    """Wrap a list of judgments as the JSON list the rubric expects.

    `FakeAnthropic` JSON-encodes the payload as `json.dumps(payload)` —
    a list payload survives that path unchanged because `dumps` accepts
    any JSON-serialisable value, not just dicts. The orchestrator's
    parser pulls the text from `.content[0].text` and parses it as a
    JSON list.
    """
    return judgments  # type: ignore[return-value]


class _FakeListResponse:
    """A `FakeResponse`-shaped object whose `.content[0].text` is a JSON list.

    `FakeAnthropic` wraps payloads in `FakeResponse(payload)` which calls
    `json.dumps(payload)` — that path expects a dict. For describe we
    need to ship a list payload, so we build a parallel response
    type that mirrors the shape `_parse_batch_response` consumes.

    `stop_reason` defaults to `"end_turn"` (the SDK's normal terminus);
    pass `stop_reason="refusal"` to simulate a safety-policy refusal,
    or any other string to simulate a max-tokens truncation etc.
    """

    def __init__(self, judgments: list[dict], *, stop_reason: str = "end_turn"):
        import json

        self.content = [type(FakeBlock(payload={}))(payload={})]  # placeholder block
        # Overwrite the text on the block with the JSON list serialisation.
        self.content[0].text = json.dumps(judgments)
        self.stop_reason = stop_reason


class _FakeRefusalResponse:
    """A response object simulating Anthropic SDK refusal.

    `stop_reason="refusal"` is the SDK's signal that the model declined
    to answer (identifiable face, NSFW, ...). The orchestrator must
    detect this and convert the batch to decorative+empty rather than
    fail it. `content` is intentionally empty — a real refusal response
    can carry no text blocks.
    """

    def __init__(self):
        self.content: list = []
        self.stop_reason = "refusal"


class _FakeTruncatedResponse:
    """A response with `stop_reason="max_tokens"` and malformed JSON text.

    Simulates the failure mode where the model hit the token cap mid-emit
    and the SDK returns a truncated, unparseable JSON fragment. The
    orchestrator must surface `stop_reason` in the warning log line so
    the operator can diagnose the cause from logs alone.
    """

    def __init__(self, truncated_text: str = '[{"index": 0, "is_dec'):
        block = type(FakeBlock(payload={}))(payload={})
        block.text = truncated_text
        self.content = [block]
        self.stop_reason = "max_tokens"


class _FakeMessagesList:
    """A fake `client.messages` that pops `_FakeListResponse` per call.

    Mirrors `tests.conftest.FakeMessages` but uses `_FakeListResponse`
    so the payload is a JSON list. Exception instances are raised
    rather than wrapped (same convention). A payload that is already a
    response object (e.g. `_FakeRefusalResponse`) is returned as-is so
    tests can simulate stop-reason variants without a wrapper.
    """

    def __init__(self, payloads: list):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        # Pass pre-built response objects through unchanged so callers
        # can ship refusal / truncation / custom-stop-reason responses
        # without a `_FakeListResponse` wrapper around a JSON list.
        if hasattr(payload, "content") and hasattr(payload, "stop_reason"):
            return payload
        return _FakeListResponse(payload)


class _FakeVisionClient:
    """Drop-in fake for `anthropic.Anthropic` over JSON-list responses."""

    def __init__(self, payloads: list):
        self.messages = _FakeMessagesList(payloads)


# --------------------------------------------------------------------- eligibility


def test_eligible_downloaded_always_eligible():
    """Downloaded entries are always eligible regardless of force/version/language."""
    entry = _downloaded()
    assert (
        _is_eligible(entry, force=False, current_version="v1", current_language="English") is True
    )
    assert _is_eligible(entry, force=True, current_version="v1", current_language="English") is True


def test_eligible_described_current_version_only_with_force():
    """A described entry on the current version+language is skipped without --force."""
    entry = _described(version="v1")
    assert (
        _is_eligible(entry, force=False, current_version="v1", current_language="English") is False
    )
    assert _is_eligible(entry, force=True, current_version="v1", current_language="English") is True


def test_eligible_described_stale_version_is_eligible():
    """A described entry on a stale version is eligible without --force."""
    entry = _described(version="v1")
    assert (
        _is_eligible(entry, force=False, current_version="v2", current_language="English") is True
    )


def test_eligible_described_stale_language_is_eligible():
    """A described entry whose language drifted vs the current config is eligible.

    Guards against the mixed-language vault regression: switching
    `output_language` from English → Spanish (or vice-versa) must mark
    previously-described entries stale on the next run so the enrich
    prompt does not splice the wrong language into the new vault.
    """
    entry = _described(version="v1")  # described_lang="English" by default
    assert (
        _is_eligible(entry, force=False, current_version="v1", current_language="Spanish") is True
    )


def test_eligible_pending_failed_video_are_never_eligible():
    """Pending / failed / video entries are out of scope for describe."""
    pending = MediaPhotoPending(url="https://pbs.twimg.com/media/X.jpg")
    failed = MediaPhotoFailed(
        url="https://pbs.twimg.com/media/X.jpg",
        failure_reason="http_4xx",
        error="HTTP 404",
        attempts=1,
        last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )
    video = MediaVideoPending(url="https://video.twimg.com/x.mp4")
    for entry in (pending, failed, video):
        assert (
            _is_eligible(entry, force=False, current_version="v1", current_language="English")
            is False
        )
        assert (
            _is_eligible(entry, force=True, current_version="v1", current_language="English")
            is False
        )


# --------------------------------------------------------------------- parser


def test_parse_batch_response_accepts_valid_list():
    """A clean JSON list with the right keys round-trips."""
    response = _FakeListResponse([_judgment(0), _judgment(1, decorative=True)])
    out = _parse_batch_response(response, batch_size=2)
    assert [e["index"] for e in out] == [0, 1]


def test_parse_batch_response_strips_markdown_fence():
    """Some models wrap JSON in ```json ... ``` despite the rubric."""
    import json

    class _Fenced:
        content = [type("B", (), {"type": "text", "text": ""})()]

    fenced_text = "```json\n" + json.dumps([_judgment(0)]) + "\n```"
    _Fenced.content[0].text = fenced_text
    out = _parse_batch_response(_Fenced(), batch_size=1)
    assert out[0]["index"] == 0


def test_parse_batch_response_rejects_non_list_root():
    """A JSON object (not list) at the root violates the wire contract."""
    import json

    class _ObjResponse:
        content = [type("B", (), {"type": "text", "text": json.dumps({"oops": True})})()]

    with pytest.raises(ValueError, match="not a JSON list"):
        _parse_batch_response(_ObjResponse(), batch_size=1)


def test_parse_batch_response_rejects_missing_text_block():
    """A response with no text block at all cannot be parsed."""

    class _Empty:
        content: list = []

    with pytest.raises(ValueError, match="no text block"):
        _parse_batch_response(_Empty(), batch_size=1)


def test_parse_batch_response_rejects_duplicate_index():
    """Duplicate indices would silently overwrite a transition — reject."""
    response = _FakeListResponse([_judgment(0), _judgment(0)])
    with pytest.raises(ValueError, match="duplicate index"):
        _parse_batch_response(response, batch_size=2)


def test_validate_judgment_entry_rejects_bool_as_int_index():
    """`bool` is a subclass of `int` in Python — exclude it explicitly."""
    with pytest.raises(ValueError, match="must be int"):
        _validate_judgment_entry(
            {"index": True, "is_decorative": False, "description": "x"},
            batch_size=2,
        )


def test_validate_judgment_entry_rejects_out_of_range_index():
    """An index outside `[0, batch_size)` is a contract violation."""
    with pytest.raises(ValueError, match="out of batch range"):
        _validate_judgment_entry(
            {"index": 5, "is_decorative": False, "description": "x"},
            batch_size=2,
        )


def test_validate_judgment_entry_rejects_non_bool_decorative():
    """`is_decorative` must be a real bool, not a truthy string."""
    with pytest.raises(ValueError, match="is_decorative"):
        _validate_judgment_entry(
            {"index": 0, "is_decorative": "yes", "description": "x"},
            batch_size=1,
        )


# --------------------------------------------------------------------- orchestrator


def test_describe_all_transitions_downloaded_to_described(tmp_path: Path):
    """The happy path: one downloaded photo becomes one described entry."""
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0, description="A diagram.")]])
    report = describe_all(
        store,
        media_root,
        model="claude-sonnet-4-6",
        output_language="English",
        description_version="v1",
        client=client,
    )
    assert isinstance(item.media[0], MediaPhotoDescribed)
    assert item.media[0].description == "A diagram."
    assert item.media[0].description_lang == "English"
    assert item.media[0].description_version == "v1"
    assert item.media[0].is_decorative is False
    assert report.photos_described == 1
    assert report.photos_failed == 0
    assert report.batches_attempted == 1


def test_describe_all_is_noop_for_already_described_current_version(tmp_path: Path):
    """Re-running over a described entry at the current version is a no-op."""
    media_root = tmp_path / "media"
    entry = _described(item_id="1", index=0, version="v1", media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    # No payloads queued: a no-op run must not call the API at all.
    client = _FakeVisionClient([])
    report = describe_all(
        store,
        media_root,
        model="claude-sonnet-4-6",
        output_language="English",
        description_version="v1",
        client=client,
    )
    assert isinstance(item.media[0], MediaPhotoDescribed)
    assert item.media[0].description == "a previously-described image"
    assert report.photos_described == 0
    assert report.photos_skipped_already_described == 1
    assert report.batches_attempted == 0
    assert client.messages.calls == []


def test_describe_all_force_redescribes_current_version(tmp_path: Path):
    """`--force` re-describes everything, current-version included."""
    media_root = tmp_path / "media"
    entry = _described(item_id="1", index=0, version="v1", media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0, description="updated description")]])
    describe_all(
        store,
        media_root,
        model="claude-sonnet-4-6",
        output_language="English",
        description_version="v1",
        force=True,
        client=client,
    )
    assert isinstance(item.media[0], MediaPhotoDescribed)
    assert item.media[0].description == "updated description"


def test_describe_all_redescribes_stale_version_without_force(tmp_path: Path):
    """Bumping `description_version` invalidates stale entries automatically."""
    media_root = tmp_path / "media"
    entry = _described(item_id="1", index=0, version="v1", media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0, description="v2 description")]])
    describe_all(
        store,
        media_root,
        model="claude-sonnet-4-6",
        output_language="English",
        description_version="v2",
        client=client,
    )
    assert isinstance(item.media[0], MediaPhotoDescribed)
    assert item.media[0].description_version == "v2"
    assert item.media[0].description == "v2 description"


def test_describe_all_batches_five_images_per_call_by_default(tmp_path: Path):
    """Default batch size is 5 — 6 photos must produce 2 API calls (5 + 1)."""
    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(6)],
    )
    store = {"1": item}
    first_batch = [_judgment(i, description=f"img {i}") for i in range(5)]
    second_batch = [_judgment(0, description="img 5")]
    client = _FakeVisionClient([first_batch, second_batch])
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        client=client,
    )
    assert len(client.messages.calls) == 2
    assert all(isinstance(m, MediaPhotoDescribed) for m in item.media)


def test_describe_all_respects_custom_batch_size(tmp_path: Path):
    """A non-default `batch_size` must control how many images per call."""
    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(4)],
    )
    store = {"1": item}
    client = _FakeVisionClient(
        [
            [_judgment(0, description="x"), _judgment(1, description="x")],
            [_judgment(0, description="x"), _judgment(1, description="x")],
        ]
    )
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        batch_size=2,
        client=client,
    )
    assert len(client.messages.calls) == 2


def test_describe_all_decorative_judgment_writes_empty_description(tmp_path: Path):
    """A decorative classification produces an empty description on the variant.

    The orchestrator enforces the rubric contract even if the model
    returned non-empty text — `is_decorative` implies empty description
    at the boundary.
    """
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    # The model put text in description despite is_decorative=True; the
    # orchestrator must overwrite it with the empty string.
    client = _FakeVisionClient(
        [[{"index": 0, "is_decorative": True, "description": "shouldnt persist"}]]
    )
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        client=client,
    )
    assert isinstance(item.media[0], MediaPhotoDescribed)
    assert item.media[0].is_decorative is True
    assert item.media[0].description == ""


def test_describe_all_isolates_a_batch_error(tmp_path: Path, capsys):
    """One failing batch must NOT abort the rest of the run."""
    from anthropic import APIError

    media_root = tmp_path / "media"
    item_a = _item("a", [_downloaded(item_id="a", index=0, media_root=media_root)])
    item_b = _item("b", [_downloaded(item_id="b", index=0, media_root=media_root)])
    store = {"a": item_a, "b": item_b}
    client = _FakeVisionClient(
        [
            APIError("503 service unavailable", request=None, body=None),
            [_judgment(0, description="ok")],
        ]
    )
    report = describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        batch_size=1,
        client=client,
    )
    # Item a stayed Downloaded; item b transitioned.
    assert isinstance(item_a.media[0], MediaPhotoDownloaded)
    assert isinstance(item_b.media[0], MediaPhotoDescribed)
    assert report.photos_described == 1
    assert report.photos_failed == 1
    assert report.batches_failed == 1
    # The orchestrator does not emit SUMMARY — the CLI does. Verify the
    # orchestrator stays silent so the CLI is the single source of truth.
    err = capsys.readouterr().err
    assert "SUMMARY:" not in err


def test_describe_all_raises_when_every_batch_fails(tmp_path: Path):
    """A total-failure run (every batch errored) must raise RuntimeError."""
    from anthropic import APIError

    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(2)],
    )
    store = {"1": item}
    client = _FakeVisionClient(
        [
            APIError("401 unauthorized", request=None, body=None),
            APIError("401 unauthorized", request=None, body=None),
        ]
    )
    with pytest.raises(RuntimeError, match=r"All 2 describe batches failed"):
        describe_all(
            store,
            media_root,
            model="m",
            output_language="English",
            description_version="v1",
            batch_size=1,
            client=client,
        )


def test_describe_all_substitutes_output_language_in_system_prompt(tmp_path: Path):
    """The rubric's `{language}` placeholder must be substituted before sending."""
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0)]])
    describe_all(
        store,
        media_root,
        model="m",
        output_language="Spanish",
        description_version="v1",
        client=client,
    )
    system = client.messages.calls[0]["system"]
    assert "{language}" not in system
    assert "Spanish" in system


def test_describe_all_records_language_on_described_variant(tmp_path: Path):
    """The persisted `description_lang` must reflect the call-time language."""
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0)]])
    describe_all(
        store,
        media_root,
        model="m",
        output_language="Spanish",
        description_version="v1",
        client=client,
    )
    assert isinstance(item.media[0], MediaPhotoDescribed)
    assert item.media[0].description_lang == "Spanish"


def test_describe_all_propagates_programmer_bugs(tmp_path: Path):
    """`AttributeError` (programmer bug) must NOT be swallowed — propagate."""
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}

    class _Boom:
        class messages:  # noqa: N801
            @staticmethod
            def create(*_args, **_kwargs):
                raise AttributeError("programmer bug — undefined attribute")

    with pytest.raises(AttributeError, match="programmer bug"):
        describe_all(
            store,
            media_root,
            model="m",
            output_language="English",
            description_version="v1",
            client=_Boom(),
        )


def test_describe_all_propagates_keyboard_interrupt(tmp_path: Path):
    """Ctrl-C must propagate — falls through the narrow Exception catch."""
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}

    class _CtrlC:
        class messages:  # noqa: N801
            @staticmethod
            def create(*_args, **_kwargs):
                raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        describe_all(
            store,
            media_root,
            model="m",
            output_language="English",
            description_version="v1",
            client=_CtrlC(),
        )


def test_describe_all_limit_caps_attempts(tmp_path: Path):
    """`--limit N` must stop after N photos have been described."""
    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(5)],
    )
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0), _judgment(1)]])
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        limit=2,
        batch_size=2,
        client=client,
    )
    described = sum(1 for m in item.media if isinstance(m, MediaPhotoDescribed))
    downloaded = sum(1 for m in item.media if isinstance(m, MediaPhotoDownloaded))
    assert described == 2
    assert downloaded == 3
    assert len(client.messages.calls) == 1


def test_describe_all_items_filter_restricts_scope(tmp_path: Path):
    """`--items <id>` must skip items not in the filter list."""
    media_root = tmp_path / "media"
    item_a = _item("a", [_downloaded(item_id="a", index=0, media_root=media_root)])
    item_b = _item("b", [_downloaded(item_id="b", index=0, media_root=media_root)])
    store = {"a": item_a, "b": item_b}
    client = _FakeVisionClient([[_judgment(0)]])
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        items_filter=["b"],
        client=client,
    )
    assert isinstance(item_a.media[0], MediaPhotoDownloaded)
    assert isinstance(item_b.media[0], MediaPhotoDescribed)


def test_describe_all_handles_missing_file_as_per_batch_failure(tmp_path: Path):
    """A missing photo file is per-photo failure, not a total abort.

    The entry says "downloaded" but the bytes are not on disk
    (operator removed `data/media/`, snapshot restored old items.json,
    etc). The orchestrator catches the OSError, marks the whole batch
    failed, and continues with the next batch.
    """
    media_root = tmp_path / "media"
    # Entry A: bytes on disk. Entry B: bytes missing on purpose.
    entry_a = _downloaded(item_id="a", index=0, media_root=media_root)
    entry_b_missing = MediaPhotoDownloaded(
        url="https://pbs.twimg.com/media/b.jpg",
        local_path="b/0.jpg",
        width=4,
        height=3,
        bytes_size=512,
        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )
    item_a = _item("a", [entry_a])
    item_b = _item("b", [entry_b_missing])
    store = {"a": item_a, "b": item_b}
    client = _FakeVisionClient([[_judgment(0)], [_judgment(0)]])
    report = describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        batch_size=1,
        client=client,
    )
    assert isinstance(item_a.media[0], MediaPhotoDescribed)
    assert isinstance(item_b.media[0], MediaPhotoDownloaded)  # stayed Downloaded
    assert report.photos_described == 1
    assert report.photos_failed == 1


def test_describe_all_does_not_emit_summary_on_partial_failure(tmp_path: Path, capsys):
    """The orchestrator never emits SUMMARY — the CLI is the single source of truth.

    Mirrors `media.download_all`: SUMMARY lives on `emit_summary_line`,
    not on the orchestrator. Asserts the count is exactly zero on a
    partial-failure run so a future regression that re-introduces a
    second emitter (double-SUMMARY) fails this test.
    """
    from anthropic import APIError

    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(2)],
    )
    store = {"1": item}
    client = _FakeVisionClient(
        [
            [_judgment(0)],
            APIError("503", request=None, body=None),
        ]
    )
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        batch_size=1,
        client=client,
    )
    err = capsys.readouterr().err
    assert err.count("SUMMARY:") == 0


def test_describe_all_emits_no_summary_when_all_succeed(tmp_path: Path, capsys):
    """A clean batch stays silent on stderr — same convention as `media`."""
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0)]])
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        client=client,
    )
    err = capsys.readouterr().err
    assert "SUMMARY:" not in err


def test_describe_all_fires_on_progress_after_each_batch(tmp_path: Path):
    """The Ctrl-C-coherent invariant: persistence fires between batches."""
    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(3)],
    )
    store = {"1": item}
    client = _FakeVisionClient(
        [
            [_judgment(0)],
            [_judgment(0)],
            [_judgment(0)],
        ]
    )
    calls: list[int] = []
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        batch_size=1,
        client=client,
        on_progress=lambda: calls.append(len(calls) + 1),
    )
    assert calls == [1, 2, 3]


def test_describe_all_skips_pending_failed_and_video_silently(tmp_path: Path):
    """Non-downloaded variants are silently out of scope — no API call, no failure."""
    media_root = tmp_path / "media"
    pending = MediaPhotoPending(url="https://pbs.twimg.com/media/p.jpg")
    failed = MediaPhotoFailed(
        url="https://pbs.twimg.com/media/f.jpg",
        failure_reason="http_4xx",
        error="HTTP 404",
        attempts=1,
        last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )
    video = MediaVideoPending(url="https://video.twimg.com/x.mp4")
    item = _item("1", [pending, failed, video])
    store = {"1": item}
    client = _FakeVisionClient([])  # no calls expected
    report = describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        client=client,
    )
    assert report.photos_attempted == 0
    assert report.photos_described == 0
    assert report.batches_attempted == 0
    assert client.messages.calls == []


def test_describe_all_total_failure_does_not_emit_summary(tmp_path: Path, capsys):
    """The all-failed branch raises BEFORE the summary print — verify."""
    from anthropic import APIError

    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([APIError("503", request=None, body=None)])
    with pytest.raises(RuntimeError):
        describe_all(
            store,
            media_root,
            model="m",
            output_language="English",
            description_version="v1",
            client=client,
        )
    err = capsys.readouterr().err
    assert "SUMMARY:" not in err


def test_describe_all_carries_over_local_path_dimensions(tmp_path: Path):
    """The described variant inherits the on-disk fields verbatim — no re-decode."""
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    # Tweak the fields so we can prove they round-tripped intact.
    entry = MediaPhotoDownloaded(
        url=entry.url,
        local_path=entry.local_path,
        width=1920,
        height=1080,
        bytes_size=123456,
        downloaded_at=entry.downloaded_at,
    )
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0)]])
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        client=client,
    )
    described = item.media[0]
    assert isinstance(described, MediaPhotoDescribed)
    assert described.local_path == "1/0.jpg"
    assert described.width == 1920
    assert described.height == 1080
    assert described.bytes_size == 123456


def test_describe_all_uses_injectable_clock(tmp_path: Path):
    """`now` injection lets tests assert the timestamp deterministically."""
    media_root = tmp_path / "media"
    entry = _downloaded(item_id="1", index=0, media_root=media_root)
    item = _item("1", [entry])
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0)]])
    fixed = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
    describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        client=client,
        now=lambda: fixed,
    )
    described = item.media[0]
    assert isinstance(described, MediaPhotoDescribed)
    assert described.described_at == fixed


# --------------------------------------------------------------------- payload shape


def test_describe_all_sends_payload_with_expected_kwarg_shape(tmp_path: Path):
    """The kwargs sent to `messages.create` must match the wire contract.

    A regression in the payload shape (wrong `max_tokens`, dropped
    `system`, role flip, image-block restructure) is otherwise invisible
    until production. Pins the structure end-to-end.
    """
    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(2)],
    )
    store = {"1": item}
    client = _FakeVisionClient([[_judgment(0), _judgment(1)]])
    describe_all(
        store,
        media_root,
        model="claude-sonnet-4-6",
        output_language="English",
        description_version="v1",
        batch_size=2,
        client=client,
    )
    kwargs = client.messages.calls[0]
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 1200
    assert isinstance(kwargs["system"], str) and kwargs["system"]
    messages = kwargs["messages"]
    assert isinstance(messages, list) and len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # 2 image blocks + exactly one trailing text directive.
    assert len(content) == 3
    image_blocks = content[:2]
    for block in image_blocks:
        assert block["type"] == "image"
        source = block["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "image/jpeg"
        assert isinstance(source["data"], str) and source["data"]
    text_block = content[2]
    assert text_block["type"] == "text"
    assert "Describe images 0 through 1" in text_block["text"]


# --------------------------------------------------------------------- refusal


def test_describe_all_handles_refusal_as_decorative_empty(tmp_path: Path):
    """SDK `stop_reason="refusal"` must transition every photo to decorative+empty.

    The Anthropic safety policy refuses identifiable faces / NSFW; the
    rubric says "if undescribable, mark decorative with empty
    description". The orchestrator applies that contract at the SDK
    level too so the batch makes progress instead of churning forever.
    """
    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(5)],
    )
    store = {"1": item}
    client = _FakeVisionClient([_FakeRefusalResponse()])
    report = describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        batch_size=5,
        client=client,
    )
    # All five photos must be Described as decorative + empty.
    for media_entry in item.media:
        assert isinstance(media_entry, MediaPhotoDescribed)
        assert media_entry.is_decorative is True
        assert media_entry.description == ""
    # Refusal handling counts as described, not failed — the photo made
    # progress out of Downloaded.
    assert report.photos_described == 5
    assert report.photos_failed == 0


def test_describe_all_logs_stop_reason_on_json_decode_failure(tmp_path: Path, caplog):
    """A `max_tokens` truncation must surface stop_reason in the warning log.

    Otherwise the operator gets only "Expecting value: line N column M"
    and cannot tell whether the cause is a truncated response, a
    refusal, or a malformed model emission.
    """
    media_root = tmp_path / "media"
    item = _item("1", [_downloaded(item_id="1", index=0, media_root=media_root)])
    store = {"1": item}
    client = _FakeVisionClient([_FakeTruncatedResponse()])
    with caplog.at_level("WARNING", logger="xbrain.describe"):
        # Total failure raises (1 batch attempted, 0 described).
        with pytest.raises(RuntimeError):
            describe_all(
                store,
                media_root,
                model="m",
                output_language="English",
                description_version="v1",
                batch_size=1,
                client=client,
            )
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "stop_reason='max_tokens'" in messages or "stop_reason=" in messages


# --------------------------------------------------------------------- fence variants


def test_parse_batch_response_strips_fence_without_language_tag():
    """Some models emit ``` ... ``` (no `json` after the opening fence)."""
    import json

    class _Fenced:
        content = [type("B", (), {"type": "text", "text": ""})()]

    fenced_text = "```\n" + json.dumps([_judgment(0)]) + "\n```"
    _Fenced.content[0].text = fenced_text
    out = _parse_batch_response(_Fenced(), batch_size=1)
    assert out[0]["index"] == 0


# --------------------------------------------------------------------- partial-data batch


def test_describe_all_partial_judgments_count_as_batch_failed(tmp_path: Path):
    """A batch with missing judgments must bump batches_failed (not 0).

    The vision model returned 1 of 2 expected judgments. One photo
    transitions to Described, the other stays Downloaded, and the
    batch counts as a failure because it did NOT return complete
    data for the API call we made.
    """
    media_root = tmp_path / "media"
    item = _item(
        "1",
        [_downloaded(item_id="1", index=i, media_root=media_root) for i in range(2)],
    )
    store = {"1": item}
    # Only one judgment for a 2-image batch.
    client = _FakeVisionClient([[_judgment(0, description="ok")]])
    report = describe_all(
        store,
        media_root,
        model="m",
        output_language="English",
        description_version="v1",
        batch_size=2,
        client=client,
    )
    assert report.photos_described == 1
    assert report.photos_failed == 1
    assert report.batches_failed == 1


# --------------------------------------------------------------------- summary line


def test_emit_summary_line_silent_when_nothing_happened(capsys):
    """No attempts, no skips → no SUMMARY noise. Same convention as media."""
    emit_summary_line(DescribeReport())
    assert "SUMMARY:" not in capsys.readouterr().err


def test_emit_summary_line_prints_when_described_or_skipped(capsys):
    """A non-zero `photos_described` triggers the SUMMARY line."""
    report = DescribeReport(
        photos_attempted=1,
        photos_described=1,
    )
    emit_summary_line(report)
    err = capsys.readouterr().err
    assert "SUMMARY: described: 1, failed: 0, skipped: 0" in err


def test_emit_summary_line_prints_when_only_skipped(capsys):
    """An all-skipped no-op (idempotency proof) emits the SUMMARY line too."""
    report = DescribeReport(photos_skipped_already_described=3)
    emit_summary_line(report)
    err = capsys.readouterr().err
    assert "skipped: 3" in err


# --------------------------------------------------------------------- response helper


def test_fake_response_block_carries_text_attribute():
    """`tests.conftest.FakeBlock` requires both `type` and `text` attributes."""
    block = FakeBlock(payload={"x": 1})
    assert block.type == "text"
    assert isinstance(block.text, str)


def test_fake_response_round_trips():
    """Smoke check on the `FakeResponse` machinery the tests rely on."""
    payload = {"summary": "x", "primary_topic": "x", "topics": ["x"]}
    response = FakeResponse(payload)
    assert response.content[0].text


def test_fake_anthropic_records_calls():
    """`FakeAnthropic.messages.calls` records every kwarg dict — relied on by other tests."""
    client = FakeAnthropic([{"x": 1}])
    client.messages.create(model="m", system="s", messages=[])
    assert client.messages.calls[0]["model"] == "m"

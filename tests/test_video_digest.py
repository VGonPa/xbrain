# tests/test_video_digest.py
import json
from datetime import datetime, timezone

import pytest

from xbrain.models import Author, Content, ContentSourceSuccess, Item, VideoFrame
from xbrain.video_digest import (
    apply_video_digest_judgments,
    export_video_digest_worksheet,
    import_video_digest_worksheet,
    items_pending_video_digest,
)


def _video_item(
    item_id: str = "7",
    *,
    text: str = "a talk about scaling laws",
    has_speech: bool = True,
    frame_descs: tuple[str, ...] = (),
    digest: str = "",
) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="watch this",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/v",
                    title="A great talk",
                    text=text,
                    has_speech=has_speech,
                    frames=[
                        VideoFrame(
                            timestamp=float(i),
                            local_path=f"{item_id}/frames/{i}.png",
                            description=d,
                        )
                        for i, d in enumerate(frame_descs)
                    ],
                    digest=digest,
                )
            ],
        ),
    )


def _text_item(item_id: str = "9") -> Item:
    """A non-video item — must never be selected for a video digest."""
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="a plain text post",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------- selection


def test_pending_selects_video_with_transcript_and_no_digest():
    store = {"7": _video_item(text="deep talk")}
    assert [it.id for it in items_pending_video_digest(store)] == ["7"]


def test_pending_selects_mute_video_with_frames():
    """A silent slide/screen video (no transcript) but WITH frames is digestible."""
    store = {"7": _video_item(text="", has_speech=False, frame_descs=("A slide.",))}
    assert [it.id for it in items_pending_video_digest(store)] == ["7"]


def test_pending_skips_silent_video_without_frames():
    """No transcript AND no frames → nothing to digest."""
    store = {"7": _video_item(text="", has_speech=False)}
    assert items_pending_video_digest(store) == []


def test_pending_skips_mute_video_whose_frames_have_empty_descriptions():
    """Selection must match the exporter: a mute video whose frames all carry EMPTY
    descriptions serialises to an empty worksheet (`_video_frame_descriptions` drops
    them), so it must NOT be selected — else it is re-exported every run forever."""
    store = {"7": _video_item(text="", has_speech=False, frame_descs=("", ""))}
    assert items_pending_video_digest(store) == []


def test_pending_skips_already_digested_video():
    store = {"7": _video_item(text="deep talk", digest="Already has a digest.")}
    assert items_pending_video_digest(store) == []


def test_pending_skips_non_video_item():
    store = {"9": _text_item()}
    assert items_pending_video_digest(store) == []


# ---------------------------------------------------------------- export


def test_export_worksheet_carries_transcript_frames_and_rubric(tmp_path):
    path = tmp_path / "ws.json"
    export_video_digest_worksheet(
        [_video_item(text="the full transcript", frame_descs=("A chart.", "A code slide."))],
        path,
        "claude-code",
        "English",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["items"][0]
    assert entry["video_transcript"] == "the full transcript"
    assert entry["video_frame_descriptions"] == ["A chart.", "A code slide."]
    assert entry["title"] == "A great talk"
    assert data["executor"] == "claude-code"
    assert data["judgments"] == []
    # The rubric ships with {language} already substituted.
    assert "{language}" not in data["rubric"]
    assert "English" in data["rubric"]


def test_export_worksheet_carries_full_untruncated_transcript(tmp_path):
    """The worksheet is agent-judged, so it carries the whole transcript (no cap)."""
    from xbrain.rubrics import TRANSCRIPT_CHAR_LIMIT

    long_text = "word " * TRANSCRIPT_CHAR_LIMIT
    path = tmp_path / "ws.json"
    export_video_digest_worksheet([_video_item(text=long_text)], path, "manual", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"][0]["video_transcript"] == long_text


# ---------------------------------------------------------------- import


def test_import_reads_judgments(tmp_path):
    path = tmp_path / "ws.json"
    export_video_digest_worksheet([_video_item()], path, "claude-code", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["judgments"] = [{"item_id": "7", "digest": "A digest."}]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert import_video_digest_worksheet(path) == [{"item_id": "7", "digest": "A digest."}]


def test_import_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_video_digest_worksheet(tmp_path / "nope.json")


def test_import_rejects_non_list_judgments(tmp_path):
    path = tmp_path / "ws.json"
    path.write_text(json.dumps({"judgments": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        import_video_digest_worksheet(path)


# ---------------------------------------------------------------- apply


def test_apply_writes_digest_onto_the_video_source():
    store = {"7": _video_item(text="talk")}
    applied, invalid = apply_video_digest_judgments(
        store, [{"item_id": "7", "digest": "  A crisp digest.  "}]
    )
    assert applied == 1
    assert invalid == []
    source = store["7"].content.sources[0]
    assert source.digest == "A crisp digest."  # stripped


def test_apply_rejects_unknown_item():
    store = {"7": _video_item()}
    applied, invalid = apply_video_digest_judgments(store, [{"item_id": "404", "digest": "x"}])
    assert applied == 0
    assert invalid == [("404", ["unknown item id"])]


def test_apply_rejects_item_without_video_source():
    store = {"9": _text_item()}
    applied, invalid = apply_video_digest_judgments(store, [{"item_id": "9", "digest": "x"}])
    assert applied == 0
    assert invalid[0][0] == "9"
    assert "no x_video source" in invalid[0][1][0]


def test_apply_rejects_empty_digest():
    store = {"7": _video_item()}
    applied, invalid = apply_video_digest_judgments(store, [{"item_id": "7", "digest": "   "}])
    assert applied == 0
    assert "empty" in invalid[0][1][0]
    assert store["7"].content.sources[0].digest == ""  # unchanged

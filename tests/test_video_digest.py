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
    # The rubric ships with {language} already substituted. NOTE: these two
    # assertions hold for EVERY rubric in the package — they do not pin WHICH
    # rubric was exported. `test_export_worksheet_rubric_carries_the_faithfulness_rules`
    # does that.
    assert "{language}" not in data["rubric"]
    assert "English" in data["rubric"]


def test_export_worksheet_rubric_carries_the_faithfulness_rules(tmp_path):
    """The exported worksheet IS the digest model's prompt — the digest flow has no
    other path to an LLM — so the payload's `rubric` is the closest a deterministic
    test gets to "the model sees the anti-inference rule".

    Guards a real, silent failure: wiring `load_rubric("summary", ...)` into
    `export_video_digest_worksheet` would strip the hardening from the prompt while
    every other assertion above stays green, since every rubric substitutes
    `{language}`. One canary per rule — name-nothing-unnamed, quote-verbatim,
    do-not-sharpen — so a deleted rule or a swapped rubric reds here.
    """
    path = tmp_path / "ws.json"
    export_video_digest_worksheet([_video_item()], path, "claude-code", "English")
    rubric = json.loads(path.read_text(encoding="utf-8"))["rubric"].lower()
    assert "neutral descriptor" in rubric, (
        "exported rubric lost the never-name-an-unnamed-entity rule"
    )
    assert "attribution" in rubric, "exported rubric lost the attribution-vs-content rule"
    assert "verbatim" in rubric, "exported rubric lost the quote-verbatim rule"
    assert "sharpen" in rubric, "exported rubric lost the do-not-sharpen rule"


def test_export_worksheet_instructions_agree_with_the_rubrics_evidence_contract(tmp_path):
    """The worksheet's `instructions` and its `rubric` are ONE prompt — they must not
    contradict each other.

    The old instruction line said to ground the digest "NOT the tweet text/caption",
    which flatly denies the rubric's rule that the tweet text and author metadata ARE
    valid evidence for attribution. A model reading both would have to pick one. The
    line must keep the caption out of the SUBSTANCE while admitting it for ATTRIBUTION.
    """
    path = tmp_path / "ws.json"
    export_video_digest_worksheet([_video_item()], path, "claude-code", "English")
    instructions = json.loads(path.read_text(encoding="utf-8"))["instructions"].lower()
    assert "attribut" in instructions, "worksheet instructions do not mention attribution"
    # The caption must still be excluded from what gets summarised.
    assert "caption" in instructions
    assert "not the tweet `text`/caption" not in instructions, (
        "instructions still blanket-forbid the tweet text, contradicting the rubric"
    )


def test_export_worksheet_carries_author_handle_and_display_name(tmp_path):
    """The digest rubric admits the author metadata as evidence for WHO is speaking,
    so the generator must be handed the same metadata the judge holds.

    Post-#86 the judge's `_source_text` opens with `@handle (Display Name)`. Shipping
    only the handle would leave the generator de-abbreviating `@lexfridman` into a
    name it never saw, while the judge validates against the display name it did —
    the two sides disagreeing about what evidence exists. 221 of the 235 video items
    in the store have a display name that differs from the handle, so this is the
    common case, not an edge one.
    """
    path = tmp_path / "ws.json"
    export_video_digest_worksheet([_video_item()], path, "claude-code", "English")
    entry = json.loads(path.read_text(encoding="utf-8"))["items"][0]
    assert entry["author"] == "a"
    assert entry["author_name"] == "A"


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


# ------- the digest's evidence set must match what the generator AND judge are given
#
# 45 of the 235 video items are ALSO quote-tweets. Since #98 the judge's `_source_text`
# carries a `[Quoted post — @handle (Name)]` block for those — so a digest rubric that
# declares "the evidence is exactly these FIVE surfaces", a worksheet that never ships
# the quoted post, and a judge that is shown it, are three different answers to one
# question. That one-field-many-rules drift is precisely what this PR exists to kill.
#
# And the attribution runs the WRONG way if left alone: the rubric says the author
# metadata attributes the clip ("posted by the speaker's own account"). On a quote-tweet
# the poster is NOT the author of the quoted content — the #86 conflation, aimed at the
# digest.

from xbrain.executors.api import quoted_attribution, quoted_text  # noqa: E402
from xbrain.rubrics import load_rubric  # noqa: E402
from xbrain.verification import _source_text  # noqa: E402


def _quoting_video_item() -> Item:
    item = _video_item()
    item.quoted_id = "999"
    item.content.sources = [
        *item.content.sources,
        ContentSourceSuccess(
            kind="quoted_tweet",
            url="https://x.com/karpathy/status/999",
            text="My talk on why RL is terrible.",
            author=Author(handle="karpathy", name="Andrej Karpathy"),
        ),
    ]
    return item


def test_the_judge_shows_the_digest_a_quoted_post_block(tmp_path):
    """The premise. If this ever stops holding, the coupling below is moot."""
    assert f"[{quoted_attribution(_quoting_video_item())}]" in _source_text(_quoting_video_item())


def test_the_digest_worksheet_ships_the_quoted_post_the_judge_will_check_against(tmp_path):
    """The generator must hold every surface the judge holds, or the rubric names a
    surface the running generator was never given — the bug this PR was written for."""
    item = _quoting_video_item()
    path = tmp_path / "ws.json"
    export_video_digest_worksheet([item], path, "claude-code", "English")
    entry = json.loads(path.read_text(encoding="utf-8"))["items"][0]

    assert entry["quoted_text"] == quoted_text(item)
    assert entry["quoted_attribution"] == quoted_attribution(item)


def test_the_digest_rubric_admits_the_quoted_post_and_denies_the_poster_its_authorship():
    """Attribution evidence, not substance — and pointed the right way round: on a
    quote-tweet the speaker may be the QUOTED account, never the poster by default."""
    text = load_rubric("video-digest").lower()

    assert "quoted post" in text
    assert "five surfaces" not in text  # the count moved; the enumeration must move with it
    assert "poster" in text

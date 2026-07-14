# tests/test_evidence_contract.py
"""The cross-component guard: four components, ONE definition of evidence.

Four components each need to know what counts as evidence for a generated output:

  1. the GENERATOR  — what the worksheet actually hands the agent
  2. the RUBRIC     — what the judge is told may support a claim
  3. the JUDGE      — what `_source_text` actually puts in front of it
  4. the CHECKER    — what the deterministic entity check searches for a name

Before `xbrain.evidence`, each maintained its own hand-written list, and 1,306 tests
passed while the four contradicted one another — because every PR tested only its own
side. The contradictions were not theoretical: the judge was handed the linked article
for a DIGEST whose generator never saw it (so it excused inventions it could not have
sourced), and neither generator shipped the author display name its rubric promised.

This file binds them. It asserts, per target:

    generator fields  ⊇  evidence_surfaces(item, target)
    judge source      ==  evidence_surfaces(item, target)
    checker evidence  ==  evidence_surfaces(item, target)
    verify rubric     declares every surface in evidence_surfaces(item, target)

Identity against the shared function — never a substring, and never a hand-written list
repeated here (a list repeated in the test is a fifth copy of the bug). Add a surface to
one component and forget the others, and this file goes red.
"""

import json
from datetime import datetime, timezone

import pytest

from xbrain.evidence import (
    SURFACE_KEYS,
    SURFACE_RUBRIC_PHRASES,
    evidence_surfaces,
    evidence_text,
)
from xbrain.executors.api import _user_prompt
from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Item,
    Link,
    MediaPhotoDescribed,
    Topic,
    VideoFrame,
)
from xbrain.rubrics import load_rubric
from xbrain.verification import _source_text
from xbrain.video_digest import export_video_digest_worksheet
from xbrain.worksheet import export_worksheet

VOCAB = [Topic(slug="misc", description="Noise.")]

# Every surface carries a UNIQUE sentinel, so "is this surface present?" is answerable
# by substring on the rendered payload without a header or another surface satisfying it
# by accident.
AUTHOR_HANDLE = "emollick"
AUTHOR_NAME = "Ethan Mollick"
TWEET = "SENTINEL-tweet-text"
VIDEO_TITLE = "SENTINEL-video-title"
TRANSCRIPT = "SENTINEL-transcript"
FRAME = "SENTINEL-frame-description"
ARTICLE_TITLE = "SENTINEL-article-title"
ARTICLE = "SENTINEL-article-body"
THREAD = "SENTINEL-thread-text"
IMAGE = "SENTINEL-image-description"


def _full_item() -> Item:
    """One item that carries EVERY surface at once — so a surface missing from a
    component is a missing sentinel, not an accident of the fixture."""
    return Item(
        id="7",
        source="bookmark",
        url="https://x.com/a/status/7",
        author=Author(handle=AUTHOR_HANDLE, name=AUTHOR_NAME),
        text=TWEET,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/a", domain="example.com")],
        media=[
            MediaPhotoDescribed(
                url="https://pbs.twimg.com/media/X.jpg",
                local_path="7/0.jpg",
                width=4,
                height=3,
                bytes_size=512,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
                is_decorative=False,
                description=IMAGE,
                description_lang="English",
                description_version="v1",
                described_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
            )
        ],
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/v",
                    title=VIDEO_TITLE,
                    text=TRANSCRIPT,
                    has_speech=True,
                    frames=[
                        VideoFrame(timestamp=0.0, local_path="7/frames/0.png", description=FRAME)
                    ],
                ),
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.com/a",
                    title=ARTICLE_TITLE,
                    text=ARTICLE,
                ),
                ContentSourceSuccess(
                    kind="thread",
                    url="https://x.com/a/status/7",
                    text=THREAD,
                ),
            ],
        ),
    )


def _generator_payload(target: str, tmp_path) -> str:
    """What the target's GENERATOR actually hands its agent, as raw text.

    `digest` → the video-digest worksheet. `summary`/`topics` → the enrich worksheet AND
    the `api` executor's prompt; both are generators for those targets, so both must
    carry every surface.
    """
    item = _full_item()
    path = tmp_path / "ws.json"
    if target == "digest":
        export_video_digest_worksheet([item], path, "claude-code", "English")
        return path.read_text(encoding="utf-8")
    export_worksheet([item], VOCAB, path, "claude-code", "English")
    return path.read_text(encoding="utf-8") + "\n" + _user_prompt(item, VOCAB)


@pytest.mark.parametrize("target", ["digest", "summary", "topics"])
def test_the_generator_ships_every_surface_the_judge_will_check(target, tmp_path):
    """generator ⊇ surfaces. A surface the judge treats as evidence but the generator
    never shipped is a FALSE FAIL waiting to happen: the output could not have used it,
    and an output that does use it (because a DIFFERENT surface carried the fact) gets
    judged against a source the generator never saw."""
    payload = _generator_payload(target, tmp_path)
    for surface in evidence_surfaces(_full_item(), target):
        for value in surface.values:
            assert value in payload, (
                f"generator for {target!r} never ships {value!r} "
                f"(surface {surface.key!r}) — the judge would check the output against "
                "evidence the generator was not handed"
            )


@pytest.mark.parametrize("target", ["digest", "summary", "topics"])
def test_the_judge_source_is_exactly_the_evidence_surfaces(target):
    """judge == surfaces, in BOTH directions.

    Missing a surface → the judge flags a claim the generator was entitled to make (the
    36+ false author flags). Carrying an EXTRA one → the judge excuses an invention the
    generator could not have sourced (the digest judged against a linked article its
    generator never received). Both directions are pinned."""
    item = _full_item()
    source = _source_text(item, target)
    admitted = evidence_surfaces(item, target)
    for surface in admitted:
        assert surface.label in source
        for value in surface.values:
            assert value in source, f"judge source for {target!r} omits {surface.key!r}"

    admitted_keys = {s.key for s in admitted}
    # Every surface this target does NOT admit must be absent from the judge's source —
    # by its sentinel, so a shared label cannot hide it.
    for surface in evidence_surfaces(item, "summary"):  # the widest set
        if surface.key not in admitted_keys:
            for value in surface.values:
                assert value not in source, (
                    f"judge source for {target!r} carries {surface.key!r}, "
                    "which that target's generator never sees"
                )


@pytest.mark.parametrize("target", ["digest", "summary", "topics"])
def test_the_checker_searches_exactly_the_evidence_surfaces(target):
    """checker == surfaces. The deterministic entity check asks "does this name appear on
    ANY evidence surface?" — if its notion of evidence drifts from the judge's, it either
    manufactures false positives or inherits the blind spot it exists to cover."""
    item = _full_item()
    evidence = evidence_text(item, target)
    admitted = evidence_surfaces(item, target)
    for surface in admitted:
        for value in surface.values:
            assert value in evidence
    admitted_keys = {s.key for s in admitted}
    for surface in evidence_surfaces(item, "summary"):
        if surface.key not in admitted_keys:
            for value in surface.values:
                assert value not in evidence


@pytest.mark.parametrize("target", ["digest", "summary", "topics"])
def test_the_verify_rubric_declares_every_surface_it_admits(target):
    """rubric == surfaces. The judge is TOLD what may support a claim; if the rubric's
    list and `_source_text` disagree, the judge either flags what it was handed or
    accepts what it was not."""
    rubric = load_rubric("verify", language="English")
    for key in SURFACE_KEYS[target]:
        phrase = SURFACE_RUBRIC_PHRASES[key]
        assert phrase in rubric, f"rubric-verify never declares the {key!r} surface ({phrase!r})"


def test_every_declared_surface_has_a_rubric_phrase():
    """No surface may exist without the rubric having a word for it — otherwise the
    binding above is vacuous for that surface."""
    for keys in SURFACE_KEYS.values():
        for key in keys:
            assert SURFACE_RUBRIC_PHRASES.get(key), f"surface {key!r} has no rubric phrase"


def test_a_url_is_evidence_for_no_target(tmp_path):
    """D1, the deepest one: the judge's source shows the links (topic signal), but a URL
    is NOT an evidence surface — so naming the publication it belongs to can never be
    grounded. `axios.com` must not ground "Axios"."""
    for target in ("digest", "summary", "topics"):
        assert "example.com" not in evidence_text(_full_item(), target)
    assert not any(
        s.key == "links" for s in evidence_surfaces(_full_item(), "summary")
    )  # no such surface exists


def test_the_contract_survives_a_json_round_trip(tmp_path):
    """The worksheet is JSON on disk; a surface must survive serialisation to reach the
    agent (a `None` field ships nothing)."""
    item = _full_item()
    path = tmp_path / "ws.json"
    export_worksheet([item], VOCAB, path, "claude-code", "English")
    entry = json.loads(path.read_text(encoding="utf-8"))["items"][0]
    serialised = json.dumps(entry, ensure_ascii=False)
    for surface in evidence_surfaces(item, "summary"):
        for value in surface.values:
            assert value in serialised

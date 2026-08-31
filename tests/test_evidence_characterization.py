# tests/test_evidence_characterization.py
"""The pin that proves the knowledge contract did NOT move `evidence.py` (Plan 01 §2).

WHY THIS FILE EXISTS, AND WHY IT IS WRITTEN FIRST. `verification.contract_fingerprint`
hashes three arms, and one of them is `_source_text(item, target)` — which is built from
`evidence.evidence_surfaces`, which reads five selectors out of `executors/api.py`. The
knowledge-contract PR refactors exactly those five selectors onto shared atomic iterators
(M2). **If that refactor changes one byte of the judge's source text, every stored
`VerificationVerdict` goes stale and every badge disappears from `generate`** — CLAUDE.md
rule 6 run backwards: invalidating derivatives without having repaired any evidence.

So this is a CHARACTERIZATION pin, not a test of a fix. It is born green on purpose,
against the tree as it stands BEFORE the refactor, and its whole value is staying green
afterwards. That makes rule 1 ("a test that passes before you write the fix is not a
test") inapplicable in its usual form — and it is replaced by an explicit obligation:
`test_the_pin_can_fail` demonstrates, in-process, that a one-byte change to any arm moves
the hash. A pin nobody has seen fail is a decoration.

TWO LEVELS, on purpose. The literal that matters is `CONTRACT_FINGERPRINTS` (the whole
contract: output + source + rubrics). But a bare contract failure cannot say WHICH arm
moved, and the rubric arm reads files this PR does not touch. So `SOURCE_FINGERPRINTS`
pins the source arm alone: when both go red the evidence surfaces moved; when only the
contract goes red a rubric file was edited.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Enrichment,
    Item,
    MediaPhotoDescribed,
    VideoFrame,
)
from xbrain.verification import _source_text, contract_fingerprint

UTC = timezone.utc
LANGUAGE = "Spanish"


def _full_item() -> Item:
    """An item that exercises EVERY enrich surface at once.

    Author, video title, transcript, frames, images, article title, article body, thread,
    quoted post (with its own third-party author) and the tweet text. If the refactor
    drops or reorders any of them, the source arm moves.
    """
    return Item(
        id="1000000000000000001",
        source="bookmark",
        url="https://x.com/poster/status/1000000000000000001",
        author=Author(handle="poster", name="The Poster"),
        text="Here is the same thought by someone that knows a bit more https://t.co/abc",
        created_at=datetime(2026, 3, 4, 10, 0, tzinfo=UTC),
        captured_at=datetime(2026, 3, 5, 10, 0, tzinfo=UTC),
        media=[
            MediaPhotoDescribed(
                url="https://pbs.twimg.com/media/photo-one.jpg",
                local_path="1000000000000000001/0.jpg",
                width=1200,
                height=800,
                bytes_size=44_000,
                downloaded_at=datetime(2026, 3, 5, 11, 0, tzinfo=UTC),
                is_decorative=False,
                description="A bar chart comparing throughput across three runtimes.",
                description_lang="Spanish",
                description_version="describe/v3",
                described_at=datetime(2026, 3, 5, 12, 0, tzinfo=UTC),
            ),
            MediaPhotoDescribed(
                url="https://pbs.twimg.com/media/avatar.jpg",
                local_path="1000000000000000001/1.jpg",
                width=200,
                height=200,
                bytes_size=9_000,
                downloaded_at=datetime(2026, 3, 5, 11, 0, tzinfo=UTC),
                is_decorative=True,
                description="",
                description_lang="Spanish",
                description_version="describe/v3",
                described_at=datetime(2026, 3, 5, 12, 0, tzinfo=UTC),
            ),
        ],
        content=Content(
            fetched_at=datetime(2026, 3, 5, 13, 0, tzinfo=UTC),
            sources=[
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.org/essay",
                    title="On Export Controls",
                    text="The essay body, paragraph one.\n\nParagraph two of the essay body.",
                    attempts=1,
                ),
                ContentSourceSuccess(
                    kind="thread",
                    url="https://x.com/poster/status/1000000000000000001",
                    text="Thread continuation written by the poster themselves.",
                    attempts=1,
                ),
                ContentSourceSuccess(
                    kind="quoted_tweet",
                    url="https://x.com/jj/status/999",
                    text="What if Mythos shipped with open weights?",
                    author=Author(handle="JosephJacks_", name="JJ"),
                    attempts=1,
                ),
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://video.twimg.com/amplify_video/1/vid.mp4",
                    title="A talk about evaluation loops",
                    text="The transcript body of the talk, verbatim.",
                    has_speech=True,
                    language="en",
                    frames=[
                        VideoFrame(
                            timestamp=12.0,
                            local_path="1000000000000000001/frames/0.jpg",
                            description="A slide titled Self-Attention.",
                        ),
                        VideoFrame(
                            timestamp=48.5,
                            local_path="1000000000000000001/frames/1.jpg",
                            description="A slide showing arXiv 2502.16982.",
                        ),
                    ],
                    digest="What it is: a talk. Key points: evaluation loops.",
                ),
            ],
        ),
        enriched=Enrichment(
            enriched_at=datetime(2026, 3, 6, 9, 0, tzinfo=UTC),
            executor="claude-code",
            summary="Un resumen conciso del hilo y del artículo enlazado.",
            primary_topic="agent-evaluation",
            topics=["agent-evaluation", "observability"],
        ),
    )


def _video_item() -> Item:
    """A video-only item — the `digest` target's narrower surface set."""
    return Item(
        id="1000000000000000002",
        source="bookmark",
        url="https://x.com/speaker/status/1000000000000000002",
        author=Author(handle="speaker", name="The Speaker"),
        text="Full talk below.",
        created_at=datetime(2026, 4, 1, 8, 0, tzinfo=UTC),
        captured_at=datetime(2026, 4, 2, 8, 0, tzinfo=UTC),
        content=Content(
            fetched_at=datetime(2026, 4, 2, 9, 0, tzinfo=UTC),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://video.twimg.com/amplify_video/2/vid.mp4",
                    title="Scaling strategy",
                    text="Transcript: token efficiency, long context, agent swarms.",
                    has_speech=True,
                    language="en",
                    frames=[
                        VideoFrame(
                            timestamp=5.0,
                            local_path="1000000000000000002/frames/0.jpg",
                            description="A slide listing three pillars.",
                        )
                    ],
                    digest="What it is: a scaling talk. Key points: three pillars.",
                )
            ],
        ),
        enriched=Enrichment(
            enriched_at=datetime(2026, 4, 3, 9, 0, tzinfo=UTC),
            executor="claude-code",
            summary="Charla sobre estrategia de escalado.",
            primary_topic="agentic-engineering",
            topics=["agentic-engineering"],
        ),
    )


FIXTURES = {"full_item": _full_item, "video_item": _video_item}

# The judge's SOURCE arm alone: sha256 of `_source_text(item, target)`. This is the arm
# the knowledge-contract refactor can move, and pinning it separately is what lets a
# failure name the culprit instead of just saying "something changed".
SOURCE_FINGERPRINTS: dict[tuple[str, str], str] = {
    ("full_item", "summary"): "3e8ff9aef2247806abfa9ee22e09d61ad543d2b876870894a65fe9ed7ab9e877",
    # Identical to `summary` on purpose: `evidence.SURFACE_KEYS` gives both enrich targets
    # the SAME surface tuple. If this pair ever diverges, the target split moved.
    ("full_item", "topics"): "3e8ff9aef2247806abfa9ee22e09d61ad543d2b876870894a65fe9ed7ab9e877",
    ("video_item", "digest"): "600e770b3a0ff3f3eeb90e2b20996d4e00f79a35af3e053093d8391dd34dc60f",
}

# The FULL contract: output + source + rubrics, exactly what a stored verdict binds to.
CONTRACT_FINGERPRINTS: dict[tuple[str, str], str] = {
    ("full_item", "summary"): "304418b287dd14e5efe4f2d72d53d1b6b7d6f33029e8260fa58bc6175c9a6e10",
    ("full_item", "topics"): "b37e6f89819096de58c168c81662b5c094e92bc29d8e23007c958f77381f883a",
    ("video_item", "digest"): "cef43c15ca29660e2f532876d52f8765df04c1dcc45599b20f6b539b4002d598",
}

CASES = sorted(CONTRACT_FINGERPRINTS)


def _source_fingerprint(item: Item, target: str) -> str:
    return hashlib.sha256(_source_text(item, target).encode("utf-8")).hexdigest()


@pytest.mark.parametrize(("fixture", "target"), CASES)
def test_judge_source_text_is_unchanged(fixture: str, target: str) -> None:
    """The judge reads byte-for-byte what it read before the knowledge refactor.

    Red means `evidence_surfaces` (or one of the five `executors/api.py` selectors it
    reads) changed what it emits. That is the change that retires every stored verdict.
    """
    item = FIXTURES[fixture]()
    assert _source_fingerprint(item, target) == SOURCE_FINGERPRINTS[(fixture, target)]


@pytest.mark.parametrize(("fixture", "target"), CASES)
def test_contract_fingerprint_is_unchanged(fixture: str, target: str) -> None:
    """The whole judging contract is unchanged, so no stored verdict goes stale.

    If this is red while `test_judge_source_text_is_unchanged` is green, a RUBRIC file was
    edited — the third arm — not the evidence surfaces.
    """
    item = FIXTURES[fixture]()
    assert contract_fingerprint(item, target, LANGUAGE) == CONTRACT_FINGERPRINTS[(fixture, target)]


def test_the_pin_can_fail() -> None:
    """The pin is a real detector: a one-byte change to any arm moves the hash.

    Without this, a pin is unfalsifiable decoration — the failure mode CLAUDE.md rule 1
    exists to name. Two arms are perturbed in-process and asserted to diverge: the OUTPUT
    (the summary text) and the SOURCE (a frame description). The third arm, the rubrics,
    is not perturbed here because doing so means editing files on disk and clearing
    `rubric_digest`'s cache; `tests/test_contract_fingerprint.py` already owns that case.

    It also asserts the SEPARATION the two levels depend on: moving the output must NOT
    move the source arm. Without that, a red `test_judge_source_text_is_unchanged` would
    no longer mean "the evidence surfaces moved".
    """
    baseline_source = _source_fingerprint(_full_item(), "summary")
    baseline_contract = contract_fingerprint(_full_item(), "summary", LANGUAGE)

    moved_output = _full_item()
    moved_output.enriched.summary += "."  # type: ignore[union-attr]
    assert _source_fingerprint(moved_output, "summary") == baseline_source, (
        "the SOURCE arm must not depend on the output text — otherwise the two levels "
        "of this pin cannot separate 'the evidence moved' from 'the claim moved'"
    )
    assert contract_fingerprint(moved_output, "summary", LANGUAGE) != baseline_contract

    moved_source = _full_item()
    moved_source.content.sources[3].frames[0].description += "!"  # type: ignore[union-attr]
    assert _source_fingerprint(moved_source, "summary") != baseline_source
    assert contract_fingerprint(moved_source, "summary", LANGUAGE) != baseline_contract

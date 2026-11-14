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

This file binds THREE of them. It asserts, per target and PER GENERATOR:

    generator fields  ⊇  evidence_surfaces(item, target)   [each generator, on its own]
    judge source      ==  evidence_surfaces(item, target)
    verify rubric     declares every surface in evidence_surfaces(item, target)

Identity against the shared function — never a substring, and never a hand-written list
repeated here (a list repeated in the test is a fifth copy of the bug). Add a surface to
one component and forget the others, and this file goes red.

THE CHECKER'S LEG IS NOT HERE, AND THIS FILE DOES NOT PRETEND OTHERWISE. The deterministic
entity check lives in #89, stacked ON this branch, so nothing here can import it. A test
comparing `evidence_text` to `evidence_surfaces` — both from `xbrain.evidence` — would
assert the module against itself: green forever, binding nothing. That is exactly the
"passes for the wrong reason" test this PR exists to end, and an earlier draft of this file
shipped it. What lives here instead is the CONTRACT #89 consumes
(`test_evidence_text_is_a_faithful_label_free_projection_of_the_surfaces`); the binding
itself — `entity_grounding` calls `evidence_text` and keeps no private list — is asserted
in #89, where it can actually run.
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
from xbrain.executors.api import BOOKMARK_FOLDER_RULE, _user_prompt
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
QUOTED_BODY = "SENTINEL-quoted-body"
QUOTED_HANDLE = "karpathy"
QUOTED_NAME = "SENTINEL-quoted-author"


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
        quoted_id="999",
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
                # The quoted post (#98) — a surface for EVERY target: 45 of the 235 video
                # items are quote-tweets too, so the digest admits it as well (#87).
                ContentSourceSuccess(
                    kind="quoted_tweet",
                    url=f"https://x.com/{QUOTED_HANDLE}/status/999",
                    text=QUOTED_BODY,
                    author=Author(handle=QUOTED_HANDLE, name=QUOTED_NAME),
                ),
            ],
        ),
    )


def _generator_payload(generator: str, tmp_path) -> str:
    """What ONE generator actually hands its agent, as raw text.

    Never a concatenation of two: `summary`/`topics` have two generators (the enrich
    worksheet and the `api` prompt) and each must carry every surface on its own.
    """
    item = _full_item()
    path = tmp_path / "ws.json"
    if generator == "video_digest_worksheet":
        export_video_digest_worksheet([item], path, "claude-code", "English")
        return path.read_text(encoding="utf-8")
    if generator == "enrich_worksheet":
        export_worksheet([item], VOCAB, path, "claude-code", "English")
        return path.read_text(encoding="utf-8")
    return _user_prompt(item, VOCAB)


# Every (target, generator) pair that actually RUNS. `summary`/`topics` have TWO
# generators — the enrich worksheet and the `api` executor's prompt — and each must
# carry every surface ON ITS OWN.
#
# The earlier contract concatenated the two and asserted `value in worksheet + prompt`,
# so a surface present in only ONE of them passed. It did: `_user_prompt` never emitted
# the video title while the contract was green, because the worksheet's copy covered for
# it. An `--executor api` summary would then be judged against a title its generator was
# never shown — the "judge sees MORE than the generator" pathology this module exists to
# kill, arriving through the one generator the guardrail could not see.
_GENERATORS: list[tuple[str, str]] = [
    ("digest", "video_digest_worksheet"),
    ("summary", "enrich_worksheet"),
    ("summary", "api_prompt"),
    ("topics", "enrich_worksheet"),
    ("topics", "api_prompt"),
]


@pytest.mark.parametrize(("target", "generator"), _GENERATORS)
def test_the_generator_ships_every_surface_the_judge_will_check(target, generator, tmp_path):
    """generator ⊇ surfaces, for EACH generator independently.

    A surface the judge treats as evidence but the generator never shipped is a FALSE FAIL
    waiting to happen: the output could not have used it, and an output that does use it
    (because a DIFFERENT surface carried the fact) gets judged against a source the
    generator never saw. Asserted per generator, so one generator cannot cover for
    another's gap."""
    payload = _generator_payload(generator, tmp_path)
    for surface in evidence_surfaces(_full_item(), target):
        for value in surface.values:
            assert value in payload, (
                f"generator {generator!r} for target {target!r} never ships {value!r} "
                f"(surface {surface.key!r}) — the judge would check the output against "
                "evidence that generator was not handed"
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
def test_evidence_text_is_a_faithful_label_free_projection_of_the_surfaces(target):
    """`evidence_text` == the ATOMIC VALUES of the admitted surfaces, and nothing else.

    NAMED FOR WHAT IT CHECKS. This is a property of THIS MODULE — `evidence_text` against
    `evidence_surfaces` — not a binding of the checker. An earlier draft called this
    `test_the_checker_searches_exactly_the_evidence_surfaces`, which was a TAUTOLOGY: both
    sides come from `xbrain.evidence`, so it asserted the module against itself and would
    have stayed green no matter what the checker did. That would have been the fifth
    "passes for the wrong reason" test in this repo — inside the PR written to end that
    class of test.

    The checker lives in #89, which is stacked ON this branch, so nothing here can import
    it. **#89 carries the real binding**: `entity_grounding` calls `evidence_text` and
    keeps no private list. What this test guarantees is the CONTRACT #89 consumes — that
    the blob it will search holds every admitted value, no unadmitted one, and no label.
    """
    item = _full_item()
    evidence = evidence_text(item, target)
    admitted = evidence_surfaces(item, target)

    for surface in admitted:
        for value in surface.values:
            assert value in evidence
        # Labels are scaffolding, not content: a name must be grounded in what the item
        # SAYS, never in what we called the box.
        assert surface.label not in evidence

    admitted_keys = {s.key for s in admitted}
    for surface in evidence_surfaces(item, "summary"):  # the widest set
        if surface.key not in admitted_keys:
            for value in surface.values:
                assert value not in evidence


def test_evidence_text_carries_the_quoted_author_that_lives_in_the_LABEL():
    """The regression the `values`-vs-`text` split exists to prevent.

    The quoted post's author is rendered into the judge's LABEL
    (`[Quoted post — @karpathy (SENTINEL-quoted-author)]`), and `evidence_text` strips
    labels. Building the blob from `text` would therefore drop the quoted author — and
    #89's checker would flag "Karpathy announces he is leaving OpenAI" as an ungrounded
    name on the very item that grounds it. Built from `values`, it cannot.
    """
    item = _full_item()
    evidence = evidence_text(item, "summary")

    assert QUOTED_NAME in evidence
    assert QUOTED_HANDLE in evidence
    assert QUOTED_BODY in evidence


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


def test_no_surface_is_derived_from_a_LINK():
    """D1, the deepest one: `item.links` grounds nothing. A URL/domain is topic signal —
    never a name, never content — so naming the publication a link belongs to can never be
    grounded. `axios.com/...-anthropic` must not ground "Axios", nor "Anthropic".

    Asserted on the LINK, which is the real invariant. The fixture's link carries a
    distinctive host, so a surface that ever started reading `item.links` goes red here.
    """
    item = _full_item()
    for target in ("digest", "summary", "topics"):
        evidence = evidence_text(item, target)
        assert "example.com" not in evidence
        assert "https://example.com/a" not in evidence
    assert not any(s.key == "links" for s in evidence_surfaces(item, "summary"))


def test_the_tweet_surface_carries_the_URLs_the_POST_ITSELF_contains():
    """And now the honest half, which the earlier draft asserted away.

    "No surface contains a URL" is FALSE: 1,281 of the 2,168 items carry one inside their
    own tweet text, and `[Tweet]` is the post's words verbatim — as it must be. So
    `evidence_text` does contain URL characters, and a checker doing a naive substring
    search could ground "Anthropic" in a slug sitting in the tweet.

    Not exploitable in the corpus today (stripping URLs leaves every grounded name still
    grounded — measured), but it is a real hole, and it belongs to the component that does
    the matching: **#89's checker must strip URLs before grounding a name.** Pinning the
    truth here keeps that hole visible instead of legislating it away.
    """
    item = _full_item()
    item.text = f"{TWEET} https://axios.com/2025/05/28/ai-jobs-anthropic"

    evidence = evidence_text(item, "summary")

    assert "axios.com" in evidence  # it rides in, inside the poster's own words
    assert any(s.key == "tweet" for s in evidence_surfaces(item, "summary"))
    # …and it got there via the TWEET, not via a link-derived surface.
    assert not any(s.key == "links" for s in evidence_surfaces(item, "summary"))


def test_the_bookmark_folder_is_NOT_evidence_and_both_generators_say_so(tmp_path):
    """The explicit decision (L: "bookmark_folder reaches both generators while being no
    surface at all; decide it").

    DECIDED: it is NOT evidence. It is the USER'S OWN filing label — organisational
    metadata about how he sorted the post, not a fact the post asserts about the world. A
    folder named "ai-industry" must never ground a name or a claim, exactly as a domain
    must not. It still ships (it is genuine topic signal for `topics`), so both generators
    must TELL the agent what it is — otherwise a generator grounds a claim in it and the
    judge, which never sees it, flags that claim: a false FAIL by construction.
    """
    item = _full_item()
    item.bookmark_folder = "ai-industry"

    assert not any(s.key == "bookmark_folder" for s in evidence_surfaces(item, "summary"))
    assert "ai-industry" not in evidence_text(item, "summary")
    assert "ai-industry" not in _source_text(item, "summary")  # the judge never sees it

    # Both generators ship it AND carry the SAME rule — compared BY REFERENCE to the
    # shared constant. A bare `"topic signal only" in payload` would pass on the LINKS
    # rule alone, which says nothing about the folder: a test green for the wrong reason,
    # in the PR written to end those. (It did, until this assertion was tightened.)
    prompt = _user_prompt(item, VOCAB)
    assert "ai-industry" in prompt
    assert BOOKMARK_FOLDER_RULE in prompt
    path = tmp_path / "ws.json"
    export_worksheet([item], VOCAB, path, "claude-code", "English")
    worksheet = path.read_text(encoding="utf-8")
    assert "ai-industry" in worksheet
    assert BOOKMARK_FOLDER_RULE in worksheet


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

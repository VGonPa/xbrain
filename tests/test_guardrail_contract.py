# tests/test_guardrail_contract.py
"""The unfetched-content guardrails are ONE contract, read by every LLM surface.

The generator (the `api` prompt, the enrich worksheet) and the judge (the verify
source) must receive the note VERBATIM and IDENTICAL. If any surface drifts to its own
wording, the judge can no longer hold the generator to what it was told — which is the
entire premise of the guardrail.

These tests assert IDENTITY against the shared note, and pin the note's semantic core.
They never assert a bare `"NOT fetched"` substring: that substring is already satisfied
by the section HEADER (`[Links — content NOT fetched]`), so a test written that way
passes even when the instruction itself has been deleted — an accidental tautology that
let a gutted guardrail through review once already.
"""

import json
from datetime import datetime, timezone

import pytest

from xbrain.executors.api import (
    QUOTED_CONTENT_UNFETCHED_NOTE,
    _user_prompt,
    unfetched_links_note,
)
from xbrain.models import Author, Content, ContentSourceSuccess, Item, Link, Topic
from xbrain.verification import _source_text
from xbrain.worksheet import export_worksheet

VOCAB = [Topic(slug="misc", description="Noise.")]


def _item(*, links: int = 0, quoted: bool = False, article: bool = False) -> Item:
    sources = (
        [
            ContentSourceSuccess(
                kind="external_article",
                url="https://example.com/0",
                title="The piece",
                text="the fetched article body",
            )
        ]
        if article
        else []
    )
    return Item(
        id="1",
        source="bookmark",
        url="https://x.com/a/status/1",
        author=Author(handle="a", name="A"),
        text="a post",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url=f"https://example.com/{i}", domain="example.com") for i in range(links)],
        quoted_id="999" if quoted else None,
        content=(
            Content(fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc), sources=sources)
            if sources
            else None
        ),
    )


def _worksheet_entry(item: Item, tmp_path) -> dict:
    path = tmp_path / "ws.json"
    export_worksheet([item], VOCAB, path, "claude-code", "English")
    return json.loads(path.read_text(encoding="utf-8"))["items"][0]


# ------------------------------------------------- one contract, every surface


@pytest.mark.parametrize(
    ("links", "article"),
    [
        (1, False),  # nothing fetched
        (2, True),  # PARTIAL fetch — the note states the counts
    ],
)
def test_every_surface_carries_the_identical_unfetched_links_note(links, article, tmp_path):
    """The generator's two surfaces and the judge's source must carry the SAME note,
    verbatim. Any surface drifting to its own wording breaks the contract silently."""
    item = _item(links=links, article=article)
    note = unfetched_links_note(item)
    assert note is not None
    assert note in _user_prompt(item, VOCAB)  # generator: api prompt
    assert _worksheet_entry(item, tmp_path)["unfetched_links_note"] == note  # generator: worksheet
    assert note in _source_text(item)  # judge: verify source


def test_every_surface_carries_the_identical_quoted_note(tmp_path):
    """Same contract for the quoted post nobody fetches."""
    item = _item(quoted=True)
    assert QUOTED_CONTENT_UNFETCHED_NOTE in _user_prompt(item, VOCAB)
    assert _worksheet_entry(item, tmp_path)["quoted_content_note"] == QUOTED_CONTENT_UNFETCHED_NOTE
    assert QUOTED_CONTENT_UNFETCHED_NOTE in _source_text(item)


def test_the_judge_source_carries_the_note_not_merely_the_header():
    """The judge's source must carry the INSTRUCTION, not just the `[Links — content
    NOT fetched]` heading. Deleting the note while keeping the header is the mutation
    that survived review: the header alone satisfies a `"NOT fetched"` substring."""
    item = _item(links=1)
    text = _source_text(item)
    header = "[Links — content NOT fetched]"
    assert header in text
    assert text.replace(header, "").count("NOT fetched") >= 1  # the note survives header removal
    assert unfetched_links_note(item) in text


def test_the_judge_source_carries_the_quoted_note_not_merely_the_header():
    item = _item(quoted=True)
    text = _source_text(item)
    header = "[Quoted post — content NOT fetched]"
    assert header in text
    assert QUOTED_CONTENT_UNFETCHED_NOTE in text.replace(header, "")


# ------------------------------------------------- the note's semantic core


@pytest.mark.parametrize(
    "note_of",
    [
        lambda: unfetched_links_note(_item(links=1)),
        lambda: unfetched_links_note(_item(links=2, article=True)),
        lambda: QUOTED_CONTENT_UNFETCHED_NOTE,
    ],
)
def test_the_note_forbids_reconstructing_content_that_was_never_downloaded(note_of):
    """The note IS the guardrail. Gutting its instruction to a bare "NOT fetched." must
    not stay green — that empties the PR of its content while CI applauds."""
    note = note_of()
    assert note is not None
    assert "NOT fetched" in note  # the state
    for forbidden in ("describe", "reconstruct", "guess"):  # the instruction
        assert forbidden in note
    for basis in ("URL", "domain", "world knowledge"):  # …and what it may not be guessed from
        assert basis in note


def test_the_partial_fetch_note_states_the_counts():
    """A partial fetch has a `[Linked article]` block sitting right there; the note must
    say how many links it does NOT cover, or that block lends it false credibility."""
    note = unfetched_links_note(_item(links=2, article=True))
    assert note is not None
    assert "1 of 2" in note

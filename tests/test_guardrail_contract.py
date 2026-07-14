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
    fetched_link_sources,
    quoted_attribution,
    unfetched_links_note,
)
from xbrain.models import Author, Content, ContentSourceSuccess, Item, Link, Topic
from xbrain.verification import _source_text
from xbrain.worksheet import export_worksheet

VOCAB = [Topic(slug="misc", description="Noise.")]


QUOTED_SOURCE = ContentSourceSuccess(
    kind="quoted_tweet",
    url="https://x.com/karpathy/status/999",
    text="I am leaving OpenAI.",
    author=Author(handle="karpathy", name="Andrej Karpathy"),
)


def _blocks(source_text: str) -> dict[str, str]:
    """Split `_source_text` into `{[Label]: body}`.

    So a test can assert WHICH label a span sits under. A bare `assert body in text`
    would pass even if the quoted post were served under `[Linked article]` — the
    exact mislabelling that tells the judge a link was downloaded when none was.
    """
    blocks: dict[str, list[str]] = {}
    label: str | None = None
    for line in source_text.split("\n"):
        if line.startswith("[") and line.endswith("]"):
            label = line
            blocks.setdefault(label, [])
        elif label is not None:
            blocks[label].append(line)
    return {key: "\n".join(value) for key, value in blocks.items()}


def _item(
    *, links: int = 0, quoted: bool = False, article: bool = False, quoted_fetched: bool = False
) -> Item:
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
    if quoted_fetched:
        sources = [*sources, QUOTED_SOURCE]
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


# ------------------------------------------- the FETCHED quote: same one contract
#
# A fetched quote and an unfetched quote are DIFFERENT states and both must be
# represented. The unfetched marker (above) is damage control; these pin the fix:
# the quoted body + its author reach every surface, under ONE shared attribution.


def test_every_surface_carries_the_fetched_quoted_body_under_the_same_attribution(tmp_path):
    """The generator's two surfaces and the judge's source must name the quoted
    post's author IDENTICALLY — via the one shared builder, not three hand-written
    labels that can drift. Compared by REFERENCE to `quoted_attribution`."""
    item = _item(quoted=True, quoted_fetched=True)
    label = quoted_attribution(item)

    assert label == "Quoted post — @karpathy (Andrej Karpathy)"
    assert label in _user_prompt(item, VOCAB)  # generator: api prompt
    entry = _worksheet_entry(item, tmp_path)  # generator: worksheet
    assert entry["quoted_attribution"] == label
    assert entry["quoted_text"] == QUOTED_SOURCE.text
    assert f"[{label}]" in _source_text(item)  # judge: verify source


def test_the_fetched_quoted_body_reaches_every_surface(tmp_path):
    item = _item(quoted=True, quoted_fetched=True)
    body = QUOTED_SOURCE.text

    assert body in _user_prompt(item, VOCAB)
    assert _worksheet_entry(item, tmp_path)["quoted_text"] == body
    assert body in _source_text(item)


def test_a_fetched_quote_silences_the_unfetched_marker(tmp_path):
    """#86's marker must fire ONLY when the content genuinely is not there. Stamping
    `content NOT fetched` over a body we DID fetch would order the generator to
    ignore its best evidence."""
    item = _item(quoted=True, quoted_fetched=True)

    assert QUOTED_CONTENT_UNFETCHED_NOTE not in _user_prompt(item, VOCAB)
    assert _worksheet_entry(item, tmp_path)["quoted_content_note"] is None
    text = _source_text(item)
    assert QUOTED_CONTENT_UNFETCHED_NOTE not in text
    assert "[Quoted post — content NOT fetched]" not in text


def test_an_unfetched_quote_still_fires_the_marker_and_carries_no_body(tmp_path):
    """The other arm: #86's guardrail stays honest when the quote could not be
    fetched (deleted, protected, not hydrated)."""
    item = _item(quoted=True, quoted_fetched=False)

    assert QUOTED_CONTENT_UNFETCHED_NOTE in _user_prompt(item, VOCAB)
    entry = _worksheet_entry(item, tmp_path)
    assert entry["quoted_content_note"] == QUOTED_CONTENT_UNFETCHED_NOTE
    assert entry["quoted_text"] is None
    assert entry["quoted_attribution"] is None
    assert "[Quoted post — content NOT fetched]" in _source_text(item)


def test_the_quoted_body_sits_under_the_quoted_label_never_the_article_label():
    """Structural, not substring. With BOTH a fetched article and a fetched quote on
    one item, each body must sit under its OWN label. Serving the quoted post as a
    `[Linked article]` would tell the judge a link was downloaded and hand it text
    that is not that link's content."""
    item = _item(quoted=True, quoted_fetched=True, links=1, article=True)
    blocks = _blocks(_source_text(item))
    quoted_block = blocks[f"[{quoted_attribution(item)}]"]
    # The article label embeds the title (`[Linked article — The piece]`), so find it
    # by its stem rather than pinning a literal that a title change would break.
    article_label = next(key for key in blocks if key.startswith("[Linked article"))
    article_block = blocks[article_label]

    assert QUOTED_SOURCE.text in quoted_block
    assert QUOTED_SOURCE.text not in article_block
    assert "the fetched article body" in article_block
    assert "the fetched article body" not in quoted_block


def test_the_quoted_post_never_counts_as_a_fetched_LINK_body():
    """It is a third party's post, not the body of any link — it must not silence the
    unfetched-links guardrail."""
    item = _item(quoted=True, quoted_fetched=True, links=1)

    assert fetched_link_sources(item) == 0
    assert unfetched_links_note(item) is not None
    assert unfetched_links_note(item) in _source_text(item)


def test_the_attribution_degrades_when_the_quoted_author_is_unknown():
    """A quoted body whose author X did not hydrate still ships — under a label that
    names NOBODY, rather than silently inheriting the poster's identity."""
    item = _item(quoted=True)
    item.content = Content(
        fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        sources=[
            ContentSourceSuccess(
                kind="quoted_tweet", url="https://x.com/i/status/999", text="a body, no author"
            )
        ],
    )

    assert quoted_attribution(item) == "Quoted post"
    assert "@" not in quoted_attribution(item)

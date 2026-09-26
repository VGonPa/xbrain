# tests/test_jev_page_browser.py
"""`jev.html` driven in a real browser: what the reader SEES matches the numbers beside it.

The blob's filter keys and counts are pinned in tests/test_jev_dashboard.py; this file checks
the other half — that the page, running, shows for each view exactly the posts its number
counts, that a topic narrows to its number, that j/k/n/p land where they say, that pages of
fifty arrive, that the URL keeps the view (including characters that need escaping), and that
a late error does not paint "the page could not draw itself" over a page that did.

It drives headless Chrome with `--dump-dom`: a probe script appended to the rendered page
clicks and presses keys, then writes what it saw into a `<pre id="probe">` as JSON.

THE SKIP IS FAIL-CLOSED ON A RUNNER. A laptop without Chrome skips these tests; the gate job
sets `XBRAIN_REQUIRE_CHROME` (tests/test_ci_workflow.py pins it), and then a missing Chrome
FAILS every page test (`_need_chrome`) instead of skipping it, and
`test_chrome_is_available_where_the_page_tests_are_required` names the cause.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess  # nosec B404 - runs a local browser binary on a file we wrote
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.test_jev_dashboard import _assessment, _corpus, _data, _item
from xbrain.jev.dashboard import render_jev_dashboard_html
from xbrain.models import Author

_REQUIRED = os.environ.get("XBRAIN_REQUIRE_CHROME", "").strip() not in ("", "0", "false", "False")
_MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _chrome() -> str | None:
    """`XBRAIN_CHROME`, else the first Chrome/Chromium on PATH, else the macOS app bundle."""
    explicit = os.environ.get("XBRAIN_CHROME")
    if explicit:
        return explicit
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return _MAC_CHROME if sys.platform == "darwin" and Path(_MAC_CHROME).exists() else None


CHROME = _chrome()


def _need_chrome() -> None:
    """No Chrome: skip on a laptop, FAIL where the page tests are required (the gate). Every
    page test goes through here, so none of them can turn a missing browser into a green
    skip on a runner."""
    if CHROME is None:
        if _REQUIRED:
            pytest.fail("XBRAIN_REQUIRE_CHROME está puesto y no hay Chrome (pon XBRAIN_CHROME)")
        pytest.skip("sin Chrome")


_requires_chrome = pytest.mark.skipif(
    CHROME is None and not _REQUIRED,
    reason="sin Chrome: las pruebas de la página en navegador se saltan",
)


def test_chrome_is_available_where_the_page_tests_are_required():
    """Named apart from the probe tests: a missing binary there fails with a subprocess
    error, which says nothing about what went unchecked."""
    if _REQUIRED:
        assert CHROME is not None, (
            "XBRAIN_REQUIRE_CHROME está puesto y no hay Chrome: las pruebas que manejan "
            "jev.html en un navegador no se pueden correr (instala google-chrome o pon "
            "XBRAIN_CHROME)."
        )


#: Runs after the page has booted. Every step goes through the page's own controls (rail
#: buttons, keys, the search box) and reads the page's own list.
_PROBE = r"""
<script>
const sleep = ms => new Promise(r => setTimeout(r, ms));
const count = () => Number(document.getElementById('count').textContent.match(/mostrando ([\d.]+)/)[1].replace(/\./g, ''));
const num = (text) => Number(text.replace(/\./g, ''));
const key = k => document.dispatchEvent(new KeyboardEvent('keydown', {key: k, bubbles: true}));
const cur = () => { const c = document.querySelector('.card.cur'); return c && c.dataset.id; };
const railButtons = () => [...document.querySelectorAll('#rail > button.f')];
const topicButtons = () => [...document.querySelectorAll('#rail .tbox button.f')];
const drawAll = async () => { while (!document.getElementById('more').hidden) { document.getElementById('more').click(); await sleep(5); } };
(async () => {
  await sleep(100);
  const out = {view: {f: view.f, count: count()}, filters: {}, topics: {}, all_topics: {}};
  for (const b of railButtons()) {
    b.click(); await sleep(10);
    out.filters[b.firstChild.textContent] = {rail: num(b.lastChild.textContent), list: count(), ids: shown.map(p => p.id), hash: location.hash};
  }
  railButtons().find(b => b.firstChild.textContent === 'Con discrepancias').click(); await sleep(10);
  for (const b of topicButtons()) {
    const parts = b.lastChild.textContent.split(' · ').map(num);
    const name = b.firstChild.textContent;
    b.click(); await sleep(10);
    out.topics[name] = {disagreeing: parts[2], list: count(), ids: shown.map(p => p.id)};
    b.click(); await sleep(10);
  }
  out.one_way = {};
  for (const view_ of ['Enrich asigna y Jev no', 'Jev añadiría topic']) {
    railButtons().find(b => b.firstChild.textContent === view_).click(); await sleep(10);
    out.one_way[view_] = {};
    for (const b of topicButtons()) {
      b.click(); await sleep(10);
      out.one_way[view_][b.firstChild.textContent] = shown.map(p => p.id);
      b.click(); await sleep(10);
    }
  }
  railButtons().find(b => b.firstChild.textContent === 'Todos').click(); await sleep(10);
  for (const b of topicButtons()) {
    b.click(); await sleep(10);
    out.all_topics[b.firstChild.textContent] = shown.map(p => p.id);
    b.click(); await sleep(10);
  }
  // Todos: fifty cards, then fifty more once j walks past them.
  out.drawn_first = document.querySelectorAll('#cards .card').length;
  for (let i = 0; i < 55; i++) key('j');
  out.drawn_after = document.querySelectorAll('#cards .card').length;
  out.j_cursor = cur(); out.j_expected = shown[54].id;
  // Re-clicking the view redraws the list and resets the cursor (the same hash would not).
  railButtons().find(b => b.firstChild.textContent === 'Todos').click(); await sleep(10);
  key('n'); const n1 = cur(); key('n'); const n2 = cur(); key('p'); const p1 = cur();
  out.n = [n1, n2, p1];
  // Sin evaluar: which cards offer the command, which say there is no evidence.
  railButtons().find(b => b.firstChild.textContent === 'Sin evaluar por Jev').click(); await sleep(10);
  await drawAll();
  out.copy_ids = [...document.querySelectorAll('#cards .card')].filter(c => c.querySelector('.ask button')).map(c => c.dataset.id);
  out.no_evidence_ids = [...document.querySelectorAll('#cards .card')].filter(c => c.textContent.includes('sin evidencia')).map(c => c.dataset.id);
  out.commands = [...document.querySelectorAll('#cards .ask code')].map(c => c.textContent).slice(0, 2);
  // The URL keeps a search with characters that need escaping.
  const box = document.getElementById('search');
  box.value = 'R&D c++ 50% ¿qué?'; box.dispatchEvent(new Event('input')); await sleep(10);
  out.hash_after_search = location.hash;
  // One unreadable post costs one line; a late error does not repaint the boot banner.
  out.error_card = safeCard({id: 'roto'}).textContent;
  setTimeout(() => { throw new Error('tarde'); }, 0); await sleep(30);
  out.banner_hidden_after_late_error = document.getElementById('banner').hidden;
  const frag = linkified('ver https://e.com/a). fin');
  out.link = frag.querySelector('a').getAttribute('href');
  const wiki = linkified('(ver https://en.wikipedia.org/wiki/Foo_(bar)) fin');
  out.wiki = wiki.querySelector('a').getAttribute('href');
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>
"""

#: For a page opened AT a given hash: what the view became and whether it drew.
_READ_VIEW = r"""
<script>
setTimeout(() => {
  const pre = document.createElement('pre'); pre.id = 'probe';
  pre.textContent = JSON.stringify({q: view.q, f: view.f, search: document.getElementById('search').value,
    banner_hidden: document.getElementById('banner').hidden, count: document.getElementById('count').textContent,
    more_shown: getComputedStyle(document.getElementById('more')).display !== 'none'});
  document.body.appendChild(pre);
}, 200);
</script>
"""


#: The Topics tab, from its index: sort, open a topic, its groups, a pair, back and forward.
_TOPICS_PROBE = r"""
<script>
const sleep = ms => new Promise(r => setTimeout(r, ms));
const indexOrder = () => [...document.querySelectorAll('#topics-index tbody tr')].map(tr => tr.dataset.slug);
const sortHeader = name => [...document.querySelectorAll('#topics-index th .th')].find(b => b.firstChild.textContent.startsWith(name));
async function groups() {
  const found = {};
  for (const sec of document.querySelectorAll('#topic-detail section[data-group]')) {
    for (;;) { const more = sec.querySelector('button.more'); if (!more || more.hidden) break; more.click(); await sleep(5); }
    found[sec.dataset.group] = {shown: Number(sec.querySelector('h3').textContent.split(' · ')[1].replace(/\./g, '')),
      cards: [...sec.querySelectorAll('.card')].map(c => c.dataset.id)};
  }
  return found;
}
const tabState = () => ({hash: location.hash, index: !document.getElementById('topics-index').hidden,
  detail: !document.getElementById('topic-detail').hidden, pair: !!document.getElementById('topic-pair')});
(async () => {
  await sleep(100);
  const out = {};
  out.default_order = indexOrder();
  sortHeader('Discrepancias').click(); await sleep(10);
  out.by_disagreeing = indexOrder();
  out.sort_hash = location.hash;
  sortHeader('Discrepancias').click(); await sleep(10);
  out.by_disagreeing_asc = indexOrder();
  location.hash = '#topics'; await sleep(20);
  document.querySelector('#topics-index a.tl[href="#topics?t=ai-coding"]').click(); await sleep(30);
  out.detail = tabState();
  out.groups = await groups();
  out.pairs = [...document.querySelectorAll('#topic-detail a[data-pair]')].map(a => a.dataset.pair);
  const first = document.querySelector('#topic-detail a[data-pair]');
  first.click(); await sleep(30);
  out.pair_clicked = first.dataset.pair;
  out.pair_cards = [...document.querySelectorAll('#topic-pair .card')].map(c => c.dataset.id);
  out.with_pair = tabState();
  history.back(); await sleep(50); out.back1 = tabState();
  history.back(); await sleep(50); out.back2 = tabState();
  history.forward(); await sleep(50); out.forward = tabState();
  // From a card on the Posts tab, its topic row jumps to the topic's page.
  location.hash = '#posts?f=all'; await sleep(20);
  const row = document.querySelector('#cards .rows a.tlink');
  out.row_topic = row.getAttribute('href');
  row.click(); await sleep(30);
  out.from_card = tabState();
  // A second topic whose three groups have different sizes.
  location.hash = '#topics?t=startups'; await sleep(30);
  out.groups_startups = await groups();
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>
"""

#: A topics page opened at a given hash.
_READ_TOPICS = r"""
<script>
setTimeout(() => {
  const pre = document.createElement('pre'); pre.id = 'probe';
  pre.textContent = JSON.stringify({detail: !document.getElementById('topic-detail').hidden,
    title: (document.querySelector('#topic-detail .th2') || {}).textContent || null,
    pair_cards: [...document.querySelectorAll('#topic-pair .card')].map(c => c.dataset.id),
    banner_hidden: document.getElementById('banner').hidden});
  document.body.appendChild(pre);
}, 200);
</script>
"""


def _fixture() -> dict[str, Any]:
    """`_corpus`'s cases, plus 110 never-asked posts (so Todos needs a third page) and one
    post with no evidence at all."""
    items, assessments = _corpus()
    # Two more posts where Jev adds startups, so its groups differ in size (1 · 1 · 3).
    for extra in ("jev-only-2", "jev-only-3"):
        item = _item(extra, topics=("ai-coding",))
        items.append(item)
        assessments[extra] = _assessment(item, membership={"ai-coding": 0.9, "startups": 0.9})
    items += [_item(f"z{n:03d}") for n in range(110)]
    empty = _item("vacio", text=" ")
    empty.author = Author(handle="", name="")
    items.append(empty)
    return _data(items, assessments)


def _open(page: Path, hash_: str = "") -> dict[str, Any]:
    assert CHROME is not None
    result = subprocess.run(  # nosec B603 - fixed argv, a file this test wrote
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--virtual-time-budget=8000",
            "--dump-dom",
            page.as_uri() + hash_,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    found = re.search(r'<pre id="probe">(.*?)</pre>', result.stdout, re.S)
    assert found, f"la página no escribió el probe (rc={result.returncode}): {result.stderr[-800:]}"
    return json.loads(html.unescape(found.group(1)))


def _page(tmp_path: Path, data: dict[str, Any], script: str) -> Path:
    page = tmp_path / "jev.html"
    page.write_text(
        render_jev_dashboard_html(data).replace("</body>", script + "</body>"), encoding="utf-8"
    )
    return page


@pytest.fixture(scope="module")
def probed(tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any]]:
    _need_chrome()
    data = _fixture()
    return data, _open(_page(tmp_path_factory.mktemp("jev"), data, _PROBE))


def _ids(data: dict[str, Any], key: str) -> set[str]:
    return {post["id"] for post in data["posts"] if key == "all" or key in post["in"]}


_KEYS = {
    "Todos": "all",
    "Con discrepancias": "disc",
    "Enrich asigna y Jev no": "enrich_only",
    "Jev añadiría topic": "adds",
    "Primario distinto": "prim",
    "Jev eligió «otro»": "fallback",
    "Sin evaluar por Jev": "uneval",
}


@_requires_chrome
def test_the_page_opens_on_the_disagreements(probed):
    data, seen = probed

    assert seen["view"] == {"f": "disc", "count": data["summary"]["posts_with_disagreement"]}


@_requires_chrome
def test_each_view_lists_exactly_the_posts_its_number_counts(probed):
    data, seen = probed

    assert set(seen["filters"]) == set(_KEYS)
    for name, key in _KEYS.items():
        view = seen["filters"][name]
        assert view["rail"] == view["list"], name
        assert set(view["ids"]) == _ids(data, key), name
        assert view["hash"] == ("#posts" if key == "disc" else f"#posts?f={key}"), name


@_requires_chrome
def test_a_topic_under_con_discrepancias_lists_its_disagreeing_posts(probed):
    data, seen = probed
    rows = {row["slug"]: row for row in data["summary"]["per_topic"]}
    labels = {topic["label"]: topic["slug"] for topic in data["topics"]}

    assert seen["topics"]
    for name, view in seen["topics"].items():
        assert view["list"] == view["disagreeing"] == rows[labels[name]]["disagreeing"], name


@_requires_chrome
def test_a_topic_under_a_one_way_view_lists_that_direction_only(probed):
    """ "Enrich asigna y Jev no" + a topic = the posts whose row for it is solo enrich
    (`per_topic.doubtful`); "Jev añadiría topic" + a topic = solo Jev (`missing`)."""
    data, seen = probed
    labels = {topic["label"]: topic["slug"] for topic in data["topics"]}
    rows = {row["slug"]: row for row in data["summary"]["per_topic"]}

    def with_verdict(slug: str, verdict: str) -> set[str]:
        return {
            p["id"]
            for p in data["posts"]
            if p["jev"]
            and any(r["slug"] == slug and r["verdict"] == verdict for r in p["jev"]["topics"])
        }

    for view_, verdict, field in (
        ("Enrich asigna y Jev no", "solo_enrich", "doubtful"),
        ("Jev añadiría topic", "solo_jev", "missing"),
    ):
        for name, ids in seen["one_way"][view_].items():
            slug = labels[name]
            assert set(ids) == with_verdict(slug, verdict), (view_, name)
            assert len(ids) == rows[slug][field], (view_, name)


@_requires_chrome
def test_a_topic_under_todos_lists_every_post_either_side_puts_there(probed):
    data, seen = probed
    labels = {topic["label"]: topic["slug"] for topic in data["topics"]}

    for name, ids in seen["all_topics"].items():
        slug = labels[name]
        assert set(ids) == {p["id"] for p in data["posts"] if slug in p["slugs"]}, name


@_requires_chrome
def test_fifty_cards_at_a_time_and_j_walks_into_the_next_fifty(probed):
    _data_, seen = probed

    assert seen["drawn_first"] == 50
    assert seen["drawn_after"] == 100
    assert seen["j_cursor"] == seen["j_expected"]


@_requires_chrome
def test_n_and_p_land_only_on_posts_that_disagree(probed):
    data, seen = probed
    disagreeing = [post["id"] for post in data["posts"] if "disc" in post["in"]]

    assert seen["n"] == [disagreeing[0], disagreeing[1], disagreeing[0]]


@_requires_chrome
def test_unevaluated_and_stale_posts_offer_the_command_and_an_empty_one_says_why(probed):
    _data_, seen = probed

    assert {"stale", "never", "z000"} <= set(seen["copy_ids"])
    assert "vacio" not in seen["copy_ids"] and seen["no_evidence_ids"] == ["vacio"]
    assert all(cmd.startswith("xbrain jev topics --id ") for cmd in seen["commands"])


@_requires_chrome
def test_a_late_error_is_a_line_on_one_card_not_the_boot_banner(probed):
    _data_, seen = probed

    assert "no se pudo dibujar" in seen["error_card"]
    assert seen["banner_hidden_after_late_error"] is True


@_requires_chrome
def test_sentence_punctuation_after_a_url_stays_outside_the_link(probed):
    assert probed[1]["link"] == "https://e.com/a"


@_requires_chrome
def test_a_balanced_parenthesis_is_part_of_the_url(probed):
    """A Wikipedia URL ends in `(bar)`; inside a parenthesised aside, only the aside's
    closing `)` stays outside the link."""
    assert probed[1]["wiki"] == "https://en.wikipedia.org/wiki/Foo_(bar)"


@_requires_chrome
def test_a_search_with_escapable_characters_survives_a_reload(probed, tmp_path):
    """The hash the search wrote, opened fresh, restores the same search — `&`, `+`, `%`
    and `?` included."""
    data, seen = probed

    reopened = _open(_page(tmp_path, data, _READ_VIEW), seen["hash_after_search"])

    assert reopened["q"] == reopened["search"] == "R&D c++ 50% ¿qué?"
    assert reopened["f"] == "uneval" and reopened["banner_hidden"] is True


@_requires_chrome
@pytest.mark.parametrize("hash_", ["#posts?q=%E0", "#%E0%A", "#posts?q=a%"])
def test_a_malformed_hash_opens_the_page_instead_of_breaking_it(tmp_path, hash_):
    _need_chrome()
    item = _item("1")
    data = _data([item], {"1": _assessment(item)})

    seen = _open(_page(tmp_path, data, _READ_VIEW), hash_)

    assert seen["banner_hidden"] is True
    assert seen["count"].startswith("mostrando")
    # One post, all drawn: no "Mostrar 0 más" button left on screen.
    assert seen["more_shown"] is False


# --------------------------------------------------------------------------- the Topics tab


@pytest.fixture(scope="module")
def topics_probed(tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any]]:
    _need_chrome()
    data = _fixture()
    return data, _open(_page(tmp_path_factory.mktemp("jev-topics"), data, _TOPICS_PROBE), "#topics")


def _per_topic(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["slug"]: row for row in data["summary"]["per_topic"]}


@_requires_chrome
def test_the_topic_index_opens_worst_agreement_first_among_topics_with_enough_posts(topics_probed):
    """`per_topic`'s order puts startups first (0 % backed), but enrich put it on ONE post:
    under the five-post floor it goes to the end, and ai-coding leads."""
    data, seen = topics_probed
    rows = _per_topic(data)
    base = [row["slug"] for row in data["summary"]["per_topic"]]

    assert (
        base[0] == "startups" and rows["startups"]["assigned"] < 5 <= rows["ai-coding"]["assigned"]
    )
    assert seen["default_order"] == ["ai-coding", "startups"]


@_requires_chrome
def test_a_column_header_sorts_the_index_both_ways_and_the_url_keeps_it(topics_probed):
    data, seen = topics_probed
    rows = _per_topic(data)
    desc = sorted(rows, key=lambda s: (-rows[s]["disagreeing"], s))

    assert seen["by_disagreeing"] == desc
    assert seen["by_disagreeing_asc"] == sorted(rows, key=lambda s: (rows[s]["disagreeing"], s))
    assert seen["sort_hash"] == "#topics?o=disagreeing&d=desc"


@_requires_chrome
@pytest.mark.parametrize(
    ("slug", "key"), [("ai-coding", "groups"), ("startups", "groups_startups")]
)
def test_a_topics_groups_hold_exactly_the_posts_their_numbers_count(topics_probed, slug, key):
    """Coinciden / Solo enrich / Solo Jev: the number shown is `backed` / `doubtful` /
    `missing`, and the cards under it are exactly that many — the cards whose row for the
    topic carries that verdict."""
    data, seen = topics_probed
    row = _per_topic(data)[slug]

    assert seen["detail"] == {
        "hash": "#topics?t=ai-coding",
        "index": False,
        "detail": True,
        "pair": False,
    }
    if slug == "startups":
        assert len({row["backed"], row["doubtful"], row["missing"]}) == 3  # the mutant's net
    for verdict, field in (
        ("coinciden", "backed"),
        ("solo_enrich", "doubtful"),
        ("solo_jev", "missing"),
    ):
        expected = {
            p["id"]
            for p in data["posts"]
            if p["jev"]
            and p["jev"]["compared"]
            and any(r["slug"] == slug and r["verdict"] == verdict for r in p["jev"]["topics"])
        }
        group = seen[key].get(verdict)
        assert group is not None, verdict
        assert group["shown"] == row[field] == len(group["cards"]), verdict
        assert set(group["cards"]) == expected, verdict


@_requires_chrome
def test_a_confusion_pair_lists_exactly_its_posts(topics_probed):
    data, seen = topics_probed
    kind, key = seen["pair_clicked"].split(":", 1)
    enrich, jev = (None if side == "-" else side for side in key.split("~"))
    rows = data["summary"]["topic_confusion" if kind == "cx" else "primary_confusion"]
    [row] = [r for r in rows if (r["enrich"], r["jev"]) == (enrich, jev)]

    assert "ai-coding" in (enrich, jev)
    assert seen["pair_cards"] == row["ids"]
    assert seen["with_pair"]["pair"] is True


@_requires_chrome
def test_back_and_forward_walk_between_the_index_the_topic_and_the_pair(topics_probed):
    _data_, seen = topics_probed

    assert seen["back1"]["detail"] and not seen["back1"]["pair"]
    assert seen["back1"]["hash"] == "#topics?t=ai-coding"
    assert seen["back2"]["index"] and not seen["back2"]["detail"]
    assert seen["forward"]["hash"] == "#topics?t=ai-coding" and seen["forward"]["detail"]


@_requires_chrome
def test_a_topic_row_on_a_post_card_opens_that_topics_page(topics_probed):
    _data_, seen = topics_probed

    assert seen["row_topic"].startswith("#topics?t=")
    assert seen["from_card"]["hash"] == seen["row_topic"] and seen["from_card"]["detail"]


@_requires_chrome
@pytest.mark.parametrize(
    ("hash_", "title", "pair"),
    [
        ("#topics?t=startups", "Startups", False),
        ("#topics?t=startups&cx=startups~-", "Startups", True),
        ("#topics?t=no-existe", None, False),
        ("#topics?t=startups&cx=%E0~", "Startups", False),
    ],
)
def test_a_topics_url_opened_fresh_shows_that_view(tmp_path, hash_, title, pair):
    _need_chrome()
    data = _fixture()
    [row] = [
        r
        for r in data["summary"]["topic_confusion"]
        if (r["enrich"], r["jev"]) == ("startups", None)
    ]

    seen = _open(_page(tmp_path, data, _READ_TOPICS), hash_)

    assert seen["banner_hidden"] is True
    assert seen["detail"] is (title is not None)
    if title:
        assert seen["title"].startswith(title)
    assert seen["pair_cards"] == (row["ids"] if pair else [])

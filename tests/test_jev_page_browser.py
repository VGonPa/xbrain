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
from datetime import datetime, timezone

from xbrain.jev.dashboard import compute_jev_dashboard_data
from xbrain.jev.models import PrimaryChoice
from xbrain.models import Author, Topic

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

_TOPIC_VOCAB = [
    Topic(slug="alpha", description="Alfa: el topic grande."),
    Topic(slug="beta", description="Beta: justo en el suelo de cinco."),
    Topic(slug="gamma", description="Gamma: a medias."),
    Topic(slug="delta", description="Delta: pocos datos."),
    Topic(slug="omega", description="Omega: solo lo pone Jev."),
]


def _topics_fixture() -> dict[str, Any]:
    """Five topics that tell every ordering apart.

    alpha 30 assigned / 25 backed (83 %), beta EXACTLY 5 / 1 (20 %), gamma 10 / 5 (50 %),
    delta 2 / 0 (under the floor), omega never assigned but added by Jev on 25 posts. Default
    order (floor, then exact ratio) = beta, gamma, alpha, delta, omega; alphabetical and
    by-assigned orders differ from it. `-~omega` is a 22-post pair and alpha's Coinciden group
    has 25 posts (both past a page of 20); `gamma~omega` is a real A~B pair of 3 posts.
    """
    specs: list[tuple[str, tuple[str, ...], dict[str, float], str]] = []
    specs += [(f"a{n:02d}", ("alpha",), {"alpha": 0.9, "omega": 0.9}, "alpha") for n in range(22)]
    specs += [(f"a{n:02d}", ("alpha",), {"alpha": 0.9}, "alpha") for n in range(22, 25)]
    specs += [(f"b{n:02d}", ("alpha",), {"alpha": 0.2}, "alpha") for n in range(5)]
    specs += [(f"g{n:02d}", ("gamma",), {"gamma": 0.9}, "gamma") for n in range(5)]
    specs += [(f"g{n:02d}", ("gamma",), {"gamma": 0.2, "omega": 0.9}, "omega") for n in range(5, 8)]
    specs += [
        ("g08", ("gamma",), {"gamma": 0.2}, "otro"),
        ("g09", ("gamma",), {"gamma": 0.2}, "gamma"),
    ]
    specs += [("e00", ("beta",), {"beta": 0.9}, "beta")]
    specs += [(f"e{n:02d}", ("beta",), {"beta": 0.1}, "beta") for n in range(1, 5)]
    specs += [(f"d{n:02d}", ("delta",), {"delta": 0.1}, "delta") for n in range(2)]
    items, assessments = [], {}
    for item_id, topics, membership, choice in specs:
        item = _item(item_id, text=f"post {item_id}", topics=topics)
        full = {topic.slug: 0.05 for topic in _TOPIC_VOCAB} | membership
        items.append(item)
        # The contract hashes the state and the questions, not the answer: swapping in this
        # vocabulary's Choice keeps the record current.
        answer = _assessment(item, membership=full, vocab=_TOPIC_VOCAB)
        probabilities = {choice: 0.7} | ({} if choice == "otro" else {"otro": 0.1})
        primary = PrimaryChoice(choice=choice, confidence=0.7, probabilities=probabilities)
        assessments[item_id] = answer.model_copy(update={"primary": primary})
    return compute_jev_dashboard_data(
        items,
        assessments,
        _TOPIC_VOCAB,
        threshold=0.85,
        fallback="otro",
        char_limit=100_000,
        id2note={},
        updated="SEP 26, 2026",
        runs=[],
        now=datetime(2026, 9, 26, tzinfo=timezone.utc),
    )


#: The Topics tab, from its index. Every step records its own error, so one broken step names
#: itself instead of leaving the whole probe silent.
_TOPICS_PROBE = r"""
<script>
const sleep = ms => new Promise(r => setTimeout(r, ms));
const indexOrder = () => [...document.querySelectorAll('#topics-index tbody tr')].map(tr => tr.dataset.slug);
const sortHeader = name => [...document.querySelectorAll('#topics-index th .th')].find(b => b.firstChild.textContent.startsWith(name));
const tabState = () => ({hash: location.hash, index: !document.getElementById('topics-index').hidden,
  detail: !document.getElementById('topic-detail').hidden, pair: !!document.getElementById('topic-pair')});
async function drawAll(root) {
  for (;;) { const more = [...root.querySelectorAll('button.more')].find(b => !b.hidden); if (!more) return; more.click(); await sleep(5); }
}
(async () => {
  const out = {errors: []};
  const step = async (name, fn) => { try { await fn(); } catch (e) { out.errors.push(name + ': ' + e); } };
  await sleep(100);
  await step('default', async () => {
    out.default_order = indexOrder();
    out.few = [...document.querySelectorAll('#topics-index tbody tr')].filter(tr => tr.querySelector('.few')).map(tr => tr.dataset.slug);
    out.acuerdo_cells = Object.fromEntries([...document.querySelectorAll('#topics-index tbody tr')].map(tr => [tr.dataset.slug, tr.children[3].textContent]));
    out.footnote = [...document.querySelectorAll('#topics-index p.muted')].map(p => p.textContent).join(' ');
  });
  await step('sort', async () => {
    const before = history.length;
    sortHeader('Acuerdo').click(); await sleep(10);
    out.acuerdo_desc = indexOrder();
    sortHeader('Acuerdo').click(); await sleep(10);
    out.acuerdo_asc = indexOrder();
    sortHeader('Topic').click(); await sleep(10);
    out.label_arrow = sortHeader('Topic').firstChild.textContent;
    out.sort_hash = location.hash;
    out.history_grew = history.length - before;
  });
  await step('open topic', async () => {
    location.hash = '#topics'; await sleep(20);
    document.querySelector('#topics-index a.tl[href="#topics?t=alpha"]').click(); await sleep(30);
    out.detail = tabState();
    const coinciden = document.querySelector('#topic-detail section[data-group="coinciden"]');
    out.coinciden_first = coinciden.querySelectorAll('.card').length;
    await drawAll(coinciden);
    out.coinciden_all = coinciden.querySelectorAll('.card').length;
    out.coinciden_shown = Number(coinciden.querySelector('h3').textContent.split(' · ')[1]);
  });
  await step('open omega and its big pair', async () => {
    location.hash = '#topics?t=omega'; await sleep(30);
    out.omega_pairs = [...document.querySelectorAll('#topic-detail a[data-pair]')].map(a => a.dataset.pair);
    out.omega_empty = [...document.querySelectorAll('#topic-detail .pairs p.muted')].map(p => p.textContent);
    document.querySelector('#topic-detail a[data-pair="cx:-~omega"]').click(); await sleep(30);
    const pair = document.getElementById('topic-pair');
    const top = pair.getBoundingClientRect().top;
    out.pair_in_view = top >= 0 && top < innerHeight;
    out.big_pair_first = pair.querySelectorAll('.card').length;
    await drawAll(pair);
    out.big_pair_all = [...pair.querySelectorAll('.card')].map(c => c.dataset.id);
  });
  await step('a real A~B pair', async () => {
    location.hash = '#topics?t=gamma'; await sleep(30);
    document.querySelector('#topic-detail a[data-pair="cx:gamma~omega"]').click(); await sleep(30);
    out.ab_pair = [...document.querySelectorAll('#topic-pair .card')].map(c => c.dataset.id);
    out.ab_hash = location.hash;
  });
  await step('back and forward', async () => {
    history.back(); await sleep(50); out.back1 = tabState();
    history.back(); await sleep(50); out.back2 = tabState();
    history.forward(); await sleep(50); out.forward = tabState();
  });
  await step('ver en Posts', async () => {
    location.hash = '#topics?t=alpha'; await sleep(30);
    [...document.querySelectorAll('#topic-detail .tnav a')].find(a => a.textContent.startsWith('ver en Posts')).click(); await sleep(30);
    out.posts_hash = location.hash;
    out.posts_ids = shown.map(p => p.id);
  });
  await step('card row to topic', async () => {
    const row = document.querySelector('#cards .rows a.tlink');
    out.row_topic = row.getAttribute('href');
    row.click(); await sleep(30);
    out.from_card = tabState();
  });
  await step('pairs past eight', async () => {
    const rows = Array.from({length: 11}, (_, i) => ({enrich: 'alpha', jev: 'x' + i, posts: 2}));
    const list = pairsList('prueba', rows, 'jev', 'cx', '');
    document.body.appendChild(list);
    out.pairs_visible_before = [...list.querySelectorAll('li')].filter(li => !li.hidden).length;
    const more = list.querySelector('button.more');
    out.pairs_more = more.textContent;
    more.click();
    out.pairs_visible_after = [...list.querySelectorAll('li')].filter(li => !li.hidden).length;
    list.remove();
  });
  await step('scroll kept on return', async () => {
    const spacer = document.createElement('div'); spacer.style.height = '4000px'; document.body.appendChild(spacer);
    const tabTop = () => Math.round(document.getElementById('tab-topics').getBoundingClientRect().top);
    location.hash = '#topics?t=alpha'; await sleep(30);
    window.scrollTo(0, document.getElementById('tab-topics').offsetTop + 900); await sleep(10);
    document.querySelector('#topic-detail a[data-pair]').click(); await sleep(30);
    history.back(); await sleep(50);
    out.topic_tab_top_after_back = tabTop();
    location.hash = '#topics'; await sleep(30);
    window.scrollTo(0, document.getElementById('tab-topics').offsetTop + 600); await sleep(10);
    document.querySelector('#topics-index a.tl').click(); await sleep(30);
    out.entered_tab_top = tabTop();
    history.back(); await sleep(50);
    out.index_tab_top_after_back = tabTop();
  });
  await step('sort survives a topic', async () => {
    location.hash = '#topics'; await sleep(30);
    sortHeader('Discrepancias').click(); await sleep(10);
    document.querySelector('#topics-index a.tl').click(); await sleep(30);
    out.back_href = [...document.querySelectorAll('#topic-detail .tnav a')].find(a => a.textContent.startsWith('← todos')).getAttribute('href');
    out.tab_href = document.querySelector('.tabs a[data-tab="topics"]').getAttribute('href');
  });
  await step('missing post', async () => {
    const gone = DATA.post_sets.cx['gamma~omega'][0];
    DATA.posts = DATA.posts.filter(p => p.id !== gone);
    location.hash = '#topics?t=gamma&cx=gamma~omega'; await sleep(30);
    out.missing_note = (document.querySelector('#topic-pair .note') || {}).textContent || null;
  });
  await step('guard', async () => {
    DATA.summary.per_topic = null;
    location.hash = '#topics?t=beta'; await sleep(30);
    out.guard = document.getElementById('topic-detail').textContent;
    out.guard_banner_hidden = document.getElementById('banner').hidden;
  });
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
    notes: [...document.querySelectorAll('#topic-detail .note')].map(p => p.textContent),
    groups: document.querySelectorAll('#topic-detail section[data-group]').length,
    order: [...document.querySelectorAll('#topics-index tbody tr')].map(tr => tr.dataset.slug),
    arrow: ([...document.querySelectorAll('#topics-index th[aria-sort="ascending"], #topics-index th[aria-sort="descending"]')][0] || {}).textContent || null,
    pair_cards: [...document.querySelectorAll('#topic-pair .card')].map(c => c.dataset.id),
    banner_hidden: document.getElementById('banner').hidden});
  document.body.appendChild(pre);
}, 200);
</script>
"""


@pytest.fixture(scope="module")
def topics_probed(tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any]]:
    _need_chrome()
    data = _topics_fixture()
    seen = _open(_page(tmp_path_factory.mktemp("jev-topics"), data, _TOPICS_PROBE), "#topics")
    assert seen["errors"] == [], seen["errors"]
    return data, seen


def _per_topic(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["slug"]: row for row in data["summary"]["per_topic"]}


@_requires_chrome
def test_the_fixture_tells_the_orders_apart():
    data = _topics_fixture()
    rows = _per_topic(data)

    assert {s: (rows[s]["assigned"], rows[s]["backed"]) for s in rows} == {
        "alpha": (30, 25),
        "beta": (5, 1),
        "gamma": (10, 5),
        "delta": (2, 0),
        "omega": (0, 0),
    }
    assert data["summary"]["primary_fallback"] == 1


@_requires_chrome
def test_the_index_opens_worst_agreement_first_above_the_floor_then_the_rest(topics_probed):
    """Not alphabetical (alpha first), not by size (alpha first): beta (20 %, exactly 5
    posts) leads, delta (0 % but 2 posts) and never-assigned omega follow the floor."""
    _data_, seen = topics_probed

    assert seen["default_order"] == ["beta", "gamma", "alpha", "delta", "omega"]
    assert seen["few"] == ["delta", "omega"]
    assert seen["acuerdo_cells"]["omega"] == "—"


@_requires_chrome
def test_acuerdo_sorts_by_the_exact_order_and_keeps_unassigned_last_both_ways(topics_probed):
    _data_, seen = topics_probed

    assert seen["acuerdo_desc"] == ["alpha", "gamma", "beta", "delta", "omega"]
    assert seen["acuerdo_asc"] == ["delta", "beta", "gamma", "alpha", "omega"]


@_requires_chrome
def test_sorting_replaces_the_url_instead_of_adding_history(topics_probed):
    _data_, seen = topics_probed

    assert seen["history_grew"] == 0
    assert seen["sort_hash"] == "#topics?o=label&d=asc"
    assert seen["label_arrow"].endswith("↑")


@_requires_chrome
def test_the_other_fallback_is_named_under_the_index(topics_probed):
    _data_, seen = topics_probed

    assert "Jev eligió «otro»" in seen["footnote"] and "1 post" in seen["footnote"]


@_requires_chrome
def test_a_group_past_one_page_reaches_its_number_with_mostrar_mas(topics_probed):
    data, seen = topics_probed

    assert seen["detail"] == {
        "hash": "#topics?t=alpha",
        "index": False,
        "detail": True,
        "pair": False,
    }
    assert seen["coinciden_first"] == 20
    assert (
        seen["coinciden_all"]
        == seen["coinciden_shown"]
        == _per_topic(data)["alpha"]["backed"]
        == 25
    )


@_requires_chrome
def test_a_pair_opens_in_view_and_lists_exactly_its_posts_past_one_page(topics_probed):
    data, seen = topics_probed

    assert "cx:-~omega" in seen["omega_pairs"] and "cx:gamma~omega" in seen["omega_pairs"]
    assert seen["pair_in_view"] is True
    assert seen["big_pair_first"] == 20
    assert seen["big_pair_all"] == data["post_sets"]["cx"]["-~omega"]
    assert len(seen["big_pair_all"]) == 22


@_requires_chrome
def test_a_real_two_topic_pair_lists_exactly_its_posts(topics_probed):
    data, seen = topics_probed

    assert seen["ab_hash"] == "#topics?t=gamma&cx=gamma%7Eomega"
    assert seen["ab_pair"] == data["post_sets"]["cx"]["gamma~omega"] == ["g05", "g06", "g07"]


@_requires_chrome
def test_an_empty_confusion_side_says_what_the_row_says(topics_probed):
    """omega: enrich never puts it and never picks it as primary — the copy says exactly that,
    not "Jev confirms it whenever enrich puts it"."""
    _data_, seen = topics_probed

    assert "Enrich no lo pone en ningún post comparado." in seen["omega_empty"]
    assert "Enrich nunca lo elige como principal." in seen["omega_empty"]


@_requires_chrome
def test_back_and_forward_walk_between_the_topic_and_the_pair(topics_probed):
    _data_, seen = topics_probed

    assert seen["back1"]["hash"] == "#topics?t=gamma" and not seen["back1"]["pair"]
    assert seen["back2"]["hash"].startswith("#topics?t=omega")
    assert seen["forward"]["hash"] == "#topics?t=gamma" and seen["forward"]["detail"]


@_requires_chrome
def test_ver_en_posts_opens_every_post_with_the_topic(topics_probed):
    data, seen = topics_probed

    assert seen["posts_hash"] == "#posts?f=all&t=alpha"
    assert set(seen["posts_ids"]) == {p["id"] for p in data["posts"] if "alpha" in p["slugs"]}


@_requires_chrome
def test_a_topic_row_on_a_post_card_opens_that_topics_page(topics_probed):
    _data_, seen = topics_probed

    assert seen["row_topic"].startswith("#topics?t=")
    assert seen["from_card"]["hash"] == seen["row_topic"] and seen["from_card"]["detail"]


@_requires_chrome
def test_pairs_past_eight_open_with_ver_todos(topics_probed):
    _data_, seen = topics_probed

    assert seen["pairs_visible_before"] == 8
    assert seen["pairs_more"] == "ver todos (3 cruces más, 6 posts)"
    assert seen["pairs_visible_after"] == 11


@_requires_chrome
def test_returning_keeps_the_scroll_and_only_entering_a_topic_jumps_to_the_tab(topics_probed):
    """Back from a pair to its topic, and back from a topic to the index, keep where the
    reader was; opening a topic from the index starts at the tab's top."""
    _data_, seen = topics_probed

    assert abs(seen["entered_tab_top"]) <= 2
    assert abs(seen["topic_tab_top_after_back"]) > 50
    assert abs(seen["index_tab_top_after_back"]) > 50


@_requires_chrome
def test_the_sort_travels_with_the_back_link_and_the_tab_link(topics_probed):
    _data_, seen = topics_probed

    assert seen["back_href"] == "#topics?o=disagreeing&d=desc"
    assert seen["tab_href"] == "#topics?o=disagreeing&d=desc"


@_requires_chrome
def test_a_pair_whose_posts_are_missing_says_how_many(topics_probed):
    _data_, seen = topics_probed

    assert seen["missing_note"] == "1 de 3 posts no están en esta página."


@_requires_chrome
def test_a_topics_tab_that_fails_after_boot_says_so_in_the_tab(topics_probed):
    _data_, seen = topics_probed

    assert "La pestaña Topics no pudo dibujarse" in seen["guard"]
    assert seen["guard_banner_hidden"] is True


@_requires_chrome
@pytest.mark.parametrize(
    ("hash_", "expect"),
    [
        ("#topics?o=disagreeing&d=asc", {"order": ["delta", "beta", "alpha", "gamma", "omega"]}),
        ("#topics?o=label", {"order": ["alpha", "beta", "delta", "gamma", "omega"], "arrow": "↑"}),
        ("#topics?t=gamma&cx=gamma~omega", {"title": "Gamma", "pair_cards": ["g05", "g06", "g07"]}),
        (
            "#topics?t=gamma&cx=alpha~-",
            {"title": "Gamma", "note": "Ese cruce ya no existe", "groups": 3},
        ),
        ("#topics?t=no-existe", {"note": "no está en el vocabulario actual"}),
        ("#topics?t=gamma&cx=%E0~", {"title": "Gamma", "note": "Ese cruce ya no existe"}),
    ],
)
def test_a_topics_url_opened_fresh_shows_that_view(tmp_path, hash_, expect):
    _need_chrome()
    data = _topics_fixture()

    seen = _open(_page(tmp_path, data, _READ_TOPICS), hash_)

    assert seen["banner_hidden"] is True
    if "order" in expect:
        assert seen["order"] == expect["order"]
    if "arrow" in expect:
        assert (
            seen["arrow"].split(" ")[-1].startswith(expect["arrow"])
            or expect["arrow"] in seen["arrow"]
        )
    if "title" in expect:
        assert seen["title"].startswith(expect["title"])
    if "pair_cards" in expect:
        assert seen["pair_cards"] == expect["pair_cards"]
    if "note" in expect:
        assert any(expect["note"] in note for note in seen["notes"]), seen["notes"]
    if "groups" in expect:
        assert seen["groups"] == expect["groups"] and seen["pair_cards"] == []

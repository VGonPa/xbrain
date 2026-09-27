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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.test_jev_dashboard import (
    _assessment,
    _compare_fixture,
    _corpus,
    _data,
    _item,
    _settings,
)
from xbrain.jev.dashboard import render_jev_dashboard_html
from datetime import datetime, timezone

from xbrain.jev.dashboard import compute_jev_dashboard_data
from xbrain.jev.models import PrimaryChoice
from xbrain.models import Author, Topic

#: Every page test runs with no network: each host but localhost resolves to NOTFOUND, so a
#: test can never pass (or hang) on what X's embed or a font server sent — X's messages are
#: handed to the page by the probe instead (tests/test_jev_page_embed_browser.py).
_NO_NETWORK = "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1"

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


#: What a reader SEES: an element that is laid out (a `display:none` anywhere up its tree has
#: no client rects), not `visibility:hidden`, and not inside a closed `<details>` — Chrome
#: folds those with `content-visibility: hidden`, which keeps client rects, so only
#: `checkVisibility()` tells a folded question from an open one. A probe that reads the DOM without this
#: certifies content nobody can see — the 9b pair list and a hidden tab both passed that way.
_SEEN = r"""
const seen = e => !!e && e.getClientRects().length > 0 && getComputedStyle(e).visibility !== 'hidden' && e.checkVisibility();
const inView = e => { if (!seen(e)) return false; const r = e.getBoundingClientRect(); return r.height > 0 && r.top >= -2 && r.top < innerHeight; };
const txt = e => (seen(e) ? e.textContent : null);
const seenCards = root => [...root.querySelectorAll('.card')].filter(seen).map(c => c.dataset.id);
"""

#: Runs after the page has booted. Every step goes through the page's own controls (rail
#: buttons, keys, the search box) and reads the page's own list.
_PROBE = (
    "<script>"
    + _SEEN
    + r"""
const sleep = ms => new Promise(r => setTimeout(r, ms));
const count = () => Number(txt(document.getElementById('count')).match(/mostrando ([\d.]+)/)[1].replace(/\./g, ''));
const num = (text) => Number(text.replace(/\./g, ''));
const key = k => document.dispatchEvent(new KeyboardEvent('keydown', {key: k, bubbles: true}));
const cur = () => { const c = document.querySelector('.card.cur'); return c && c.dataset.id; };
const railButtons = () => [...document.querySelectorAll('#rail > button.f')].filter(seen);
const topicButtons = () => [...document.querySelectorAll('#rail .tbox button.f')];
const drawAll = async () => { while (seen(document.getElementById('more'))) { document.getElementById('more').click(); await sleep(5); } };
const cards = async () => { await drawAll(); return seenCards(document.getElementById('cards')); };
(async () => {
  await sleep(100);
  const out = {view: {f: view.f, count: count()}, filters: {}, topics: {}, all_topics: {}};
  for (const name of railButtons().map(b => b.firstChild.textContent)) {
    // The rail is redrawn on every click: read the number beside the button before pressing it.
    const b = railButtons().find(x => x.firstChild.textContent === name);
    const rail = num(txt(b.lastChild));
    b.click(); await sleep(10);
    out.filters[name] = {rail: rail, list: count(), ids: await cards(), hash: location.hash};
  }
  railButtons().find(b => b.firstChild.textContent === 'Con discrepancias').click(); await sleep(10);
  for (const b of topicButtons()) {
    // «confirma/pone» then the posts that disagree about it: read as the reader reads them.
    const c = b.lastChild;
    // (Read as text: on this narrow window the Topics list starts folded.)
    const rail = {text: c.textContent, ok: c.querySelector('.ok').textContent, d: c.querySelector('.dn').textContent};
    const name = b.firstChild.textContent;
    b.click(); await sleep(10);
    out.topics[name] = {disagreeing: num(rail.d), rail: rail, list: count(), ids: await cards()};
    b.click(); await sleep(10);
  }
  document.querySelector('#rail .tbox').open = true; await sleep(10);
  out.rail_head = txt(document.querySelector('#rail .tbox .thead'));
  out.rail_legend = [...document.querySelectorAll('#rail .tbox > .legend')].filter(seen).map(p => ({text: p.textContent, title: p.title}));
  out.one_way = {};
  for (const view_ of ['Enrich asigna y Jev no', 'Jev añadiría topic']) {
    railButtons().find(b => b.firstChild.textContent === view_).click(); await sleep(10);
    out.one_way[view_] = {};
    for (const b of topicButtons()) {
      b.click(); await sleep(10);
      out.one_way[view_][b.firstChild.textContent] = await cards();
      b.click(); await sleep(10);
    }
  }
  railButtons().find(b => b.firstChild.textContent === 'Todos').click(); await sleep(10);
  for (const b of topicButtons()) {
    b.click(); await sleep(10);
    out.all_topics[b.firstChild.textContent] = await cards();
    b.click(); await sleep(10);
  }
  // Todos: fifty cards, then fifty more once j walks past them.
  railButtons().find(b => b.firstChild.textContent === 'Todos').click(); await sleep(10);
  out.drawn_first = seenCards(document.getElementById('cards')).length;
  for (let i = 0; i < 55; i++) key('j');
  out.drawn_after = seenCards(document.getElementById('cards')).length;
  out.j_cursor = cur(); out.j_expected = shown[54].id;
  // Re-clicking the view redraws the list and resets the cursor (the same hash would not).
  railButtons().find(b => b.firstChild.textContent === 'Todos').click(); await sleep(10);
  key('n'); const n1 = cur(); key('n'); const n2 = cur(); key('p'); const p1 = cur();
  out.n = [n1, n2, p1];
  // Sin evaluar: which cards offer the command, which say there is no evidence.
  railButtons().find(b => b.firstChild.textContent === 'Sin evaluar por Jev').click(); await sleep(10);
  await drawAll();
  out.copy_ids = [...document.querySelectorAll('#cards .card')].filter(c => seen(c) && seen(c.querySelector('.ask button'))).map(c => c.dataset.id);
  out.no_evidence_ids = [...document.querySelectorAll('#cards .card')].filter(c => seen(c) && c.textContent.includes('sin evidencia')).map(c => c.dataset.id);
  // What a card says about Jev's review of its topics, and where: its head and its Jev block.
  out.status = Object.fromEntries(['stale', 'never'].map(id => {
    const c = document.getElementById('post-' + id);
    return [id, {text: c.textContent, head: txt(c.querySelector('.twh .st')), block: txt(c.querySelector('.jev .badge'))}];
  }));
  out.commands = [...document.querySelectorAll('#cards .ask code')].filter(seen).map(c => c.textContent).slice(0, 2);
  // The cost strip's first number in its three cases, drawn from the same data reworked.
  const saved = JSON.stringify(DATA.cost);
  const totalOf = () => { const n = document.querySelector('#kpis .kpi .num');
    return {num: txt(n), amber: n.classList.contains('accent'), ctx: txt(document.querySelector('#kpis .kpi .ctx')),
      note: txt(document.getElementById('before-log'))}; };
  DATA.cost.total = Object.assign({}, DATA.cost.total, {runs: 1, requests: 3, cost_usd: 0.0123, input_tokens: 900});
  DATA.cost.out_of_log = {assessments: 2, cost_usd: 0.0002, input_tokens: 10, input_tokens_unknown: 0, unpriced_providers: []};
  renderCost(); out.cost_logged = totalOf();
  DATA.cost.total = Object.assign({}, DATA.cost.total, {runs: 0, requests: 0, cost_usd: 0, input_tokens: 0});
  renderCost(); out.cost_unlogged = totalOf();
  DATA.cost.out_of_log = Object.assign({}, DATA.cost.out_of_log, {assessments: 0, cost_usd: 0});
  renderCost(); out.cost_nothing = totalOf();
  // Each figure once: with no logged pass and every saved answer current, the Total already is
  // what the current answers cost, so their tile does not say the number again.
  const storedCtx = () => txt(document.querySelectorAll('#kpis .kpi .ctx')[2]);
  DATA.cost.out_of_log = Object.assign({}, DATA.cost.out_of_log, {assessments: 2, cost_usd: 0.0769});
  DATA.cost.current = Object.assign({}, DATA.cost.current, {assessments: 2, cost_usd: 0.0769});
  renderCost(); out.cost_echo = {kpis: txt(document.getElementById('kpis')), stored: storedCtx()};
  DATA.cost.current = Object.assign({}, DATA.cost.current, {cost_usd: 0.05});
  renderCost(); out.cost_apart = {stored: storedCtx()};
  // One answer with no token count, in a logged total: the noun agrees with the number.
  DATA.cost.total = Object.assign({}, DATA.cost.total, {runs: 1, requests: 3, cost_usd: 0.0123, input_tokens: 900, input_tokens_unknown: 1});
  renderCost(); out.cost_one_unknown = totalOf();
  DATA.cost = JSON.parse(saved); renderCost();
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
)

#: For a page opened AT a given hash: what the view became and whether it drew.
_READ_VIEW = (
    "<script>"
    + _SEEN
    + r"""
setTimeout(() => {
  const pre = document.createElement('pre'); pre.id = 'probe';
  pre.textContent = JSON.stringify({q: view.q, f: view.f, search: document.getElementById('search').value,
    banner_hidden: document.getElementById('banner').hidden, count: txt(document.getElementById('count')),
    cards: seenCards(document.getElementById('cards')), more_shown: seen(document.getElementById('more'))});
  document.body.appendChild(pre);
}, 200);
</script>
"""
)


#: The page's tabs as a reader sees them: which top tab and which Revisar sub-tab are current,
#: the URL it settled on, and WHERE the cost strip and Preguntar's cost line are (their top tab),
#: each only when seen. `history.length` says a rewritten old route added no entry. Then, from
#: a topic's page, Preguntar and back through «Revisar topics»: the sub-tab's view is kept.
_ROUTE_PROBE = (
    "<script>"
    + _SEEN
    + r"""
const sleep = ms => new Promise(r => setTimeout(r, ms));
const cur = (id) => { const a = document.querySelector('#' + id + ' a[aria-current="page"]'); return seen(a) ? a.dataset.tab : null; };
const topOfEl = (e) => (seen(e) ? e.closest('.toptab').id : null);
const read = () => ({hash: location.hash, top: cur('toptabs'), sub: cur('subtabs'),
  cost: topOfEl(document.getElementById('cost')), numbers: topOfEl(document.getElementById('numbers')),
  ask_cost: topOfEl(document.getElementById('ask-cost')), ask_cost_text: txt(document.getElementById('ask-cost')),
  shown: [...document.querySelectorAll('.toptab')].filter(seen).map(e => e.id),
  history: history.length});
(async () => {
  await sleep(150);
  const out = {opened: read()};
  location.hash = '#revisar/posts?f=all&t=startups'; await sleep(60);
  document.querySelector('#toptabs a[data-tab="ask"]').click(); await sleep(60);
  out.on_ask = read();
  document.querySelector('#toptabs a[data-tab="revisar"]').click(); await sleep(60);
  out.back = read();
  // A bare «#revisar» after another sub-tab was shown returns to that sub-tab.
  location.hash = '#revisar/compare'; await sleep(60);
  location.hash = '#revisar'; await sleep(60);
  out.bare = read();
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>
"""
)


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
    return _dump(page.as_uri() + hash_)


def _dump(url: str, budget_ms: int = 8000, window: str = "") -> dict[str, Any]:
    """`window` ("1280,900") sets the window's size; left empty, Chrome's own default."""
    assert CHROME is not None
    result = subprocess.run(  # nosec B603 - fixed argv, a file or local server this test made
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            _NO_NETWORK,
            f"--virtual-time-budget={budget_ms}",
            *([f"--window-size={window}"] if window else []),
            "--dump-dom",
            url,
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


@pytest.fixture(scope="module")
def route_page(tmp_path_factory) -> Path:
    _need_chrome()
    return _page(tmp_path_factory.mktemp("routes"), _fixture(), _ROUTE_PROBE)


@pytest.fixture(scope="module")
def route_history(route_page) -> int:
    """`history.length` of a page opened at a current route: headless Chrome's own start."""
    return _open(route_page, "#revisar/compare")["opened"]["history"]


_ROUTES = {
    # opened at (a reload there) → the URL it settles on, the top tab, the sub-tab
    "": ("#revisar/posts", "revisar", "posts"),
    "#revisar/posts?f=all": ("#revisar/posts?f=all", "revisar", "posts"),
    "#revisar/topics?t=startups": ("#revisar/topics?t=startups", "revisar", "topics"),
    "#revisar/compare": ("#revisar/compare", "revisar", "compare"),
    "#ask": ("#ask", "ask", None),
    "#config": ("#config", "config", None),
    # Old bookmarks: the same view, the URL rewritten in place.
    "#posts?f=all": ("#revisar/posts?f=all", "revisar", "posts"),
    "#topics?t=startups": ("#revisar/topics?t=startups", "revisar", "topics"),
    "#compare": ("#revisar/compare", "revisar", "compare"),
    "#revisar": ("#revisar/posts", "revisar", "posts"),
}


@_requires_chrome
@pytest.mark.parametrize("opened", list(_ROUTES))
def test_three_top_tabs_and_their_sub_tabs_route_and_survive_a_reload(
    route_page, route_history, opened
):
    seen = _open(route_page, opened)["opened"]
    hash_, top, sub = _ROUTES[opened]

    assert (seen["hash"], seen["top"], seen["sub"]) == (hash_, top, sub)
    assert seen["shown"] == [f"top-{top}"]
    # An old route is rewritten in place: no history entry more than a current route opens with.
    assert seen["history"] == route_history
    # The cost strip and the three numbers live in Revisar only; Preguntar has its own line.
    in_revisar = "top-revisar" if top == "revisar" else None
    assert (seen["cost"], seen["numbers"]) == (in_revisar, in_revisar)
    assert seen["ask_cost"] == ("top-ask" if top == "ask" else None)


@_requires_chrome
def test_leaving_revisar_and_coming_back_through_its_tab_keeps_the_sub_tabs_view(route_page):
    seen = _open(route_page, "#config")

    assert (seen["on_ask"]["top"], seen["on_ask"]["hash"]) == ("ask", "#ask")
    assert seen["on_ask"]["ask_cost_text"] == "Aún no se ha pagado ninguna pregunta."
    assert (seen["back"]["top"], seen["back"]["sub"]) == ("revisar", "posts")
    assert seen["back"]["hash"] == "#revisar/posts?f=all&t=startups"
    # A bare «#revisar» goes back to the sub-tab last shown, not always to Posts.
    assert (seen["bare"]["top"], seen["bare"]["sub"]) == ("revisar", "compare")
    assert seen["bare"]["hash"].startswith("#revisar/compare")


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
        assert view["hash"] == ("#revisar/posts" if key == "disc" else f"#revisar/posts?f={key}"), (
            name
        )


@_requires_chrome
def test_a_topic_under_con_discrepancias_lists_its_disagreeing_posts(probed):
    data, seen = probed
    rows = {row["slug"]: row for row in data["summary"]["per_topic"]}
    labels = {topic["label"]: topic["slug"] for topic in data["topics"]}

    assert seen["topics"]
    for name, view in seen["topics"].items():
        assert view["list"] == view["disagreeing"] == rows[labels[name]]["disagreeing"], name


@_requires_chrome
def test_each_rail_topic_reads_confirmed_of_placed_then_its_disagreements(probed):
    data, seen = probed
    rows = {r["slug"]: r for r in data["summary"]["per_topic"]}
    labels = {t["slug"]: t["label"] for t in data["topics"]}

    for slug, label in labels.items():
        r = rows.get(slug, {"assigned": 0, "backed": 0, "disagreeing": 0})
        rail = seen["topics"][label]["rail"]
        # Confirmed of placed — never a bigger third number that reads as a count gone wrong.
        assert rail["ok"] == f"{r['backed']}/{r['assigned']}", label
        assert rail["d"] == str(r["disagreeing"]), label
        assert "·" not in rail["text"], label
    assert seen["rail_head"] == "confirma/pone · discrepan"
    [legend] = seen["rail_legend"]
    assert legend["text"] == "Solo los posts comparados, con el filtro de arriba."
    assert "posts que discrepan sobre él" in legend["title"]


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
def test_the_cost_strip_opens_on_a_number_and_an_amber_dash_never(probed):
    _data_, seen = probed
    logged, unlogged, nothing = seen["cost_logged"], seen["cost_unlogged"], seen["cost_nothing"]

    # Logged passes: their total, and the answers out of the log said apart, not in it.
    assert logged["num"] == "~0,0123 $" and logged["amber"] is True
    assert logged["note"].startswith("2 evaluaciones fuera del registro de pasadas (~0,0002 $")
    assert logged["note"].endswith("No están en el total.")
    # None logged: what the saved answers cost is the total, named as an estimate (no note).
    assert unlogged["num"] == "~0,0002 $" and unlogged["amber"] is True
    assert unlogged["ctx"].startswith(
        "sin pasadas registradas: lo que costaron las 2 evaluaciones guardadas, estimado"
    )
    assert unlogged["note"] is None
    # Nothing paid at all: «—», in ink, not in the colour of money.
    assert nothing["num"] == "—" and nothing["amber"] is False
    assert nothing["ctx"] == "sin pasadas registradas: aún no se ha pagado nada"


@_requires_chrome
def test_the_cost_strip_says_each_figure_once(probed):
    _data_, seen = probed

    # No logged pass, every saved answer current: 0,0769 is said once, by the Total.
    echo = seen["cost_echo"]
    assert echo["kpis"].count("0,0769") == 1
    assert echo["stored"].startswith("lo que costaron: el Total · ")
    # Current answers that cost something else: their own figure.
    assert seen["cost_apart"]["stored"].startswith("lo que costaron: ~0,0500 $ · ")
    # «+1 respuesta», as the Preguntar line says it.
    assert "(+1 respuesta sin recuento)" in seen["cost_one_unknown"]["ctx"]


@_requires_chrome
def test_a_card_says_once_that_jev_has_not_reviewed_its_topics(probed):
    _data_, seen = probed

    # Once per card, in the Jev block it qualifies; the head does not repeat it. The words say
    # what was not done — the topics were not reviewed — so they cannot contradict an ask's score.
    never, stale = seen["status"]["never"], seen["status"]["stale"]
    assert never["block"] == "topics sin revisar por Jev" and never["head"] is None
    assert never["text"].count("sin revisar por Jev") == 1
    assert stale["block"] == "topics revisados con datos antiguos" and stale["head"] is None
    assert "sin revisar por Jev" not in stale["text"] and "caducada" not in stale["text"]


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
        settings=_settings(model="jev-latest", concurrency=8),
        id2note={},
        updated="2026-09-26",
        runs=[],
        now=datetime(2026, 9, 26, tzinfo=timezone.utc),
    )


#: The Topics tab, from its index. Every step records its own error, so one broken step names
#: itself instead of leaving the whole probe silent.
_TOPICS_PROBE = (
    "<script>"
    + _SEEN
    + r"""
const sleep = ms => new Promise(r => setTimeout(r, ms));
const indexOrder = () => [...document.querySelectorAll('#topics-index tbody tr')].filter(seen).map(tr => tr.dataset.slug);
const sortHeader = name => [...document.querySelectorAll('#topics-index th .th')].find(b => b.firstChild.textContent.startsWith(name));
const tabState = () => ({hash: location.hash, index: seen(document.getElementById('topics-index')),
  detail: seen(document.getElementById('topic-detail')), pair: seen(document.getElementById('topic-pair'))});
async function drawAll(root) {
  for (;;) { const more = [...root.querySelectorAll('button.more')].find(seen); if (!more) return; more.click(); await sleep(5); }
}
// Registered AFTER the page's own listener, so when it counts a change the page has drawn it.
let hashChanges = 0;
addEventListener('hashchange', () => { hashChanges++; });
/** Go back or forward and wait for the page to have handled it — never a fixed sleep. */
async function travel(go) {
  const before = hashChanges;
  go();
  for (let t = 0; hashChanges === before && t < 300; t++) await sleep(10);
  if (hashChanges === before) throw new Error('no hashchange after history navigation');
  await sleep(0);
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
    document.querySelector('#topics-index a.tl[href="#revisar/topics?t=alpha"]').click(); await sleep(30);
    out.detail = tabState();
    const coinciden = document.querySelector('#topic-detail section[data-group="coinciden"]');
    out.coinciden_first = seenCards(coinciden).length;
    await drawAll(coinciden);
    out.coinciden_all = seenCards(coinciden).length;
    out.coinciden_shown = Number(txt(coinciden.querySelector('h3')).split(' · ')[1]);
  });
  await step('open omega and its big pair', async () => {
    location.hash = '#topics?t=omega'; await sleep(30);
    out.omega_pairs = [...document.querySelectorAll('#topic-detail a[data-pair]')].filter(seen).map(a => a.dataset.pair);
    out.omega_empty = [...document.querySelectorAll('#topic-detail .pairs p.muted')].map(p => p.textContent);
    document.querySelector('#topic-detail a[data-pair="cx:-~omega"]').click(); await sleep(30);
    const pair = document.getElementById('topic-pair');
    out.pair_in_view = inView(pair);
    out.big_pair_first = seenCards(pair).length;
    await drawAll(pair);
    out.big_pair_all = seenCards(pair);
  });
  await step('a real A~B pair', async () => {
    location.hash = '#topics?t=gamma'; await sleep(30);
    document.querySelector('#topic-detail a[data-pair="cx:gamma~omega"]').click(); await sleep(30);
    out.ab_pair = seenCards(document.getElementById('topic-pair'));
    out.ab_hash = location.hash;
  });
  await step('back and forward', async () => {
    await travel(() => history.back()); out.back1 = tabState();
    await travel(() => history.back()); out.back2 = tabState();
    await travel(() => history.forward()); out.forward = tabState();
  });
  await step('ver en Posts', async () => {
    location.hash = '#topics?t=alpha'; await sleep(30);
    [...document.querySelectorAll('#topic-detail .tnav a')].find(a => a.textContent.startsWith('ver en Posts')).click(); await sleep(30);
    out.posts_hash = location.hash;
    while (seen(document.getElementById('more'))) { document.getElementById('more').click(); await sleep(5); }
    out.posts_ids = seenCards(document.getElementById('cards'));
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
    document.getElementById('topic-detail').appendChild(list);
    // What the reader SEES: a CSS `display` on the rows would override the `hidden` attribute.
    out.pairs_visible_before = [...list.querySelectorAll('li')].filter(seen).length;
    const more = list.querySelector('button.more');
    out.pairs_more = more.textContent;
    more.click();
    out.pairs_visible_after = [...list.querySelectorAll('li')].filter(seen).length;
    list.remove();
  });
  await step('scroll kept on return', async () => {
    const spacer = document.createElement('div'); spacer.style.height = '4000px'; document.body.appendChild(spacer);
    const tabTop = () => Math.round(document.getElementById('tab-topics').getBoundingClientRect().top);
    location.hash = '#topics?t=alpha'; await sleep(30);
    window.scrollTo(0, document.getElementById('tab-topics').offsetTop + 900); await sleep(10);
    document.querySelector('#topic-detail a[data-pair]').click(); await sleep(30);
    await travel(() => history.back());
    out.topic_tab_top_after_back = tabTop();
    location.hash = '#topics'; await sleep(30);
    window.scrollTo(0, document.getElementById('tab-topics').offsetTop + 600); await sleep(10);
    document.querySelector('#topics-index a.tl').click(); await sleep(30);
    out.entered_tab_top = tabTop();
    await travel(() => history.back());
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
    out.missing_note = txt(document.querySelector('#topic-pair .note'));
  });
  await step('guard', async () => {
    DATA.summary.per_topic = null;
    location.hash = '#topics?t=beta'; await sleep(30);
    out.guard = txt(document.getElementById('topic-detail'));
    out.guard_banner_hidden = document.getElementById('banner').hidden;
  });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>
"""
)

#: A topics page opened at a given hash.
_READ_TOPICS = (
    "<script>"
    + _SEEN
    + r"""
setTimeout(() => {
  const pre = document.createElement('pre'); pre.id = 'probe';
  const pair = document.getElementById('topic-pair');
  pre.textContent = JSON.stringify({detail: seen(document.getElementById('topic-detail')),
    title: txt(document.querySelector('#topic-detail .th2')),
    notes: [...document.querySelectorAll('#topic-detail .note')].filter(seen).map(p => p.textContent),
    groups: [...document.querySelectorAll('#topic-detail section[data-group]')].filter(seen).length,
    order: [...document.querySelectorAll('#topics-index tbody tr')].filter(seen).map(tr => tr.dataset.slug),
    arrow: txt([...document.querySelectorAll('#topics-index th[aria-sort="ascending"], #topics-index th[aria-sort="descending"]')][0]),
    pair_cards: pair ? seenCards(pair) : [],
    banner_hidden: document.getElementById('banner').hidden});
  document.body.appendChild(pre);
}, 200);
</script>
"""
)


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
    assert seen["sort_hash"] == "#revisar/topics?o=label&d=asc"
    assert seen["label_arrow"].endswith("↑")


@_requires_chrome
def test_the_other_fallback_is_named_under_the_index(topics_probed):
    _data_, seen = topics_probed

    assert "Jev eligió «otro»" in seen["footnote"] and "1 post" in seen["footnote"]


@_requires_chrome
def test_a_group_past_one_page_reaches_its_number_with_mostrar_mas(topics_probed):
    data, seen = topics_probed

    assert seen["detail"] == {
        "hash": "#revisar/topics?t=alpha",
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

    assert seen["ab_hash"] == "#revisar/topics?t=gamma&cx=gamma%7Eomega"
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

    assert seen["back1"]["hash"] == "#revisar/topics?t=gamma" and not seen["back1"]["pair"]
    assert seen["back2"]["hash"].startswith("#revisar/topics?t=omega")
    assert seen["forward"]["hash"] == "#revisar/topics?t=gamma" and seen["forward"]["detail"]


@_requires_chrome
def test_ver_en_posts_opens_every_post_with_the_topic(topics_probed):
    data, seen = topics_probed

    assert seen["posts_hash"] == "#revisar/posts?f=all&t=alpha"
    assert set(seen["posts_ids"]) == {p["id"] for p in data["posts"] if "alpha" in p["slugs"]}


@_requires_chrome
def test_a_topic_row_on_a_post_card_opens_that_topics_page(topics_probed):
    _data_, seen = topics_probed

    assert seen["row_topic"].startswith("#revisar/topics?t=")
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

    assert seen["back_href"] == "#revisar/topics?o=disagreeing&d=desc"
    assert seen["tab_href"] == "#revisar/topics?o=disagreeing&d=desc"


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


# --------------------------------------------------------------------------- the Comparar tab

_COMPARE_PROBE = (
    "<script>"
    + _SEEN
    + r"""
const sleep = ms => new Promise(r => setTimeout(r, ms));
const cnum = t => Number(String(t).match(/[\d.]+/)[0].replace(/\./g, ''));
const sec = key => document.querySelector('#tab-compare [data-sec="' + key + '"]');
async function drawAll(root) {
  for (;;) { const more = [...root.querySelectorAll('button.more')].find(b => seen(b) && b.textContent.startsWith('Mostrar')); if (!more) return; more.click(); await sleep(5); }
}
async function postsList() { await drawAll(document.getElementById('cards')); return {count: txt(document.getElementById('count')), ids: seenCards(document.getElementById('cards'))}; }
async function openList(a) {
  a.click(); await sleep(30);
  const list = document.getElementById('compare-list');
  if (!seen(list)) return {hash: location.hash, list: false};
  const in_view = inView(list);
  const first = seenCards(list).length;
  await drawAll(list);
  return {hash: location.hash, head: txt(list.querySelector('h3')), first: first, ids: seenCards(list), in_view: in_view,
    under: list.previousElementSibling && list.previousElementSibling.dataset.sec,
    notes: [...list.querySelectorAll('.note')].filter(seen).map(p => p.textContent)};
}
(async () => {
  const out = {errors: [], kinds: {}, sentences: {}, bands: {}, px: {}, pd: {}};
  const step = async (name, fn) => { try { await fn(); } catch (e) { out.errors.push(name + ': ' + e); } };
  await sleep(100);
  const home = async () => { location.hash = '#compare'; await sleep(30); };
  await step('shell', async () => {
    out.tab_seen = seen(document.getElementById('tab-compare'));
    out.sections = [...document.querySelectorAll('#tab-compare [data-sec]')].filter(seen).map(s => s.dataset.sec);
  });
  await step('kinds', async () => {
    for (const k of ['enrich_only', 'adds', 'prim']) {
      await home();
      const card = document.querySelector('#tab-compare [data-kind="' + k + '"]');
      const shownCount = txt(card.querySelector('.cnum'));
      const why = txt(card.querySelector('p'));
      const a = card.querySelector('a');
      const link = txt(a);
      a.click(); await sleep(30);
      out.kinds[k] = Object.assign({shown: shownCount, why: why, link: link, hash: location.hash}, await postsList());
    }
  });
  await step('sentences', async () => {
    // The three facts have one home, «Lo que dice Jev», above every Revisar sub-tab; Comparar
    // does not say them a second time.
    await home();
    out.headings = [...document.querySelectorAll('#top-revisar h2, #top-revisar h3')].filter(seen).map(h => h.textContent);
    for (const go of ['enrich_only', 'adds', 'prim']) {
      await home();
      const box = [...document.querySelectorAll('#numbers .fact')].find(b => b.querySelector('a[data-go="' + go + '"]'));
      const a = box.querySelector('a[data-go]');
      const read = {big: txt(box.querySelector('.big')), why: txt(box.querySelector('.why')), link: txt(a), href: a.getAttribute('href')};
      a.click(); await sleep(30);
      out.sentences[go] = Object.assign(read, {opened: location.hash}, await postsList());
    }
  });
  await step('topics table', async () => {
    await home();
    out.topic_rows = [...sec('topics').querySelectorAll('tbody tr')].filter(seen).map(tr => tr.dataset.slug);
    out.topic_more = txt(sec('topics').querySelector('a.all'));
    out.gamma_cells = [...sec('topics').querySelectorAll('tr[data-slug="gamma"] td')].map(txt);
    out.f01_cells = [...sec('topics').querySelectorAll('tr[data-slug="f01"] td')].map(txt);
    const disc = sec('topics').querySelector('tr[data-slug="gamma"] a.disc');
    disc.click(); await sleep(30);
    out.gamma_disc = Object.assign({hash: location.hash}, await postsList());
  });
  await step('bands', async () => {
    await home();
    out.band_heads = [...sec('bands').querySelectorAll('h4')].map(txt);
    out.band_rows = Object.fromEntries([...sec('bands').querySelectorAll('[data-band-row]')].map(r =>
      [r.dataset.bandRow, {label: txt(r.querySelector('b')), range: txt(r.querySelector('.range')), pairs: txt(r.querySelector('.pairs-n')), link: txt(r.querySelector('a[data-band]'))}]));
    for (const key of Object.keys(out.band_rows)) {
      await home();
      out.bands[key] = await openList(sec('bands').querySelector('a[data-band="' + key + '"]'));
    }
  });
  await step('px', async () => {
    await home();
    const rows = () => [...sec('cross').querySelectorAll('a[data-px]')];
    out.px_order = rows().filter(seen).map(a => a.dataset.px);
    for (const key of out.px_order) {
      await home();
      const a = rows().find(x => x.dataset.px === key);
      out.px[key] = Object.assign({name: txt(a), count: txt(a.parentNode.querySelector('.c'))}, await openList(a));
    }
  });
  await step('pd', async () => {
    await home();
    const rows = () => [...sec('cross').querySelectorAll('a[data-pd]')];
    out.pd_order = rows().filter(seen).map(a => a.dataset.pd);
    for (const key of out.pd_order) {
      await home();
      const a = rows().find(x => x.dataset.pd === key);
      out.pd[key] = Object.assign({name: txt(a), count: txt(a.parentNode.querySelector('.c'))}, await openList(a));
    }
    await home();
    out.agree = txt(sec('cross').querySelector('.agree'));
  });
  await step('otro', async () => {
    await home();
    out.otro_count = txt(sec('otro').querySelector('.cnum'));
    out.otro_rows = [...sec('otro').querySelectorAll('a[data-px]')].filter(seen).map(a => [a.dataset.px, a.textContent, txt(a.parentNode.querySelector('.c'))]);
    out.otro_open = await openList(sec('otro').querySelector('a[data-px]'));
    await home();
    sec('otro').querySelector('a.go').click(); await sleep(30);
    out.otro_posts = Object.assign({hash: location.hash}, await postsList());
  });
  await step('lists past ten', async () => {
    await home();
    const rows = Array.from({length: 12}, (_, i) => ({name: 'x' + i, posts: 2, href: '#compare', data: {}}));
    const box = linkList('prueba', rows, COMPARE_PAIRS_SHOWN, ['topic más', 'topics más']);
    document.getElementById('compare-body').appendChild(box);
    const visible = () => [...box.querySelectorAll('li')].filter(seen).length;
    out.list_visible_before = visible();
    out.list_more = txt(box.querySelector('button.more'));
    box.querySelector('button.more').click();
    out.list_visible_after = visible();
    box.remove();
  });
  await step('round trips', async () => {
    out.round = {};
    for (const h of ['#compare?px=gamma~omega', '#compare?pd=delta', '#compare?b=e-lo']) {
      location.hash = h; await sleep(30);
      location.hash = '#posts?f=adds'; await sleep(30);
      document.querySelector('.tabs a[data-tab="compare"]').click(); await sleep(30);
      out.round[h] = {hash: location.hash, ids: seenCards(document.getElementById('compare-list') || document.body)};
    }
    out.posts_tab_href = document.querySelector('.tabs a[data-tab="posts"]').getAttribute('href');
    out.posts_f_kept = view.f;
  });
  await step('resolution order', async () => {
    location.hash = '#compare?px=gamma~omega&pd=alpha&b=e-near'; await sleep(30);
    out.order_ids = seenCards(document.getElementById('compare-list'));
    out.order_tab_href = document.querySelector('.tabs a[data-tab="compare"]').getAttribute('href');
  });
  await step('same list keeps the scroll', async () => {
    location.hash = '#compare?b=e-mid'; await sleep(30);
    window.scrollTo(0, 0); await sleep(10);
    location.hash = '#compare?b=e-mid&z=1'; await sleep(30);
    out.same_list_scroll = window.scrollY;
  });
  await step('clear', async () => {
    location.hash = '#compare?b=e-lo'; await sleep(30);
    document.querySelector('#compare-list a.clear').click(); await sleep(30);
    out.cleared = {hash: location.hash, list: seen(document.getElementById('compare-list'))};
  });
  await step('stale', async () => {
    location.hash = '#compare?b=no-existe'; await sleep(30);
    out.stale_notes = [...document.querySelectorAll('#tab-compare .note')].filter(seen).map(p => p.textContent);
  });
  await step('count differs', async () => {
    DATA.summary.confidence_bands.find(b => b.key === 'j-hi').posts = 99;
    location.hash = '#compare?b=j-hi'; await sleep(30);
    const list = document.getElementById('compare-list');
    out.differs = {head: txt(list.querySelector('h3')), notes: [...list.querySelectorAll('.note')].filter(seen).map(p => p.textContent)};
  });
  await step('missing post', async () => {
    const gone = DATA.post_sets.bands['e-mid'][0];
    DATA.posts = DATA.posts.filter(p => p.id !== gone);
    location.hash = '#compare?b=e-mid'; await sleep(30);
    out.missing_note = [...document.querySelectorAll('#compare-list .note')].filter(seen).map(p => p.textContent);
  });
  await step('loadData', async () => {
    location.hash = '#posts'; await sleep(20);
    loadData(Object.assign({}, DATA, {threshold: 0.875, fallback: 'nada'}));
    location.hash = '#compare'; await sleep(30);
    out.after_load = {range: txt(document.querySelector('#tab-compare [data-band-row="e-lo"] .range')),
      filter: FILTERS.find(f => f.key === 'fallback').name, otro_title: txt(sec('otro').querySelector('h3'))};
  });
  await step('guard', async () => {
    location.hash = '#posts'; await sleep(20);
    DATA.summary.confidence_bands = null;
    location.hash = '#compare'; await sleep(30);
    out.guard = txt(document.getElementById('tab-compare'));
    out.guard_banner_hidden = document.getElementById('banner').hidden;
  });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>
"""
)

#: A Comparar page opened at a given hash.
_READ_COMPARE = (
    "<script>"
    + _SEEN
    + r"""
setTimeout(() => {
  const pre = document.createElement('pre'); pre.id = 'probe';
  const list = document.getElementById('compare-list');
  const tab = document.getElementById('tab-compare');
  pre.textContent = JSON.stringify({tab: view.tab, visible: seen(tab), text: txt(tab),
    list: seen(list) ? seenCards(list) : null, head: list ? txt(list.querySelector('h3')) : null,
    in_view: inView(list),
    notes: [...document.querySelectorAll('#tab-compare .note, #tab-compare .empty')].filter(seen).map(p => p.textContent),
    ranges: [...document.querySelectorAll('#tab-compare .range')].map(txt),
    stamp: txt(document.getElementById('stamp')),
    facts: txt(document.getElementById('facts')),
    banner_hidden: document.getElementById('banner').hidden});
  document.body.appendChild(pre);
}, 200);
</script>
"""
)


@pytest.fixture(scope="module")
def compare_probed(tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any]]:
    _need_chrome()
    data = _compare_fixture()
    seen = _open(_page(tmp_path_factory.mktemp("jev-compare"), data, _COMPARE_PROBE), "#compare")
    assert seen["errors"] == [], seen["errors"]
    return data, seen


_KIND_COUNTS = {
    "enrich_only": ("posts_enrich_only", "doubtful_pairs"),
    "adds": ("posts_jev_only", "missing_pairs"),
    "prim": ("posts_primary_differs", None),
}


@_requires_chrome
def test_the_tab_and_its_six_sections_are_on_screen(compare_probed):
    _data_, seen = compare_probed

    assert seen["tab_seen"] is True
    assert seen["sections"] == ["kinds", "topics", "cross", "otro", "bands"]


@_requires_chrome
def test_each_disagreement_kind_says_its_numbers_and_opens_exactly_its_posts(compare_probed):
    data, seen = compare_probed
    s = data["summary"]

    for key, (posts, pairs) in _KIND_COUNTS.items():
        kind = seen["kinds"][key]
        assert kind["shown"] == str(s[posts]), key
        assert set(kind["ids"]) == _ids(data, key) and len(kind["ids"]) == s[posts], key
        assert kind["count"].startswith(f"mostrando {s[posts]} de"), key
        assert kind["hash"] == f"#revisar/posts?f={key}" and kind["link"] == "ver los posts →", key
        if pairs:
            assert kind["why"].endswith(f": {s[pairs]} topics en total."), key


@_requires_chrome
def test_the_three_numbers_are_said_once_and_open_their_posts(compare_probed):
    data, seen = compare_probed
    s = data["summary"]
    one, add, prim = (seen["sentences"][k] for k in ("enrich_only", "adds", "prim"))

    # Said once, above the sub-tabs: Comparar has no second, reworded copy of them.
    assert "En tres frases" not in seen["headings"]
    assert seen["headings"].count("Lo que dice Jev, en tres números") == 1
    assert one["big"] == "Jev confirma 39 de 58 topics de enrich (67,2 %)"
    assert (s["assigned_backed"], s["assigned_pairs"], s["enrich_backed_pct"]) == (39, 58, 67.2)
    assert one["why"].endswith(
        f"Los otros {s['doubtful_pairs']} son candidatos a quitar. "
        f"{s['assigned_unjudged']} topic ya no está en el vocabulario y no se juzga."
    )
    assert (
        one["link"] == f"ver los {s['posts_enrich_only']} posts con un topic que Jev no confirma →"
    )
    assert add["big"] == f"Jev añadiría {s['missing_pairs']} topics que enrich no puso"
    assert add["link"] == f"ver los {s['posts_jev_only']} posts donde Jev añadiría →"
    assert prim["big"] == "El topic principal coincide en 71,7 %"
    assert s["primary_agree_pct"] == 71.7
    assert prim["why"].startswith(
        f"En {s['primary_agree']} de {s['items_compared']} posts comparados"
    )
    assert prim["link"] == f"ver los {s['posts_primary_differs']} posts donde no coincide →"
    assert {k: v["href"] for k, v in seen["sentences"].items()} == {
        "enrich_only": "#revisar/posts?f=enrich_only",
        "adds": "#revisar/posts?f=adds",
        "prim": "#revisar/posts?f=prim",
    }
    for key, sentence in seen["sentences"].items():
        assert sentence["opened"] == f"#revisar/posts?f={key}", key
        assert set(sentence["ids"]) == _ids(data, key) and sentence["ids"], key


@_requires_chrome
def test_the_topic_table_is_the_ten_worst_in_the_topics_tabs_order(compare_probed):
    data, seen = compare_probed
    gamma = {r["slug"]: r for r in data["summary"]["per_topic"]}["gamma"]

    assert seen["topic_rows"] == [
        "gamma",
        "delta",
        "alpha",
        "beta",
        "f01",
        "f02",
        "f03",
        "f04",
        "f05",
        "f06",
    ]
    assert seen["topic_more"] == "ver los 12 en Topics →"
    assert seen["gamma_cells"] == ["Gamma", "10", "0", "0,0 %", "10"]
    # Never assigned: no agreement to be 0 % of.
    assert seen["f01_cells"] == ["F01pocos datos", "0", "0", "—", "0"]
    assert gamma["disagreeing"] == 10 == len(seen["gamma_disc"]["ids"])
    assert seen["gamma_disc"]["hash"] == "#revisar/posts?t=gamma"


@_requires_chrome
def test_each_band_says_its_numbers_and_opens_exactly_its_posts(compare_probed):
    data, seen = compare_probed
    rows = {row["key"]: row for row in data["summary"]["confidence_bands"]}

    assert seen["band_heads"] == [
        "Enrich asigna y Jev no · 18 desacuerdos",
        "Jev añadiría · 24 desacuerdos",
    ]
    assert set(seen["bands"]) == set(rows) == {"e-lo", "e-mid", "e-near", "j-near", "j-hi"}
    for key, band in seen["bands"].items():
        row = seen["band_rows"][key]
        assert row["label"] == rows[key]["label"], key
        assert row["pairs"] == f"{rows[key]['pairs']} desacuerdos", key
        assert row["link"] == f"{rows[key]['posts']} post" + (
            "s" if rows[key]["posts"] != 1 else ""
        ), key
        assert (
            band["ids"] == data["post_sets"]["bands"][key]
            and len(band["ids"]) == rows[key]["posts"]
        ), key
        assert band["head"].startswith(row["link"] + " · " + rows[key]["label"]), key
        assert band["hash"] == f"#revisar/compare?b={key}" and band["under"] == "bands", key
        assert band["in_view"] is True, key
    assert seen["bands"]["j-near"]["first"] == 20 and len(seen["bands"]["j-near"]["ids"]) == 22


@_requires_chrome
def test_the_band_edges_are_stated_as_probabilities(compare_probed):
    _data_, seen = compare_probed

    assert {k: r["range"] for k, r in seen["band_rows"].items()} == {
        "e-lo": "probabilidad < 0,20",
        "e-mid": "probabilidad 0,20 – 0,50",
        "e-near": "probabilidad 0,50 – 0,85",
        "j-near": "probabilidad 0,85 – 0,95",
        "j-hi": "probabilidad ≥ 0,95",
    }


@_requires_chrome
def test_each_primary_cross_pair_is_ranked_named_and_opens_exactly_its_posts(compare_probed):
    data, seen = compare_probed

    assert seen["px_order"] == ["gamma~omega", "delta~otro", "-~alpha", "alpha~beta"]
    names = {k: v["name"] for k, v in seen["px"].items()}
    assert names == {
        "gamma~omega": "Gamma → Omega",
        "delta~otro": "Delta → otro (ninguno del vocabulario)",
        "-~alpha": "sin principal → Alpha",
        "alpha~beta": "Alpha → Beta",
    }
    for key, pair in seen["px"].items():
        n = len(data["post_sets"]["px"][key])
        assert pair["ids"] == data["post_sets"]["px"][key], key
        assert pair["count"] == f"{n} post" + ("s" if n != 1 else ""), key
        assert pair["head"].startswith(pair["count"] + " · Principal · enrich: "), key
        assert pair["hash"] == "#revisar/compare?px=" + key.replace("~", "%7E"), key
    assert seen["px"]["delta~otro"]["under"] == "otro"
    assert seen["px"]["gamma~omega"]["under"] == "cross"


@_requires_chrome
def test_the_diagonal_is_ranked_ties_by_label_and_opens_its_posts(compare_probed):
    data, seen = compare_probed
    s = data["summary"]

    assert seen["pd_order"] == ["alpha", "beta", "delta", "gamma"]
    assert {k: v["count"] for k, v in seen["pd"].items()} == {
        "alpha": "28 posts",
        "beta": "4 posts",
        "delta": "3 posts",
        "gamma": "3 posts",
    }
    for key, pd in seen["pd"].items():
        assert pd["ids"] == data["post_sets"]["pd"][key] and pd["under"] == "cross", key
    assert seen["agree"] == (
        f"Coinciden en {s['primary_agree']} posts de {s['items_compared']} comparados; "
        f"no coinciden en {s['posts_primary_differs']} posts."
    )


@_requires_chrome
def test_the_other_fallback_is_counted_with_its_posts_and_what_enrich_had(compare_probed):
    data, seen = compare_probed

    assert seen["otro_count"] == str(data["summary"]["primary_fallback"]) == "5"
    assert seen["otro_rows"] == [["delta~otro", "Delta", "5 posts"]]
    assert seen["otro_open"]["under"] == "otro"
    assert seen["otro_open"]["ids"] == data["post_sets"]["px"]["delta~otro"]
    assert seen["otro_posts"]["hash"] == "#revisar/posts?f=fallback"
    assert (
        set(seen["otro_posts"]["ids"]) == _ids(data, "fallback")
        and len(seen["otro_posts"]["ids"]) == 5
    )


@_requires_chrome
def test_a_long_list_shows_ten_and_ver_todos_names_its_unit(compare_probed):
    _data_, seen = compare_probed

    assert seen["list_visible_before"] == 10
    assert seen["list_more"] == "ver todos (2 topics más, 4 posts)"
    assert seen["list_visible_after"] == 12


@_requires_chrome
def test_leaving_and_returning_keeps_each_tabs_view(compare_probed):
    data, seen = compare_probed
    sets = data["post_sets"]

    assert seen["round"] == {
        "#compare?px=gamma~omega": {
            "hash": "#revisar/compare?px=gamma%7Eomega",
            "ids": sets["px"]["gamma~omega"],
        },
        "#compare?pd=delta": {"hash": "#revisar/compare?pd=delta", "ids": sets["pd"]["delta"]},
        "#compare?b=e-lo": {"hash": "#revisar/compare?b=e-lo", "ids": sets["bands"]["e-lo"]},
    }
    assert seen["posts_tab_href"] == "#revisar/posts?f=adds" and seen["posts_f_kept"] == "adds"
    assert seen["cleared"] == {"hash": "#revisar/compare", "list": False}


@_requires_chrome
def test_one_list_at_a_time_band_first_and_the_tab_link_says_which(compare_probed):
    data, seen = compare_probed

    assert seen["order_ids"] == data["post_sets"]["bands"]["e-near"]
    assert seen["order_tab_href"] == "#revisar/compare?b=e-near"


@_requires_chrome
def test_the_same_list_again_does_not_scroll(compare_probed):
    _data_, seen = compare_probed

    assert seen["same_list_scroll"] == 0


@_requires_chrome
def test_a_stale_group_says_so_in_this_tab(compare_probed):
    _data_, seen = compare_probed

    assert seen["stale_notes"] == [
        "Ese grupo ya no existe en estos datos; en esta pestaña, la comparación completa."
    ]


@_requires_chrome
def test_a_list_heading_counts_what_it_lists_and_names_a_differing_summary(compare_probed):
    _data_, seen = compare_probed

    assert seen["differs"]["head"].startswith("1 post · ")
    assert seen["differs"]["notes"] == [
        "El resumen cuenta 99 posts en este grupo; la lista tiene 1."
    ]


@_requires_chrome
def test_a_compare_list_whose_posts_are_missing_says_how_many(compare_probed):
    _data_, seen = compare_probed

    assert seen["missing_note"] == ["1 de 7 posts no están en esta página."]


@_requires_chrome
def test_new_data_through_loaddata_reaches_every_derived_value(compare_probed):
    _data_, seen = compare_probed

    assert seen["after_load"] == {
        "range": "probabilidad < 0,200",
        "filter": "Jev eligió «nada»",
        "otro_title": "Jev eligió «nada»",
    }


@_requires_chrome
def test_a_compare_tab_that_fails_after_boot_says_so_in_the_tab(compare_probed):
    _data_, seen = compare_probed

    assert "La pestaña Comparar no pudo dibujarse" in seen["guard"]
    assert seen["guard_banner_hidden"] is True


@_requires_chrome
@pytest.mark.parametrize(
    ("hash_", "expect"),
    [
        ("#compare?b=e-near", {"list": ["n00", "n01", "n02"]}),
        ("#compare?px=gamma~omega", {"list": [f"m{n:02d}" for n in range(7)]}),
        ("#compare?px=-~alpha", {"list": ["s00", "s01"]}),
        ("#compare?pd=gamma", {"list": ["n00", "n01", "n02"]}),
        ("#compare?b=no-existe", {"note": "Ese grupo ya no existe en estos datos"}),
        ("#compare?px=zz~yy", {"note": "Ese grupo ya no existe en estos datos"}),
        ("#compare?pd=%E0", {"note": "Ese grupo ya no existe en estos datos"}),
        # A key every JS object inherits must not open a list.
        ("#compare?pd=constructor", {"note": "Ese grupo ya no existe en estos datos"}),
        ("#compare?b=toString", {"note": "Ese grupo ya no existe en estos datos"}),
    ],
)
def test_a_compare_url_opened_fresh_shows_that_view(tmp_path, hash_, expect):
    _need_chrome()
    data = _compare_fixture()

    seen = _open(_page(tmp_path, data, _READ_COMPARE), hash_)

    assert seen["banner_hidden"] is True and seen["tab"] == "compare" and seen["visible"]
    if "list" in expect:
        assert seen["list"] == expect["list"]
        assert seen["in_view"] is True
    if "note" in expect:
        assert seen["list"] is None
        assert any(expect["note"] in note for note in seen["notes"]), seen["notes"]


@_requires_chrome
def test_nothing_compared_says_so_instead_of_zero_percent(tmp_path):
    """A vocabulary edit retires every answer: the tab says nothing is compared and why,
    and shows no «0,0 %» that reads as a measured disagreement."""
    _need_chrome()
    items = [_item(f"{n}") for n in range(3)]
    data = _data(items, {i.id: _assessment(i, contract="0" * 64) for i in items})
    assert data["summary"]["items_compared"] == 0 and data["totals"]["stale"] == 3

    seen = _open(_page(tmp_path, data, _READ_COMPARE), "#compare")

    assert "Ningún post comparado todavía" in seen["text"]
    assert "3 evaluaciones caducadas" in seen["text"]
    assert "0,0 %" not in seen["text"] and "%" not in seen["text"]
    # The header's three numbers too: a share of nothing is «—», not 0,0 %.
    assert "%" not in seen["facts"] and seen["facts"].count("—") == 2


@_requires_chrome
def test_the_edges_print_at_the_thresholds_own_precision(tmp_path):
    _need_chrome()
    items, assessments = _corpus()
    data = _data(items, assessments, threshold=0.875)

    seen = _open(_page(tmp_path, data, _READ_COMPARE), "#compare")

    assert "Umbral de topics 0,875" in seen["stamp"]
    assert "probabilidad 0,500 – 0,875" in seen["ranges"]


# --------------------------------------------------------------------------- the Configuración tab

#: Where the fixture's files live. The run log does not exist yet (no paid pass): the page
#: must say so rather than list it as if it did.
_CONFIG_FILES = [
    {
        "key": "topics",
        "label": "jev/topics.json",
        "path": "/repo/data/jev/topics.json",
        "exists": True,
        "snapshotted": False,
    },
    {
        "key": "runs",
        "label": "jev/runs.jsonl",
        "path": "/repo/data/jev/runs.jsonl",
        "exists": False,
        "snapshotted": False,
    },
    {
        "key": "vocab",
        "label": "vocab.yaml",
        "path": "/repo/data/vocab.yaml",
        "exists": True,
        "snapshotted": True,
    },
    {
        "key": "report_json",
        "label": "jev/topics-report.json",
        "path": "/repo/data/jev/topics-report.json",
        "exists": True,
        "snapshotted": False,
    },
    {
        "key": "report_md",
        "label": "jev/topics-report.md",
        "path": "/repo/data/jev/topics-report.md",
        "exists": True,
        "snapshotted": False,
    },
    {
        # A first render is built before its file is written: the page must not call
        # itself missing while the reader is looking at it.
        "key": "page",
        "label": "jev.html",
        "path": "/vault/x-knowledge/jev.html",
        "exists": False,
        "snapshotted": False,
        "served": False,
    },
]
#: Every setting away from its default, so a value the template hard-codes cannot pass.
_CONFIG_SETTINGS = {
    "threshold": 0.875,
    "fallback": "ninguno",
    "char_limit": 50_000,
    "serve_max_usd": 0.25,
    "ask_max_usd": 0.5,
    "ask_top": 7,
}
#: Four answers from one model and one from another, named so that name order and count
#: order disagree.
_MANY, _FEW = "jev-9.0.0", "jev-1.0.0"


def _config_fixture(
    files: list[dict[str, Any]] | None = _CONFIG_FILES, tokens=None
) -> dict[str, Any]:
    """Five current answers at the fixture's settings: one without a token count, one from a
    provider nobody prices (so tokens average 4 answers and dollars 3), four never-asked posts
    and one with no evidence."""
    fallback, limit = _CONFIG_SETTINGS["fallback"], _CONFIG_SETTINGS["char_limit"]
    specs = tokens or [
        ("c0", "typesafe", 1000, _FEW),
        ("c1", "typesafe", 3000, _MANY),
        ("c2", "typesafe", None, _MANY),
        ("c3", "fake", 5000, _MANY),
        ("c4", "typesafe", 2000, _MANY),
    ]
    items, assessments = [], {}
    for item_id, provider, n, model in specs:
        item = _item(item_id, text=f"post {item_id}")
        items.append(item)
        answer = _assessment(
            item, provider=provider, input_tokens=n, fallback=fallback, char_limit=limit
        )
        assessments[item_id] = answer.model_copy(update={"model": model})
    items += [_item(f"n{n}") for n in range(4)]
    empty = _item("vacio", text=" ")
    empty.author = Author(handle="", name="")
    items.append(empty)
    return _data(items, assessments, files=files, **_CONFIG_SETTINGS)


#: The Configuración tab, opened at #config: what a reader sees in each section, the questions
#: before and after unfolding them, a vocabulary link followed, and the tab's error guard.
_CONFIG_PROBE = (
    "<script>"
    + _SEEN
    + r"""
const sleep = ms => new Promise(r => setTimeout(r, ms));
const cfgBody = () => document.getElementById('config-body');
const secs = () => [...cfgBody().querySelectorAll('section.cfg')];
const sec = title => secs().find(s => s.querySelector('h3').textContent === title);
const cells = tr => [...tr.querySelectorAll('td')].map(txt);
const rows = s => [...s.querySelectorAll('tbody tr')].filter(seen).map(cells);
const wireOf = w => ({instructions: txt(w.querySelector('.wi')),
  criteria: [...w.querySelectorAll('dt')].map(dt => [txt(dt), txt(dt.nextElementSibling)])});
const wires = root => [...root.querySelectorAll('.wire')].filter(seen).map(wireOf);
let hashChanges = 0;
addEventListener('hashchange', () => { hashChanges++; });
async function travel(go) {
  const before = hashChanges;
  go();
  for (let t = 0; hashChanges === before && t < 300; t++) await sleep(10);
  if (hashChanges === before) throw new Error('no hashchange');
  await sleep(0);
}
(async () => {
  const out = {errors: []};
  const step = async (name, fn) => { try { await fn(); } catch (e) { out.errors.push(name + ': ' + e); } };
  await sleep(100);
  await step('shell', async () => {
    out.tab_seen = seen(document.getElementById('tab-config'));
    out.posts_seen = seen(document.getElementById('tab-posts'));
    out.titles = secs().filter(seen).map(s => txt(s.querySelector('h3')));
  });
  await step('settings', async () => {
    const s = sec('Ajustes');
    out.settings = rows(s);
    out.settings_notes = [...s.querySelectorAll('p')].filter(seen).map(p => p.textContent);
  });
  await step('questions', async () => {
    const s = sec('Las preguntas exactas');
    const [yes, pick] = [...s.querySelectorAll('.q')];
    out.yes_head = txt(yes.querySelector('.lab'));
    out.yes_gloss = txt(yes.querySelector('.gloss'));
    out.pick_gloss = txt(pick.querySelector('.gloss'));
    out.yes_folded = wires(yes);
    out.pick_folded = wires(pick);
    yes.querySelector('summary').click(); pick.querySelector('summary').click(); await sleep(10);
    out.yes_open = wires(yes.querySelector('details'));
    out.pick_open = wires(pick.querySelector('details'));
    out.intro = txt(s.querySelector('p.muted'));
    out.digest = txt(s.querySelector('.digest'));
  });
  await step('vocab', async () => {
    const s = sec('El vocabulario');
    out.vocab = [...s.querySelectorAll('tbody tr')].filter(seen).map(tr => ({
      label: txt(tr.querySelector('a')), href: tr.querySelector('a').getAttribute('href'),
      slug: txt(tr.querySelector('.slug')), desc: txt(tr.querySelector('td.cw'))}));
  });
  await step('evidence', async () => {
    const s = sec('Qué ve Jev de cada post');
    out.surfaces = [...s.querySelectorAll('li')].map(txt);
    out.marker = txt(s.querySelector('code.marker'));
    out.cut = txt(s.querySelectorAll('p.muted')[1]);
    const docs = s.querySelector('a.lk');
    out.docs = {text: txt(docs), href: docs.getAttribute('href')};
  });
  await step('files', async () => {
    const s = sec('Ficheros');
    out.files = rows(s);
    out.files_notes = [...s.querySelectorAll('p')].filter(seen).map(txt);
  });
  await step('cost', async () => {
    const s = sec('Coste de una pasada (estimación)');
    out.cost = [...s.querySelectorAll('.kind')].map(k => ({lab: txt(k.querySelector('.lab')),
      big: txt(k.querySelector('.cnum')), why: txt(k.querySelector('p'))}));
    out.cost_intro = txt(s.querySelector('p.muted'));
  });
  await step('link', async () => {
    const a = sec('El vocabulario').querySelector('tbody a');
    await travel(() => a.click());
    out.after_link = {hash: location.hash, topics_seen: seen(document.getElementById('tab-topics')),
      config_seen: seen(document.getElementById('tab-config')),
      heading: txt(document.querySelector('#topic-detail .th2'))};
    await travel(() => history.back());
    out.back = {hash: location.hash, config_seen: seen(document.getElementById('tab-config')),
      sections: secs().filter(seen).length};
  });
  await step('stamp', async () => {
    // The header names the threshold and links to where it is explained: from another tab,
    // the link opens Configuración at that row; on Configuración, it scrolls there.
    out.stamp = txt(document.getElementById('stamp'));
    await travel(() => { location.hash = '#revisar/posts'; });
    const a = document.querySelector('#stamp a[data-cfg]');
    out.stamp_link = {text: txt(a), href: a.getAttribute('href')};
    // Centred: the row is where the eye lands, not merely somewhere on screen.
    const centred = () => { const r = document.getElementById('cfg-threshold').getBoundingClientRect();
      return Math.abs((r.top + r.bottom) / 2 - innerHeight / 2) < 40; };
    await travel(() => a.click());
    await sleep(600);
    out.stamp_opened = {hash: location.hash, config_seen: seen(document.getElementById('tab-config')),
      row: centred(), row_text: txt(document.querySelector('#cfg-threshold td'))};
    scrollTo(0, 0); await sleep(10);
    out.stamp_same_tab_before = centred();
    a.click(); await sleep(600);
    out.stamp_same_tab = centred();
  });
  await step('guard', async () => {
    const saved = DATA.config;
    DATA.config = null;
    renderConfig();
    out.guard = txt(cfgBody());
    out.guard_banner_hidden = document.getElementById('banner').hidden;
    DATA.config = saved;
  });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>
"""
)


@pytest.fixture(scope="module")
def config_probed(tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any]]:
    _need_chrome()
    data = _config_fixture()
    seen = _open(_page(tmp_path_factory.mktemp("jev-config"), data, _CONFIG_PROBE), "#config")
    assert seen["errors"] == [], seen["errors"]
    return data, seen


def _wire_view(question: dict[str, Any], only: list[str] | None = None) -> dict[str, Any]:
    """A shipped question as the probe reads it off the screen."""
    return {
        "instructions": question["instructions"],
        "criteria": [[k, v] for k, v in question["criteria"] if only is None or k in only],
    }


def _nf(n: int) -> str:
    """es-ES grouping as the page prints it: from five digits only (2000, but 12.000)."""
    return str(n) if abs(n) < 10_000 else f"{n:,}".replace(",", ".")


def _usd(x: float, decimals: int = 4) -> str:
    return "~" + f"{x:.{decimals}f}".replace(".", ",") + " $"


@_requires_chrome
def test_the_config_tab_shows_its_six_sections(config_probed):
    _data_, seen = config_probed

    assert seen["tab_seen"] is True and seen["posts_seen"] is False
    assert seen["titles"] == [
        "Ajustes",
        "Las preguntas exactas",
        "El vocabulario",
        "Qué ve Jev de cada post",
        "Ficheros",
        "Coste de una pasada (estimación)",
    ]


@_requires_chrome
def test_each_setting_shows_the_value_in_effect_and_its_default(config_probed):
    """Every setting away from its default: the value shown is the one handed in, the default
    is `JEV_DEFAULTS`' — so neither can be typed into the template."""
    data, seen = config_probed
    prices = data["config"]["prices"]

    by_name = {row[0]: row for row in seen["settings"]}
    assert list(by_name) == [
        "Umbral",
        "Modelo que se pedirá",
        "Opción de escape",
        "Peticiones a la vez",
        "Límite de evidencia",
        "Tope por trabajo de xbrain jev serve",
        "Tope sin preguntar de xbrain jev ask",
        "Resultados que se muestran de una pregunta",
        *[f"Precio de entrada · {p}" for p in sorted(prices)],
    ]
    assert by_name["Umbral"][1] == "0,875por defecto: 0,850"
    # The header: the day in the page's words, the threshold as a link to its row here.
    assert seen["stamp"].startswith("Actualizado 22 sept 2026")
    assert "(config)" not in seen["stamp"]
    assert seen["stamp_link"] == {"text": "Umbral de topics 0,875", "href": "#config"}
    assert seen["stamp_opened"] == {
        "hash": "#config",
        "config_seen": True,
        "row": True,
        "row_text": "Umbral",
    }
    assert seen["stamp_same_tab_before"] is False and seen["stamp_same_tab"] is True
    assert by_name["Umbral"][3] == "[jev].threshold"
    assert by_name["Opción de escape"][1] == "«ninguno»por defecto: «otro»"
    assert by_name["Peticiones a la vez"][1] == "3por defecto: 8"
    assert by_name["Límite de evidencia"][1] == "50.000 caracterespor defecto: 100.000 caracteres"
    assert by_name["Tope por trabajo de xbrain jev serve"][1] == "0,25 $por defecto: 1,00 $"
    assert by_name["Tope por trabajo de xbrain jev serve"][3] == "[jev].serve_max_usd"
    assert by_name["Tope sin preguntar de xbrain jev ask"][1] == "0,50 $por defecto: 0,25 $"
    assert by_name["Tope sin preguntar de xbrain jev ask"][3] == "[jev].ask_max_usd"
    assert by_name["Resultados que se muestran de una pregunta"][1] == "7por defecto: 20"
    assert by_name["Resultados que se muestran de una pregunta"][3] == "[jev].ask_top"
    # Most answers first, then by name: here the reverse of name order.
    assert by_name["Modelo que se pedirá"][1] == (
        "jev-9.9.9por defecto: jev-latest · las evaluaciones guardadas no registran qué modelo "
        f"se pidió, solo el que respondió: {_MANY} (4 evaluaciones vigentes), "
        f"{_FEW} (1 evaluación vigente)"
    )
    assert data["config"]["models_answered"] == [[_MANY, 4], [_FEW, 1]]
    assert by_name["Precio de entrada · typesafe"][1] == (
        f"{str(prices['typesafe']).replace('.', ',')} $por millón de tokens de entrada"
    )
    # Two short lines and where they change, not eight names run into one sentence.
    built, next_, where = seen["settings_notes"][:3]
    assert built == (
        "Cambian lo que ves en esta página: Umbral, Opción de escape, Límite de evidencia y "
        "Resultados que se muestran de una pregunta."
    )
    assert next_ == (
        "Se usan en la próxima pasada (xbrain jev topics o un trabajo de xbrain jev serve): "
        "Modelo que se pedirá, Peticiones a la vez, Tope por trabajo de xbrain jev serve y "
        "Tope sin preguntar de xbrain jev ask."
    )
    assert where == "Se cambian en config.toml, nunca aquí."
    assert seen["settings_notes"][-1] == "Los tokens de salida son gratis."


@_requires_chrome
def test_the_questions_on_screen_are_the_wire_text_the_blob_carries(config_probed):
    """Folded: one sí/no example (the first in wire order) and the Choice with only its escape
    option. Unfolded: every sí/no question and every option, exactly as shipped."""
    data, seen = config_probed
    questions = data["config"]["questions"]
    yes_no = [q for q in questions if q["type"] == "yes_no"]
    [choice] = [q for q in questions if q["type"] == "choice"]
    fallback = choice["criteria"][-1][0]

    assert fallback == "ninguno"
    assert seen["yes_folded"] == [_wire_view(yes_no[0])]
    assert seen["pick_folded"] == [_wire_view(choice, [fallback])]
    assert seen["yes_open"] == [_wire_view(q) for q in yes_no]
    assert seen["pick_open"] == [_wire_view(choice)]
    assert seen["yes_head"] == f"{len(yes_no)} preguntas sí/no"
    assert "sí/no por topic = pertenencia, varios posibles" in seen["yes_gloss"]
    assert "elección = el principal, uno solo; sus probabilidades suman 1" in seen["pick_gloss"]
    assert seen["pick_gloss"].endswith("y al final «ninguno»:")
    assert f"con {len(questions)} preguntas" in seen["intro"]
    assert seen["digest"] == data["config"]["questions_digest"]


@_requires_chrome
def test_the_vocabulary_lists_every_topic_and_links_its_topics_page(config_probed):
    data, seen = config_probed

    assert seen["vocab"] == [
        {
            "label": t["label"],
            "href": f"#revisar/topics?t={t['slug']}",
            "slug": t["slug"],
            "desc": t["description"],
        }
        for t in data["topics"]
    ]
    first = data["topics"][0]
    assert seen["after_link"]["hash"] == f"#revisar/topics?t={first['slug']}"
    assert seen["after_link"]["topics_seen"] is True
    assert seen["after_link"]["config_seen"] is False
    assert seen["after_link"]["heading"].startswith(first["label"])
    assert seen["back"] == {"hash": "#config", "config_seen": True, "sections": 6}


@_requires_chrome
def test_what_jev_reads_is_the_state_order_and_the_cut_quotes_the_real_marker(config_probed):
    data, seen = config_probed
    config = data["config"]

    assert seen["surfaces"] == [s["label"] for s in config["surfaces"]]
    assert seen["surfaces"][0] == "Tweet"
    assert seen["marker"] == config["cut_marker"]
    assert seen["cut"].startswith("El corte: si pasa de 50.000 caracteres, se corta")
    assert "un corte solo llega a él si el propio tweet pasa del límite" in seen["cut"]
    assert seen["docs"]["href"] == config["docs_url"]


@_requires_chrome
def test_the_files_say_where_each_lives_whether_it_exists_and_whether_snapshots_keep_it(
    config_probed,
):
    data, seen = config_probed

    [runs] = [row for row in seen["files"] if row[0] == "jev/runs.jsonl"]
    assert runs[1] == (
        "/repo/data/jev/runs.jsonl"
        "(aún no existe: lo crea la primera xbrain jev topics que envíe peticiones)"
    )
    assert [(row[0], row[3]) for row in seen["files"]] == [
        (f["label"], "sí" if f["snapshotted"] else "no") for f in data["config"]["files"]
    ]
    assert [row[1] for row in seen["files"] if row[0] != "jev/runs.jsonl"] == [
        f["path"] for f in data["config"]["files"] if f["key"] != "runs"
    ]
    assert seen["files"][-1][1] == "/vault/x-knowledge/jev.html"
    assert seen["files"][-1][2] == "esta página; la escribe xbrain jev dashboard"
    assert seen["files_notes"] == [
        "data/ no está en git. Las evaluaciones de Jev (jev/topics.json) y su registro "
        "(jev/runs.jsonl) no entran en los snapshots; el topics.json de enrich y vocab.yaml sí.",
        "Cuidado: rehacer jev/topics.json cuesta dinero, y xbrain snapshot restore no lo "
        "devuelve; runs.jsonl es el histórico de costes.",
    ]


@_requires_chrome
def test_the_cost_of_a_pass_is_the_estimate_python_computed(config_probed):
    """Tokens average four answers and dollars three (one provider has no price): the line
    names both, and the dollar figures say they leave that provider out."""
    data, seen = config_probed
    estimate = data["config"]["estimate"]
    per_post = estimate["per_post"]

    assert per_post["tokens_n"] == 4 and per_post["n"] == 3
    assert per_post["unpriced_providers"] == ["fake"]
    labs = [c["lab"] for c in seen["cost"]]
    assert labs == ["Media por post", "Lo que falta", "Todo el corpus"]
    mean, pending, corpus = seen["cost"]
    assert mean["big"] == _nf(round(per_post["mean_tokens"])) + " tokens"
    assert mean["why"] == (
        "tokens: media de 4 de las 5 evaluaciones vigentes (las que traen recuento) · $: "
        + _usd(per_post["mean_usd"], 5)
        + ", media de 3 (las que además tienen tarifa) · sin tarifa: fake"
    )
    assert pending["big"] == _usd(estimate["pending"]["usd"])
    assert corpus["big"] == _usd(estimate["corpus"]["usd"])
    for shown, key in ((pending, "pending"), (corpus, "corpus")):
        e = estimate[key]
        assert shown["why"] == (
            f"{e['posts']} posts · ~{_nf(e['tokens'])} tokens de entrada · el $ solo promedia "
            "proveedores con tarifa (sin tarifa: fake)"
        )
    # Recounted from the cards: the posts that offer the `jev topics --id` command, and every
    # post with evidence.
    asks = [p for p in data["posts"] if p["status"] in ("stale", "unevaluated")]
    assert estimate["pending"]["posts"] == sum(not p["no_evidence"] for p in asks) == 4
    assert estimate["corpus"]["posts"] == len(data["posts"]) - 1
    assert "como si se volviera a preguntar todo" in seen["cost_intro"]


@_requires_chrome
def test_a_config_tab_that_fails_after_boot_says_so_in_the_tab(config_probed):
    _data_, seen = config_probed

    assert seen["guard"].startswith("La pestaña Configuración no pudo dibujarse: ")
    assert seen["guard_banner_hidden"] is True


_READ_CONFIG = (
    "<script>"
    + _SEEN
    + r"""
setTimeout(() => {
  const s = t => [...document.querySelectorAll('#config-body section.cfg')].find(x => x.querySelector('h3').textContent === t);
  const pre = document.createElement('pre'); pre.id = 'probe';
  pre.textContent = JSON.stringify({
    files: [...s('Ficheros').querySelectorAll('p')].map(txt),
    cost: [...s('Coste de una pasada (estimación)').querySelectorAll('.kind')].map(k =>
      [txt(k.querySelector('.cnum')), txt(k.querySelector('p'))])});
  document.body.appendChild(pre);
}, 200);
</script>
"""
)


@_requires_chrome
def test_a_page_without_paths_or_token_counts_says_so(tmp_path):
    """A pure build ships no paths; answers with no token count leave nothing to average —
    the page says both instead of printing zeros."""
    _need_chrome()
    no_tokens = [(f"c{n}", "typesafe", None, _MANY) for n in range(2)]
    data = _config_fixture(files=None, tokens=no_tokens)

    seen = _open(_page(tmp_path, data, _READ_CONFIG), "#config")

    assert seen["files"] == ["Esta página se generó sin rutas de ficheros."]
    assert seen["cost"] == [
        ["—", "ninguna evaluación vigente trae recuento de tokens"],
        ["—", "4 posts · sin media que multiplicar"],
        ["—", f"{len(data['posts']) - 1} posts · sin media que multiplicar"],
    ]


_TO_CONFIG = (
    "<script>"
    + _SEEN
    + r"""
setTimeout(() => {
  document.querySelector('.tabs a[data-tab="config"]').click();
  setTimeout(() => {
    const pre = document.createElement('pre'); pre.id = 'probe';
    pre.textContent = JSON.stringify({hash: location.hash,
      titles: [...document.querySelectorAll('#config-body section.cfg h3')].map(txt).filter(Boolean)});
    document.body.appendChild(pre);
  }, 100);
}, 100);
</script>
"""
)


@_requires_chrome
def test_the_config_tab_draws_when_reached_from_another_tab(tmp_path):
    """Opened on Posts, then the tab link: the tab draws on the hash change, not only at boot."""
    _need_chrome()

    seen = _open(_page(tmp_path, _config_fixture(), _TO_CONFIG))

    assert seen["hash"] == "#config"
    assert len(seen["titles"]) == 6 and seen["titles"][0] == "Ajustes"


# --------------------------------------------------------------------------- served (PR 10b)
#
# The page as `xbrain jev serve` serves it, driven in Chrome against a REAL local server whose
# jobs ask `tests/jev_fakes.FakeJevClient` — never TypeSafe. Each wait in the probe is a fetch
# to the server, so real time passes for the job thread while Chrome's virtual clock waits.

#: The served probes' helpers, all prefixed `s` (the probe shares the page's scope). Every wait
#: fetches `/probe-wait` (a paced look, never a call the page makes: `_probe_handler`), so real
#: time passes for the job thread while Chrome's clock waits, and gives up after 30 real
#: seconds by the server's `Date` (the virtual budget is large: the waits bound the probe).
#: `refresh` is wrapped to count the page's reloads; `fetch` to count estimate replies and keep
#: the last one (the server's own numbers, to compare the panel with).
_SERVE_JS = (
    _SEEN
    + r"""
const sOut = {};
const sId = id => document.getElementById(id);
// Bounded by REAL time — the server's `Date` header — since the page's clock is virtual: 30 s
// for one wait, and one deadline for the WHOLE probe (from its first wait), so a probe whose
// first step fails names that step instead of running out Chrome's own timeout.
const sBudgetMs = /*BUDGET*/75000;
let sDeadline = null;
const sWait = async (cond, what) => {
  let start = null;
  for (;;) {
    if (cond()) return;
    const r = await sFetch0.call(window, '/probe-wait');
    const now = Date.parse(r.headers.get('Date'));
    if (start === null) start = now;
    if (sDeadline === null) sDeadline = now + sBudgetMs;
    if (now > sDeadline) throw new Error('nunca (se acabó el plazo del probe): ' + what);
    if (now - start > 30000) throw new Error('nunca: ' + what + ' · ' + JSON.stringify(sPanel()));
  }
};
const sPost = (path, body) => sFetch0.call(window, path, {method: 'POST', body: body}).catch(() => null);
const sStep = async (name, fn) => {
  await sPost('/probe-step', name);
  try { sOut[name] = await fn(); } catch (e) { sOut[name] = 'ERROR ' + e.message; }
};
const sCard = id => sId('post-' + id);
const sButtons = (root) => (root ? [...root.querySelectorAll('button.evalb')].filter(seen).map(b => b.textContent) : []);
const sPress = (root, text) => [...root.querySelectorAll('button.evalb')].find(b => seen(b) && b.textContent === text).click();
const sPanel = () => ({
  shown: seen(sId('jobp')),
  title: txt(sId('ask-title')),
  est: txt(sId('ask-est')),
  error: txt(sId('ask-error')),
  force: seen(sId('ask-force')) ? sId('ask-force').checked : null,
  force_label: seen(sId('ask-force')) ? txt(sId('ask-force').parentNode) : null,
  go: seen(sId('ask-go')) ? !sId('ask-go').disabled : null,
  go_text: txt(sId('ask-go')),
  stop: txt(sId('ask-stop')),
  reload: txt(sId('ask-reload')),
  close: txt(sId('ask-cancel')),
  progress: txt(sId('ask-progress')),
});
// The top tabs whose job block is out of sight, and what their badge says («tab:words»).
const sBadges = () => [...document.querySelectorAll('#toptabs .tbadge')].filter(seen).map(b => b.parentNode.dataset.tab + ':' + b.textContent);
// Which top tab the job block is in (`top-revisar`, `top-ask`, `top-config`).
const sPanelTop = () => { const t = sId('jobp') && sId('jobp').closest('.toptab'); return t ? t.id : null; };
const sBar = () => {
  const t = document.querySelector('#jobp [role=progressbar]');
  const f = t && t.firstElementChild;
  return {now: t && t.getAttribute('aria-valuenow'), width: f && f.style.width,
    fill: seen(f) ? f.getBoundingClientRect().width : null, track: seen(t) ? t.getBoundingClientRect().width : null};
};
const sHeader = () => {
  sId('hist').open = true;
  return {kpis: txt(sId('kpis')), numbers: txt(sId('numbers-sub')), runs: txt(sId('hist')),
    runs_rows: [...document.querySelectorAll('#hist tbody tr')].filter(seen).length,
    before_log: txt(sId('before-log')), count: txt(sId('count')),
    total: txt(document.querySelector('#kpis .kpi .num')), total_amber: document.querySelector('#kpis .kpi .num').classList.contains('accent')};
};
let sRefreshed = 0;
const sRefresh0 = refresh;
refresh = function (blob) { sRefresh0(blob); sRefreshed++; };
let sEstimates = 0, sLastEstimate = null;
const sFetchRaw = window.fetch;
// The probe's own requests: one that fails at the network says which it was (a bare «Failed
// to fetch» names nothing). The page's requests keep the browser's own error.
const sFetch0 = (u, init) => sFetchRaw.call(window, u, init).catch((e) => {
  throw new Error(e.message + ' (' + ((init && init.method) || 'GET') + ' ' + u + ')');
});
let sPosts = 0;
window.fetch = function (u, init) {
  const p = sFetchRaw.call(window, u, init);
  if (init && init.method === 'POST') sPosts++;
  if (String(u).includes('/estimate')) {
    p.then(r => r.clone().json()).then(j => { sLastEstimate = j; sEstimates++; }, () => { sEstimates++; });
  }
  return p;
};
const sNext = async (n) => {
  await sWait(() => seen(sId('evalnext')), 'el control de los siguientes');
  const box = sId('evalnext');
  const input = box.querySelector('input');
  input.value = String(n);
  input.dispatchEvent(new Event('input'));
  sPress(box, 'Evaluar los ' + n + ' siguientes sin evaluar');
  await sWait(() => sPanel().go, 'la estimación');
};
const sQuery = async (n) => {
  location.hash = '#ask';
  await sWait(() => seen(sId('ask-form')), 'el formulario de Preguntar');
  for (const [id, value] of [['ask-q', 'hooks'], ['ask-limit', String(n)]]) {
    sId(id).value = value;
    sId(id).dispatchEvent(new Event('input', {bubbles: true}));
  }
  sPress(sId('ask-form'), 'Estimar lo que cuesta');
  await sWait(() => sPanel().go, 'la estimación');
};
// The output goes to the test server, which ends Chrome as soon as it has it: `--dump-dom`
// alone waits for the virtual-time budget to drain, which a hung page never lets happen.
const sDone = () => {
  const pre = document.createElement('pre');
  pre.id = 'probe';
  pre.textContent = JSON.stringify(sOut);
  document.body.appendChild(pre);
  sPost('/probe-done', pre.textContent);
};
"""
)

_SERVE_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('before', async () => {
    await sWait(() => seen(sCard('3')), 'la tarjeta 3');
    return {
      card3: sButtons(sCard('3')),
      card1: sButtons(sCard('1')),
      copy: [...document.querySelectorAll('.ask code')].filter(seen).length,
      next: sButtons(sId('evalnext')),
      order: seenCards(sId('cards')),
    };
  });
  await sStep('estimate', async () => {
    sPress(sCard('3'), 'Evaluar este post');
    await sWait(() => sPanel().go, 'la estimación');
    return Object.assign(sPanel(), {focused: document.activeElement && document.activeElement.id});
  });
  await sStep('done', async () => {
    sId('ask-go').click();
    await sWait(() => sRefreshed > 0, 'el final del trabajo');
    return Object.assign(sPanel(), {
      card3: sButtons(sCard('3')),
      card3text: txt(sCard('3')),
      order: seenCards(sId('cards')),
    });
  });
  await sStep('force', async () => {
    sPress(sCard('3'), 'Re-evaluar');
    await sWait(() => (sPanel().error || '').includes('nada que evaluar'), 'la negativa sin forzar');
    const unforced = sPanel();
    sId('ask-force').click();
    await sWait(() => (sPanel().est || '').includes('re-evalúa') && sPanel().go, 'la estimación forzada');
    const forced = sPanel();
    // The box changed after the estimate with no new one (no change event): Confirmar refuses.
    sId('ask-force').checked = false;
    sId('ask-go').click();
    await sWait(() => (sPanel().error || '').includes('casilla'), 'la negativa de la casilla');
    const mismatch = sPanel();
    // Two estimates in flight, the forced one slower (the server delays it): the last counts.
    const n0 = sEstimates;
    sId('ask-force').checked = true;
    estimate();
    sId('ask-force').checked = false;
    estimate();
    const held = sId('ask-force').disabled;
    await sWait(() => sEstimates >= n0 + 2, 'las dos estimaciones');
    for (let i = 0; i < 5; i++) await sFetch0.call(window, '/probe-wait');
    const raced = Object.assign(sPanel(), {held: held, released: !sId('ask-force').disabled});
    // Ticked by hand: a forced estimate, then the forced job with what was estimated.
    sId('ask-force').click();
    await sWait(() => (sPanel().est || '').includes('re-evalúa') && sPanel().go, 'la estimación forzada otra vez');
    const r0 = sRefreshed;
    sId('ask-go').click();
    await sWait(() => sRefreshed > r0, 'el trabajo forzado');
    const ran = sPanel();
    sId('ask-cancel').click();
    return {unforced, forced, mismatch, raced, ran, closed: !seen(sId('jobp'))};
  });
  await sStep('escape', async () => {
    sPress(sCard('4'), 'Evaluar este post');
    await sWait(() => sPanel().go, 'la estimación del 4');
    const open = sPanel().shown;
    const onTitle = document.activeElement === sId('ask-title');
    const opener = [...sCard('4').querySelectorAll('button.evalb')].find(b => b.textContent === 'Evaluar este post');
    // In the page, right under the card whose button opened it — not floating over it.
    const box = sId('jobp'), r = box.getBoundingClientRect(), card = sCard('4').getBoundingClientRect();
    const placed = {position: getComputedStyle(box).position, top: sPanelTop(), tab: box.closest('.tab').id,
      after: box.previousElementSibling && box.previousElementSibling.id, below_card: r.top >= card.bottom - 1,
      in_flow: box.offsetParent !== null && getComputedStyle(box).position === 'static'};
    sId('jobp').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    await sFetch0.call(window, '/probe-wait');
    return {open, placed, closed: !seen(sId('jobp')), on_title: onTitle,
      focus_back: document.activeElement === opener, roles: {
      region: sId('jobp').getAttribute('role'), error: sId('ask-error').getAttribute('role'),
      live: sId('ask-progress').getAttribute('aria-live'),
      bar: !!document.querySelector('#jobp [role=progressbar][aria-valuemin="0"][aria-valuemax="100"]')}};
  });
  await sStep('places', async () => {
    const go = async (hash, root) => {
      location.hash = hash;
      await sWait(() => sButtons(sId(root)).length > 0, hash);
      return sButtons(sId(root));
    };
    const band = Object.keys(DATA.post_sets.bands).find(k => DATA.post_sets.bands[k].length);
    const cx = Object.keys(DATA.post_sets.cx)[0];
    const places = {
      posts_topic: await go('#posts?f=all&t=startups', 'active'),
      topic: await go('#topics?t=startups', 'topic-detail'),
      topic_pair: await go('#topics?t=ai-coding&cx=' + encodeURIComponent(cx), 'topic-pair'),
      band: await go('#compare?b=' + encodeURIComponent(band), 'compare-list'),
    };
    // Where the block opens for each origin: right under the row of buttons it came from.
    const where = () => { const box = sId('jobp'), prev = box.previousElementSibling;
      return {in: box.parentNode.id, after: prev ? prev.id || prev.className : null, shown: seen(box)}; };
    const pressIn = async (hash, root) => {
      location.hash = hash;
      await sWait(() => sButtons(sId(root)).length > 0, hash);
      [...sId(root).querySelectorAll('button.evalb')].find(seen).click();
      await sWait(() => sPanel().go || sPanel().error, 'la estimación de ' + root);
      return where();
    };
    places.placed = {};
    places.placed.active = await pressIn('#posts?f=all&t=startups', 'active');
    sId('ask-cancel').click();
    places.placed.topic = await pressIn('#topics?t=startups', 'topic-detail');
    sId('ask-cancel').click();
    places.placed.topic_pair = await pressIn('#topics?t=ai-coding&cx=' + encodeURIComponent(cx), 'topic-pair');
    sId('ask-cancel').click();
    places.placed.band = await pressIn('#compare?b=' + encodeURIComponent(band), 'compare-list');
    places.band_panel = sPanel();
    // Its row no longer on screen (another sub-tab): the block goes to the top of Revisar.
    location.hash = '#revisar/topics';
    await sWait(() => seen(sId('tab-topics')), 'Topics');
    places.placed.hidden_anchor = where();
    sId('ask-cancel').click();
    return places;
  });
  sDone();
})();
</script>"""
)


def _page_saw(done: int = 0) -> tuple[Any, Any]:
    """A `JevService` subclass that notes when the PAGE has been told a job is running with at
    least `done` posts done, and the event it sets: a fake answer waits on it, so what the
    page must see mid-job is there however slowly Chrome runs (never a guessed sleep)."""
    import threading

    from xbrain.jev.service import JevService

    saw = threading.Event()

    class _Watched(JevService):
        def job_view(self) -> dict[str, Any]:
            view = super().job_view()
            if view["state"] == "running" and view["done"] >= done:
                saw.set()
            return view

    return _Watched, saw


def _served_dump(
    root: Path,
    probe: str,
    *,
    client: Any = None,
    make_client: Any = None,
    jev: str = "",
    seed_tokens: int = 100,
    prepare: Any = None,
    base: Any = None,
    patch: Any = None,
    before_dump: Any = None,
    repo: Any = None,
) -> dict[str, Any]:
    """`probe` run in Chrome against a real `JevService` (or `base`, a subclass) over `_repo`
    (or `repo(root, monkeypatch)`, another repo builder),
    every job asking `client` — a fake, never TypeSafe. `prepare(cfg)` edits the repo first,
    `patch(monkeypatch)` swaps what a scenario needs, `before_dump(service, port)` runs with
    the server up. Returns the probe's output with the last job's view as `job`."""
    import threading

    from tests.test_jev_serve import _repo
    from xbrain.jev.serve import make_server
    from xbrain.jev.service import JevService

    _need_chrome()
    monkeypatch = pytest.MonkeyPatch()
    try:
        cfg = (
            repo(root, monkeypatch)
            if repo is not None
            else _repo(root, monkeypatch, jev=jev, seed_tokens=seed_tokens)
        )
        if prepare is not None:
            prepare(cfg)
        if patch is not None:
            patch(monkeypatch)

        class _Probed(base or JevService):
            def page_html(self) -> str:
                return super().page_html().replace("</body>", probe + "</body>")

        service = _Probed(cfg, make_client or (lambda: client))
        server = make_server(service, 0)
        server.RequestHandlerClass = _probe_handler()
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        thread.start()
        try:
            port = server.server_address[1]
            if before_dump is not None:
                before_dump(service, port)
            seen = _dump_served(server, f"http://127.0.0.1:{port}/#posts?f=all")
            service.wait(15)
            seen["job"] = JevService.job_view(service)
        finally:
            server.shutdown()
            server.server_close()
            service.stop()
        return seen
    finally:
        monkeypatch.undo()


#: Real seconds a served probe may take before Chrome is ended and the last step is named.
_SERVED_DEADLINE_S = 110


#: Real seconds `/probe-wait` holds each look. A wait is a loop of looks while Chrome's clock
#: stands still; unpaced, one that never comes true opened ~2,500 connections a second and
#: ran the machine out of ephemeral ports (TIME_WAIT) long before its 30 s bound — a bare
#: «Failed to fetch» on Linux, a stalled Chrome on macOS (16,377 looks: its 16,384 ports).
#: Paced, 30 s of looks is ~3,000 connections.
_PROBE_WAIT_S = 0.01


def _probe_handler() -> Any:
    """The server's handler, plus the routes only a probe uses: `/probe-step` (the step now
    running) and `/probe-done` (the output), which land on the server object, and
    `/probe-wait` (one paced look: 204 with the server's `Date`, for `sWait`)."""
    from xbrain.jev.serve import _Handler

    class _ProbeHandler(_Handler):
        def do_GET(self) -> None:  # noqa: N802 — the stdlib's name
            if self.path != "/probe-wait":
                super().do_GET()
                return
            time.sleep(_PROBE_WAIT_S)
            self.send_response(204)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 — the stdlib's name
            if self.path not in ("/probe-step", "/probe-done"):
                super().do_POST()
                return
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
            if self.path == "/probe-step":
                self.server.probe_step = body  # type: ignore[attr-defined]
            else:
                self.server.probe_out = body  # type: ignore[attr-defined]
            self.send_response(204)
            self.end_headers()

    return _ProbeHandler


def _dump_served(server: Any, url: str) -> dict[str, Any]:
    """Chrome on `url` until the probe POSTs its output (then Chrome is ended), or until
    `_SERVED_DEADLINE_S` — a failure that names the step the probe was in."""
    assert CHROME is not None
    server.probe_out = server.probe_step = None
    chrome = subprocess.Popen(  # nosec B603 - fixed argv, a local server this test made
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            _NO_NETWORK,
            "--virtual-time-budget=900000",
            "--dump-dom",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + _SERVED_DEADLINE_S
    try:
        while server.probe_out is None and chrome.poll() is None:
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"el probe no terminó en {_SERVED_DEADLINE_S} s; último paso: "
                    f"{server.probe_step!r}"
                )
            time.sleep(0.05)
    finally:
        if chrome.poll() is None:
            chrome.kill()
        stdout, stderr = chrome.communicate()
    if server.probe_out is not None:
        seen = json.loads(server.probe_out)
    else:
        found = re.search(r'<pre id="probe">(.*?)</pre>', stdout, re.S)
        assert found, (
            f"la página no escribió el probe (rc={chrome.returncode}, último paso: "
            f"{server.probe_step!r}): {stderr[-800:]}"
        )
        seen = json.loads(html.unescape(found.group(1)))
    if any(_failed_step(v) for v in seen.values()):
        # What Chrome said, for the step that failed (`_step` shows it).
        seen[_CHROME_SAID] = f"{len(stderr)} caracteres; el final: {stderr[-3000:]}"
    return seen


#: The key `_dump_served` adds, when a step failed, with the end of Chrome's stderr.
_CHROME_SAID = "chrome_stderr"


def _failed_step(value: Any) -> bool:
    """A probe step that threw: `sStep` stores it as the string `ERROR <message>`."""
    return isinstance(value, str) and value.startswith("ERROR ")


def _step(seen: dict[str, Any], name: str) -> Any:
    """Step `name` of a served probe's output; a step that threw fails HERE, naming the step,
    its error (the request, when a request failed) and what Chrome said — not later, as a
    TypeError on a string indexed like the dict it should have been."""
    assert name in seen, f"el probe no llegó al paso {name!r}: {sorted(seen)}"
    value = seen[name]
    assert not _failed_step(value), (
        f"el paso {name!r} del probe falló: {value} · Chrome: {seen.get(_CHROME_SAID)}"
    )
    return value


def _stale_post_2(cfg: Any) -> None:
    """Post 2 («startups», answered by the seed pass) edited since: its answer is stale."""
    from xbrain.store import load_store, save_store

    items = load_store(cfg.items_path)
    items["2"].text = "Seed round tips, revisado"
    save_store(items, cfg.items_path)


@pytest.fixture(scope="module")
def serve_probed(tmp_path_factory) -> dict[str, Any]:
    import time

    from tests.test_jev_serve import _Recorder
    from xbrain.jev.service import JevService

    class _SlowForce(JevService):
        """A forced estimate answers late, so two estimates in flight come back reversed."""

        def estimate(self, kind: str, body: Any) -> dict[str, Any]:
            if isinstance(body, dict) and body.get("force"):
                time.sleep(0.5)
            return super().estimate(kind, body)

    client = _Recorder()
    seen = _served_dump(
        tmp_path_factory.mktemp("served"),
        _SERVE_PROBE,
        client=client,
        prepare=_stale_post_2,
        base=_SlowForce,
    )
    seen["asked"] = client.asked
    return seen


#: What the force checkbox says, word for word: what it costs and what it replaces.
_FORCE_LABEL = (
    "Volver a evaluar también los posts que ya tienen evaluación vigente: se pagan otra vez y "
    "se sustituye su evaluación (antes se guarda una copia)"
)


@_requires_chrome
def test_served_cards_offer_to_evaluate_instead_of_a_command(serve_probed):
    before = serve_probed["before"]

    assert before["card3"] == ["Evaluar este post"]
    assert before["card1"] == ["Re-evaluar"]
    assert before["copy"] == 0
    assert before["next"] == ["Evaluar los 20 siguientes sin evaluar"]


@_requires_chrome
def test_served_estimate_says_what_it_will_pay_before_anything_is_spent(serve_probed):
    estimate = serve_probed["estimate"]

    assert estimate["title"] == "Evaluar este post"
    # Post 2 is stale: the mean is the one current priced answer's.
    assert estimate["est"] == (
        "1 post por evaluar. Coste estimado: ~0,00000 $ (coste medio por post de la "
        "evaluación ya pagada × 1 post). Tope: 1,00 $. Al llegar se para; lo que ya esté en "
        "vuelo termina y puede pasarlo por poco (como mucho 1 post)."
    )
    assert estimate["go_text"] == "Evaluar y pagar ~0,00000 $"
    assert (estimate["force"], estimate["go"], estimate["error"]) == (False, True, None)
    assert estimate["force_label"] == _FORCE_LABEL
    assert estimate["focused"] == "ask-title"


@_requires_chrome
def test_served_confirm_runs_one_job_and_the_card_updates_in_place(serve_probed):
    done = serve_probed["done"]

    assert serve_probed["asked"][0] == "3"
    assert done["progress"].startswith("1 evaluación guardada · ~0,00000 $ gastado")
    assert done["card3"] == ["Re-evaluar"]
    assert "sin evaluar por Jev" not in done["card3text"]
    # In place: the same cards, in the same order, the evaluated one swapped where it was.
    assert done["order"] == serve_probed["before"]["order"]


@_requires_chrome
def test_served_force_is_unticked_until_the_reader_ticks_it(serve_probed):
    force = serve_probed["force"]

    assert force["unforced"]["force"] is False and force["unforced"]["go"] is False
    assert force["unforced"]["error"].startswith("No se puede evaluar: nada que evaluar")
    assert force["forced"]["force"] is True and force["forced"]["go"] is True
    assert "1 vigente que se re-evalúa" in force["forced"]["est"]
    # The forced job ran with what was estimated, and names the copy made before it.
    assert force["ran"]["progress"].startswith("1 evaluación guardada")
    assert "copia previa de las evaluaciones: " in force["ran"]["progress"]
    assert "/data/jev/topics." in force["ran"]["progress"] and force["ran"]["progress"].endswith(
        ".bak"
    )
    assert serve_probed["asked"] == ["3", "3"]
    assert force["closed"] is True


@_requires_chrome
def test_served_confirm_refuses_when_the_checkbox_changed_since_the_estimate(serve_probed):
    mismatch = serve_probed["force"]["mismatch"]

    assert mismatch["force"] is False and mismatch["go"] is False
    assert "casilla" in mismatch["error"] and "vuelve a estimar" in mismatch["error"]


@_requires_chrome
def test_served_a_late_estimate_reply_never_overwrites_the_last_one(serve_probed):
    """The forced estimate was sent first and answered last: the panel keeps the unforced
    answer, as the checkbox shows."""
    raced = serve_probed["force"]["raced"]

    assert raced["force"] is False and raced["go"] is False
    # The box is held while an estimate is out, and given back with the answer.
    assert raced["held"] is True and raced["released"] is True
    assert "re-evalúa" not in (raced["est"] or "")
    assert raced["error"].startswith("No se puede evaluar: nada que evaluar")


@_requires_chrome
def test_served_panel_is_a_labelled_region_that_escape_closes(serve_probed):
    escape = serve_probed["escape"]

    assert escape["open"] is True and escape["closed"] is True
    # Focus is on the panel while it is open, and back on the very button that opened it.
    assert escape["on_title"] is True and escape["focus_back"] is True
    assert escape["roles"] == {"region": "region", "error": "alert", "live": "polite", "bar": True}


@_requires_chrome
def test_served_the_estimate_block_is_in_the_page_right_under_the_card_that_opened_it(
    serve_probed,
):
    placed = serve_probed["escape"]["placed"]

    # Víctor, 2026-09-27: «que esté integrado en la página» — never the floating corner panel.
    assert placed["position"] == "static" and placed["in_flow"] is True
    assert (placed["top"], placed["tab"], placed["after"]) == ("top-revisar", "tab-posts", "post-4")
    assert placed["below_card"] is True


@_requires_chrome
def test_served_topic_pair_and_band_views_offer_to_evaluate_their_posts(serve_probed):
    places = serve_probed["places"]

    # Under «startups»: post 2 (its answer is stale) and post 4 (never asked).
    assert places["posts_topic"] == ["Evaluar este topic (2 sin evaluar)"]
    assert "Evaluar este topic (2 sin evaluar)" in places["topic"]
    assert places["topic_pair"][0] == "Evaluar estos posts"
    assert places["band"][0] == "Evaluar estos posts"
    # Posts on a band are current: unticked, the panel says there is nothing to ask.
    assert places["band_panel"]["force"] is False
    assert places["band_panel"]["error"].startswith("No se puede evaluar: nada que evaluar")
    assert "la casilla «Volver a evaluar también…»" in places["band_panel"]["error"]
    # Nothing to ask costs nothing: no «Coste estimado» of zero posts.
    assert "Coste estimado" not in places["band_panel"]["est"]
    # Each origin opens the block right under its own row of buttons, never elsewhere; with
    # that row off screen (another sub-tab), the block is at the top of Revisar.
    assert places["placed"] == {
        "active": {"in": "", "after": "active", "shown": True},
        "topic": {"in": "topic-detail", "after": "evalrow", "shown": True},
        "topic_pair": {"in": "topic-pair", "after": "evalrow", "shown": True},
        "band": {"in": "compare-list", "after": "evalrow", "shown": True},
        "hidden_anchor": {"in": "jobslot-revisar", "after": None, "shown": True},
    }


# --------------------------------------------------------------------------- one job, followed


_PROGRESS_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('before', async () => { await sWait(() => seen(sCard('3')), 'la tarjeta 3'); return sHeader(); });
  await sStep('estimate', async () => {
    await sNext(3);
    return Object.assign(sPanel(), {server: sLastEstimate});
  });
  await sStep('mid', async () => {
    sId('ask-go').click();
    await sWait(() => /^[12] de 3 posts/.test(sPanel().progress || ''), 'la mitad del trabajo');
    // Escape does not close a running job's block; and the block sits under the Posts toolbar
    // whose button started it.
    sId('jobp').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    const kept = sPanel().shown;
    const after = sId('jobp').previousElementSibling;
    return Object.assign(sPanel(), sBar(), {escape_kept: kept, top: sPanelTop(),
      under_toolbar: !!after && after.classList.contains('controls')});
  });
  await sStep('end', async () => {
    await sWait(() => sRefreshed > 0, 'la recarga');
    return Object.assign(sPanel(), sHeader(), {bar: sBar(),
      cards: ['3', '4', '5'].map(i => sButtons(sCard(i)))});
  });
  await sStep('elsewhere', async () => {
    const d = await (await fetch('/api/data')).json();
    location.hash = '#topics?t=ai-coding';
    await sWait(() => sButtons(sId('topic-detail')).length > 0, 'el topic');
    const topic = sButtons(sId('topic-detail')).filter(b => b.startsWith('Evaluar este topic'));
    location.hash = '#config';
    const kinds = () => [...document.querySelectorAll('#config-body .kind')].filter(seen);
    await sWait(() => kinds().length > 0, 'la configuración');
    const pending = kinds().find(k => txt(k.querySelector('.lab')) === 'Lo que falta');
    return {topic, pending_big: txt(pending.querySelector('.cnum')), pending_why: txt(pending.querySelector('p')),
      data_pending: d.config.estimate.pending,
      data_uneval: d.posts.filter(p => p.slugs.includes('ai-coding') && p.in.includes('uneval') && !p.no_evidence).length};
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def progress_probed(tmp_path_factory) -> dict[str, Any]:
    """Víctor's machine: priced answers and NO runs.jsonl yet. A non-default cap, and a mean
    that shows at five decimals. The second answer waits until the page has been told one
    post is done, so the half-way state is on screen."""
    from tests.test_jev_serve import _Recorder

    watched, saw = _page_saw(done=1)

    class _HalfWay(_Recorder):
        def ask(self, state, questions):
            if len(self.asked) == 1:
                saw.wait(60)
                time.sleep(0.3)
            return super().ask(state, questions)

    def _no_run_log(cfg: Any) -> None:
        cfg.jev_runs_path.unlink()

    return _served_dump(
        tmp_path_factory.mktemp("progress"),
        _PROGRESS_PROBE,
        client=_HalfWay(input_tokens=1000),
        base=watched,
        jev="serve_max_usd = 0.25\nconcurrency = 2\n",
        seed_tokens=1000,
        prepare=_no_run_log,
    )


@_requires_chrome
def test_served_money_in_the_panel_is_the_servers_estimate_and_cap(progress_probed):
    estimate = progress_probed["estimate"]
    server = estimate["server"]

    # 1000 tokens per answer at 0.042 $/MTok = 4.2e-5 $ a post; three posts.
    assert server["usd"] == pytest.approx(3 * 4.2e-5) and server["max_usd"] == 0.25
    assert estimate["go_text"] == "Evaluar y pagar ~0,00013 $"
    assert (
        "Coste estimado: ~0,00013 $ (coste medio por post de las 2 evaluaciones ya pagadas × "
        "3 posts). Tope: 0,25 $. Al llegar se para; lo que ya esté en vuelo termina y puede "
        "pasarlo por poco (como mucho 2 posts)."
    ) in estimate["est"]
    assert estimate["est"].startswith("3 posts por evaluar")


@_requires_chrome
def test_served_progress_mid_job_shows_the_share_done(progress_probed):
    mid = progress_probed["mid"]
    done = int(mid["progress"][0])

    assert mid["progress"].startswith(f"{done} de 3 posts · ")
    assert mid["now"] == str(round(100 * done / 3)) and mid["width"] == mid["now"] + "%"
    # The fill is drawn short of its track (no rule from the cost strip stretches it).
    assert 0 <= mid["fill"] < 0.9 * mid["track"]
    assert mid["stop"] == "Parar (se guarda lo ya pagado)" and mid["close"] == "Ocultar"


@_requires_chrome
def test_served_a_running_job_stays_under_the_toolbar_that_started_it_and_escape_keeps_it(
    progress_probed,
):
    mid = progress_probed["mid"]

    assert mid["escape_kept"] is True and mid["shown"] is True
    assert mid["top"] == "top-revisar" and mid["under_toolbar"] is True


@_requires_chrome
def test_served_a_job_without_a_run_log_redraws_the_header_and_the_history(progress_probed):
    before, end = progress_probed["before"], progress_probed["end"]

    assert "sin pasadas registradas" in before["kpis"]
    assert "Aún no hay pasadas registradas" in before["runs"]
    # No logged pass, but 2 paid evaluations: the first number is what they cost, said as an
    # estimate from their own tokens — never an amber «—» with the cost in a note below it.
    assert before["total"] == "~0,0001 $" and before["total_amber"] is True
    assert (
        "sin pasadas registradas: lo que costaron las 2 evaluaciones guardadas, "
        "estimado por sus propios tokens" in before["kpis"]
    )
    assert before["before_log"] is None
    assert end["error"] is None
    assert end["progress"].startswith("3 evaluaciones guardadas · ~0,00013 $ gastado")
    assert end["bar"]["now"] == "100"
    assert "3 peticiones" in end["kpis"] and "1 pasada" in end["kpis"]
    assert "sin pasadas registradas" not in end["kpis"]
    assert end["runs_rows"] == 1 and "Aún no hay pasadas registradas" not in end["runs"]
    assert "2 posts comparados" in before["numbers"] and "5 posts comparados" in end["numbers"]
    assert end["count"] == before["count"]
    assert end["cards"] == [["Re-evaluar"]] * 3


@_requires_chrome
def test_served_other_tabs_show_the_data_after_the_job(progress_probed):
    elsewhere = progress_probed["elsewhere"]

    assert elsewhere["data_uneval"] == 0
    assert elsewhere["topic"] == ["Evaluar este topic (0 sin evaluar)"]
    posts = elsewhere["data_pending"]["posts"]
    assert posts == 0
    assert elsewhere["pending_why"].startswith(f"{posts} posts")


# --------------------------------------------------------------------------- a job not started here

_RESUME_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('resumed', async () => {
    await sWait(() => sPanel().shown && /de 3 posts/.test(sPanel().progress || ''), 'el panel del trabajo en curso');
    return sPanel();
  });
  await sStep('end', async () => {
    await sWait(() => sRefreshed > 0, 'la recarga');
    return Object.assign(sPanel(), {cards: ['3', '4', '5'].map(i => sButtons(sCard(i)))});
  });
  sDone();
})();
</script>"""
)


def _start_job(service: Any, port: int, body: dict[str, Any], kind: str = "topics") -> None:
    """Start a job of `kind` over HTTP, as another tab would."""
    import http.client

    from xbrain.jev.serve import TOKEN_HEADER

    headers = {
        "Content-Type": "application/json",
        TOKEN_HEADER: service.token,
        "Origin": f"http://127.0.0.1:{port}",
    }
    replies = []
    for path, sent in ((f"/api/{kind}/estimate", body), (f"/api/{kind}/evaluate", None)):
        if sent is None:
            sent = {**body, "confirm_token": replies[-1]["confirm_token"]}
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", path, body=json.dumps(sent), headers=headers)
        replies.append(json.loads(conn.getresponse().read()))
        conn.close()
    assert replies[-1]["state"] == "running", replies


@pytest.fixture(scope="module")
def resume_probed(tmp_path_factory) -> dict[str, Any]:
    """The page opened (or reloaded) while a job runs: its first answer waits until the page
    has been told the job is running, however long Chrome takes to start."""
    from tests.test_jev_serve import _Recorder

    watched, saw = _page_saw()

    class _SlowFirst(_Recorder):
        def ask(self, state, questions):
            if not self.asked:
                saw.wait(60)
            return super().ask(state, questions)

    return _served_dump(
        tmp_path_factory.mktemp("resume"),
        _RESUME_PROBE,
        client=_SlowFirst(),
        base=watched,
        before_dump=lambda service, port: _start_job(service, port, {"unevaluated": 3}),
    )


@_requires_chrome
def test_served_a_page_opened_mid_job_follows_it(resume_probed):
    resumed, end = resume_probed["resumed"], resume_probed["end"]

    assert resumed["title"] == "Evaluación de topics en curso"
    assert re.match(r"^[0-2] de 3 posts · ", resumed["progress"])
    assert resumed["est"].endswith(
        "Tope: 1,00 $. Al llegar se para; lo que ya esté en vuelo termina y puede pasarlo por "
        "poco (como mucho 1 post)."
    )
    assert resumed["stop"] == "Parar (se guarda lo ya pagado)" and resumed["go"] is None
    assert end["progress"].startswith("3 evaluaciones guardadas")
    assert end["cards"] == [["Re-evaluar"]] * 3


_WATCH_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
const sAsk = async (body) => {
  const h = {'Content-Type': 'application/json', 'X-Xbrain-Token': DATA.serve.token};
  const e = await (await sFetch0('/api/topics/estimate', {method: 'POST', headers: h, body: JSON.stringify(body)})).json();
  const r = await sFetch0('/api/topics/evaluate', {method: 'POST', headers: h,
    body: JSON.stringify(Object.assign({}, body, {confirm_token: e.confirm_token}))});
  return r.status;
};
(async () => {
  await sStep('finished_elsewhere', async () => {
    await sWait(() => seen(sCard('3')), 'la tarjeta 3');
    const idle = sPanel().shown;
    const status = await sAsk({ids: ['3']});
    await sWait(() => sButtons(sCard('3')).includes('Re-evaluar'), 'la tarjeta 3 recargada');
    return {idle, status, card3: sButtons(sCard('3'))};
  });
  await sStep('running_elsewhere', async () => {
    // An estimate opened here under card 5 and cancelled: nothing of it is kept, so another
    // tab's job is never placed under that card.
    sPress(sCard('5'), 'Evaluar este post');
    await sWait(() => sPanel().go, 'la estimación del 5');
    sId('ask-cancel').click();
    const status = await sAsk({ids: ['4']});
    await sWait(() => sPanel().shown && (sPanel().progress || '').includes('de 1 post '), 'el panel del trabajo de otra pestaña');
    const box = sId('jobp'), prev = box.previousElementSibling;
    const during = Object.assign(sPanel(), {placed: {in: box.parentNode.id, after: prev ? prev.id : null}});
    await sWait(() => sButtons(sCard('4')).includes('Re-evaluar'), 'la tarjeta 4 recargada');
    return {status, during, after: sPanel(), card4: sButtons(sCard('4'))};
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def watch_probed(tmp_path_factory) -> dict[str, Any]:
    """An idle page while another tab runs jobs: one quick, and one (post 4) whose answer
    waits until the page has been told it is running."""
    from tests.test_jev_serve import _Recorder

    watched, saw = _page_saw()

    class _Slow4(_Recorder):
        def ask(self, state, questions):
            if state["post"].startswith("Series A"):
                saw.wait(60)
            return super().ask(state, questions)

    return _served_dump(
        tmp_path_factory.mktemp("watch"), _WATCH_PROBE, client=_Slow4(), base=watched
    )


@_requires_chrome
def test_served_an_idle_page_reloads_when_another_tab_s_job_ends(watch_probed):
    done = watch_probed["finished_elsewhere"]

    assert done["idle"] is False and done["status"] == 202
    assert done["card3"] == ["Re-evaluar"]


@_requires_chrome
def test_served_an_idle_page_follows_a_job_another_tab_started(watch_probed):
    running = watch_probed["running_elsewhere"]

    assert running["status"] == 202
    assert running["during"]["title"] == "Evaluación de topics en curso"
    assert re.match(r"^0 de 1 post · ", running["during"]["progress"])
    # Not under card 5, whose estimate was opened here and cancelled: at the top of Revisar.
    assert running["during"]["placed"] == {"in": "jobslot-revisar", "after": None}
    assert running["card4"] == ["Re-evaluar"]
    assert running["after"]["progress"].startswith("1 evaluación guardada")


# --------------------------------------------------------------------------- how a job can end

_END_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
const sScenario = /*SCENARIO*/;
(async () => {
  await sStep('estimate', async () => {
    await (sScenario.kind === 'ask' ? sQuery : sNext)(sScenario.n);
    return sPanel();
  });
  await sStep('run', async () => {
    sId('ask-go').click();
    await sWait(() => sPanel().stop !== null || sPanel().close === 'Cerrar', 'el trabajo');
    let hidden = null;
    if (sScenario.hide) { sId('ask-cancel').click(); hidden = !sPanel().shown; }
    if (sScenario.stop) {
      await sWait(() => /[1-9]\d* respuesta/.test(sPanel().progress || ''), 'la primera respuesta');
      sPress(sId('jobp'), 'Parar (se guarda lo ya pagado)');
    }
    return {hidden};
  });
  await sStep('end', async () => {
    // A lost server is declared only after POLL_MS + ΣRETRY_MS of retries on the PAGE's clock,
    // which a request in flight holds still — and each look of `sWait` is one. So the page's
    // clock runs first, idle, past every retry; then the look. (A scenario that ends sooner
    // is simply there on the first look.)
    await new Promise(r => setTimeout(r, POLL_MS + RETRY_MS.reduce((a, b) => a + b, 0) + 500));
    await sWait(() => sRefreshed > 0 || sPanel().reload !== null, 'el final');
    return sPanel();
  });
  sDone();
})();
</script>"""
)


def _ended(tmp_path: Path, scenario: str, kind: str = "topics") -> dict[str, Any]:
    """One job of `kind` (a topics pass from the Posts toolbar, or an ask from the Preguntar
    tab) that ends in `scenario`, the panel hidden by the reader right after Confirmar (except
    `stop`, which presses «Parar»). The ask's posts are 1–5 in store order, the topics' 3–5."""

    def _end_probe(n: int, *, hide: bool = False, stop: bool = False) -> str:
        if kind == "ask" and n == 3 and not stop:
            n = 5  # the failing post (4) and the unpriced one (5) are the ask's 4th and 5th
        scenario_ = {"n": n, "hide": hide, "stop": stop, "kind": kind}
        return _END_PROBE.replace("/*SCENARIO*/", json.dumps(scenario_))

    import threading
    from dataclasses import replace

    from tests.test_jev_serve import _Recorder
    from xbrain.jev import service as service_module
    from xbrain.jev.client import JevError
    from xbrain.jev.service import JevService

    def _outcome(change: Any) -> Any:
        name = "run_ask" if kind == "ask" else "run_topics"
        real = getattr(service_module, name)

        def run(*args: Any, **kwargs: Any) -> Any:
            return change(real(*args, **kwargs))

        return lambda monkeypatch: monkeypatch.setattr(service_module, name, run)

    if scenario == "tope":
        # Answers cost 3× the mean: after the first, the next one's reservation passes the cap.
        return _served_dump(
            tmp_path,
            _end_probe(2, hide=True),
            client=_Recorder(input_tokens=3000),
            seed_tokens=1000,
            jev="serve_max_usd = 0.0001\n",
        )
    if scenario == "error":

        def _no_key() -> Any:
            raise JevError("TYPESAFE_API_KEY no encontrada")

        return _served_dump(tmp_path, _end_probe(1, hide=True), make_client=_no_key)
    if scenario == "failures":

        class _Mixed(_Recorder):
            def ask(self, state, questions):
                if state["post"].startswith("Series A"):
                    raise JevError("respuesta ilegible")
                result = super().ask(state, questions)
                if state["post"].startswith("Agents"):
                    return replace(result, provider="fake", input_tokens=None)
                return result

        return _served_dump(
            tmp_path,
            _end_probe(3, hide=True),
            client=_Mixed(),
            patch=_outcome(lambda o: replace(o, logged=o.logged.model_copy(update={"unsaved": 2}))),
        )
    if scenario == "history":
        # The answers are paid, saved and logged; then the history cannot be written.
        def _no_history(*args: Any, **kwargs: Any) -> Any:
            raise OSError(28, "No space left on device")

        return _served_dump(
            tmp_path,
            _end_probe(2, hide=True),
            client=_Recorder(),
            patch=lambda monkeypatch: monkeypatch.setattr(
                service_module, "finish_ask", _no_history
            ),
        )
    if scenario == "not-logged":
        return _served_dump(
            tmp_path,
            _end_probe(1, hide=True),
            client=_Recorder(),
            patch=_outcome(lambda o: replace(o, logged=None)),
        )
    if scenario == "stop":
        stopped = threading.Event()

        class _Stoppable(JevService):
            def cancel_job(self) -> dict[str, Any]:
                view = super().cancel_job()
                stopped.set()
                return view

        class _UntilStopped(_Recorder):
            """The second answer waits for «Parar»: the third post is never sent."""

            def ask(self, state, questions):
                if len(self.asked) == 1:
                    stopped.wait(60)
                return super().ask(state, questions)

        return _served_dump(
            tmp_path, _end_probe(3, stop=True), client=_UntilStopped(), base=_Stoppable
        )
    if scenario == "flaky":

        class _Flaky(JevService):
            """`/api/job` fails twice once a job exists, then answers again."""

            failures = 0

            def job_view(self) -> dict[str, Any]:
                view = super().job_view()
                if view["state"] != "idle" and _Flaky.failures < 2:
                    _Flaky.failures += 1
                    raise RuntimeError("un momento")
                return view

        return _served_dump(tmp_path, _end_probe(1), client=_Recorder(delay=0.3), base=_Flaky)
    assert scenario == "lost"

    class _Lost(JevService):
        """`/api/job` breaks once a job exists: the page's polling loses the server."""

        def job_view(self) -> dict[str, Any]:
            view = super().job_view()
            if view["state"] != "idle":
                raise RuntimeError("se cayó")
            return view

    return _served_dump(tmp_path, _end_probe(1, hide=True), client=_Recorder(delay=0.3), base=_Lost)


@pytest.fixture(scope="module")
def ended(tmp_path_factory) -> Callable[[str], dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}

    def get(scenario: str) -> dict[str, Any]:
        if scenario not in cache:
            cache[scenario] = _ended(tmp_path_factory.mktemp(scenario), scenario)
        return cache[scenario]

    return get


@_requires_chrome
@pytest.mark.parametrize("scenario", ["tope", "error", "failures", "not-logged", "lost"])
def test_served_a_job_that_did_not_end_cleanly_is_shown_even_if_hidden(ended, scenario):
    seen = ended(scenario)

    assert seen["run"]["hidden"] is True
    assert seen["end"]["shown"] is True and seen["end"]["close"] == "Cerrar"


@_requires_chrome
def test_served_a_job_cut_by_the_cap_says_so(ended):
    seen = ended("tope")

    assert seen["job"]["reason"] == "tope"
    assert seen["end"]["progress"].startswith(
        "Interrumpido (se alcanzó el tope por trabajo): 1 evaluación guardada · "
    )


@_requires_chrome
def test_served_a_job_that_failed_says_why(ended):
    seen = ended("error")

    assert seen["job"]["state"] == "error"
    assert seen["end"]["error"] == "El trabajo falló: TYPESAFE_API_KEY no encontrada"


@_requires_chrome
def test_served_failures_unsaved_and_unpriced_answers_are_named(ended):
    end = ended("failures")["end"]

    assert end["progress"].startswith("2 evaluaciones guardadas · ")
    assert "1 respuesta cobrada a la media estimada" in end["progress"]
    assert "1 respuesta sin recuento de tokens" in end["progress"]
    assert "sin tarifa: fake" in end["progress"]
    assert "1 fallida: 4 (" in end["progress"] and "respuesta ilegible" in end["progress"]
    assert "2 respuestas pagadas sin guardar" in end["progress"]


@_requires_chrome
def test_served_a_pass_that_did_not_reach_the_run_log_says_so(ended):
    end = ended("not-logged")["end"]

    assert end["progress"].startswith("1 evaluación guardada")
    assert "la pasada no quedó en runs.jsonl" in end["progress"]


@_requires_chrome
def test_served_the_page_can_stop_a_job_and_what_was_paid_is_kept(ended):
    seen = ended("stop")

    assert seen["job"]["state"] == "interrupted" and seen["job"]["reason"] == "cancelado"
    assert seen["end"]["progress"].startswith("Interrumpido (lo paraste desde la página): ")
    kept = seen["job"]["outcome"]["ok"]
    assert 1 <= kept < 3
    assert f"{kept} evaluaci" in seen["end"]["progress"]


@_requires_chrome
def test_served_a_moment_without_the_server_is_retried_not_given_up(ended):
    end = ended("flaky")["end"]

    assert end["error"] is None and end["reload"] is None
    assert end["progress"].startswith("1 evaluación guardada · ")


@_requires_chrome
def test_served_a_lost_server_is_said_and_offers_to_reload(ended):
    end = ended("lost")["end"]

    assert end["error"].startswith("Se perdió el contacto con el servidor")
    assert end["reload"] == "Recargar la página"


# --------------------------------------------------------------------------- the static page

_STATIC_PROBE = (
    "<script>"
    + _SEEN
    + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const slug = DATA.posts.find(p => p.slugs.length).slugs[0];
  const band = Object.keys(DATA.post_sets.bands).find(k => DATA.post_sets.bands[k].length);
  const cx = Object.keys(DATA.post_sets.cx)[0];
  const views = ['#posts?f=all', '#posts?f=all&t=' + slug, '#posts?f=uneval', '#topics', '#topics?t=' + slug,
    '#topics?t=' + slug + '&cx=' + encodeURIComponent(cx), '#compare', '#compare?b=' + encodeURIComponent(band), '#ask', '#config'];
  const out = {};
  for (const hash of views) {
    location.hash = hash;
    await sleep(60);
    out[hash.split('?')[0].slice(1) + (hash.includes('?') ? '?' + hash.split('?')[1] : '')] = {
      evaluate: [...document.querySelectorAll('button')].filter(b => seen(b) && /^(Evaluar|Re-evaluar|Estimar|Preguntar)/.test(b.textContent)).map(b => b.textContent),
      next: seen(document.getElementById('evalnext')) || seen(document.getElementById('ask-form')),
      copy: [...document.querySelectorAll('#cards .ask button')].filter(b => seen(b) && b.textContent === 'copiar comando').length,
    };
  }
  const pre = document.createElement('pre');
  pre.id = 'probe';
  pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
)


@_requires_chrome
def test_the_static_page_has_no_evaluate_control_on_any_tab(tmp_path):
    """The file `jev dashboard` writes is not served: «copiar comando», never a button that
    would POST to a server that is not there."""
    _need_chrome()
    data = _fixture()
    seen = _open(_page(tmp_path, data, _STATIC_PROBE))

    assert data["serve"] is None
    assert len(seen) == 10 and "ask" in seen
    for view, found in seen.items():
        assert found["evaluate"] == [], view
        assert found["next"] is False, view
    assert seen["posts?f=uneval"]["copy"] > 0

# tests/test_jev_page_listview_browser.py
"""The list view (PR 17, VGonPa/xbrain#211) in a real browser: Preguntar's results and Revisar ›
Posts SORTED, GROUPED and PAGED over what the page already has — nothing asked, nothing paid.

Every order, group, count and page is compared with a brute force written here in Python,
straight from the rules (`docs/jev.md`, «The list view»), over a repo shaped like Víctor's first
real ask but with five authors, ~2 months of posts in an order unrelated to the ranking, ties
in time, and posts under two topics (`victor_shaped_repo(varied=True)`).

Probes follow `test_jev_page_browser.py`'s rules: they read what a reader SEES, go through the
page's own controls, and are fail-closed (a step that throws is an `ERROR` value the assertions
then reject).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from tests.jev_ask_fixtures import VARIED_HANDLES, VICTOR_QUERY, victor_shaped_repo
from tests.test_jev_page_ask_browser import _ASK_JS, _REFINE_JS, DT
from tests.test_jev_page_browser import (
    _SEEN,
    _SERVE_JS,
    _dump,
    _need_chrome,
    _page,
    _requires_chrome,
    _served_dump,
    _step,
)
from xbrain.config import Config
from xbrain.jev.ask import AskFilters, AskQuery, normalise_author, post_topics

SHA = AskQuery.of(VICTOR_QUERY).sha
LABELS = {
    "agentic-engineering": "Agentic Engineering",
    "startups": "Startups",
    "nutricion": "Nutricion",
}

# --------------------------------------------------------------------------- Python's side


def _refined(cfg: Config, *, minimum: float = 0.0, **refine: Any):
    """`ask.refine_results` of the one saved query: the list the view starts from (base order)."""
    from xbrain.jev.ask import load_history, refine_results, reopen_results
    from xbrain.jev.load import load_jev_pairs

    jev = load_jev_pairs(cfg)
    [entry] = load_history(cfg).queries.values()
    found = reopen_results(cfg, entry, jev)
    return refine_results(
        found, AskFilters(**refine), minimum, store=jev.store, jev=jev, threshold=cfg.jev_threshold
    ).ranked


def _ask_acc(cfg: Config) -> dict[str, Any]:
    """An answer's fields as the rules read them: time and primary from the post, day, handle
    and topics as `ask.refine_keys` computes them."""
    from xbrain.jev.load import load_jev_pairs

    current = load_jev_pairs(cfg).current_by_id()
    return {
        "id": lambda pair: pair[0].id,
        "time": lambda pair: pair[0].created_at.timestamp(),
        "author": lambda pair: normalise_author(pair[0].author.handle),
        "topics": lambda pair: sorted(
            post_topics(pair[0], current.get(pair[0].id), cfg.jev_threshold)
        ),
        "primary": lambda pair: pair[0].enriched.primary_topic if pair[0].enriched else "",
        "month": lambda pair: pair[0].created_at.astimezone(timezone.utc).strftime("%Y-%m"),
        "cost": lambda pair: 0,
    }


def _posts_acc() -> dict[str, Any]:
    """A Posts card's fields as the rules read them (the blob's own card)."""
    return {
        "id": lambda p: p["id"],
        "time": lambda p: datetime.fromisoformat(p["created"]).timestamp(),
        "author": lambda p: normalise_author(p["author"]["handle"]),
        "topics": lambda p: p["slugs"],
        "primary": lambda p: (p["enrich"] or {}).get("primary") or "",
        "month": lambda p: (
            datetime.fromisoformat(p["created"]).astimezone(timezone.utc).strftime("%Y-%m")
        ),
        "cost": lambda p: (
            p["jev"]["cost_usd"] if p["jev"] and p["jev"].get("cost_usd") is not None else -1
        ),
    }


def _sorted(base: list, sort: str, acc: dict[str, Any]) -> list:
    """A stable sort of the base order: its ties keep the base order (relevance, then id)."""
    keys = {
        "recent": lambda x: -acc["time"](x),
        "old": lambda x: acc["time"](x),
        "author": lambda x: (acc["author"](x) == "", acc["author"](x)),
        "topic": lambda x: (
            acc["primary"](x) == "",
            LABELS.get(acc["primary"](x), acc["primary"](x)).casefold(),
        ),
        "cost": lambda x: -acc["cost"](x),
    }
    return sorted(base, key=keys[sort]) if sort in keys else list(base)


def _grouped(base: list, ordered: list, group: str, go: str, acc: dict[str, Any]):
    """`[(key, items)]`: each group's items in `ordered`'s order (a post under each of its
    topics), groups by `go` — the best base position, the biggest, or the name (months newest
    first) — ties by name, the keyless group last."""
    if not group:
        return [("", ordered)]
    of = {
        "topic": lambda x: acc["topics"](x) or [""],
        "author": lambda x: [acc["author"](x)],
        "month": lambda x: [acc["month"](x)],
    }[group]
    groups: dict[str, list] = {}
    for x in ordered:
        for key in of(x):
            groups.setdefault(key, []).append(x)
    position = {acc["id"](x): i for i, x in enumerate(base)}
    name = {
        "topic": lambda k: LABELS.get(k, k).casefold(),
        "author": lambda k: k,
        "month": lambda k: k,
    }[group]
    keys = sorted(groups, key=name, reverse=group == "month")
    if go == "best":
        keys.sort(key=lambda k: min(position[acc["id"](x)] for x in groups[k]))
    elif go == "size":
        keys.sort(key=lambda k: -len(groups[k]))
    keys.sort(key=lambda k: k == "")
    return [(k, groups[k]) for k in keys]


def _rows(groups, acc) -> list[dict[str, Any]]:
    """What a page holding every entry draws: each group's head, then its cards."""
    out: list[dict[str, Any]] = []
    for key, items in groups:
        if len(groups) > 1 or key:
            out.append({"head": key, "count": _nf(len(items))})
        out += [{"id": acc["id"](x)} for x in items]
    return out


def _flat(groups, acc) -> list[str]:
    return [acc["id"](x) for _, items in groups for x in items]


# --------------------------------------------------------------------------- the page's side

#: The list view as a reader sees it: the list's rows (group heads and cards, in order), both
#: pagers, the view bar's state, the line saying what is shown and the note on duplicated entries.
_LV_JS = r"""
const lvSeenPager = (id) => { const p = document.getElementById(id); if (!seen(p)) return null;
  return {range: txt(p.querySelector('.prange')), items: [...p.querySelectorAll('.pnums > *')].filter(seen).map(a => a.textContent),
    current: txt(p.querySelector('[aria-current="page"]')), off: [...p.querySelectorAll('.off')].map(a => a.textContent),
    links: [...p.querySelectorAll('a')].map(a => a.getAttribute('href'))}; };
const lvView = (listId, barId, pagerIds, dupId) => {
  const list = document.getElementById(listId);
  const bar = document.getElementById(barId);
  const pressed = (sel) => { const b = [...bar.querySelectorAll(sel)].find(x => x.getAttribute('aria-pressed') === 'true'); return b ? b.id.replace(barId + '-', '') : null; };
  return {
    hash: location.hash,
    rows: [...list.children].filter(seen).filter(k => k.matches('.card, .lvhead')).map(k => k.matches('.lvhead')
      ? {head: k.dataset.group, count: txt(k.querySelector('.lvn')), cont: !!k.querySelector('.lvcont'),
         open: k.querySelector('button').getAttribute('aria-expanded') === 'true', name: txt(k.querySelector('.lvname'))}
      : {id: k.dataset.id}),
    ids: [...list.children].filter(seen).filter(k => k.matches('.card')).map(k => k.dataset.id),
    top: lvSeenPager(pagerIds[0]), bottom: lvSeenPager(pagerIds[1]),
    bar: {sort: pressed('.segtrack button'), group: pressed('.lvgrp button'), size: document.getElementById(barId + '-size').value,
      go: bar.querySelector('select[id$="-go"]') ? bar.querySelector('select[id$="-go"]').value : null,
      sizes: [...document.getElementById(barId + '-size').options].map(o => o.value)},
    dup: txt(document.getElementById(dupId)),
  };
};
const lvAsk = () => Object.assign(lvView('ask-list', 'ask-view', ['ask-pager-top', 'ask-pager'], 'ask-dup'), {reach: txt(document.getElementById('ask-reach'))});
const lvPosts = () => Object.assign(lvView('cards', 'posts-bar', ['pager-top', 'pager'], 'posts-dup'), {count: txt(document.getElementById('count'))});
const lvAuthors = () => { const b = document.getElementById('refine-who-open'); if (b.getAttribute('aria-expanded') !== 'true') b.click();
  return [...document.querySelectorAll('#refine-authors button')].filter(seen).map(x => [x.dataset.author, txt(x.querySelector('.tcount')), x.getAttribute('aria-pressed')]); };
"""

_READ = (
    "<script>"
    + _SEEN
    + _ASK_JS
    + _REFINE_JS
    + _LV_JS
    + r"""
setTimeout(() => {
  const pre = document.createElement('pre'); pre.id = 'probe';
  let out;
  try { out = location.hash.startsWith('#ask') ? Object.assign(lvAsk(), {authors: lvAuthors(), refine: sQRefine()}) : lvPosts(); }
  catch (e) { out = 'ERROR ' + e.message; }
  pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
}, 300);
</script>"""
)


@pytest.fixture(scope="module")
def varied(tmp_path_factory):
    """The varied repo, its page (with the reading probe) and its blob."""
    from xbrain.jev.dashboard import build_page_data

    _need_chrome()
    root = tmp_path_factory.mktemp("lv-varied")
    cfg = victor_shaped_repo(root, varied=True)
    data = build_page_data(cfg, now=DT)
    return cfg, data, _page(root, data, _READ)


def _open(page: Path, hash_: str) -> dict[str, Any]:
    seen = _dump(page.as_uri() + hash_)
    assert not isinstance(seen, str), seen
    return seen


def _ask(params: dict[str, str]) -> str:
    return "#ask?" + urlencode({"q": SHA, **params})


def _nf(n: int) -> str:
    """A count as the page writes it (`es-ES`): a thousands dot only from five digits on."""
    return f"{n:,}".replace(",", ".") if n >= 10_000 else str(n)


# --------------------------------------------------------------------------- Preguntar: sorts


def test_the_varied_fixture_has_what_the_list_view_needs(tmp_path):
    """Guard on the fixture: several authors, months and topics, posts under two topics, and
    the recency order not the ranking — or a sort test could pass on a list that never moves."""
    cfg = victor_shaped_repo(tmp_path, varied=True)
    acc = _ask_acc(cfg)
    base = _refined(cfg)
    assert len(base) == 808
    assert {acc["author"](x) for x in base} == {h.casefold() for h in VARIED_HANDLES}
    assert len({acc["month"](x) for x in base}) >= 3
    assert sum(len(acc["topics"](x)) > 1 for x in base) > 100
    assert _sorted(base, "recent", acc) != base and _sorted(base, "recent", acc)[::-1] != base
    times = [acc["time"](x) for x in base]
    assert len(set(times)) < len(times)  # ties in time: the base order breaks them


@_requires_chrome
@pytest.mark.parametrize("sort", ["rel", "recent", "old", "author", "topic"])
def test_each_sort_of_the_results_is_pythons_order(varied, sort):
    cfg, _, page = varied
    acc = _ask_acc(cfg)
    base = _refined(cfg)
    params = {"size": "900"} | ({} if sort == "rel" else {"sort": sort})

    seen = _open(page, _ask(params))

    assert seen["ids"] == [acc["id"](x) for x in _sorted(base, sort, acc)]
    assert seen["bar"]["sort"] == f"sort-{sort}" and seen["bar"]["group"] == "group-none"
    words = {
        "rel": "de mayor a menor probabilidad",
        "recent": "los más recientes primero",
        "old": "los más antiguos primero",
        "author": "por autor, de la A a la Z",
        "topic": "por topic principal, de la A a la Z",
    }[sort]
    assert seen["reach"] == f"808 leídos por Jev, {words}."


@_requires_chrome
@pytest.mark.parametrize(
    "group,go,sort",
    [
        ("topic", "best", "rel"),
        ("topic", "size", "recent"),
        ("topic", "name", "rel"),
        ("author", "best", "rel"),
        ("author", "size", "old"),
        ("author", "name", "rel"),
        ("month", "best", "author"),
        ("month", "name", "rel"),
        ("month", "size", "rel"),
    ],
)
def test_each_grouping_is_pythons_groups_counts_and_order(varied, group, go, sort):
    """Heads with their counts, in their order, each group's cards sorted within it; a post with
    two topics is under both, and the page says so."""
    cfg, _, page = varied
    acc = _ask_acc(cfg)
    base = _refined(cfg)
    params = {"size": "2000", "group": group} | ({} if go == "best" else {"go": go})
    params |= {} if sort == "rel" else {"sort": sort}

    seen = _open(page, _ask(params))

    groups = _grouped(base, _sorted(base, sort, acc), group, go, acc)
    heads = [r for r in seen["rows"] if "head" in r]
    assert [
        {k: r[k] for k in ("head", "count")} if "head" in r else r for r in seen["rows"]
    ] == _rows(groups, acc)
    assert all(h["open"] and not h["cont"] for h in heads)
    assert seen["bar"]["group"] == f"group-{group}" and seen["bar"]["go"] == go
    entries = sum(len(items) for _, items in groups)
    if group == "topic":
        assert entries > 808
        assert seen["dup"] == (
            f"Un post con varios topics sale en cada uno: {_nf(entries)} entradas de 808 posts."
        )
        assert seen["bottom"]["range"] == f"1–{_nf(entries)} de {_nf(entries)} entradas"
    else:
        assert entries == 808 and seen["dup"] is None
    names = [row["name"] for row in seen["rows"] if "head" in row]
    if group == "author":
        assert names == [f"@{key}" for key, _ in groups]
    if group == "month":
        # A header: capitalised («Mayo de 2026», UX m4).
        assert names[0].endswith(" de 2026") and names[0].split()[0] in ("Mayo", "Junio", "Julio")


# --------------------------------------------------------------------------- Preguntar: pages


@_requires_chrome
@pytest.mark.parametrize(
    "page,items,current",
    [
        (None, ["‹", "1", "2", "…", "41", "›"], "1"),
        ("3", ["‹", "1", "2", "3", "4", "…", "41", "›"], "3"),
        ("20", ["‹", "1", "…", "19", "20", "21", "…", "41", "›"], "20"),
        ("41", ["‹", "1", "…", "40", "41", "›"], "41"),
        ("99", ["‹", "1", "…", "40", "41", "›"], "41"),
    ],
)
def test_the_pager_pages_pythons_ranking_twenty_at_a_time(varied, page, items, current):
    cfg, _, page_file = varied
    order = [x[0].id for x in _refined(cfg)]

    seen = _open(page_file, _ask({} if page is None else {"page": page}))

    n = int(current)
    start, end = (n - 1) * 20, min(n * 20, 808)
    assert seen["ids"] == order[start:end]
    for pager in (seen["top"], seen["bottom"]):
        assert pager["items"] == items and pager["current"] == current
        assert pager["range"] == f"{start + 1}–{end} de 808 resultados"
        # ‹ on the first page and › on the last are there, off.
        assert pager["off"] == (["‹"] if n == 1 else ["›"] if n == 41 else [])
    assert seen["reach"] == "808 leídos por Jev, de mayor a menor probabilidad."
    assert seen["bar"]["size"] == "20" and seen["bar"]["sizes"] == ["10", "20", "50", "100"]
    # Every page link is a link: the keyboard, the back button and a reload know it.
    assert all(h.startswith(f"#ask?q={SHA}") for h in seen["bottom"]["links"])
    assert f"#ask?q={SHA}&page=2" in seen["bottom"]["links"] or n != 1


@_requires_chrome
def test_a_link_from_before_the_pager_opens_as_one_page_of_its_size(varied):
    """`top=40` (the old «Ver más» link) is the first page of 40: the same 40 results."""
    cfg, _, page = varied
    order = [x[0].id for x in _refined(cfg)]

    seen = _open(page, _ask({"top": "40"}))

    assert seen["ids"] == order[:40]
    assert seen["bar"]["size"] == "40" and "40" in seen["bar"]["sizes"]
    assert seen["bottom"]["range"] == "1–40 de 808 resultados"


# --------------------------------------------------------------------------- Preguntar: facets


@_requires_chrome
@pytest.mark.parametrize(
    "params,refine",
    [
        ({}, {}),
        ({"min": "0.5"}, {"minimum": 0.5}),
        (
            {"t": "nutricion", "since": "2026-06-01"},
            {"topics": ("nutricion",), "since": "2026-06-01"},
        ),
        # The author's own filter does not count its facet: every author is still offered.
        ({"author": "@@Dana"}, {"author": "@@Dana"}),
    ],
)
def test_the_author_facet_counts_what_each_author_keeps_under_the_rest(varied, params, refine):
    cfg, _, page = varied
    minimum = refine.pop("minimum", 0.0)
    if "since" in refine:
        refine["since"] = datetime.fromisoformat(refine["since"]).date()
    others = {k: v for k, v in refine.items() if k != "author"}

    seen = _open(page, _ask(params))

    kept = _refined(cfg, minimum=minimum, **others)
    counts: dict[str, int] = {}
    for item, _ in kept:
        handle = normalise_author(item.author.handle)
        counts[handle] = counts.get(handle, 0) + 1
    expected = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    picked = normalise_author(refine.get("author", ""))
    assert seen["authors"] == [
        [handle, _nf(n), "true" if handle == picked else "false"] for handle, n in expected
    ]


@_requires_chrome
def test_several_leading_at_signs_are_read_the_same_on_both_sides(varied):
    """PR 14 re-review m3: `@@Dana` is `dana` in Python (`lstrip("@")`) and on the page."""
    cfg, _, page = varied

    seen = _open(page, _ask({"author": "@@Dana", "size": "900"}))

    expected = [x[0].id for x in _refined(cfg, author="@@Dana")]
    assert expected and seen["ids"] == expected
    assert normalise_author("@@Dana") == "dana"


# --------------------------------------------------------------------------- Preguntar: moving


_ASK_MOVE = (
    "<script>"
    + _SEEN
    + _ASK_JS
    + _REFINE_JS
    + _LV_JS
    + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  const pick = (id, value) => { const s = document.getElementById(id); s.value = value; s.dispatchEvent(new Event('change', {bubbles: true})); };
  const pageLink = (n) => [...document.querySelectorAll('#ask-pager a.pn')].find(a => a.textContent === String(n));
  const h0 = history.length;
  await step('first', async () => { if (!location.hash.startsWith('#ask')) location.hash = '#ask'; await sleep(80); return lvAsk(); });
  await step('page2', async () => { pageLink(2).click(); await sleep(80); return lvAsk(); });
  await step('next', async () => { document.querySelector('#ask-pager a.pa[aria-label="Página siguiente"]').click(); await sleep(80); return lvAsk(); });
  await step('size_down', async () => { pick('ask-view-size', '10'); await sleep(80); return lvAsk(); });
  await step('size_default', async () => { pick('ask-view-size', '20'); await sleep(80); return lvAsk(); });
  await step('size_up', async () => { pick('ask-view-size', '50'); await sleep(80); return lvAsk(); });
  await step('sorted', async () => { document.getElementById('ask-view-sort-recent').click(); await sleep(80); return lvAsk(); });
  await step('sorted_p2', async () => { pageLink(2).click(); await sleep(80); return lvAsk(); });
  await step('filtered', async () => { sQSet({'refine-min': '0.3'}); await sleep(80); return lvAsk(); });
  await step('grouped', async () => { document.getElementById('ask-view-group-author').click(); await sleep(80); return lvAsk(); });
  await step('go', async () => { pick('ask-view-go', 'size'); await sleep(80); return lvAsk(); });
  await step('folded', async () => {
    document.querySelector('#ask-list > .lvhead button').click(); await sleep(80);
    return lvAsk();
  });
  await step('unfolded', async () => { document.querySelector('#ask-list > .lvhead button').click(); await sleep(80); return lvAsk(); });
  await step('back', async () => { history.back(); await sleep(150); return lvAsk(); });
  await step('forward', async () => { history.forward(); await sleep(150); return lvAsk(); });
  await step('reset', async () => { document.getElementById('refine-clear').click(); await sleep(80); return Object.assign(lvAsk(), {refine: sQRefine()}); });
  await step('pushed', async () => history.length - h0);
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
)


@pytest.fixture(scope="module")
def moved(tmp_path_factory):
    from xbrain.jev.dashboard import build_page_data

    _need_chrome()
    root = tmp_path_factory.mktemp("lv-move")
    cfg = victor_shaped_repo(root, varied=True)
    page = _page(root, build_page_data(cfg, now=DT), _ASK_MOVE)
    return cfg, _dump(page.as_uri())


def _no_error(seen: dict[str, Any]) -> None:
    bad = {k: v for k, v in seen.items() if isinstance(v, str) and v.startswith("ERROR")}
    assert not bad, bad


@_requires_chrome
def test_the_reader_pages_through_the_pager_and_the_hash_keeps_the_page(moved):
    cfg, seen = moved
    _no_error(seen)
    order = [x[0].id for x in _refined(cfg)]

    assert seen["first"]["ids"] == order[:20] and seen["first"]["bottom"]["current"] == "1"
    assert seen["page2"]["ids"] == order[20:40] and seen["page2"]["hash"] == f"#ask?q={SHA}&page=2"
    assert seen["page2"]["bottom"]["current"] == "2"
    assert seen["next"]["ids"] == order[40:60] and seen["next"]["hash"].endswith("&page=3")
    # A new size keeps the first result on screen on the page shown: result 41 is on page 5 of
    # 10, then on page 1 of 50.
    assert seen["size_down"]["ids"] == order[40:50]
    assert seen["size_down"]["hash"] == f"#ask?q={SHA}&size=10&page=5"
    # Back to the default size (`[jev].ask_top`, out of the hash): result 41 is on page 3.
    assert seen["size_default"]["ids"] == order[40:60]
    assert seen["size_default"]["hash"] == f"#ask?q={SHA}&page=3"
    assert seen["size_up"]["ids"] == order[0:50] and seen["size_up"]["hash"] == (
        f"#ask?q={SHA}&size=50"
    )


@_requires_chrome
def test_a_sort_or_a_filter_starts_again_at_page_one(moved):
    cfg, seen = moved
    _no_error(seen)
    acc = _ask_acc(cfg)
    recent = [acc["id"](x) for x in _sorted(_refined(cfg), "recent", acc)]

    assert seen["sorted"]["ids"] == recent[:50] and "page=" not in seen["sorted"]["hash"]
    assert "sort=recent" in seen["sorted"]["hash"]
    assert seen["sorted_p2"]["ids"] == recent[50:100]
    kept = [acc["id"](x) for x in _sorted(_refined(cfg, minimum=0.3), "recent", acc)]
    assert seen["filtered"]["ids"] == kept[:50] and "page=" not in seen["filtered"]["hash"]
    assert "min=0.3" in seen["filtered"]["hash"] and "sort=recent" in seen["filtered"]["hash"]


@_requires_chrome
def test_grouping_folding_and_the_group_order_follow_python(moved):
    cfg, seen = moved
    _no_error(seen)
    acc = _ask_acc(cfg)
    base = _refined(cfg, minimum=0.3)
    ordered = _sorted(base, "recent", acc)

    by_best = _grouped(base, ordered, "author", "best", acc)
    assert seen["grouped"]["rows"][0] == {
        "head": by_best[0][0],
        "count": _nf(len(by_best[0][1])),
        "cont": False,
        "open": True,
        "name": f"@{by_best[0][0]}",
    }
    assert seen["grouped"]["ids"] == _flat(by_best, acc)[:50]
    by_size = _grouped(base, ordered, "author", "size", acc)
    assert [r["head"] for r in seen["go"]["rows"] if "head" in r][:1] == [by_size[0][0]]
    assert "go=size" in seen["go"]["hash"] and "group=author" in seen["go"]["hash"]
    # Folding the first group: its head stays (its count too, closed), its cards go, the next
    # group's cards take the page.
    folded = seen["folded"]
    first_key, first_items = by_size[0]
    assert folded["rows"][0] == {
        "head": first_key,
        "count": _nf(len(first_items)),
        "cont": False,
        "open": False,
        "name": f"@{first_key}",
    }
    rest = [(k, items) for k, items in by_size[1:]]
    assert folded["ids"] == _flat(rest, acc)[:50]
    total = sum(len(items) for _, items in rest)
    assert folded["bottom"]["range"] == f"1–50 de {_nf(total)} entradas"
    assert seen["unfolded"]["ids"] == seen["go"]["ids"]


@_requires_chrome
def test_back_and_forward_walk_the_views_and_restablecer_clears_them_all(moved):
    _, seen = moved
    _no_error(seen)

    assert seen["back"]["hash"] == seen["grouped"]["hash"]
    assert seen["back"]["ids"] == seen["grouped"]["ids"]
    assert seen["forward"]["hash"] == seen["go"]["hash"]
    reset = seen["reset"]
    assert reset["hash"] == f"#ask?q={SHA}"
    assert (reset["bar"]["sort"], reset["bar"]["group"], reset["bar"]["size"]) == (
        "sort-rel",
        "group-none",
        "20",
    )
    assert reset["refine"]["min"] == "" and reset["bottom"]["current"] == "1"
    # Every move was a history entry (a page, a size, a sort, a filter, a grouping, the reset);
    # folding a group is not a view of its own.
    assert seen["pushed"] >= 10


@_requires_chrome
def test_a_zero_minimum_in_the_link_stays_zero_over_a_query_asked_with_one(tmp_path):
    """PR 14 re-review m2: a query asked with `--min 0.5`, opened with `min=0`, shows every
    answer — and a change after it (a sort) keeps `min=0` in the hash, never the 0.5 default."""
    from xbrain.jev.dashboard import build_page_data

    _need_chrome()
    cfg = victor_shaped_repo(tmp_path, varied=True)
    path = cfg.jev_asks_dir / "index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    for entry in index["queries"].values():
        entry["last_min"] = 0.5
    path.write_text(json.dumps(index), encoding="utf-8")
    probe = (
        "<script>"
        + _SEEN
        + _ASK_JS
        + _REFINE_JS
        + _LV_JS
        + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  await sleep(80);
  out.opened = lvAsk();
  document.getElementById('ask-view-sort-old').click(); await sleep(80);
  out.sorted = lvAsk();
  location.hash = '#ask?q=' + DATA.asks.history[0].sha; await sleep(80);
  out.plain = lvAsk();
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
    )
    page = _page(tmp_path, build_page_data(cfg, now=DT), probe)

    seen = _dump(page.as_uri() + _ask({"min": "0"}))

    assert seen["opened"]["reach"].startswith("808 leídos por Jev")
    assert "min=0" in seen["sorted"]["hash"] and "sort=old" in seen["sorted"]["hash"]
    assert seen["sorted"]["reach"].startswith("808 leídos por Jev")
    # Without it, the query's own minimum is the default.
    assert seen["plain"]["reach"].startswith("156 con relevancia ≥ 0,50")


# --------------------------------------------------------------------------- Revisar › Posts


@_requires_chrome
@pytest.mark.parametrize("sort", ["disc", "recent", "old", "author", "topic", "cost"])
def test_each_sort_of_the_posts_list_is_pythons_order(varied, sort):
    _, data, page = varied
    acc = _posts_acc()
    params = {"f": "all", "size": "900"} | ({} if sort == "disc" else {"s": sort})

    seen = _open(page, "#revisar/posts?" + urlencode(params))

    assert seen["ids"] == [p["id"] for p in _sorted(data["posts"], sort, acc)]
    assert seen["bar"]["sort"] == f"sort-{sort}"
    assert seen["count"] == "mostrando 813 de 813 posts"


@_requires_chrome
@pytest.mark.parametrize(
    "group,go,sort",
    [("topic", "best", "disc"), ("author", "size", "recent"), ("month", "name", "old")],
)
def test_each_grouping_of_the_posts_list_is_pythons(varied, group, go, sort):
    _, data, page = varied
    acc = _posts_acc()
    params = {"f": "all", "size": "2000", "group": group, "s": sort}
    params |= {} if go == "best" else {"go": go}

    seen = _open(page, "#revisar/posts?" + urlencode(params))

    groups = _grouped(data["posts"], _sorted(data["posts"], sort, acc), group, go, acc)
    assert [
        {k: r[k] for k in ("head", "count")} if "head" in r else r for r in seen["rows"]
    ] == _rows(groups, acc)
    if group == "topic":
        entries = sum(len(items) for _, items in groups)
        assert seen["dup"] == (
            f"Un post con varios topics sale en cada uno: {_nf(entries)} entradas de 813 posts."
        )


@_requires_chrome
def test_the_posts_list_pages_fifty_at_a_time_and_opens_on_its_page(varied):
    _, data, page = varied
    order = [p["id"] for p in data["posts"]]

    first = _open(page, "#revisar/posts?f=all")
    third = _open(page, "#revisar/posts?f=all&page=3")

    assert first["ids"] == order[:50] and first["bottom"]["range"] == "1–50 de 813 posts"
    assert first["bottom"]["items"] == ["‹", "1", "2", "…", "17", "›"]
    assert third["ids"] == order[100:150] and third["bottom"]["current"] == "3"
    assert third["bar"]["sizes"] == ["10", "20", "50", "100"] and third["bar"]["size"] == "50"


_POSTS_MOVE = (
    "<script>"
    + _SEEN
    + _LV_JS
    + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  const key = k => document.dispatchEvent(new KeyboardEvent('keydown', {key: k, bubbles: true}));
  const cur = () => { const c = document.querySelector('.card.cur'); return c && c.dataset.id; };
  await step('first', async () => { await sleep(80); return lvPosts(); });
  await step('walk', async () => {
    for (let i = 0; i < 12; i++) key('j');
    await sleep(40);
    return Object.assign(lvPosts(), {cur: cur()});
  });
  await step('back_up', async () => { key('k'); key('k'); await sleep(40); return Object.assign(lvPosts(), {cur: cur()}); });
  await step('sorted', async () => { document.getElementById('posts-bar-sort-author').click(); await sleep(80); return lvPosts(); });
  await step('search', async () => {
    const box = document.getElementById('search');
    box.value = 'agentic'; box.dispatchEvent(new Event('input')); await sleep(40);
    return lvPosts();
  });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
)


@_requires_chrome
def test_j_walks_across_pages_and_a_sort_or_a_search_starts_at_page_one(tmp_path):
    from xbrain.jev.dashboard import build_page_data

    _need_chrome()
    cfg = victor_shaped_repo(tmp_path, varied=True)
    data = build_page_data(cfg, now=DT)
    page = _page(tmp_path, data, _POSTS_MOVE)

    seen = _dump(page.as_uri() + "#revisar/posts?f=all&size=10")

    _no_error(seen)
    order = [p["id"] for p in data["posts"]]
    assert seen["first"]["ids"] == order[:10]
    # The twelfth j is post 12: on page 2, which the page turned to.
    assert seen["walk"]["cur"] == order[11] and seen["walk"]["ids"] == order[10:20]
    assert "page=2" in seen["walk"]["hash"] and seen["walk"]["bottom"]["current"] == "2"
    assert seen["back_up"]["cur"] == order[9] and seen["back_up"]["ids"] == order[:10]
    acc = _posts_acc()
    by_author = [p["id"] for p in _sorted(data["posts"], "author", acc)]
    assert seen["sorted"]["ids"] == by_author[:10] and "page=" not in seen["sorted"]["hash"]
    assert "s=author" in seen["sorted"]["hash"] and "size=10" in seen["sorted"]["hash"]
    assert (
        seen["search"]["bottom"]["current"] in ("1", None)
        and "page=" not in (seen["search"]["hash"])
    )


# --------------------------------------------------------------------------- served: free


_SERVED_LV = (
    "<script>"
    + _SERVE_JS
    + _ASK_JS
    + _REFINE_JS
    + _LV_JS
    + r"""
(async () => {
  await sStep('listview', async () => {
    location.hash = '#ask';
    await sWait(() => sQResults().length === 20, 'los resultados');
    const posts0 = sPosts, e0 = sEstimates;
    const reqs = [];
    const f0 = window.fetch;
    window.fetch = function (u, init) { reqs.push([String(u), (init && init.method) || 'GET']); return f0.call(window, u, init); };
    // Each move waits for what the reader then SEES (the bar, the pager), not only the hash.
    sId('ask-view-sort-old').click();
    await sWait(() => lvAsk().bar.sort === 'sort-old', 'el orden');
    sId('ask-view-group-month').click();
    await sWait(() => lvAsk().bar.group === 'group-month', 'los grupos');
    [...document.querySelectorAll('#ask-pager a.pn')].find(a => a.textContent === '2').click();
    await sWait(() => (lvAsk().bottom || {}).current === '2', 'la página 2');
    const s = sId('ask-view-size'); s.value = '50'; s.dispatchEvent(new Event('change', {bubbles: true}));
    await sWait(() => lvAsk().bar.size === '50', 'el tamaño');
    document.querySelector('#ask-list > .lvhead button').click();
    await sWait(() => lvAsk().rows[0].open === false, 'el grupo plegado');
    const view = lvAsk();
    location.hash = '#revisar/posts?f=all';
    await sWait(() => seen(document.querySelector('#cards .card')), 'Posts');
    sId('posts-bar-sort-recent').click();
    await sWait(() => lvPosts().bar.sort === 'sort-recent', 'el orden de Posts');
    [...document.querySelectorAll('#pager a.pn')].find(a => a.textContent === '2').click();
    await sWait(() => (lvPosts().bottom || {}).current === '2', 'la página 2 de Posts');
    window.fetch = f0;
    return {posts: sPosts - posts0, estimates: sEstimates - e0, requests: reqs.filter(r => r[1] !== 'GET' || !r[0].includes('/api/job')), view: view};
  });
  sDone();
})();
</script>"""
)


@_requires_chrome
def test_sorting_grouping_and_paging_ask_the_server_nothing(tmp_path):
    """Served: every move of the list view is the page's own — zero POSTs, zero estimates, and
    no request but the page's idle watch of `/api/job`."""
    from tests.jev_fakes import FakeJevClient

    def _repo(root: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
        monkeypatch.setenv("XBRAIN_REPO_ROOT", str(root))
        return victor_shaped_repo(root, varied=True)

    built: list[int] = []

    def _make() -> FakeJevClient:
        built.append(1)
        return FakeJevClient()

    seen = _served_dump(tmp_path, _SERVED_LV, make_client=_make, repo=_repo)

    step = _step(seen, "listview")
    assert (step["posts"], step["estimates"], step["requests"]) == (0, 0, [])
    assert built == []
    assert step["view"]["hash"].count("page=2") == 0  # a new size put the reader on page 1
    assert "group=month" in step["view"]["hash"] and "size=50" in step["view"]["hash"]


# --------------------------------------------------------------------------- the small things


_SMALL = (
    "<script>"
    + _SEEN
    + r"""
setTimeout(() => {
  const out = {};
  try {
    // A video with no frame: its ▶ above its words, never over them.
    const grid = mediaGrid([{type: 'video', why: 'no_frame'}, {type: 'video', src: 'x.jpg'}]);
    grid.style.width = '550px';
    document.body.appendChild(grid);
    const none = grid.children[0].querySelector('.noimg');
    const play = none.querySelector('.noplay'), words = none.lastElementChild;
    const a = play.getBoundingClientRect(), b = words.getBoundingClientRect();
    out.video = {words: words.textContent, overlap: !(a.bottom <= b.top || b.bottom <= a.top || a.right <= b.left || b.right <= a.left),
      over: !!grid.children[0].querySelector(':scope > .play'), framed_play: !!grid.children[1].querySelector(':scope > .play')};
    // A page ask's end names the minimum its count is cut at (PR 14 re-review m4).
    const job = (min) => ({kind: 'ask', state: 'done', usd: 0.001, outcome: {ok: 5, results: 3, min: min, failed: 0, unsaved: 0, logged: true, sent: 5}});
    out.outcome = {cut: outcomeText(job(0.5)), whole: outcomeText(job(0))};
    out.items = [[1, 1], [1, 2], [1, 3], [2, 5], [3, 5], [4, 9], [5, 9], [1, 7]].map(([p, n]) => lvPageItems(p, n).join(' '));
  } catch (e) { out.error = e.message; }
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
}, 200);
</script>"""
)


@_requires_chrome
def test_a_videos_play_mark_never_covers_why_it_has_no_frame_and_an_end_names_its_minimum(
    varied,
):
    _, data, page_file = varied
    root = page_file.parent / "small"
    root.mkdir(exist_ok=True)
    page = _page(root, data, _SMALL)

    seen = _open(page, "")

    assert "error" not in seen, seen
    assert seen["video"] == {
        "words": "vídeo sin fotograma extraído",
        "overlap": False,
        "over": False,
        "framed_play": True,
    }
    assert "3 resultados con relevancia ≥ 0,50 · " in seen["outcome"]["cut"]
    assert "3 resultados · " in seen["outcome"]["whole"] and "≥" not in seen["outcome"]["whole"]
    # The pager's numbers: the ends, the page and its neighbours; a gap of one is that number.
    assert seen["items"] == [
        "1",
        "1 2",
        "1 2 3",
        "1 2 3 4 5",
        "1 2 3 4 5",
        "1 2 3 4 5 … 9",
        "1 … 4 5 6 … 9",
        "1 2 … 7",
    ]


# --------------------------------------------------------------------------- PR 17 fix wave
#: The reader turns a page from the BOTTOM pager (the one used under twenty cards): the list's
#: head lands at the top of the screen, and the keyboard focus is on the new current page of the
#: top pager (a screen reader says «Página 3, página actual»). A sort from the bar does not
#: scroll (the reader is at the bar already). A fold keeps the focus on its own button.
_LAND_PROBE = (
    "<script>"
    + _SEEN
    + _LV_JS
    + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  const top = (id) => Math.round(document.getElementById(id).getBoundingClientRect().top);
  const focus = () => { const a = document.activeElement;
    return {id: a.id || null, tag: a.tagName, label: a.getAttribute('aria-label'), current: a.getAttribute('aria-current'),
      nav: a.closest('nav') ? a.closest('nav').id : null, expanded: a.getAttribute('aria-expanded')}; };
  const bottomNext = (pager) => {
    const next = document.querySelector('#' + pager + ' a.pa[aria-label="Página siguiente"]');
    next.scrollIntoView({block: 'end', behavior: 'instant'});
    next.focus();
    return {before: Math.round(scrollY), pager_top: Math.round(next.getBoundingClientRect().top), click: () => next.click()};
  };
  await step('ask_bottom', async () => {
    await sleep(80);
    const b = bottomNext('ask-pager');
    b.click(); await sleep(150);
    return {before: b.before, pager_top: b.pager_top, refine_top: top('ask-refine'), scroll: Math.round(scrollY),
      hash: location.hash, focus: focus()};
  });
  await step('ask_sort', async () => {
    // The view bar on screen, the refine bar's head just above the top: a sort does not scroll.
    scrollTo({top: scrollY + document.getElementById('ask-refine').getBoundingClientRect().top + 40, behavior: 'instant'});
    const before = Math.round(scrollY);
    document.getElementById('ask-view-sort-recent').click(); await sleep(150);
    return {before: before, scroll: Math.round(scrollY), hash: location.hash};
  });
  await step('ask_fold', async () => {
    document.getElementById('ask-view-group-author').click(); await sleep(150);
    const b = document.querySelector('#ask-list > .lvhead button');
    const id = b.id;
    b.focus(); b.click(); await sleep(150);
    const folded = focus();
    document.activeElement.click(); await sleep(150);
    return {id: id, folded: folded, unfolded: focus()};
  });
  await step('posts_bottom', async () => {
    location.hash = '#revisar/posts?f=all&size=50&group=author'; await sleep(150);
    const b = bottomNext('pager');
    b.click(); await sleep(150);
    return {before: b.before, bar_top: top('posts-bar'), scroll: Math.round(scrollY), hash: location.hash, focus: focus()};
  });
  await step('posts_fold', async () => {
    // Page 2 opens on a group continued from page 1: folded from that head, the group's head
    // is on page 1, and the reader (and the focus) go there with it — no history entry.
    const b = document.querySelector('#cards > .lvhead button');
    const id = b.id, cont = !!b.querySelector('.lvcont'), h0 = history.length;
    b.focus(); b.click(); await sleep(150);
    const folded = Object.assign(focus(), {hash: location.hash, on_screen: inView(document.activeElement)});
    document.activeElement.click(); await sleep(150);
    return {id: id, cont: cont, folded: folded, unfolded: focus(), pushed: history.length - h0};
  });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
)


@_requires_chrome
def test_a_page_turned_from_the_bottom_lands_on_the_lists_head_with_the_focus_on_its_page(
    varied,
):
    """PR 17 review C1 + I1 (UX): from the bottom pager, the list's head is at the top of the
    screen and the focus is on the top pager's new current page; a sort does not scroll; a
    fold keeps the focus on its button (a stable id), so a second Enter unfolds it."""
    _, data, page_file = varied
    root = page_file.parent / "land"
    root.mkdir(exist_ok=True)
    page = _page(root, data, _LAND_PROBE)

    seen = _dump(page.as_uri() + _ask({"size": "50"}), window="1280,900")

    _no_error(seen)
    ask = seen["ask_bottom"]
    assert ask["before"] > 2000 and ask["hash"].endswith("&size=50&page=2")
    assert 0 <= ask["refine_top"] <= 10, ask
    assert ask["focus"] == {
        "id": "ask-pager-top-p2",
        "tag": "A",
        "label": "Página 2",
        "current": "page",
        "nav": "ask-pager-top",
        "expanded": None,
    }
    sort = seen["ask_sort"]
    assert "sort=recent" in sort["hash"] and sort["scroll"] == sort["before"]
    fold = seen["ask_fold"]
    assert fold["id"].startswith("ask-list-fold-")
    assert (fold["folded"]["id"], fold["folded"]["expanded"]) == (fold["id"], "false")
    assert (fold["unfolded"]["id"], fold["unfolded"]["expanded"]) == (fold["id"], "true")
    posts = seen["posts_bottom"]
    assert posts["before"] > 2000 and "page=2" in posts["hash"]
    assert 0 <= posts["bar_top"] <= 10, posts
    assert (posts["focus"]["id"], posts["focus"]["current"]) == ("pager-top-p2", "page")
    pfold = seen["posts_fold"]
    assert pfold["id"].startswith("cards-fold-") and pfold["cont"] is True
    assert (pfold["folded"]["id"], pfold["folded"]["expanded"]) == (pfold["id"], "false")
    assert "page=" not in pfold["folded"]["hash"] and pfold["folded"]["on_screen"] is True
    assert (pfold["unfolded"]["id"], pfold["unfolded"]["expanded"]) == (pfold["id"], "true")
    assert pfold["pushed"] == 0


#: Posts' back and forward: page 3, then 4 (each a history entry), then a search that leaves
#: one page (it replaces the entry, like any typing). Back is page 3 of the whole list again.
_POSTS_BACK = (
    "<script>"
    + _SEEN
    + _LV_JS
    + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  const next = () => document.querySelector('#pager a.pa[aria-label="Página siguiente"]').click();
  await step('p3', async () => { await sleep(80); next(); await sleep(120); next(); await sleep(120); return lvPosts(); });
  await step('p4', async () => { next(); await sleep(120); return lvPosts(); });
  await step('search', async () => {
    const box = document.getElementById('search');
    box.value = document.querySelector('#cards .card').dataset.id; box.dispatchEvent(new Event('input')); await sleep(120);
    return lvPosts();
  });
  await step('back', async () => { history.back(); await sleep(200); return Object.assign(lvPosts(), {q: document.getElementById('search').value}); });
  await step('forward', async () => { history.forward(); await sleep(200); return lvPosts(); });
  await step('past', async () => { location.hash = '#revisar/posts?f=all&size=10&page=999'; await sleep(200); return lvPosts(); });
  await step('filter', async () => {
    // A rail filter is a history entry like a sort (arch m1: one rule): Back undoes it.
    const before = location.hash;
    const b = [...document.querySelectorAll('#rail > button.f')].find(x => !x.classList.contains('on') && x.getAttribute('aria-pressed') !== 'true');
    b.click(); await sleep(200);
    const moved = location.hash;
    history.back(); await sleep(200);
    return {before: before, moved: moved, back: location.hash};
  });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
)


@_requires_chrome
def test_posts_back_to_a_bigger_view_keeps_its_page_and_its_history_entry(varied):
    """PR 17 review I2 (arch): the page of a hash is checked against the view it opens, never
    the one on screen before — so Back from a one-page search returns to page 3 of the whole
    list, and the entry is not rewritten. A page past the last still opens the last."""
    _, data, page_file = varied
    root = page_file.parent / "back"
    root.mkdir(exist_ok=True)
    page = _page(root, data, _POSTS_BACK)
    order = [p["id"] for p in data["posts"]]

    seen = _dump(page.as_uri() + "#revisar/posts?f=all&size=10")

    _no_error(seen)
    assert seen["p3"]["ids"] == order[20:30] and seen["p4"]["ids"] == order[30:40]
    assert seen["search"]["ids"] == [order[30]]
    back = seen["back"]
    assert back["hash"].endswith("size=10&page=3") and "q=" not in back["hash"]
    assert back["ids"] == order[20:30] and back["bottom"]["current"] == "3" and back["q"] == ""
    assert "q=" in seen["forward"]["hash"] and seen["forward"]["ids"] == [order[30]]
    last = (len(order) + 9) // 10
    assert seen["past"]["bottom"]["current"] == str(last)
    assert seen["past"]["hash"].endswith(f"size=10&page={last}")
    f = seen["filter"]
    assert f["moved"] != f["before"] and "page=" not in f["moved"] and f["back"] == f["before"]


#: A 375 px frame of the page (headless Chrome keeps a window ≥ 500 px; served, so the frame is
#: same-origin): the controls over the results and over Posts, measured.
_PHONE_LV = (
    "<script>"
    + r"""
(async () => {
  if (window !== window.top) return;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  const measure = async (hash, bar, pager, first) => {
    const phone = document.createElement('iframe');
    phone.style.cssText = 'width:375px;height:812px;border:0';
    phone.src = location.pathname + hash;
    document.body.prepend(phone);
    await new Promise(r => phone.addEventListener('load', r, {once: true}));
    const d = phone.contentDocument, w = phone.contentWindow;
    for (let i = 0; i < 200 && !d.querySelector(first); i++) await sleep(25);
    await sleep(200);
    const r = (e) => e.getBoundingClientRect();
    const track = d.querySelector('#' + bar + ' .segtrack');
    const buttons = [...track.querySelectorAll('button')];
    const pressed = track.querySelector('[aria-pressed="true"]');
    const tops = new Set(buttons.map(b => Math.round(r(b).top)));
    const range = d.querySelector('#' + pager + ' .prange'), nums = d.querySelector('#' + pager + ' .pnums');
    const end = d.querySelector('#' + (pager === 'pager-top' ? 'pager' : 'ask-pager') + ' select');
    const out = {width: w.innerWidth, page: d.documentElement.scrollWidth,
      controls: Math.round(r(d.querySelector(first)).top - r(d.getElementById(bar)).top),
      track_rows: tops.size, track_scrolls: track.scrollWidth > track.clientWidth,
      pressed_seen: r(pressed).left >= r(track).left - 1 && r(pressed).right <= r(track).right + 1,
      label_inline: Math.abs(r(d.querySelector('#' + bar + ' .lvseg .lvlab')).top - r(track).top) < 12,
      size_in_pager: !!end && end.checkVisibility(),
      pager_rows: range && nums ? Math.abs(r(range).top - r(nums).top) < 12 : null};
    phone.remove();
    return out;
  };
  try {
    out.ask = await measure('#ask?q=__SHA__&sort=author&group=author', 'ask-view', 'ask-pager-top', '#ask-list > .lvhead');
    out.posts = await measure('#revisar/posts?f=all&s=cost&group=month', 'posts-bar', 'pager-top', '#cards > .lvhead');
  } catch (e) { out.error = e.message; }
  fetch('/probe-done', {method: 'POST', body: JSON.stringify(out)});
})();
</script>"""
).replace("__SHA__", SHA)


@_requires_chrome
def test_at_375_px_the_controls_are_compact_and_the_sort_track_is_one_scrolling_row(varied):
    """PR 17 review I3 (UX): on a phone the sort track stays ONE row that scrolls (the pressed
    sort scrolled into view), «Ordenar» sits beside it, «Por página» lives in the bottom pager's
    row, the top pager's range and page numbers share a line, and the bar plus the pager take
    ≤ 200 px before the list's first row — grouped, which is the tallest, with the last sort
    pressed."""
    from tests.test_jev_page_embed_browser import _live

    _, data, page_file = varied
    root = page_file.parent / "phone"
    root.mkdir(exist_ok=True)
    page = _page(root, data, _PHONE_LV)

    seen = _live(page, "#revisar/posts?f=all")

    assert "error" not in seen, seen
    for name in ("ask", "posts"):
        view = seen[name]
        assert view["width"] == 375 and view["page"] <= 375, (name, view)
        assert view["track_rows"] == 1 and view["track_scrolls"], (name, view)
        assert view["pressed_seen"] and view["label_inline"], (name, view)
        assert view["size_in_pager"] and view["pager_rows"], (name, view)
        assert view["controls"] <= 200, (name, view)

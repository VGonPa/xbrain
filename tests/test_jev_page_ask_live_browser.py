# tests/test_jev_page_ask_live_browser.py
"""A served ask's results fill in and re-rank WHILE it runs, in a real browser, with a fake.

The job's answers come from a GATED fake: each call waits for a permit the probe hands out
(`/probe-release`), so what the page must show mid-job is there however slowly Chrome runs —
never a guessed sleep, and no request is ever held while the page waits on its own timers
(the gate blocks the job's thread, not a reply). The fake answers each post with a planned
probability rising with its place in the selection, so every wave of answers ranks ABOVE the
cards already shown: each new card must land in its ranked place, the cards already there must
be kept (the same nodes), and the reader's place must hold.

Probes follow `test_jev_page_browser.py`'s rules: they read what a reader SEES, go through the
page's own controls, and are fail-closed (a step that throws is an `ERROR` value the assertions
reject). The page's own functions are wrapped only to OBSERVE (`askAbsorb`, `finished`).
"""

from __future__ import annotations

import json
import random
import re
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.jev_ask_fixtures import _START, VOCAB, _Planned, victor_item
from xbrain.jev.ask import AskFilters, AskQuery, finish_ask, plan_ask
from xbrain.jev.run import run_ask
from xbrain.jev.lock import pass_lock
from tests.test_jev_page_ask_browser import _ASK_JS, _REFINE_JS, _static_page
from tests.test_jev_page_browser import (
    _SEEN,
    _SERVE_JS,
    _dump,
    _need_chrome,
    _requires_chrome,
    _served_dump,
    _step,
)
from xbrain.config import Config, load_config
from xbrain.jev import run as run_module
from xbrain.jev import service as service_module
from xbrain.jev.service import JevService
from xbrain.models import Item
from xbrain.rubrics import save_vocab
from xbrain.store import save_store

#: Posts in the repo, all asked (no filter, no answer yet).
POSTS = 24
#: `[jev].ask_top` here: the list shows six, «Ver más» the rest.
TOP = 6
#: How many answers the probe lets through, wave by wave.
WAVES = (4, 4, 2, 6, 8)
LIVE_QUERY = "¿Qué posts cuentan cómo trabajar con agentes?"


def _p(n: int) -> float:
    """Post n's planned probability: rising with n, so a later wave outranks the earlier ones.
    n=5 ties n=4 (inside a wave) and n=9 ties n=7 (across waves): the id breaks both ties."""
    return {5: _base(4), 9: _base(7)}.get(n, _base(n))


def _base(n: int) -> float:
    return round(0.2 + 0.03 * n, 2)


def _items() -> list[Item]:
    return [
        victor_item(
            n,
            topic="agentic-engineering" if n % 2 else "startups",
            created=_START + timedelta(hours=n),
        )
        for n in range(POSTS)
    ]


PLANNED = {item.id: _p(n) for n, item in enumerate(_items())}


def _ranked(ids: set[str] | list[str], minimum: float = 0.0) -> list[str]:
    """THE expected order: `ask._rank`'s key `(-p, id)` over `ids` at or above `minimum`."""
    kept = [post for post in ids if PLANNED[post] >= minimum]
    return sorted(kept, key=lambda post: (-PLANNED[post], post))


def _live_repo(root: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    vault = root / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(
        f'[paths]\nvault = "{vault}"\noutput_subdir = "x"\ndata_dir = "data"\n'
        f'[x]\nhandle = "v"\n[jev]\nconcurrency = 3\nask_top = {TOP}\n',
        encoding="utf-8",
    )
    (root / "data").mkdir(exist_ok=True)
    save_store({item.id: item for item in _items()}, root / "data" / "items.json")
    save_vocab(VOCAB, root / "data" / "vocab.yaml")
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(root))
    return load_config(root)


class _Gated(_Planned):
    """`_Planned` (each post's planned probability) whose call about post n waits until the
    probe has let n+1 answers through (`release`). By POSITION, not by permit: a permit goes
    to whichever worker takes it first (a worker freed by one answer can take the next
    permit before a waiting one wakes), so a wave would not be the first k posts. The pool
    holds the lowest posts not yet answered, so the one allowed next is always in flight."""

    def __init__(self) -> None:
        super().__init__({f"p{n:04d}": _p(n) for n in range(POSTS)})
        self.allowed = 0
        self.moved = threading.Condition()

    def release(self, n: int) -> None:
        """`n` more posts may answer; a negative `n` starts the count again (a new job)."""
        with self.moved:
            self.allowed = 0 if n < 0 else self.allowed + n
            self.moved.notify_all()

    def ask(self, state, questions):
        n = int(state["post"].split()[0][1:])
        with self.moved:
            assert self.moved.wait_for(lambda: n < self.allowed, timeout=90), (
                f"the probe never let post {n} answer"
            )
        return super().ask(state, questions)


#: What the live list shows, read as a reader sees it.
_LIVE_JS = r"""
const sLiveView = () => ({
  reach: txt(sId('ask-reach')),
  ids: sQResults().map(r => r.id),
  nuevo: [...document.querySelectorAll('#ask-list > .card')].filter(c => seen(c.querySelector('.askr .nuevo'))).map(c => c.dataset.id),
  more: txt(sId('ask-more')),
  empty: txt(sId('ask-empty')),
  number: askLive ? askLive.number : null,
  live: sId('ask-list') ? sId('ask-list').dataset.live || null : null,
  min: askView.refine.min,
  hash: location.hash,
  scroll: Math.round(scrollY),
});
// Observed, never changed: what the list showed right after each look added answers, and
// right before the job's end replaced it.
const sAbsorbed = [];
const sAbsorb0 = askAbsorb;
askAbsorb = function (job) {
  // The cards of THIS job's live list before the look (another list's are not its cards).
  const l = sId('ask-list');
  const before = l && askLive && l.dataset.live === askLive.sha ? [...l.querySelectorAll(':scope > .card')].map(c => c.dataset.id) : [];
  const took = sAbsorb0(job);
  if (took && job.stream.answers.length) {
    sAbsorbed.push(Object.assign(sLiveView(), {got: job.stream.answers.map(a => a.id), before: before,
      job: job.number, more: job.stream.more, state: job.state, at: performance.now(),
      cached: job.stream.answers.filter(a => a.cached).map(a => a.id)}));
  }
  return took;
};
let sBeforeEnd = null, sEndList = null;
const sFinished0 = finished;
finished = async function (job) {
  sBeforeEnd = sLiveView();
  // The cards and the place right before the end replaces the live list.
  sEndList = sId('ask-list');
  document.querySelectorAll('#ask-list > .card').forEach(c => { c.__end = true; });
  sBeforeEnd.list_top = sEndList ? Math.round(sEndList.getBoundingClientRect().top) : null;
  const firstSeen = [...document.querySelectorAll('#ask-list > .card')].find(c => c.getBoundingClientRect().bottom > 0);
  sBeforeEnd.reader = firstSeen ? {id: firstSeen.dataset.id, top: Math.round(firstSeen.getBoundingClientRect().top)} : null;
  return sFinished0(job);
};
// When (page clock) the ask was confirmed and each look at its stream went out.
const sLooks = [];
let sConfirmed = null;
// While `sHold` is a promise, the page's looks at its stream wait for it before they go out.
let sHold = null;
// While `sHoldWatch` is a promise, the idle watch's looks (`/api/job`, no cursor) wait for it.
let sHoldWatch = null;
const sFetchLooks = window.fetch;
window.fetch = function (u, init) {
  const look = String(u).includes('/api/job?since=');
  const idle = /\/api\/job$/.test(String(u));
  const held = look ? sHold : idle ? sHoldWatch : null;
  const p = held ? held.then(() => sFetchLooks.call(window, u, init)) : sFetchLooks.call(window, u, init);
  if (look) sLooks.push(performance.now());
  if (String(u).includes('/api/ask/evaluate')) p.then(() => { sConfirmed = performance.now(); });
  return p;
};
const sRelease = (n) => sFetch0.call(window, '/probe-release', {method: 'POST', body: String(n)});
const sAnswered = (k) => sWait(() => (txt(sId('ask-reach')) || '').includes(' de ' + k + ' respondidos hasta ahora'), k + ' respuestas');
// The job block while an ask runs: one status line right over the gallery.
const sStatus = () => { const box = sId('jobp'), bar = box.querySelector('[role=progressbar]');
  return {text: txt(sId('ask-progress')), slim: box.classList.contains('slim'), shown: seen(box),
    bar: seen(bar), title: seen(sId('ask-title')), prev: box.previousElementSibling && box.previousElementSibling.id,
    next: box.nextElementSibling && box.nextElementSibling.id, height: Math.round(box.getBoundingClientRect().height),
    top: Math.round(box.getBoundingClientRect().top), stop: txt(sId('ask-stop')), stop_title: sId('ask-stop').title}; };
// The ids of the list's cards on screen now.
const sOnScreen = () => [...document.querySelectorAll('#ask-list > .card')].filter(c => {
  const r = c.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; }).map(c => c.dataset.id);
// The job, asked of the server the way the page does (never one of the page's looks), until
// `ok(job)`: each try paced by one `/probe-wait`.
const sJobUntil = async (ok, what) => {
  for (let i = 0; i < 3000; i++) {
    const j = await api(API.job);
    if (ok(j)) return j;
    await sFetch0.call(window, '/probe-wait');
  }
  throw new Error('nunca: ' + what);
};
// Another tab's ask: estimate and confirm through the API, as a page would.
const sAskElsewhere = async (query) => {
  const e = await api(API.ask.estimate, {query: query});
  return api(API.ask.evaluate, {query: query, confirm_token: e.confirm_token});
};
const sTag = (tag) => document.querySelectorAll('#ask-list > .card').forEach(c => { c.__probe = c.__probe || tag; });
const sTags = () => [...document.querySelectorAll('#ask-list > .card')].map(c => c.__probe || null);
"""

_LIVE_PROBE = (
    "<script>"
    + _SERVE_JS
    + _ASK_JS
    + _REFINE_JS
    + _LIVE_JS
    + r"""
(async () => {
  await sStep('start', async () => {
    location.hash = '#ask';
    await sWait(() => seen(sId('ask-form')), 'el formulario');
    sId('ask-q').value = '__QUERY__';
    sId('ask-q').dispatchEvent(new Event('input', {bubbles: true}));
    sPress(sId('ask-form'), 'Estimar lo que cuesta');
    await sWait(() => sPanel().go, 'la estimación');
    sId('ask-go').click();
    await sWait(() => sId('ask-list') && sId('ask-list').dataset.live && location.hash.startsWith('#ask?q='), 'la lista en vivo');
    return Object.assign(sLiveView(), {history: sQHistory(), head: txt(sId('ask-head')), status: sStatus()});
  });
  await sStep('wave1', async () => {
    await sRelease(4);
    await sAnswered(4);
    await sWait(() => /^Preguntando… 4 de /.test(sStatus().text || ''), 'la línea de estado');
    sTag('w1');
    return Object.assign(sLiveView(), {status: sStatus()});
  });
  await sStep('anchored', async () => {
    // The reader has scrolled into the list: its second card at the top of the screen.
    const cards = [...document.querySelectorAll('#ask-list > .card')];
    const anchor = cards[1];
    scrollTo({top: scrollY + anchor.getBoundingClientRect().top, behavior: 'instant'});
    const before = {id: anchor.dataset.id, top: anchor.getBoundingClientRect().top,
      list_top: sId('ask-list').getBoundingClientRect().top, scroll: Math.round(scrollY), on_screen: sOnScreen()};
    await sRelease(4);
    await sAnswered(8);
    const again = sId('ask-post-' + before.id);
    return Object.assign(sLiveView(), {before, after: again ? again.getBoundingClientRect().top : null,
      same_node: again === anchor, tags: sTags(), status: sStatus()});
  });
  await sStep('top', async () => {
    // The list's start on screen, 40 px down: new answers are seen landing there — nothing is
    // held, the page does not move.
    scrollTo({top: scrollY + sId('ask-list').getBoundingClientRect().top - 40, behavior: 'instant'});
    const before = {scroll: Math.round(scrollY), list_top: sId('ask-list').getBoundingClientRect().top,
      first_top: document.querySelector('#ask-list > .card').getBoundingClientRect().top};
    await sRelease(2);
    await sAnswered(10);
    return Object.assign(sLiveView(), {before, list_top: sId('ask-list').getBoundingClientRect().top});
  });
  await sStep('refined', async () => {
    // The free refine, mid-run: a minimum. Later answers are kept by it as they arrive.
    sQSet({'refine-min': '0.6'});
    await sWait(() => (txt(sId('ask-reach')) || '').includes('≥ 0,60'), 'el mínimo en vivo');
    const set = sLiveView();
    // The reader is typing in the free refine («Autor») while answers land: left alone.
    sId('refine-who-open').click();
    const input = sId('refine-author');
    input.focus();
    input.value = 'som';
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.setSelectionRange(2, 2);
    await sRelease(6);
    await sAnswered(16);
    await sWait(() => /^Preguntando… 16 de /.test(sStatus().text || ''), 'la línea de estado');
    const typing = {same: sId('refine-author') === input, connected: input.isConnected, value: input.value,
      active: document.activeElement === input, caret: [input.selectionStart, input.selectionEnd]};
    // Put back as it was before leaving it: no change, so nothing is refined by it.
    input.value = '';
    input.blur();
    return {set, typing, after: Object.assign(sLiveView(), {refine: sQRefine(), status: sStatus()})};
  });
  await sStep('rail', async () => {
    // The page's data built mid-run (a reload does this): the history has the running query's
    // row, rebuilt from its checkpoint. The rail says the job's live numbers, not that row's.
    await reloadData();
    const row = askRows().find(r => r.sha === askLive.sha);
    return {history: sQHistory(), rebuilt: row ? row.rebuilt : null, status: sStatus()};
  });
  await sStep('end', async () => {
    const r0 = sRefreshed;
    await sRelease(8);
    await sWait(() => sRefreshed > r0, 'el final del trabajo');
    return Object.assign(sLiveView(), {before_end: sBeforeEnd, head: txt(sId('ask-head')),
      history: sQHistory(), panel: sPanel(), status: sStatus(),
      same_list: sId('ask-list') === sEndList, kept: [...document.querySelectorAll('#ask-list > .card')].map(c => !!c.__end),
      list_top: Math.round(sId('ask-list').getBoundingClientRect().top),
      reader_top: sBeforeEnd && sBeforeEnd.reader && sId('ask-post-' + sBeforeEnd.reader.id)
        ? Math.round(sId('ask-post-' + sBeforeEnd.reader.id).getBoundingClientRect().top) : null});
  });
  await sStep('absorbed', async () => sAbsorbed);
  await sStep('looks', async () => ({confirmed: sConfirmed, looks: sLooks}));
  sDone();
})();
</script>"""
).replace("__QUERY__", LIVE_QUERY)


@pytest.fixture(scope="module")
def live(tmp_path_factory) -> dict[str, Any]:
    client = _Gated()
    finished: list[Any] = []
    real = service_module.finish_ask

    def _spy(*args: Any, **kw: Any):
        found = real(*args, **kw)
        finished.append(found)
        return found

    class _Releasing(JevService):
        def probe_release(self, n: int) -> None:
            client.release(n)

    seen = _served_dump(
        tmp_path_factory.mktemp("ask-live"),
        _LIVE_PROBE,
        make_client=lambda: client,
        repo=_live_repo,
        base=_Releasing,
        patch=lambda mp: (
            mp.setattr(service_module, "finish_ask", _spy),
            # A checkpoint per answer: the page's data built mid-run has the running query's
            # row, rebuilt from it (`reconstruida`), as a long real ask does.
            mp.setattr(run_module, "CHECKPOINT_EVERY", 1),
        ),
    )
    seen["finished"] = [
        [(item.id, record.probability) for item, record in found.ranked] for found in finished
    ]
    return seen


def _ids(n: int) -> list[str]:
    """The ids of the first `n` posts the pool asks (selection order = store order): after
    `n` released, exactly these have answered."""
    return [item.id for item in _items()[:n]]


#: The money a line says: four decimals, as the history says it.
_USD = r"~\d,\d{4} \$"


def _nuevo_rule(look: dict[str, Any], order: list[str]) -> None:
    """«nuevo» marks a new answer that lands ABOVE a card already on the list — never one that
    only extends the list's end, and nothing on a first fill. `order` is the ranking so far."""
    rank = {post: i for i, post in enumerate(order)}
    before = [post for post in look["before"] if post in rank]
    lowest = max((rank[post] for post in before), default=-1)
    new = set(look["got"]) & set(look["ids"])
    above = {post for post in new if rank[post] < lowest}
    assert above <= set(look["nuevo"]), look
    assert not (new - above) & set(look["nuevo"]), look


@_requires_chrome
def test_the_live_list_opens_empty_under_the_asked_query_and_the_rail_says_it_runs(live):
    start = _step(live, "start")

    assert start["hash"].startswith("#ask?q=") and start["live"]
    assert start["ids"] == [] and start["more"] is None
    assert start["empty"].startswith("Aún no ha llegado ninguna respuesta")
    # The gallery's line says what it shows; k of N is the status line's, not said twice.
    assert (
        start["reach"] == "Mostrando 0 de 0 respondidos hasta ahora, de mayor a menor probabilidad."
    )
    # No bar-only phase: the job block is already one status line — under the query's title,
    # over the refine bar and the (empty) gallery.
    status = start["status"]
    assert status["slim"] and status["shown"] and not status["bar"] and not status["title"]
    assert (status["prev"], status["next"]) == ("ask-head", "ask-refine")
    assert (status["stop"], status["stop_title"]) == ("Parar", "se guarda lo ya pagado")
    assert start["head"].startswith(f"«{LIVE_QUERY}»Preguntando ahora · Sin filtros")
    assert "cada respuesta entra" not in start["head"]
    assert re.fullmatch(
        rf"{re.escape(LIVE_QUERY)} · en curso · 0 de {POSTS} · {_USD}", start["history"][0]["text"]
    )
    assert start["history"][0]["current"] is True


@_requires_chrome
def test_every_answer_arrives_once_and_each_look_shows_the_ranked_top(live):
    """After EVERY look that brought answers: the cards on screen are the top of the answers so
    far, ranked by `(-p, id)` (and kept by the refine in effect) — plus, briefly, cards the
    reader has on screen that better answers pushed past the top — and «nuevo» marks exactly
    the new ones landing above a card already there."""
    absorbed = _step(live, "absorbed")
    so_far: list[str] = []

    assert len(absorbed) >= len(WAVES)
    assert absorbed[0]["nuevo"] == []  # the first fill marks nothing
    for look in absorbed:
        so_far += look["got"]
        expected = _ranked(so_far, look["min"] or 0.0)
        assert look["ids"][:TOP] == expected[:TOP], look
        # Past the top: only cards that were already there, still in rank order.
        assert set(look["ids"][TOP:]) <= set(look["before"]), look
        assert look["ids"] == [post for post in expected if post in set(look["ids"])], look
        assert look["reach"].startswith(
            f"Mostrando {min(len(expected), TOP)} de {len(so_far)} respondidos hasta ahora"
        )
        _nuevo_rule(look, expected)
    assert sorted(so_far) == sorted(PLANNED) and len(so_far) == len(set(so_far))


@_requires_chrome
def test_a_new_answer_lands_in_its_ranked_place_and_the_cards_there_stay(live):
    wave1, anchored = _step(live, "wave1"), _step(live, "anchored")

    assert wave1["ids"] == _ranked(_ids(4))
    assert wave1["nuevo"] == []  # the list's first fill: nothing to be news against
    # No answer reaches the bar yet (`[jev].threshold`): the status line does not say «0».
    assert re.fullmatch(rf"Preguntando… 4 de {POSTS} · {_USD} gastado", wave1["status"]["text"])
    assert wave1["status"]["slim"] and wave1["status"]["height"] < 90
    ranked8 = _ranked(_ids(8))
    # n=4..7 outrank n=0..3: four new cards above, all «nuevo». The old ones still in the top
    # are the SAME nodes (never drawn again); of the two pushed past it, those on the reader's
    # screen stay, briefly, below them.
    held = [post for post in ranked8[TOP:] if post in anchored["before"]["on_screen"]]
    assert anchored["ids"] == ranked8[:TOP] + held
    assert anchored["tags"] == [("w1" if post in _ids(4) else None) for post in anchored["ids"]]
    assert set(_ids(8)[4:]) <= set(anchored["nuevo"])
    left = 2 - len(held)
    assert anchored["more"] == (f"Ver {left} más (quedan {left})" if left else None)


@_requires_chrome
def test_the_readers_place_holds_while_answers_land_above_it(live):
    anchored = _step(live, "anchored")
    before = anchored["before"]

    assert before["list_top"] < 0 and abs(before["top"]) <= 1  # scrolled into the list
    assert anchored["same_node"] is True
    assert anchored["after"] == pytest.approx(before["top"], abs=1)
    # The page moved under the reader (four cards above), the reader did not.
    assert anchored["scroll"] > before["scroll"]


@_requires_chrome
def test_the_status_line_and_parar_stay_on_screen_while_the_gallery_scrolls(live):
    anchored = _step(live, "anchored")

    assert anchored["before"]["list_top"] < 0
    assert anchored["status"]["shown"] and anchored["status"]["top"] == 0
    assert anchored["status"]["stop"] == "Parar"


@_requires_chrome
def test_with_the_lists_start_on_screen_new_answers_are_seen_landing(live):
    top = _step(live, "top")
    before = top["before"]

    # The list's first card was on screen (so a hold would have had a card to hold) …
    assert before["list_top"] == pytest.approx(40, abs=1) and before["first_top"] < 600
    # … and nothing was held: the page did not move, the list did not either.
    assert top["scroll"] == before["scroll"] > 0
    assert top["list_top"] == pytest.approx(before["list_top"], abs=1)
    assert top["ids"] == _ranked(_ids(10))[:TOP]
    # n=9 ties n=7 at 0.41 and ranks after it (ids break ties); n=8 is first.
    assert top["ids"][0] == _ids(10)[8]


@_requires_chrome
def test_the_free_refine_works_while_the_ask_runs(live):
    refined = _step(live, "refined")
    at_set, after = refined["set"], refined["after"]

    assert "min=0.6" in at_set["hash"]
    assert at_set["ids"] == _ranked(_ids(10), 0.6)[:TOP] == []
    assert after["ids"] == _ranked(_ids(16), 0.6)[:TOP]
    over = len(_ranked(_ids(16), 0.6))
    assert after["reach"] == (
        f"Mostrando {over} de 16 respondidos hasta ahora · "
        f"{over} con relevancia ≥ 0,60, de mayor a menor probabilidad."
    )
    # The status line counts at the refine's minimum once one is set.
    assert after["status"]["text"].startswith(
        f"Preguntando… 16 de {POSTS} · {over} con relevancia ≥ 0,60 · "
    )
    assert after["refine"]["min"] == "0.6"
    assert after["refine"]["author"] == ""


@_requires_chrome
def test_typing_in_the_free_refine_is_left_alone_while_answers_land(live):
    """PR 16 tests I2: six answers landed while the reader typed «som» in «Autor» (no change
    yet): the field is the same node, still focused, with what they typed and the caret."""
    typing = _step(live, "refined")["typing"]

    assert typing == {
        "same": True,
        "connected": True,
        "value": "som",
        "active": True,
        "caret": [2, 2],
    }


@_requires_chrome
def test_the_rail_says_the_running_querys_live_numbers_never_reconstruida(live):
    rail = _step(live, "rail")

    # The mid-run data really had the checkpoint's rebuilt row …
    assert rail["rebuilt"] is True
    # … and the rail says the job, as the status line does.
    first = rail["history"][0]["text"]
    assert re.fullmatch(rf"{re.escape(LIVE_QUERY)} · en curso · 16 de {POSTS} · {_USD}", first)
    assert "reconstruida" not in first
    assert rail["status"]["text"].startswith(f"Preguntando… 16 de {POSTS} · ")


@_requires_chrome
def test_the_final_list_is_finish_asks_ranking_in_the_live_order(live):
    end = _step(live, "end")
    live_order = end["before_end"]["ids"]
    (ranking,) = live["finished"]

    # `finish_ask` ranked every answer (a new query: no minimum), the order the CLI prints.
    assert [post for post, _ in ranking] == _ranked(list(PLANNED))
    assert [p for _, p in ranking] == [PLANNED[post] for post, _ in ranking]
    # Complete, the live list already WAS that ranking under the reader's refine; the end
    # replaced it with the blob's row and nothing moved.
    expected = [post for post, p in ranking if p >= 0.6][:TOP]
    assert f" de {POSTS} respondidos hasta ahora" in end["before_end"]["reach"]
    assert live_order == expected == end["ids"]
    # Its refine stayed; the head is the history's now.
    assert "min=0.6" in end["hash"] and end["live"] is None
    assert end["reach"].startswith(f"Mostrando {TOP} de ")
    assert end["history"][0]["text"].startswith(LIVE_QUERY + " · ")
    assert "en curso" not in end["history"][0]["text"]


@_requires_chrome
def test_at_the_end_the_list_is_refilled_in_place_and_the_status_line_says_how_it_went(live):
    """UX I4: no progress bar comes back and the gallery does not move — the same list, the
    same card nodes, the same place — and one quiet line says what the ask did."""
    end = _step(live, "end")
    status = end["status"]
    over = len(_ranked(list(PLANNED), 0.6))

    assert end["same_list"] is True and end["kept"] == [True] * TOP
    # The reader's first card on screen stays where it was (the head above it may change).
    assert end["before_end"]["reader"] is not None
    assert end["reader_top"] == pytest.approx(end["before_end"]["reader"]["top"], abs=1)
    assert status["slim"] and status["shown"] and not status["bar"] and not status["title"]
    assert status["prev"] == "ask-head"
    assert re.fullmatch(
        rf"Preguntado: {POSTS} leídos · {over} con relevancia ≥ 0,60 · {_USD} gastado",
        status["text"],
    )
    assert end["panel"]["close"] == "Cerrar" and end["panel"]["stop"] is None


# --------------------------------------------------------------------------- asked again

#: The query's first use (before the page loads): two posts, at a minimum of 0.5 — both
#: current when it is asked again, with probabilities of their own.
SEEDED = {0: 0.95, 1: 0.9}
#: Two more queries, each asked by «another tab» through the API while the page watches.
QUERY_B = "¿Qué posts hablan de startups?"
QUERY_C = "¿Qué posts hablan de precios?"


def _reasked(n: int) -> float:
    """What the re-asked query's answer for post n is: the seed's for 0 and 1, else planned."""
    return SEEDED.get(n, _p(n))


def _reask_repo(root: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """`_live_repo`, and LIVE_QUERY already asked in a terminal: `--limit 2 --min 0.5`."""
    cfg = _live_repo(root, monkeypatch)
    with pass_lock(cfg.jev_lock_path, "xbrain jev ask") as lock:
        plan = plan_ask(cfg, AskQuery.of(LIVE_QUERY), AskFilters(), 2)
        seeded = _Planned({f"p{n:04d}": p for n, p in SEEDED.items()})
        outcome = run_ask(cfg, plan, lambda: seeded, lock=lock)
        finish_ask(cfg, plan, outcome, minimum=0.5, now=_START + timedelta(days=3))
    return cfg


_REASK_PROBE = (
    (
        "<script>"
        + _SERVE_JS
        + _ASK_JS
        + _REFINE_JS
        + _LIVE_JS
        + r"""
// Holds the page's looks at its stream (`sHold`) until the returned function is called.
const sHoldLooks = () => { let go; sHold = new Promise(r => { go = r; }); return () => { sHold = null; go(); }; };
const sHoldWatchLooks = () => { let go; sHoldWatch = new Promise(r => { go = r; }); return () => { sHoldWatch = null; go(); }; };
(async () => {
  await sStep('first', async () => {
    location.hash = '#ask';
    await sWait(() => seen(sId('ask-form')), 'el formulario');
    sId('ask-q').value = '__QUERY__';
    sId('ask-q').dispatchEvent(new Event('input', {bubbles: true}));
    sPress(sId('ask-form'), 'Estimar lo que cuesta');
    await sWait(() => sPanel().go, 'la estimación');
    sId('ask-go').click();
    await sWait(() => sAbsorbed.length >= 1, 'la primera mirada');
    return Object.assign(sLiveView(), {live_min: askLive.min, refine: sQRefine(), status: sStatus()});
  });
  await sStep('more', async () => {
    // Twelve answers banked while the page's looks wait: more than a reply carries (3 here).
    const unhold = sHoldLooks();
    await sRelease(14);
    await sJobUntil(j => j.done >= 12, '12 respuestas');
    unhold();
    await sAnswered(14);
    return {looks: sLooks.slice()};
  });
  await sStep('reask_end', async () => {
    const r0 = sRefreshed;
    await sRelease(10);
    await sWait(() => sRefreshed > r0, 'el final del trabajo');
    return Object.assign(sLiveView(), {before_end: sBeforeEnd, status: sStatus()});
  });
  let b = null;
  await sStep('follower', async () => {
    // Another tab asks QUERY_B; four answers are in before this page meets the job.
    location.hash = '#ask';
    await sRelease(-1);
    b = await sAskElsewhere('__QUERY_B__');
    await sRelease(4);
    await sJobUntil(j => j.number === b.number && j.done >= 4, 'B: 4');
    await sWait(() => askLive && askLive.number === b.number && sAbsorbed.some(l => l.job === b.number), 'la página sigue B');
    await sRelease(4);
    await sWait(() => askLive.answers.length >= 8, 'B: 8');
    return {number: b.number, looks: sAbsorbed.filter(l => l.job === b.number)};
  });
  await sStep('switch', async () => {
    // B ends and C starts between two of this page's looks.
    const unhold = sHoldLooks();
    await sRelease(16);
    await sJobUntil(j => j.number === b.number && j.state !== 'running', 'B termina');
    await sRelease(-1);
    const c = await sAskElsewhere('__QUERY_C__');
    await sRelease(7);
    await sJobUntil(j => j.number === c.number && j.done >= 7, 'C: 7');
    const n0 = sLooks.length;
    // The idle watch is held: only the page's own switch (`switched` → `resume`) can follow C,
    // not the watch finding it within WATCH_MS (PR 16 re-review M3).
    const unwatch = sHoldWatchLooks();
    unhold();
    const followed = await sWait(() => askLive && askLive.number === c.number && askLive.answers.length >= 7, 'la página sigue C')
      .then(() => true, () => false);
    unwatch();
    if (!followed) return {b: b.number, c: c.number, followed: false};
    const t1 = performance.now();
    await sWait(() => performance.now() - t1 >= 1500, 'un segundo y medio');
    return {b: b.number, c: c.number, followed: true, looks: sLooks.length - n0, live: askLive.number,
      answers: askLive.answers.map(r => r.id), history: sQHistory(), status: sStatus()};
  });
  await sStep('c_end', async () => {
    const r0 = sRefreshed;
    await sRelease(17);
    await sWait(() => sRefreshed > r0, 'el final de C');
    return {status: sStatus(), history: sQHistory()};
  });
  await sStep('absorbed', async () => sAbsorbed);
  sDone();
})();
</script>"""
    )
    .replace("__QUERY__", LIVE_QUERY)
    .replace("__QUERY_B__", QUERY_B)
    .replace("__QUERY_C__", QUERY_C)
)


@pytest.fixture(scope="module")
def reasked(tmp_path_factory) -> dict[str, Any]:
    client = _Gated()

    class _Releasing(JevService):
        def probe_release(self, n: int) -> None:
            client.release(n)

    return _served_dump(
        tmp_path_factory.mktemp("ask-again"),
        _REASK_PROBE,
        make_client=lambda: client,
        repo=_reask_repo,
        base=_Releasing,
        patch=lambda mp: mp.setattr(service_module, "STREAM_PAGE", 3),
    )


def _reranked(ids: list[str], minimum: float = 0.0) -> list[str]:
    probs = {item.id: _reasked(n) for n, item in enumerate(_items())}
    return sorted(
        (post for post in ids if probs[post] >= minimum), key=lambda post: (-probs[post], post)
    )


@_requires_chrome
def test_asked_again_the_live_list_opens_at_the_querys_minimum_with_its_answers(reasked):
    """PR 16 tests I1: the answers the query had come first, ranked, at the stream's minimum
    (the query's last one, 0.5) from the first look — and none of them is «nuevo»."""
    first = _step(reasked, "first")
    cached = _ids(2)

    assert first["live_min"] == 0.5 and first["refine"]["min"] == "0.5"
    assert first["ids"] == _reranked(cached, 0.5) == cached
    assert first["nuevo"] == []
    assert first["reach"] == (
        "Mostrando 2 de 2 respondidos hasta ahora · 2 con relevancia ≥ 0,50, "
        "de mayor a menor probabilidad."
    )
    # The status line counts this job's answers only: the two it had are not «relevantes» of it.
    assert re.fullmatch(rf"Preguntando… 0 de {POSTS - 2} · {_USD} gastado", first["status"]["text"])


@_requires_chrome
def test_asked_again_cached_answers_are_never_nuevo_and_the_list_never_recuts_at_the_end(reasked):
    absorbed = [
        look
        for look in _step(reasked, "absorbed")
        if look["job"] == _step(reasked, "follower")["number"] - 1
    ]
    end = _step(reasked, "reask_end")
    so_far: list[str] = []

    for look in absorbed:
        so_far += look["got"]
        assert not set(look["cached"]) & set(look["nuevo"]), look
        assert look["ids"][:TOP] == _reranked(so_far, 0.5)[:TOP], look
        _nuevo_rule(look, _reranked(so_far, 0.5))
    # Each answer once — the ones still waiting when the job ended too: read to the end first.
    assert sorted(so_far) == sorted(item.id for item in _items()) and len(so_far) == len(
        set(so_far)
    )
    assert set(absorbed[0]["cached"]) == set(_ids(2))
    # The end takes the blob's row at the SAME minimum: the live list was already it.
    final = _reranked(so_far, 0.5)[:TOP]
    assert end["before_end"]["ids"] == final == end["ids"]
    assert "min=" not in end["hash"] and end["reach"].startswith(f"Mostrando {TOP} de ")


@_requires_chrome
def test_a_reply_saying_more_is_followed_by_a_look_at_once(reasked):
    """PR 16 tests M1: STREAM_PAGE is 3 here and twelve answers wait: each reply that says
    `more` is followed by the next look at once, never a second later — and nothing twice."""
    looks = _step(reasked, "more")["looks"]
    absorbed = [
        look
        for look in _step(reasked, "absorbed")
        if look["job"] == _step(reasked, "follower")["number"] - 1
    ]
    more = [look for look in absorbed if look["more"] and look["state"] == "running"]

    assert len(more) >= 3
    for look in more:
        after = [t for t in looks if t >= look["at"]]
        assert after and after[0] - look["at"] < 100, look
    got = [post for look in absorbed for post in look["got"]]
    assert len(got) == len(set(got))


@_requires_chrome
def test_a_page_that_meets_a_job_mid_way_marks_only_what_lands_above_its_cards(reasked):
    """PR 16 tests M2: the first look of a job another tab started fills the list, no «nuevo»;
    later answers landing above those cards say it."""
    follower = _step(reasked, "follower")
    looks = follower["looks"]

    assert looks[0]["nuevo"] == [] and len(looks[0]["got"]) >= 3
    so_far: list[str] = []
    for look in looks:
        so_far += look["got"]
        _nuevo_rule(look, _ranked(so_far))
    assert any(look["nuevo"] for look in looks[1:])


@_requires_chrome
def test_when_the_followed_job_changes_the_page_follows_the_new_one_without_a_hot_loop(reasked):
    """PR 16 arch I1: B ended and C started between two looks; the held look's reply is about
    C, with `more` (7 answers, 3 a reply). The page switches to C and reads its stream from 0 —
    a bounded number of looks, not a 0 ms loop on a reply it did not take."""
    switch = _step(reasked, "switch")

    # With the idle watch held, the page still follows C: its own switch resumed it.
    assert switch["followed"] is True, switch
    assert switch["c"] == switch["b"] + 1 and switch["live"] == switch["c"]
    assert sorted(switch["answers"]) == sorted(_ids(7))
    # The held look, three pages of C, and a look or two a second after.
    assert switch["looks"] <= 8, switch["looks"]
    # B's end was seen through the new data: its row is in the history, not «en curso».
    rows = {row["text"].split(" · ")[0]: row["text"] for row in switch["history"]}
    assert "en curso" not in rows[QUERY_B]
    assert rows[QUERY_C].startswith(QUERY_C + " · en curso · 7 de ")
    assert switch["status"]["text"].startswith(f"Preguntando… 7 de {POSTS}")
    c_end = _step(reasked, "c_end")
    assert c_end["status"]["text"].startswith(f"Preguntado: {POSTS} leídos")


# --------------------------------------------------------------------------- the page alone

#: Feeds the page's live results by hand (`askLiveStart` / `askAbsorb`, what `poll` calls),
#: with answers the test built — a static page, so nothing is served and nothing is asked.
_FEED_PROBE = (
    "<script>"
    + _SEEN
    + _ASK_JS
    + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  const FEED = __FEED__;
  const job = (answers, from, next) => ({number: 1, kind: 'ask', state: 'running',
    stream: {answers: answers, from: '1-' + from, next: '1-' + next, expected: FEED.expected, min: 0, more: false}});
  await step('order', async () => FEED.order.map(pairs => pairs.slice().sort(askOrder).map(r => r.id)));
  await step('fed', async () => {
    location.hash = '#ask';
    await sleep(50);
    askLiveStart({number: 1, kind: 'ask', query_sha: FEED.sha, total: FEED.expected, pick: {query: FEED.query}}, true);
    location.hash = askHref(FEED.sha);
    await sleep(50);
    // The first answer fills an empty list (nothing to be news against); the next two land
    // above it — the one with refine keys says «nuevo».
    const first = askAbsorb(job(FEED.answers.slice(0, 1), 0, 1));
    const firstMarks = document.querySelectorAll('#ask-list .nuevo').length;
    // A reply that does not continue the cursor is not taken.
    const stray = askAbsorb(job(FEED.answers.slice(1), 0, 2));
    const rest = askAbsorb(job(FEED.answers.slice(1), 1, FEED.answers.length));
    await sleep(10);
    const mark = document.querySelector('#ask-list .askr .nuevo');
    return {took: [first, stray, rest], first_marks: firstMarks,
      ids: [...document.querySelectorAll('#ask-list > .card')].map(c => c.dataset.id),
      keyless: txt(document.getElementById('ask-keyless')),
      mark: mark ? {text: txt(mark), animation: getComputedStyle(mark).animationName,
        color: getComputedStyle(mark).color} : null,
      reduced: matchMedia('(prefers-reduced-motion: reduce)').matches};
  });
  await step('later', async () => {
    await sleep(6200);
    return {marks: document.querySelectorAll('#ask-list .nuevo').length,
      ids: [...document.querySelectorAll('#ask-list > .card')].map(c => c.dataset.id)};
  });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
)


def _order_cases() -> list[list[dict[str, Any]]]:
    """Random answer sets with MANY ties and ids of every length (so string order differs from
    number order: "10" < "9"), plus letters and mixed case."""
    rng = random.Random(211)
    cases = []
    for _ in range(40):
        n = rng.randint(1, 60)
        ids = {
            rng.choice(["", "x", "X", "a"]) + str(rng.randint(0, 10 ** rng.randint(1, 6)))
            for _ in range(n)
        }
        cases.append(
            [{"id": i, "p": rng.choice([0.0, 0.1, 0.5, 0.85, 0.9, 1.0, 1 / 3])} for i in ids]
        )
    return cases


def _feed_page(root: Path, reduced: bool) -> tuple[dict[str, Any], list[list[str]]]:
    """The static ask page, fed three answers of a new query about posts 3, 4, 5 (5 without
    refine keys: a server bug the page must name), and the comparator's cases."""
    cases = _order_cases()
    feed = {
        "sha": "f" * 64,
        "query": "posts nuevos",
        "expected": 3,
        "order": [[{"id": c["id"], "p": c["p"]} for c in case] for case in cases],
        "answers": [
            {
                "id": "3",
                "p": 0.4,
                "model": "jev-1",
                "asked_at": "2026-09-27T10:00Z",
                "cached": False,
                "keys": {"d": "2026-09-22", "a": "alice", "t": [], "n": 10},
            },
            {
                "id": "4",
                "p": 0.8,
                "model": "jev-1",
                "asked_at": "2026-09-27T10:00Z",
                "cached": False,
                "keys": {"d": "2026-09-22", "a": "alice", "t": [], "n": 10},
            },
            {
                "id": "5",
                "p": 0.9,
                "model": "jev-1",
                "asked_at": "2026-09-27T10:00Z",
                "cached": False,
            },
        ],
    }
    probe = _FEED_PROBE.replace("__FEED__", json.dumps(feed))
    page, _ = _static_page(root, probe)
    flags = ("--force-prefers-reduced-motion",) if reduced else ()
    expected = [[c["id"] for c in sorted(case, key=lambda c: (-c["p"], c["id"]))] for case in cases]
    return _dump(page.as_uri(), budget_ms=20000, flags=flags), expected


@pytest.fixture(scope="module")
def fed(tmp_path_factory) -> dict[bool, tuple[dict[str, Any], list[list[str]]]]:
    _need_chrome()
    return {
        reduced: _feed_page(tmp_path_factory.mktemp(f"feed-{reduced}"), reduced)
        for reduced in (False, True)
    }


@_requires_chrome
def test_the_pages_order_is_pythons_rank_on_every_tie_and_id_length(fed):
    """PR 14 arch I4: ONE JS comparator (`askOrder`), brute-forced against `ask._rank`'s key."""
    seen, expected = fed[False]

    assert seen["order"] == expected
    assert any(len(case) > 30 for case in expected)


@_requires_chrome
def test_a_streamed_answer_without_refine_keys_is_named_not_dropped(fed):
    seen, _ = fed[False]
    got = seen["fed"]

    assert got["ids"] == ["4", "3"]
    assert got["took"] == [True, False, True]
    assert got["keyless"] == (
        "1 respuesta llegó sin sus claves de refinado y no se muestran (un fallo del servidor): 5."
    )


@_requires_chrome
@pytest.mark.parametrize("reduced", [False, True])
def test_a_new_answer_says_nuevo_then_the_mark_goes_and_with_reduced_motion_it_never_fades(
    fed, reduced
):
    seen, _ = fed[reduced]
    got, later = seen["fed"], seen["later"]

    assert got["reduced"] is reduced
    assert got["first_marks"] == 0
    assert got["mark"]["text"] == "nuevo"
    assert got["mark"]["animation"] == ("none" if reduced else "nuevo")
    # Verdigris: `--agree`, dark or light.
    assert got["mark"]["color"] in ("rgb(70, 184, 166)", "rgb(27, 116, 102)")
    assert later == {"marks": 0, "ids": ["4", "3"]}


@_requires_chrome
def test_a_running_ask_is_looked_at_at_once_then_about_every_second(live):
    """Live without hammering: the first look right after the confirm, then ~1 s apart (at
    once only when a reply says more answers wait — never here: fewer than a page)."""
    looks = _step(live, "looks")
    times = looks["looks"]
    gaps = sorted(b - a for a, b in zip(times, times[1:], strict=False))

    assert looks["confirmed"] is not None and len(times) >= 5
    assert times[0] - looks["confirmed"] < 100
    # A retry after a failed look would be 500 ms; a look is never hurried or lagging.
    assert 950 <= gaps[len(gaps) // 2] <= 1100
    assert gaps[0] >= 900

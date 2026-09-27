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
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.jev_ask_fixtures import _START, VOCAB, _Planned, victor_item
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
        with self.moved:
            self.allowed += n
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
  sAbsorb0(job);
  if (job && job.stream && job.stream.answers.length) {
    sAbsorbed.push(Object.assign(sLiveView(), {got: job.stream.answers.map(a => a.id)}));
  }
};
let sBeforeEnd = null;
const sFinished0 = finished;
finished = async function (job) { sBeforeEnd = sLiveView(); return sFinished0(job); };
// When (page clock) the ask was confirmed and each look at its stream went out.
const sLooks = [];
let sConfirmed = null;
const sFetchLooks = window.fetch;
window.fetch = function (u, init) {
  const p = sFetchLooks.call(window, u, init);
  if (String(u).includes('/api/job?since=')) sLooks.push(performance.now());
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
    stop: txt(sId('ask-stop'))}; };
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
      list_top: sId('ask-list').getBoundingClientRect().top, scroll: Math.round(scrollY)};
    await sRelease(4);
    await sAnswered(8);
    const again = sId('ask-post-' + before.id);
    return Object.assign(sLiveView(), {before, after: again ? again.getBoundingClientRect().top : null,
      same_node: again === anchor, tags: sTags()});
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
    await sRelease(6);
    await sAnswered(16);
    await sWait(() => /^Preguntando… 16 de /.test(sStatus().text || ''), 'la línea de estado');
    return {set, after: Object.assign(sLiveView(), {refine: sQRefine(), status: sStatus()})};
  });
  await sStep('end', async () => {
    const r0 = sRefreshed;
    await sRelease(8);
    await sWait(() => sRefreshed > r0, 'el final del trabajo');
    return Object.assign(sLiveView(), {before_end: sBeforeEnd, head: txt(sId('ask-head')),
      history: sQHistory(), panel: sPanel()});
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
        patch=lambda mp: mp.setattr(service_module, "finish_ask", _spy),
    )
    seen["finished"] = [
        [(item.id, record.probability) for item, record in found.ranked] for found in finished
    ]
    return seen


def _ids(n: int) -> list[str]:
    """The ids of the first `n` posts the pool asks (selection order = store order): after
    `n` released, exactly these have answered."""
    return [item.id for item in _items()[:n]]


@_requires_chrome
def test_the_live_list_opens_empty_under_the_asked_query_and_the_rail_says_it_runs(live):
    start = _step(live, "start")

    assert start["hash"].startswith("#ask?q=") and start["live"]
    assert start["ids"] == [] and start["more"] is None
    assert start["empty"].startswith("Aún no ha llegado ninguna respuesta")
    assert start["reach"] == (
        f"Mostrando 0 de 0 respondidos hasta ahora (de {POSTS}), de mayor a menor probabilidad."
    )
    # No bar-only phase: the job block is already one status line over the (empty) gallery.
    status = start["status"]
    assert status["slim"] and status["shown"] and not status["bar"] and not status["title"]
    assert (status["prev"], status["next"]) == ("ask-refine", "ask-list")
    assert status["stop"] == "Parar (se guarda lo ya pagado)"
    assert "Preguntando ahora" in start["head"]
    assert start["history"][0]["text"] == LIVE_QUERY + " · en curso"
    assert start["history"][0]["current"] is True


@_requires_chrome
def test_every_answer_arrives_once_and_each_look_shows_the_ranked_top(live):
    """After EVERY look that brought answers: the cards on screen are the top of the answers so
    far, ranked by `(-p, id)` (and kept by the refine in effect), and every new one says
    «nuevo»."""
    absorbed = _step(live, "absorbed")
    so_far: list[str] = []

    assert len(absorbed) >= len(WAVES)
    for look in absorbed:
        so_far += look["got"]
        expected = _ranked(so_far, look["min"] or 0.0)
        assert look["ids"] == expected[:TOP], look
        shown = len(look["ids"])
        assert look["reach"].startswith(
            f"Mostrando {shown} de {len(so_far)} respondidos hasta ahora (de {POSTS})"
        )
        assert set(look["got"]) & set(look["ids"]) <= set(look["nuevo"])
    assert sorted(so_far) == sorted(PLANNED) and len(so_far) == len(set(so_far))


@_requires_chrome
def test_a_new_answer_lands_in_its_ranked_place_and_the_cards_there_stay(live):
    wave1, anchored = _step(live, "wave1"), _step(live, "anchored")

    assert wave1["ids"] == _ranked(_ids(4))
    strong = sum(PLANNED[post] >= 0.7 for post in _ids(4))
    assert wave1["status"]["text"].startswith(
        f"Preguntando… 4 de {POSTS} · {strong} muy relevantes (≥ 0,70) · "
    )
    assert wave1["status"]["slim"] and wave1["status"]["height"] < 90
    assert anchored["ids"] == _ranked(_ids(8))[:TOP]
    # n=4..7 outrank n=0..3: four new cards above. The two old ones still shown are the SAME
    # nodes (never drawn again); the two pushed past the top are gone.
    assert anchored["tags"] == [None, None, None, None, "w1", "w1"]
    # Each new card says «nuevo» (the older ones may still, for the rest of their NUEVO_MS).
    assert set(_ids(8)[4:]) <= set(anchored["nuevo"])
    assert anchored["more"] == "Ver 2 más (quedan 2)"


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
        f"Mostrando {over} de 16 respondidos hasta ahora (de {POSTS}) · "
        f"{over} con relevancia ≥ 0,60, de mayor a menor probabilidad."
    )
    # The status line counts at the refine's minimum once one is set.
    assert after["status"]["text"].startswith(
        f"Preguntando… 16 de {POSTS} · {over} con relevancia ≥ 0,60 · "
    )
    assert after["refine"]["min"] == "0.6"


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
    assert f" de {POSTS} respondidos hasta ahora (de {POSTS})" in end["before_end"]["reach"]
    assert live_order == expected == end["ids"]
    # Its refine stayed; the head is the history's now.
    assert "min=0.6" in end["hash"] and end["live"] is None
    assert end["reach"].startswith(f"Mostrando {TOP} de ")
    assert end["history"][0]["text"].startswith(LIVE_QUERY + " · ")
    assert "en curso" not in end["history"][0]["text"]
    assert end["panel"]["progress"].startswith(f"{POSTS} respuestas guardadas · {POSTS} resultados")


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
  const job = (answers, next) => ({number: 1, kind: 'ask', state: 'running',
    stream: {answers: answers, next: '1-' + next, expected: FEED.expected, min: 0, more: false}});
  await step('order', async () => FEED.order.map(pairs => pairs.slice().sort(askOrder).map(r => r.id)));
  await step('fed', async () => {
    location.hash = '#ask';
    await sleep(50);
    askLiveStart({number: 1, kind: 'ask', query_sha: FEED.sha, total: FEED.expected, pick: {query: FEED.query}}, true);
    location.hash = askHref(FEED.sha);
    await sleep(50);
    askAbsorb(job(FEED.answers, FEED.answers.length));
    await sleep(10);
    const mark = document.querySelector('#ask-list .askr .nuevo');
    return {ids: [...document.querySelectorAll('#ask-list > .card')].map(c => c.dataset.id),
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

# tests/test_jev_page_ask_browser.py
"""The «Preguntar» tab in a real browser: the static page (history and saved results, no way
to launch) and the served page (query → estimate → confirm → progress → results; every way
an ask job can end), with the fake client — never TypeSafe.

Probes follow `test_jev_page_browser.py`'s rules: they read what a reader SEES (`_SEEN`), go
through the page's own controls, and are fail-closed (a step that throws is an `ERROR` value
the assertions then reject).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.test_jev_page_browser import (
    _SEEN,
    _SERVE_JS,
    _dump,
    _ended,
    _need_chrome,
    _page,
    _page_saw,
    _requires_chrome,
    _served_dump,
    _start_job,
    _step,
)
from tests.test_jev_serve_ask import PROBS, QUERY, _Asker, _PerChar, ranked
from xbrain.config import Config
from xbrain.jev.ask import AskFilters, AskQuery, finish_ask, plan_ask
from xbrain.jev.lock import pass_lock
from xbrain.jev.report import ask_cost_by_query
from xbrain.jev.run import run_ask
from xbrain.jev.store import load_runs

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
#: A query asked only over «startups»: its posts (2, 4) answer 0.1 and 0.2 — both results,
#: ranked (nothing is cut at `[jev].threshold` any more).
SEED_QUERY = "¿Quién cuenta cómo levantó una ronda seed?"
#: A query whose file is later broken: its row says so, and costs nothing else.
BROKEN_QUERY = "posts con datos de estudios sobre creatina"


def _asked(cfg: Config, text: str, *, when: datetime, **filters: Any) -> AskQuery:
    """`xbrain jev ask` in a terminal, with the fake: plan → run → finish, under the lock."""
    query = AskQuery.of(text)
    with pass_lock(cfg.jev_lock_path, "xbrain jev ask") as lock:
        plan = plan_ask(cfg, query, AskFilters(**filters), None)
        # Enough tokens that a query's cost shows at four decimals.
        outcome = run_ask(cfg, plan, lambda: _Asker(input_tokens=200_000), lock=lock)
        finish_ask(cfg, plan, outcome, now=when)
    return query


def _asked_repo(cfg: Config) -> dict[str, str]:
    """Three queries: the hooks one (results 1 and 3), a later «startups» one with no result,
    and an earlier one whose file is then broken."""
    hooks = _asked(cfg, QUERY, when=DT + timedelta(hours=1))
    seed = _asked(cfg, SEED_QUERY, when=DT + timedelta(hours=2), topics=("startups",))
    broken = _asked(cfg, BROKEN_QUERY, when=DT + timedelta(minutes=30), author="bob")
    (cfg.jev_asks_dir / f"{broken.sha}.json").write_text("{", encoding="utf-8")
    return {"hooks": hooks.sha, "seed": seed.sha, "broken": broken.sha}


#: Reads the tab as a reader sees it. Shared by the static and the served probes.
_ASK_JS = r"""
const sQHistory = () => [...document.querySelectorAll('#ask-history li')].filter(seen).map(li => ({
  text: [...li.querySelectorAll('b, small')].map(e => e.textContent).join(' · '),
  current: li.querySelector('a').getAttribute('aria-current') === 'true'}));
const sQResults = () => [...document.querySelectorAll('#ask-results .card')].filter(seen).map(c => {
  const bar = c.querySelector('.askr .apbar');
  const fill = bar && bar.firstElementChild;
  const slot = c.querySelector('.pv .xembed');
  return {id: c.dataset.id, x: slot ? slot.dataset.id : null, local: txt(c.querySelector('.pv .twt')),
    order: [...c.children].map(k => k.className), p: txt(c.querySelector('.askr .ap')), saw: txt(c.querySelector('.askr details.saw > summary')),
    jev: txt(c.querySelector('.jev .badge')), meta: txt(c.querySelector('.askr .jh .meta')),
    bar: seen(bar) ? {width: fill.style.width, fill: fill.getBoundingClientRect().width,
      track: bar.getBoundingClientRect().width, label: bar.getAttribute('aria-label')} : null};
});
const sQView = () => ({
  hash: location.hash,
  tab: txt(document.querySelector('.tabs a[data-tab="ask"]')),
  head: txt(document.getElementById('ask-head')),
  empty: txt(document.getElementById('ask-empty')),
  error: txt(document.getElementById('ask-row-error')),
  banner: txt(document.getElementById('ask-history-error')),
  form: seen(document.getElementById('ask-form')),
  launch: txt(document.getElementById('ask-launch')),
  history: sQHistory(),
  results: sQResults(),
  more: txt(document.getElementById('ask-more')),
  keyless: txt(document.getElementById('ask-keyless')),
  absent: txt(document.getElementById('refine-absent')),
});
const sQOpen = (words) => [...document.querySelectorAll('#ask-history li a')].find(a => seen(a) && a.textContent.includes(words)).click();
"""

_ASK_STATIC_PROBE = (
    "<script>"
    + _SEEN
    + _ASK_JS
    + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  await step('default', async () => { location.hash = '#ask'; await sleep(80); return sQView(); });
  await step('hooks', async () => { sQOpen('hooks'); await sleep(80); return sQView(); });
  await step('saw3', async () => {
    const d = document.querySelector('#ask-results .card[data-id="3"] .askr details.saw');
    d.open = true; await sleep(10);
    return txt(d);
  });
  await step('broken', async () => { sQOpen('creatina'); await sleep(80); return sQView(); });
  await step('elsewhere', async () => {
    location.hash = '#posts?f=all'; await sleep(80);
    return {cards: [...document.querySelectorAll('#cards .card')].filter(seen).length,
      ask_cards: [...document.querySelectorAll('#ask-results .card')].filter(seen).length};
  });
  await step('back', async () => { location.hash = document.querySelector('.tabs a[data-tab="ask"]').getAttribute('href'); await sleep(80); return sQView(); });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
)

_ASK_READ = (
    "<script>"
    + _SEEN
    + _ASK_JS
    + r"""
setTimeout(() => {
  const pre = document.createElement('pre'); pre.id = 'probe';
  pre.textContent = JSON.stringify(sQView());
  document.body.appendChild(pre);
}, 200);
</script>"""
)


def _static_page(root: Path, probe: str) -> tuple[Path, dict[str, str]]:
    """The file `jev dashboard` writes, over `_repo` with `_asked_repo`'s three queries."""
    from tests.test_jev_serve import _repo
    from xbrain.jev.dashboard import build_page_data

    _need_chrome()
    monkeypatch = pytest.MonkeyPatch()
    try:
        cfg = _repo(root, monkeypatch)
        shas = _asked_repo(cfg)
        data = build_page_data(cfg, now=DT)
        costs = ask_cost_by_query(load_runs(cfg.jev_runs_path))
        shas.update(
            {
                f"cost:{name}": _usd4(costs[sha]["cost_usd"])
                for name, sha in list(shas.items())
                if sha in costs
            }
        )
    finally:
        monkeypatch.undo()
    assert data["serve"] is None
    return _page(root, data, probe), shas


@pytest.fixture(scope="module")
def ask_static(tmp_path_factory) -> tuple[dict[str, Any], dict[str, str]]:
    page, shas = _static_page(tmp_path_factory.mktemp("ask-static"), _ASK_STATIC_PROBE)
    return _dump(page.as_uri()), shas


def _usd4(x: float) -> str:
    return f"~{x:.4f}".replace(".", ",") + " $"


def _shown(expected: list[tuple[str, float]]) -> list[tuple[str, str]]:
    """`ranked()` as the page prints it: two decimals, a comma."""
    return [(post, f"{p:.2f}".replace(".", ",")) for post, p in expected]


def _probs(results: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(r["id"], r["p"]) for r in results]


@_requires_chrome
def test_static_the_tab_lists_every_query_last_asked_first(ask_static):
    seen, shas = ask_static
    view = seen["default"]

    assert view["tab"] == "Preguntar"
    assert [h["text"].split(" · ")[0] for h in view["history"]] == [
        SEED_QUERY,
        QUERY,
        BROKEN_QUERY,
    ]
    seed_cost, hooks_cost = shas["cost:seed"], shas["cost:hooks"]
    assert seed_cost != "~0,0000 $" and hooks_cost != seed_cost
    assert f" · 2 de 2 leídos por Jev · {seed_cost}" in view["history"][0]["text"]
    assert f" · 5 de 5 leídos por Jev · {hooks_cost}" in view["history"][1]["text"]
    assert view["history"][0]["current"] is True
    # The broken file stopped the history's rebuild: said above the list, its row kept.
    assert view["banner"].startswith("No se pudo leer el historial: ")
    assert f"{shas['broken']}.json" in view["banner"]


@_requires_chrome
def test_static_the_last_query_opens_by_default_and_says_what_it_found(ask_static):
    seen, shas = ask_static
    view = seen["default"]

    assert view["hash"] == "#ask"
    assert view["head"].startswith(f"«{SEED_QUERY}»")
    assert "Filtros: topic Startups" in view["head"]
    # Both answers are results, ranked — the old cut at 0,85 showed none.
    assert _probs(view["results"]) == [("4", "0,20"), ("2", "0,10")]
    assert view["empty"] is None and view["more"] is None
    assert "Mostrando 2 de 2 leídos por Jev, de mayor a menor probabilidad." in view["head"]


@_requires_chrome
def test_static_results_are_the_post_cards_ranked_by_probability(ask_static):
    seen, shas = ask_static
    view = seen["hooks"]

    assert view["hash"] == f"#ask?q={shas['hooks']}"
    assert _probs(view["results"]) == _shown(ranked())
    assert [h["current"] for h in view["history"]] == [False, True, False]
    assert "Mostrando 5 de 5 leídos por Jev, de mayor a menor probabilidad." in view["head"]
    assert "0,85" not in view["head"]
    for result in view["results"]:
        assert re.fullmatch(r"Lo que vio Jev · 2 fuentes, \d+ caracteres", result["saw"])
    # Each result is X's own view of the post (its frame is made near the viewport), with the
    # answer strip between the card's head and it, and the Jev block under it.
    for result in view["results"]:
        assert result["x"] == result["id"]
        assert result["local"] is None
        assert result["order"] == ["twh", "askr", "pv", "jev"]
    # Each result's probability is drawn: the bar's fill is the number, over its track.
    for result, (_, p) in zip(view["results"], ranked(), strict=True):
        bar = result["bar"]
        assert bar["width"] == f"{round(p * 100)}%"
        assert bar["fill"] == pytest.approx(bar["track"] * p, abs=1.5)
        assert bar["label"] == "probabilidad " + f"{p:.2f}".replace(".", ",")


@_requires_chrome
def test_static_what_jev_saw_is_there_for_a_post_topics_never_asked(ask_static):
    seen, _ = ask_static

    # Post 3 has no topics answer (its card says so), and the ask still shows what Jev read.
    assert (
        next(r for r in seen["hooks"]["results"] if r["id"] == "3")["jev"] == "sin evaluar por Jev"
    )
    assert "= el texto del post, arriba" in seen["saw3"]


@_requires_chrome
def test_static_a_query_whose_answers_cannot_be_read_says_so(ask_static):
    view = ask_static[0]["broken"]

    assert view["results"] == []
    assert view["error"].startswith("No se pudieron leer las respuestas de esta consulta: ")
    assert "ilegible" in view["error"]


@_requires_chrome
def test_static_launching_needs_the_server_and_says_how(ask_static):
    view = ask_static[0]["default"]

    assert view["form"] is False
    assert view["launch"].startswith(
        "Para preguntar desde esta página, ábrela con xbrain jev serve. Desde la terminal: "
    )
    assert 'xbrain jev ask "' in view["launch"]


@_requires_chrome
def test_static_the_tab_keeps_its_query_when_the_reader_comes_back(ask_static):
    seen, shas = ask_static

    assert seen["elsewhere"]["ask_cards"] == 0 and seen["elsewhere"]["cards"] > 0
    assert seen["back"]["hash"] == f"#ask?q={shas['broken']}"


@_requires_chrome
def test_static_a_query_url_opened_fresh_shows_that_query(tmp_path):
    page, shas = _static_page(tmp_path, _ASK_READ)

    hooks = _dump(page.as_uri() + f"#ask?q={shas['hooks']}")
    unknown = _dump(page.as_uri() + "#ask?q=" + "0" * 64)

    assert _probs(hooks["results"]) == _shown(ranked())
    assert hooks["history"][1]["current"] is True
    # A sha the history does not have is said so: never another query's results.
    assert unknown["head"] is None and unknown["results"] == []
    assert unknown["empty"] == "Esta consulta no está en el historial."
    assert not any(h["current"] for h in unknown["history"])


# --------------------------------------------------------------------------- the served tab

_ASK_SERVE_PROBE = (
    "<script>"
    + _SERVE_JS
    + _ASK_JS
    + r"""
const sQForm = (fields) => {
  for (const [id, value] of Object.entries(fields)) {
    const box = sId(id);
    if (box.type === 'checkbox') box.checked = value; else box.value = value;
    box.dispatchEvent(new Event('input', {bubbles: true}));
    box.dispatchEvent(new Event('change', {bubbles: true}));
  }
  sPress(sId('ask-form'), 'Estimar lo que cuesta');
};
(async () => {
  await sStep('empty', async () => {
    location.hash = '#ask';
    await sWait(() => seen(sId('ask-form')), 'el formulario');
    return sQView();
  });
  await sStep('estimate', async () => {
    sQForm({'ask-q': '  ¿Cómo configuro   hooks en Claude Code? ', 'ask-evaluated': true});
    await sWait(() => sPanel().go, 'la estimación');
    const box = sId('jobp'), form = sId('ask-form').getBoundingClientRect();
    const placed = {position: getComputedStyle(box).position, top: sPanelTop(),
      after: box.previousElementSibling && box.previousElementSibling.id,
      below_form: box.getBoundingClientRect().top >= form.bottom - 1};
    return Object.assign(sPanel(), {estimate: sLastEstimate, placed});
  });
  await sStep('done', async () => {
    sId('ask-go').click();
    await sWait(() => sRefreshed > 0 && location.hash.startsWith('#ask?q='), 'los resultados');
    return Object.assign(sQView(), {panel: sPanel()});
  });
  await sStep('again', async () => {
    const r0 = sRefreshed;
    sQForm({'ask-q': '¿Cómo configuro hooks en Claude Code?', 'ask-evaluated': true});
    await sWait(() => sPanel().go, 'la estimación gratis');
    const panel = sPanel();
    sId('ask-go').click();
    await sWait(() => ['Ocultar', 'Cerrar'].includes(sPanel().close), 'el trabajo gratis');
    sId('ask-cancel').click();  // «Ocultar»: a clean end leaves it hidden
    const hidden = !sPanel().shown;
    await sWait(() => sRefreshed > r0, 'la consulta gratis');
    for (let i = 0; i < 3; i++) await sFetch0.call(window, '/probe-wait');
    // The end text the panel would show, from the page's own function over the job it saw.
    const endText = outcomeText(await (await sFetch0('/api/job')).json());
    return Object.assign(sQView(), {panel, hidden, after: sPanel(), end_text: endText});
  });
  await sStep('reopen', async () => {
    const posts0 = sPosts, r0 = sRefreshed;
    location.hash = '#posts?f=all';
    await sWait(() => seen(sCard('1')), 'Posts');
    sId('ask-cancel').click();
    document.querySelector('.tabs a[data-tab="ask"]').click();
    await sWait(() => sQHistory().length === 1, 'el historial');
    sQOpen('hooks');
    await sWait(() => sQResults().length > 0, 'los resultados reabiertos');
    return Object.assign(sQView(), {posts: sPosts - posts0, reloads: sRefreshed - r0});
  });
  await sStep('similar', async () => {
    sQForm({'ask-q': '¿cómo configuro hooks en claude code', 'ask-evaluated': true});
    await sWait(() => (sPanel().est || '').includes('casi igual'), 'la consulta parecida');
    const panel = Object.assign(sPanel(), {estimate: sLastEstimate});
    sId('ask-cancel').click();
    return panel;
  });
  await sStep('refused', async () => {
    sQForm({'ask-q': 'otra', 'ask-evaluated': false, 'ask-author': 'nadie'});
    await sWait(() => (sPanel().error || '').includes('ningún post'), 'la negativa');
    const panel = sPanel();
    sId('ask-cancel').click();
    return panel;
  });
  await sStep('blank', async () => {
    sQForm({'ask-q': '   ', 'ask-author': ''});
    await sWait(() => (sPanel().error || '').includes('vacía'), 'la consulta vacía');
    const panel = sPanel();
    sId('ask-cancel').click();
    return panel;
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def ask_served(tmp_path_factory) -> dict[str, Any]:
    # Billed per character, so after the first use the cost model is a measured fit.
    client = _PerChar()
    seen = _served_dump(tmp_path_factory.mktemp("ask-served"), _ASK_SERVE_PROBE, client=client)
    seen["asked"] = client.asked
    return seen


@_requires_chrome
def test_served_the_tab_offers_the_query_box_and_its_filters(ask_served):
    empty = _step(ask_served, "empty")

    assert empty["form"] is True and empty["launch"] is None
    assert empty["history"] == []
    assert empty["head"] is None
    assert empty["empty"] == "Aún no se ha preguntado nada."


@_requires_chrome
def test_served_an_ask_estimate_is_the_servers_and_says_what_it_will_pay(ask_served):
    estimate = _step(ask_served, "estimate")
    e = estimate["estimate"]
    money = f"~{e['usd']:.5f}".replace(".", ",") + " $"

    assert estimate["title"] == "Preguntar: «¿Cómo configuro hooks en Claude Code?»"
    # In Preguntar, right under the form that asked for it — never the floating corner panel.
    assert estimate["placed"] == {
        "position": "static",
        "top": "top-ask",
        "after": "ask-form",
        "below_form": True,
    }
    assert e["kind"] == "ask" and e["ids"] == ["1", "2"]
    assert e["pick"] == {"query": AskQuery.of(QUERY).text, "only_evaluated": True}
    assert estimate["force"] is None
    assert estimate["go_text"] == "Preguntar y pagar " + money
    assert e["cost_model"] == {
        "per_call": 1000.0,
        "chars_per_token": 4.0,
        "measured": False,
        "answers": 0,
    }
    assert estimate["est"] == (
        f"2 posts por preguntar · 4 descartados por los filtros. {_cost_line(e)} Tope: 1,00 $. "
        "Al llegar se para; lo que ya esté en vuelo termina y puede pasarlo por poco (como "
        "mucho 1 post)."
    )
    assert "Aún sin preguntas pagadas: cifras de partida." in estimate["est"]


@_requires_chrome
def test_served_an_ask_estimate_after_paid_answers_says_the_fitted_figures(ask_served):
    similar = _step(ask_served, "similar")
    e = similar["estimate"]

    assert e["cost_model"]["measured"] is True and e["cost_model"]["answers"] == 2
    assert _cost_line(e) in similar["est"]
    assert "Cifras ajustadas con 2 respuestas ya pagadas." in similar["est"]


def _cost_line(e: dict[str, Any]) -> str:
    """The panel's cost sentence for the server's estimate `e`, in the page's own words."""
    m = e["cost_model"]

    def grouped(n: float) -> str:
        # `Math.round`: halves go up (Python's `round` sends them to the even number).
        return f"{math.floor(n + 0.5):,}".replace(",", ".")

    per_token = f"{m['chars_per_token']:.2f}".rstrip("0").rstrip(".").replace(".", ",")
    money = f"~{e['usd']:.5f}".replace(".", ",") + " $"
    posts = f"{e['posts']} post" + ("" if e["posts"] == 1 else "s")
    fitted = (
        f"Cifras ajustadas con {m['answers']} respuestas ya pagadas."
        if m["measured"]
        else "Aún sin preguntas pagadas: cifras de partida."
    )
    return (
        f"Coste estimado: {money} — {posts} × ~{grouped(m['per_call'])} tokens por llamada, más "
        f"el texto de los posts ({grouped(e['chars'])} caracteres ≈ "
        f"{grouped(e['chars'] / m['chars_per_token'])} tokens, a {per_token} caracteres por "
        f"token). {fitted}"
    )


@_requires_chrome
def test_served_an_ask_job_ends_on_its_results_and_its_history(ask_served):
    done = _step(ask_served, "done")
    sha = AskQuery.of(QUERY).sha

    assert sorted(ask_served["asked"]) == ["1", "2"]
    assert done["hash"] == f"#ask?q={sha}"
    assert _probs(done["results"]) == _shown(ranked(posts="12"))
    assert done["panel"]["progress"].startswith("2 respuestas guardadas · 2 resultados · ")
    assert [h["text"].split(" · ")[0] for h in done["history"]] == [QUERY]
    assert done["history"][0]["current"] is True
    assert ask_served["job"]["query_sha"] == sha


@_requires_chrome
def test_served_asking_again_what_is_answered_is_free_and_counted(ask_served):
    again = _step(ask_served, "again")

    assert again["panel"]["est"].startswith(
        "0 posts por preguntar · 2 ya respondidos (gratis) · 4 descartados por los filtros. "
    )
    assert again["panel"]["go_text"] == "Ver resultados (gratis)"
    assert sorted(ask_served["asked"]) == ["1", "2"]
    assert "2 veces" in again["history"][0]["text"]
    assert _probs(again["results"]) == _shown(ranked(posts="12"))
    assert again["panel"]["est"].endswith(
        "No se paga nada; se anota como una consulta más en el historial. Tope: 1,00 $. Al "
        "llegar se para; lo que ya esté en vuelo termina y puede pasarlo por poco (como mucho "
        "1 post)."
    )
    assert again["end_text"].startswith("0 respuestas guardadas · 2 resultados · ")
    assert "runs.jsonl" not in again["end_text"]
    assert "no quedó en el historial" not in again["end_text"]
    # Nothing was sent, so nothing to log: a clean end — no alarm, and the panel stays hidden.
    assert again["hidden"] is True and again["after"]["shown"] is False
    # The probe's last job is this free one.
    job = ask_served["job"]
    assert (job["state"], job["outcome"]["sent"], job["outcome"]["logged"]) == ("done", 0, False)


@_requires_chrome
def test_served_reopening_a_query_from_the_history_asks_the_server_nothing(ask_served):
    reopen = _step(ask_served, "reopen")

    assert (reopen["posts"], reopen["reloads"]) == (0, 0)
    assert reopen["hash"] == f"#ask?q={AskQuery.of(QUERY).sha}"
    assert _probs(reopen["results"]) == _shown(ranked(posts="12"))


@_requires_chrome
def test_served_a_query_asked_before_in_other_words_is_named(ask_served):
    similar = _step(ask_served, "similar")
    assert isinstance(similar, dict), similar
    est = similar["est"]

    assert (
        "Ya preguntaste algo casi igual (cambia solo en mayúsculas, puntuación o espacios, y se "
        f"paga aparte): «{AskQuery.of(QUERY).text}»." in est
    )


@_requires_chrome
def test_served_an_ask_with_nothing_to_ask_is_refused_before_confirm(ask_served):
    refused = _step(ask_served, "refused")

    assert refused["go"] is False
    assert refused["error"].startswith("No se puede preguntar: ningún post que preguntar")


@_requires_chrome
def test_served_a_blank_query_is_refused_by_the_server(ask_served):
    blank = _step(ask_served, "blank")

    assert blank["go"] is False
    assert blank["error"] == "No se puede preguntar: la consulta está vacía"


# --------------------------------------------------------------------------- how an ask can end


@pytest.fixture(scope="module")
def ask_ended(tmp_path_factory):
    cache: dict[str, dict[str, Any]] = {}

    def get(scenario: str) -> dict[str, Any]:
        if scenario not in cache:
            cache[scenario] = _ended(tmp_path_factory.mktemp("ask-" + scenario), scenario, "ask")
        return cache[scenario]

    return get


@_requires_chrome
@pytest.mark.parametrize("scenario", ["tope", "error", "failures", "not-logged", "lost", "history"])
def test_served_an_ask_that_did_not_end_cleanly_is_shown_even_if_hidden(ask_ended, scenario):
    seen = ask_ended(scenario)

    assert _step(seen, "estimate")["title"].startswith("Preguntar: «"), seen
    assert _step(seen, "run")["hidden"] is True
    assert _step(seen, "end")["shown"] is True and _step(seen, "end")["close"] == "Cerrar"
    assert seen["job"]["kind"] == "ask"


@_requires_chrome
def test_served_an_ask_cut_by_the_cap_says_so_and_keeps_what_it_paid(ask_ended):
    seen = ask_ended("tope")

    assert (seen["job"]["state"], seen["job"]["reason"]) == ("interrupted", "tope")
    assert _step(seen, "end")["progress"].startswith(
        "Interrumpido (se alcanzó el tope por trabajo): 1 respuesta guardada · 1 resultado · "
    )


@_requires_chrome
def test_served_an_ask_whose_history_could_not_be_written_keeps_its_paid_answers_in_view(
    ask_ended,
):
    seen = ask_ended("history")
    job = seen["job"]

    assert job["state"] == "done" and job["outcome"]["recorded"] is False
    assert _step(seen, "end")["progress"].startswith("2 respuestas guardadas · ")
    assert (
        f"2 respuestas pagadas y guardadas en {job['outcome']['file']}; el historial no se pudo "
        "escribir: [Errno 28] No space left on device"
    ) in _step(seen, "end")["progress"]


@_requires_chrome
def test_served_an_ask_that_failed_says_why(ask_ended):
    seen = ask_ended("error")

    assert seen["job"]["state"] == "error"
    assert _step(seen, "end")["error"] == "El trabajo falló: TYPESAFE_API_KEY no encontrada"


@_requires_chrome
def test_served_an_asks_failures_unsaved_and_unpriced_answers_are_named(ask_ended):
    end = ask_ended("failures")["end"]

    assert end["progress"].startswith("4 respuestas guardadas · 4 resultados · ")
    assert "1 respuesta cobrada a la media estimada" in end["progress"]
    assert "sin tarifa: fake" in end["progress"]
    assert "1 fallida: 4 (" in end["progress"] and "respuesta ilegible" in end["progress"]
    assert "2 respuestas pagadas sin guardar" in end["progress"]


@_requires_chrome
def test_served_an_ask_that_did_not_reach_the_run_log_says_so(ask_ended):
    end = ask_ended("not-logged")["end"]

    assert end["progress"].startswith("1 respuesta guardada · 1 resultado")
    assert "la pasada no quedó en runs.jsonl" in end["progress"]


@_requires_chrome
def test_served_the_page_can_stop_an_ask_and_what_was_paid_is_kept(ask_ended):
    seen = ask_ended("stop")

    assert (seen["job"]["state"], seen["job"]["reason"]) == ("interrupted", "cancelado")
    kept = seen["job"]["outcome"]["ok"]
    assert 1 <= kept < 3
    assert _step(seen, "end")["progress"].startswith(
        f"Interrumpido (lo paraste desde la página): {kept} respuesta"
    )


@_requires_chrome
def test_served_an_ask_survives_a_moment_without_the_server(ask_ended):
    end = ask_ended("flaky")["end"]

    assert end["error"] is None and end["reload"] is None
    assert end["progress"].startswith("1 respuesta guardada · ")


@_requires_chrome
def test_served_an_ask_that_lost_the_server_says_so(ask_ended):
    end = ask_ended("lost")["end"]

    assert end["error"].startswith("Se perdió el contacto con el servidor")
    assert end["reload"] == "Recargar la página"


_ASK_RESUME_PROBE = (
    "<script>"
    + _SERVE_JS
    + _ASK_JS
    + r"""
(async () => {
  await sStep('resumed', async () => {
    // Opened on Revisar: the running ask is announced on Preguntar's tab, and its block is there.
    await sWait(() => sBadges().includes('ask:en curso'), 'el aviso en la pestaña Preguntar');
    const badges = sBadges(), top = sPanelTop();
    location.hash = '#ask';
    await sWait(() => sPanel().shown && /de 5 posts/.test(sPanel().progress || ''), 'el panel de la pregunta en curso');
    return Object.assign(sPanel(), {badges, top});
  });
  await sStep('end', async () => {
    await sWait(() => sRefreshed > 0, 'la recarga');
    location.hash = '#ask';
    await sWait(() => sQHistory().length === 1, 'el historial');
    return Object.assign(sQView(), {panel: sPanel()});
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def ask_resumed(tmp_path_factory) -> dict[str, Any]:
    watched, saw = _page_saw()

    class _SlowFirst(_Asker):
        def ask(self, state, questions):
            if not self.asked:
                saw.wait(60)
            return super().ask(state, questions)

    return _served_dump(
        tmp_path_factory.mktemp("ask-resume"),
        _ASK_RESUME_PROBE,
        client=_SlowFirst(),
        base=watched,
        before_dump=lambda service, port: _start_job(service, port, {"query": QUERY}, kind="ask"),
    )


@_requires_chrome
def test_served_a_page_opened_mid_ask_follows_it_and_lists_it_after(ask_resumed):
    resumed, end = _step(ask_resumed, "resumed"), _step(ask_resumed, "end")

    assert resumed["title"] == "Pregunta en curso"
    assert resumed["badges"] == ["ask:en curso"] and resumed["top"] == "top-ask"
    assert re.match(r"^[0-4] de 5 posts · ", resumed["progress"])
    assert end["panel"]["progress"].startswith("5 respuestas guardadas · 5 resultados")
    assert [h["text"].split(" · ")[0] for h in end["history"]] == [QUERY]


def test_the_fake_answers_what_the_assertions_expect():
    """The results the browser tests read: probability order is not id order, and 1 and 5 tie."""
    assert ranked() == [("3", 0.97), ("1", 0.9), ("5", 0.9), ("4", 0.2), ("2", 0.1)]
    assert json.dumps(PROBS)


@_requires_chrome
def test_static_a_rebuilt_query_and_an_unreadable_run_log_say_so(tmp_path):
    """The index lost the query (it is rebuilt from its answers) and runs.jsonl has a bad
    line: the row says «reconstruida», and its cost is «—» with why — never «0,0000 $»."""
    from tests.test_jev_serve import _repo
    from xbrain.jev.dashboard import build_page_data

    _need_chrome()
    monkeypatch = pytest.MonkeyPatch()
    try:
        cfg = _repo(tmp_path, monkeypatch)
        _asked(cfg, QUERY, when=DT)
        (cfg.jev_asks_dir / "index.json").unlink()
        with cfg.jev_runs_path.open("a", encoding="utf-8") as log:
            log.write("{roto\n")
        data = build_page_data(cfg, now=DT)
    finally:
        monkeypatch.undo()

    seen = _dump(_page(tmp_path, data, _ASK_READ).as_uri() + "#ask")

    assert "reconstruida desde sus respuestas (el historial la había perdido)" in seen["head"]
    assert "coste: — (" in seen["head"] and "runs.jsonl" in seen["head"]
    assert "~0,0000 $" not in seen["head"]
    assert " · — · reconstruida" in seen["history"][0]["text"]
    assert _probs(seen["results"]) == _shown(ranked())


# --------------------------------------------------------------------------- task 14: Víctor's ask
# His first real ask answered 808 posts with a max of 0.80, and the tab said «0 de 808 llegan a
# 0,85». The same shape (`jev_ask_fixtures`), reopened: ranked, the first `[jev].ask_top`
# shown, «Ver más» adding as many, each probability drawn — and REFINED for free: the saved
# answers filtered by top / minimum / topics / days / author with no request and no cost, the
# state in the hash, and the list equal to what Python's `ask.refine_results` keeps.

_REFINE_JS = r"""
const sQRefine = () => {
  const box = document.getElementById('ask-refine');
  if (!seen(box)) return null;
  const val = id => document.getElementById(id).value;
  return {title: txt(box.querySelector('h4')), top: val('refine-top'), min: val('refine-min'),
    since: val('refine-since'), until: val('refine-until'), author: val('refine-author'),
    topics: [...box.querySelectorAll('input[type=checkbox]')].filter(seen).map(b => ({
      id: b.id, checked: b.checked, count: txt(b.parentNode.querySelector('.tcount'))}))};
};
const sQSet = (fields) => {
  for (const [id, value] of Object.entries(fields)) {
    const box = document.getElementById(id);
    if (box.type === 'checkbox') box.checked = value; else box.value = value;
    box.dispatchEvent(new Event('input', {bubbles: true}));
    box.dispatchEvent(new Event('change', {bubbles: true}));
  }
};
"""

_VICTOR_STATIC_PROBE = (
    "<script>"
    + _SEEN
    + _ASK_JS
    + _REFINE_JS
    + r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = {};
  const view = () => Object.assign(sQView(), {refine: sQRefine()});
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  await step('first', async () => { if (!location.hash.startsWith('#ask')) location.hash = '#ask'; await sleep(80); return view(); });
  await step('more', async () => { document.getElementById('ask-more').click(); await sleep(60); return view(); });
  await step('more2', async () => { document.getElementById('ask-more').click(); await sleep(60); return view(); });
  await step('away_back', async () => {
    location.hash = '#posts?f=all'; await sleep(60);
    document.querySelector('.tabs a[data-tab="ask"]').click(); await sleep(80);
    return view();
  });
  await step('refined', async () => {
    sQSet({'refine-min': '0.5', 'refine-topic-agentic-engineering': true});
    await sleep(80); return view();
  });
  await step('cleared', async () => { document.getElementById('refine-clear').click(); await sleep(80); return view(); });
  await step('clear_race', async () => {
    // A refine, «Quitar el refinado» and another change in the same moment: the last change
    // builds on the cleared state, never on the refine the clear removed.
    sQSet({'refine-min': '0.5'});
    await sleep(80);
    document.getElementById('refine-clear').click();
    sQSet({'refine-top': '30'});
    await sleep(80);
    return view();
  });
  const pre = document.createElement('pre'); pre.id = 'probe'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>"""
)

_VICTOR_READ = (
    "<script>"
    + _SEEN
    + _ASK_JS
    + _REFINE_JS
    + r"""
setTimeout(() => {
  const pre = document.createElement('pre'); pre.id = 'probe';
  pre.textContent = JSON.stringify(Object.assign(sQView(), {refine: sQRefine()}));
  document.body.appendChild(pre);
}, 200);
</script>"""
)


def _victor_page(
    root: Path,
    probe: str,
    *,
    jev: str = "",
    minimum: float | None = None,
    topics: bool = False,
    mutate: Any = None,
):
    from tests.jev_ask_fixtures import victor_shaped_repo
    from xbrain.jev.dashboard import build_page_data

    _need_chrome()
    cfg = victor_shaped_repo(root, jev=jev, topics=topics)
    if minimum is not None:
        path = cfg.jev_asks_dir / "index.json"
        index = json.loads(path.read_text(encoding="utf-8"))
        for entry in index["queries"].values():
            entry["last_min"] = minimum
        path.write_text(json.dumps(index), encoding="utf-8")
    data = build_page_data(cfg, now=DT)
    if mutate is not None:
        mutate(data)
    return _page(root, data, probe), data, cfg


def _shown2(pairs) -> list[tuple[str, str]]:
    return [(item.id, f"{record.probability:.2f}".replace(".", ",")) for item, record in pairs]


def _refined(cfg: Config, *, minimum: float = 0.0, **refine: Any):
    """Python's refine of Víctor's saved query: THE list the page must show."""
    from xbrain.jev.ask import load_history, refine_results, reopen_results
    from xbrain.jev.load import load_jev_pairs

    jev = load_jev_pairs(cfg)
    [entry] = load_history(cfg).queries.values()
    found = reopen_results(cfg, entry, jev)
    return refine_results(
        found, AskFilters(**refine), minimum, store=jev.store, jev=jev, threshold=cfg.jev_threshold
    )


@pytest.fixture(scope="module")
def victor_static(tmp_path_factory):
    page, data, cfg = _victor_page(tmp_path_factory.mktemp("victor-static"), _VICTOR_STATIC_PROBE)
    return _dump(page.as_uri()), data, cfg


@_requires_chrome
def test_static_victors_ask_opens_on_the_first_20_ranked_not_on_nothing(victor_static):
    seen, data, cfg = victor_static
    first = seen["first"]
    order = _shown2(_refined(cfg).ranked)

    assert len(order) == 808
    assert _probs(first["results"]) == order[:20]
    assert [p for _, p in order[:3]] == ["0,80", "0,79", "0,79"]
    assert first["head"].startswith("«En qué topics hablo de agentic engineering?»")
    assert "Filtros: desde 2026-05-07" in first["head"]
    assert "Mostrando 20 de 808 leídos por Jev, de mayor a menor probabilidad." in first["head"]
    assert "0,85" not in first["head"] and first["empty"] is None
    assert first["more"] == "Ver 20 más (quedan 788)"
    assert " · 20 de 808 leídos por Jev · " in first["history"][0]["text"]
    for result in first["results"]:
        assert result["bar"] is not None and result["bar"]["fill"] > 0
    assert first["refine"]["title"] == "Refinar resultados (gratis)"
    assert (first["refine"]["top"], first["refine"]["min"]) == ("20", "")


@_requires_chrome
def test_static_see_more_adds_20_and_the_hash_keeps_it(victor_static):
    seen, data, cfg = victor_static
    order = _shown2(_refined(cfg).ranked)

    assert _probs(seen["more"]["results"]) == order[:40]
    assert seen["more"]["more"] == "Ver 20 más (quedan 768)"
    assert "top=40" in seen["more"]["hash"] and seen["more"]["refine"]["top"] == "40"
    assert "Mostrando 40 de 808" in seen["more"]["head"]
    assert _probs(seen["more2"]["results"]) == order[:60]
    # Leaving the tab and coming back by its link keeps the refine state.
    assert _probs(seen["away_back"]["results"]) == order[:60]
    assert seen["away_back"]["hash"] == seen["more2"]["hash"]


@_requires_chrome
def test_static_refining_filters_the_saved_answers_like_python(victor_static):
    seen, data, cfg = victor_static
    refined = seen["refined"]
    expected = _refined(cfg, minimum=0.5, topics=("agentic-engineering",))

    assert _probs(refined["results"]) == _shown2(expected.ranked)[:60]
    assert "min=0.5" in refined["hash"] and "t=agentic-engineering" in refined["hash"]
    assert (
        f"Mostrando {min(60, len(expected.ranked))} de {len(expected.ranked)} con relevancia "
        "≥ 0,50 que pasan el refinado · 808 leídos por Jev" in refined["head"]
    )
    # Quitar el refinado: back to the query's defaults, the first 20 of every answer.
    cleared = seen["cleared"]
    assert _probs(cleared["results"]) == _shown2(_refined(cfg).ranked)[:20]
    assert (
        cleared["hash"]
        == f"#ask?q={AskQuery.of('En qué topics hablo de agentic engineering?').sha}"
    )


@_requires_chrome
def test_static_clearing_the_refine_resets_it_before_the_next_change(victor_static):
    seen, data, cfg = victor_static
    race = seen["clear_race"]
    sha = AskQuery.of("En qué topics hablo de agentic engineering?").sha

    assert race["hash"] == f"#ask?q={sha}&top=30"
    assert _probs(race["results"]) == _shown2(_refined(cfg).ranked)[:30]
    assert race["refine"]["min"] == ""


@_requires_chrome
@pytest.mark.parametrize(
    "params,refine",
    [
        ({"min": "0.5", "top": "7"}, {"minimum": 0.5}),
        (
            # Víctor's posts run 2026-05-07 … 2026-06-10: this `until` cuts a third of them.
            {"t": "startups", "since": "2026-05-20", "until": "2026-06-01", "top": "900"},
            {
                "topics": ("startups",),
                "since": datetime(2026, 5, 20).date(),
                "until": datetime(2026, 6, 1).date(),
            },
        ),
        (
            {"since": "2026-05-20", "until": "2026-05-20", "top": "900"},
            {"since": datetime(2026, 5, 20).date(), "until": datetime(2026, 5, 20).date()},
        ),
        (
            {"until": "2026-05-10", "min": "0.1", "top": "900"},
            {"until": datetime(2026, 5, 10).date(), "minimum": 0.1},
        ),
        ({"t": "nutricion,startups", "top": "900"}, {"topics": ("nutricion", "startups")}),
        (
            {"t": "agentic-engineering,startups", "author": "@SomeOne", "min": "0.3", "top": "900"},
            {"topics": ("agentic-engineering", "startups"), "author": "@SomeOne", "minimum": 0.3},
        ),
        ({"author": "nadie"}, {"author": "nadie"}),
        ({"author": " @ SOMEONE ", "min": "0.6"}, {"author": " @ SOMEONE ", "minimum": 0.6}),
    ],
)
def test_a_refined_view_opened_fresh_from_its_url_is_pythons_refine(tmp_path, params, refine):
    """The refine state lives in the hash: a bookmark (or a reload) shows the same list, and
    that list is `ask.refine_results`'. Two cases draw every kept answer (`top=900`), so the
    whole list is compared — a minimum or a day edge cannot hide below the first 20."""
    from urllib.parse import urlencode

    page, _, cfg = _victor_page(tmp_path, _VICTOR_READ)
    sha = AskQuery.of("En qué topics hablo de agentic engineering?").sha

    seen = _dump(page.as_uri() + "#ask?" + urlencode({"q": sha, **params}))

    minimum = refine.pop("minimum", 0.0)
    expected = _shown2(_refined(cfg, minimum=minimum, **refine).ranked)
    top = int(params.get("top", 20))
    assert _probs(seen["results"]) == expected[:top]
    if not expected:
        assert seen["empty"].startswith("Ningún post pasa el refinado")
    state = seen["refine"]
    assert state["top"] == str(top) and state["since"] == params.get("since", "")
    assert state["author"] == params.get("author", "")
    ticked = sorted(t["id"].removeprefix("refine-topic-") for t in state["topics"] if t["checked"])
    assert ticked == sorted(params["t"].split(",")) if "t" in params else ticked == []
    _assert_refine_counts(cfg, state, minimum, refine)
    # A topic of the link that no answer is in is listed (ticked, 0) and named.
    if "nutricion" in params.get("t", ""):
        assert "nutricion" in seen["absent"].casefold()
    else:
        assert seen["absent"] is None


def _assert_refine_counts(cfg: Config, state: dict[str, Any], minimum: float, refine: dict) -> None:
    """Each refine-bar topic's count is Python's refine by that topic alone under the rest of
    the refine (brute force, topic by topic); every topic of an answer is offered, and every
    ticked one too."""
    others = {k: v for k, v in refine.items() if k != "topics"}
    whole = _refined(cfg)
    from xbrain.jev.load import load_jev_pairs
    from xbrain.jev.ask import post_topics

    jev = load_jev_pairs(cfg)
    current = jev.current_by_id()
    offered = {
        t
        for item, _ in whole.ranked
        for t in post_topics(item, current.get(item.id), cfg.jev_threshold)
    } | set(refine.get("topics", ()))
    shown = {t["id"].removeprefix("refine-topic-"): t["count"] for t in state["topics"]}
    assert set(shown) == offered
    for topic in offered:
        count = len(_refined(cfg, minimum=minimum, topics=(topic,), **others).ranked)
        assert shown[topic] == f"{count:,}".replace(",", "."), topic


@_requires_chrome
@pytest.mark.parametrize(
    "params,refine",
    [
        ({"t": "agentic-engineering", "top": "900"}, {"topics": ("agentic-engineering",)}),
        (
            # The 0.30 answers start on 2026-06-01: some Jev-only posts stay, some are cut.
            {"t": "agentic-engineering", "min": "0.2", "until": "2026-06-05", "top": "900"},
            {
                "topics": ("agentic-engineering",),
                "minimum": 0.2,
                "until": datetime(2026, 6, 5).date(),
            },
        ),
    ],
)
def test_the_jev_side_of_a_topic_refine_matches_python_at_the_bar(tmp_path, params, refine):
    """With a current topics side-car, Jev puts some enrich-«startups» posts in
    agentic-engineering at exactly 0.85 (in) and others at 0.84 (out): the page keeps the
    same ones Python keeps."""
    from urllib.parse import urlencode
    from tests.jev_ask_fixtures import JEV_AGENTIC

    page, data, cfg = _victor_page(tmp_path, _VICTOR_READ, topics=True)
    sha = AskQuery.of("En qué topics hablo de agentic engineering?").sha

    seen = _dump(page.as_uri() + "#ask?" + urlencode({"q": sha, **params}))

    minimum = refine.pop("minimum", 0.0)
    ranked = _refined(cfg, minimum=minimum, **refine).ranked
    assert _probs(seen["results"]) == _shown2(ranked)
    # The case is live: posts only Jev puts in the topic (at 0.85) are kept; posts it scored
    # 0.84 are not.
    from xbrain.jev.load import load_jev_pairs

    jev = load_jev_pairs(cfg)
    current = jev.current_by_id()
    kept = {item.id for item, _ in ranked}
    at = {0.85: set(), 0.84: set()}
    for n, item in enumerate(jev.store.values()):
        if (
            item.id in current
            and item.enriched.primary_topic == "startups"
            and n % 7 in JEV_AGENTIC
        ):
            at[JEV_AGENTIC[n % 7]].add(item.id)
    assert at[0.85] & kept and at[0.84] and not at[0.84] & kept
    _assert_refine_counts(cfg, seen["refine"], minimum, refine)


@_requires_chrome
def test_an_answer_asked_by_another_model_or_pass_says_its_own(tmp_path):
    """The model and minute asked travel once per query; an answer that differs carries its
    own in `exceptions`, and its card says it."""

    def _second_pass(data: dict[str, Any]) -> None:
        answers = data["asks"]["history"][0]["answers"]
        answers["exceptions"][answers["ids"][1]] = {"model": "jev-2.0.0"}

    page, data, _ = _victor_page(tmp_path, _VICTOR_READ, mutate=_second_pass)

    seen = _dump(page.as_uri() + "#ask")

    metas = [r["meta"] for r in seen["results"]]
    assert metas[1].startswith("jev-2.0.0 · ")
    assert all(m.startswith("jev-1.13.0 · ") for i, m in enumerate(metas) if i != 1)
    assert len(set(metas)) == 2


@_requires_chrome
def test_an_answer_without_refine_keys_is_an_error_not_a_missing_result(tmp_path):
    """Every answer reaches the page through `ask.answer_view`, keys included; one that
    arrives without them (a bug, or a stream gone wrong) is said, never silently dropped."""

    def _drop_first(data: dict[str, Any]) -> None:
        first = data["asks"]["history"][0]["answers"]["ids"][0]
        del data["asks"]["keys"][first]

    page, data, _ = _victor_page(tmp_path, _VICTOR_READ, mutate=_drop_first)

    seen = _dump(page.as_uri() + "#ask")

    assert seen["keyless"] is not None and "1 respuesta" in seen["keyless"]
    assert "sin sus claves" in seen["keyless"]
    assert len(seen["results"]) == 20


@_requires_chrome
def test_static_a_query_asked_with_a_minimum_opens_at_it_and_ask_top_shapes_the_list(tmp_path):
    page, data, cfg = _victor_page(tmp_path, _VICTOR_STATIC_PROBE, jev="ask_top = 7\n", minimum=0.5)
    seen = _dump(page.as_uri())
    first, more = seen["first"], seen["more"]
    order = _shown2(_refined(cfg, minimum=0.5).ranked)

    assert len(order) == 156 and len(data["asks"]["history"][0]["answers"]["ids"]) == 808
    assert _probs(first["results"]) == order[:7]
    assert (
        "Mostrando 7 de 156 con relevancia ≥ 0,50 · 808 leídos por Jev, de mayor a menor "
        "probabilidad." in first["head"]
    )
    assert first["refine"]["min"] == "0.5" and first["refine"]["top"] == "7"
    assert first["more"] == "Ver 7 más (quedan 149)"
    assert _probs(more["results"]) == order[:14]
    assert " · 7 de 156 ≥ 0,50 · 808 leídos por Jev · " in first["history"][0]["text"]


_VICTOR_SERVE_PROBE = (
    "<script>"
    + _SERVE_JS
    + _ASK_JS
    + _REFINE_JS
    + r"""
const sQForm = (fields) => { sQSet(fields); sPress(sId('ask-form'), 'Estimar lo que cuesta'); };
let sCounts = 0;
const sFetchCounts = window.fetch;
window.fetch = function (u, init) { if (String(u).includes('/api/ask/counts')) sCounts++; return sFetchCounts.call(window, u, init); };
const sTopics = () => [...document.querySelectorAll('#ask-topics input[type=checkbox]')].filter(seen).map(b => ({
  id: b.id, checked: b.checked, text: txt(b.parentNode), count: txt(b.parentNode.querySelector('.tcount'))}));
const sNextEstimate = async (n0, what) => { await sWait(() => sEstimates > n0 && sPanel().go !== null, what); return sLastEstimate; };
(async () => {
  await sStep('form', async () => {
    location.hash = '#ask';
    await sWait(() => seen(sId('ask-form')) && sQResults().length > 0, 'el formulario y los resultados');
    return Object.assign(sQView(), {topics: sTopics(), explain: txt(sId('ask-explain')), tip: txt(sId('ask-tip')),
      form_title: txt(sId('ask-form').querySelector('h4')), min: sId('ask-min') ? 'HAY' : null, posts: sPosts});
  });
  await sStep('counts', async () => {
    const posts0 = sPosts, counts0 = sCounts;
    const box = sId('ask-since');
    box.value = '2026-06-01';
    box.dispatchEvent(new Event('input', {bubbles: true}));
    box.dispatchEvent(new Event('change', {bubbles: true}));
    await sWait(() => sTopics().some(t => t.id === 'ask-topic-startups' && t.count === '220'), 'los recuentos con «desde»');
    return {topics: sTopics(), posts: sPosts - posts0, counts: sCounts - counts0};
  });
  await sStep('reload', async () => {
    // New data (what a job's end brings) with «Desde» still typed: the counts are asked
    // again for it, never left as the old data's.
    const c0 = sCounts;
    document.querySelectorAll('#ask-topics .tcount').forEach(n => { n.textContent = 'viejo'; });
    await reloadData();
    await sWait(() => sCounts > c0 && sTopics().some(t => t.id === 'ask-topic-startups' && t.count === '220'), 'los recuentos tras recargar');
    return {topics: sTopics(), counts: sCounts - c0};
  });
  await sStep('one', async () => {
    const n0 = sEstimates, c0 = sCounts;
    sQForm({'ask-q': 'posts que explican cómo trabajar con agentes', 'ask-since': '', 'ask-topic-agentic-engineering': true});
    const e = await sNextEstimate(n0, 'la estimación con un topic');
    const panel = sPanel();
    sId('ask-cancel').click();
    // Clearing «Desde» asks the counts again, 250 ms after the keystroke (`askRecount`): wait
    // for that request here, or it lands inside a later step's window and is counted there.
    await sWait(() => sCounts > c0, 'el recuento de «Desde» vacío');
    return {estimate: e, panel: panel};
  });
  await sStep('two', async () => {
    const n0 = sEstimates;
    sQForm({'ask-topic-startups': true});
    const e = await sNextEstimate(n0, 'la estimación con dos topics');
    const panel = sPanel();
    sId('ask-cancel').click();
    return {estimate: e, panel: panel};
  });
  await sStep('refine', async () => {
    const posts0 = sPosts, r0 = sRefreshed, e0 = sEstimates, c0 = sCounts;
    sId('ask-more').click();
    await sWait(() => sQResults().length === 40, 'veinte más');
    const more = sQView();
    sQSet({'refine-min': '0.5'});
    await sWait(() => (sQView().head || '').includes('con relevancia ≥ 0,50'), 'el mínimo');
    const min = sQView();
    sQSet({'refine-topic-startups': true, 'refine-min': '0.3'});
    await sWait(() => (sQView().head || '').includes('que pasan el refinado'), 'el topic');
    return {more: more, min: min, topic: Object.assign(sQView(), {refine: sQRefine()}),
      posts: sPosts - posts0, reloads: sRefreshed - r0, estimates: sEstimates - e0, counts: sCounts - c0};
  });
  await sStep('reopen', async () => {
    const posts0 = sPosts, r0 = sRefreshed;
    location.hash = '#posts?f=all';
    await sWait(() => seen(document.querySelector('#cards .card')), 'Posts');
    document.querySelector('.tabs a[data-tab="ask"]').click();
    await sWait(() => sQResults().length > 0, 'los resultados reabiertos');
    sQOpen('agentic engineering');
    await sWait(() => sQResults().length === 20, 'la consulta sin refinado');
    return Object.assign(sQView(), {posts: sPosts - posts0, reloads: sRefreshed - r0});
  });
  await sStep('bad_counts', async () => {
    sQSet({'ask-since': '2026-06-10', 'ask-until': '2026-06-01'});
    await sWait(() => txt(sId('ask-topics-error')) !== null, 'el motivo del recuento fallido');
    return {error: txt(sId('ask-topics-error')), topics: sTopics()};
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def victor_served(tmp_path_factory) -> dict[str, Any]:
    from tests.jev_ask_fixtures import victor_shaped_repo
    from tests.jev_fakes import FakeJevClient

    def _repo_builder(root: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
        monkeypatch.setenv("XBRAIN_REPO_ROOT", str(root))
        return victor_shaped_repo(root)

    built: list[int] = []

    def _make() -> FakeJevClient:
        built.append(1)
        return FakeJevClient()

    seen = _served_dump(
        tmp_path_factory.mktemp("victor-served"),
        _VICTOR_SERVE_PROBE,
        make_client=_make,
        repo=_repo_builder,
    )
    seen["built"] = len(built)
    return seen


def _victor_expected(tmp_path: Path):
    """What Python says for the same repo: the config, and the topic counts under filters."""
    from tests.jev_ask_fixtures import victor_shaped_repo
    from xbrain.jev.ask import topic_counts
    from xbrain.jev.load import load_jev_pairs

    cfg = victor_shaped_repo(tmp_path)
    jev = load_jev_pairs(cfg)
    return cfg, lambda f: topic_counts(jev.store, f, jev=jev, threshold=cfg.jev_threshold)


@_requires_chrome
def test_served_the_tab_splits_what_is_asked_from_what_is_refined(victor_served):
    form = _step(victor_served, "form")

    assert form["form_title"] == "Qué posts preguntar"
    assert form["min"] is None  # the minimum is a free refine, never part of what is paid
    assert form["explain"] == (
        "Los filtros eligen qué posts se preguntan (y lo que cuesta); los resultados se ordenan "
        "por lo seguro que está Jev de que el post responde. Topic: posts que enrich o Jev "
        "(≥ 0,85) ponen en alguno de los topics marcados."
    )
    assert form["tip"] == (
        'Pregunta por el contenido que buscas ("posts que explican…"), no por los topics.'
    )


@_requires_chrome
def test_served_topics_are_a_multi_select_each_with_its_count(victor_served, tmp_path):
    form = _step(victor_served, "form")
    _, counts = _victor_expected(tmp_path)
    expected = counts(AskFilters())

    shown = {t["id"].removeprefix("ask-topic-"): t for t in form["topics"]}
    assert set(shown) == set(expected)
    assert {slug: int(t["count"].replace(".", "")) for slug, t in shown.items()} == expected
    assert shown["agentic-engineering"]["text"].startswith("Agentic Engineering")
    assert not any(t["checked"] for t in form["topics"])
    assert form["posts"] == 0  # the counts the page opens with are the blob's: no request


@_requires_chrome
def test_served_the_counts_follow_the_other_filters_from_the_server(victor_served, tmp_path):
    counts_step = _step(victor_served, "counts")
    _, counts = _victor_expected(tmp_path)
    expected = counts(AskFilters(since=datetime(2026, 6, 1).date()))

    got = {t["id"].removeprefix("ask-topic-"): int(t["count"]) for t in counts_step["topics"]}
    assert got == expected and expected["agentic-engineering"] == 0
    # Asked the server (`GET /api/ask/counts`), not computed here — and a GET: no POST.
    assert counts_step["counts"] >= 1 and counts_step["posts"] == 0


@_requires_chrome
def test_served_new_data_asks_the_counts_again_for_the_filters_typed(victor_served, tmp_path):
    step = _step(victor_served, "reload")
    _, counts = _victor_expected(tmp_path)
    expected = counts(AskFilters(since=datetime(2026, 6, 1).date()))

    assert step["counts"] >= 1
    assert {t["id"].removeprefix("ask-topic-"): int(t["count"]) for t in step["topics"]} == expected


@_requires_chrome
def test_served_a_count_the_server_refuses_says_why(victor_served):
    step = _step(victor_served, "bad_counts")

    assert "posterior" in step["error"] and "2026-06-10" in step["error"]
    assert all(t["count"] == "—" for t in step["topics"])


@_requires_chrome
def test_served_ticking_a_second_topic_widens_the_estimate(victor_served, tmp_path):
    one, two = _step(victor_served, "one")["estimate"], _step(victor_served, "two")["estimate"]
    cfg, _ = _victor_expected(tmp_path)
    query = AskQuery.of("posts que explican cómo trabajar con agentes")
    plan_one = plan_ask(cfg, query, AskFilters(topics=("agentic-engineering",)), None)
    plan_two = plan_ask(cfg, query, AskFilters(topics=("agentic-engineering", "startups")), None)

    assert one["pick"]["topics"] == ["agentic-engineering"]
    assert two["pick"]["topics"] == ["agentic-engineering", "startups"]
    assert (one["posts"], two["posts"]) == (plan_one.estimate.posts, plan_two.estimate.posts)
    assert (one["posts"], two["posts"]) == (156, 813)
    assert (one["usd"], two["usd"]) == (plan_one.estimate.usd, plan_two.estimate.usd)
    assert _step(victor_served, "one")["panel"]["est"].startswith("156 posts por preguntar")
    assert _step(victor_served, "two")["panel"]["est"].startswith("813 posts por preguntar")


@_requires_chrome
def test_served_refining_changes_the_list_with_no_request_and_no_client(victor_served, tmp_path):
    step = _step(victor_served, "refine")
    cfg, _ = _victor_expected(tmp_path)

    assert len(step["more"]["results"]) == 40
    assert _probs(step["min"]["results"]) == _shown2(_refined(cfg, minimum=0.5).ranked)[:40]
    assert "Mostrando 40 de 156 con relevancia ≥ 0,50 · 808 leídos por Jev" in step["min"]["head"]
    topic = step["topic"]
    expected = _shown2(_refined(cfg, minimum=0.3, topics=("startups",)).ranked)
    assert _probs(topic["results"]) == expected[:40]
    assert "min=0.3" in topic["hash"] and "t=startups" in topic["hash"]
    # Free: no POST, no estimate, no reload, and no client was ever built.
    assert (step["posts"], step["estimates"], step["reloads"], step["counts"]) == (0, 0, 0, 0)
    assert victor_served["built"] == 0


@_requires_chrome
def test_served_reopening_a_query_from_the_history_costs_nothing_and_resets_refine(victor_served):
    reopen = _step(victor_served, "reopen")

    assert (reopen["posts"], reopen["reloads"]) == (0, 0)
    assert len(reopen["results"]) == 20
    assert "min=" not in reopen["hash"] and "t=" not in reopen["hash"]


_EXPLAIN_READ = (
    "<script>"
    + _SERVE_JS
    + _ASK_JS
    + r"""
(async () => {
  await sStep('explain', async () => {
    location.hash = '#ask';
    await sWait(() => seen(sId('ask-explain')), 'la explicación');
    return txt(sId('ask-explain'));
  });
  sDone();
})();
</script>"""
)


@_requires_chrome
def test_served_the_explanation_says_the_configured_topic_bar(tmp_path):
    """The «(≥ …)» of the explanation is `[jev].threshold` from the blob, never a literal."""
    seen = _served_dump(tmp_path, _EXPLAIN_READ, client=_Asker(), jev="threshold = 0.9\n")

    assert "Topic: posts que enrich o Jev (≥ 0,90) ponen en alguno" in _step(seen, "explain")

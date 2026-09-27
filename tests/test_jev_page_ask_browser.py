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
)
from tests.test_jev_serve_ask import PROBS, QUERY, _Asker, _PerChar, ranked
from xbrain.config import Config
from xbrain.jev.ask import AskFilters, AskQuery, finish_ask, plan_ask
from xbrain.jev.lock import pass_lock
from xbrain.jev.report import ask_cost_by_query
from xbrain.jev.run import run_ask
from xbrain.jev.store import load_runs

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
#: A query asked only over «startups»: its posts (2, 4) answer 0.1 and 0.2 — no result.
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
        finish_ask(cfg, plan, outcome, threshold=cfg.jev_threshold, now=when)
    return query


def _asked_repo(cfg: Config) -> dict[str, str]:
    """Three queries: the hooks one (results 1 and 3), a later «startups» one with no result,
    and an earlier one whose file is then broken."""
    hooks = _asked(cfg, QUERY, when=DT + timedelta(hours=1))
    seed = _asked(cfg, SEED_QUERY, when=DT + timedelta(hours=2), topic="startups")
    broken = _asked(cfg, BROKEN_QUERY, when=DT + timedelta(minutes=30), author="bob")
    (cfg.jev_asks_dir / f"{broken.sha}.json").write_text("{", encoding="utf-8")
    return {"hooks": hooks.sha, "seed": seed.sha, "broken": broken.sha}


#: Reads the tab as a reader sees it. Shared by the static and the served probes.
_ASK_JS = r"""
const sQHistory = () => [...document.querySelectorAll('#ask-history li')].filter(seen).map(li => ({
  text: [...li.querySelectorAll('b, small')].map(e => e.textContent).join(' · '),
  current: li.querySelector('a').getAttribute('aria-current') === 'true'}));
const sQResults = () => [...document.querySelectorAll('#ask-results .card')].filter(seen).map(c => ({
  id: c.dataset.id, p: txt(c.querySelector('.askr .ap')), saw: txt(c.querySelector('.askr details.saw > summary')),
  jev: txt(c.querySelector('.jev .badge'))}));
const sQView = () => ({
  hash: location.hash,
  tab: txt(document.querySelector('.tabs a[data-tab="ask"]')),
  head: txt(document.getElementById('ask-head')),
  empty: txt(document.getElementById('ask-empty')),
  error: txt(document.getElementById('ask-row-error')),
  form: seen(document.getElementById('ask-form')),
  launch: txt(document.getElementById('ask-launch')),
  history: sQHistory(),
  results: sQResults(),
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
    assert f"0 resultados · 2 posts con respuesta · {seed_cost}" in view["history"][0]["text"]
    assert f"3 resultados · 5 posts con respuesta · {hooks_cost}" in view["history"][1]["text"]
    assert view["history"][0]["current"] is True


@_requires_chrome
def test_static_the_last_query_opens_by_default_and_says_what_it_found(ask_static):
    seen, shas = ask_static
    view = seen["default"]

    assert view["hash"] == "#ask"
    assert view["head"].startswith(f"«{SEED_QUERY}»")
    assert "Filtros: topic Startups" in view["head"]
    assert view["results"] == []
    assert view["empty"] == (
        "Ningún post llega a 0,85: Jev no ve una respuesta a esta pregunta en los 2 posts con "
        "respuesta vigente."
    )


@_requires_chrome
def test_static_results_are_the_post_cards_ranked_by_probability(ask_static):
    seen, shas = ask_static
    view = seen["hooks"]

    assert view["hash"] == f"#ask?q={shas['hooks']}"
    assert _probs(view["results"]) == _shown(ranked())
    assert [h["current"] for h in view["history"]] == [False, True, False]
    assert "3 de 5 posts con respuesta vigente llegan a 0,85" in view["head"]
    for result in view["results"]:
        assert re.fullmatch(r"Lo que vio Jev · 2 fuentes, \d+ caracteres", result["saw"])


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
    return Object.assign(sPanel(), {estimate: sLastEstimate});
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
    return Object.assign(sQView(), {panel, hidden, after: sPanel()});
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
    empty = ask_served["empty"]

    assert empty["form"] is True and empty["launch"] is None
    assert empty["history"] == []
    assert empty["head"] is None
    assert empty["empty"] == "Aún no se ha preguntado nada."


@_requires_chrome
def test_served_an_ask_estimate_is_the_servers_and_says_what_it_will_pay(ask_served):
    estimate = ask_served["estimate"]
    e = estimate["estimate"]
    money = f"~{e['usd']:.5f}".replace(".", ",") + " $"

    assert estimate["title"] == "Preguntar: «¿Cómo configuro hooks en Claude Code?»"
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
    similar = ask_served["similar"]
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
    done = ask_served["done"]
    sha = AskQuery.of(QUERY).sha

    assert sorted(ask_served["asked"]) == ["1", "2"]
    assert done["hash"] == f"#ask?q={sha}"
    assert _probs(done["results"]) == _shown(ranked(posts="12"))
    assert done["panel"]["progress"].startswith("2 respuestas guardadas · 1 resultado · ")
    assert [h["text"].split(" · ")[0] for h in done["history"]] == [QUERY]
    assert done["history"][0]["current"] is True
    assert ask_served["job"]["query_sha"] == sha


@_requires_chrome
def test_served_asking_again_what_is_answered_is_free_and_counted(ask_served):
    again = ask_served["again"]

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
    # Nothing was sent, so nothing to log: a clean end — no alarm, and the panel stays hidden.
    assert again["hidden"] is True and again["after"]["shown"] is False
    # The probe's last job is this free one.
    job = ask_served["job"]
    assert (job["state"], job["outcome"]["sent"], job["outcome"]["logged"]) == ("done", 0, False)


@_requires_chrome
def test_served_reopening_a_query_from_the_history_asks_the_server_nothing(ask_served):
    reopen = ask_served["reopen"]

    assert (reopen["posts"], reopen["reloads"]) == (0, 0)
    assert reopen["hash"] == f"#ask?q={AskQuery.of(QUERY).sha}"
    assert _probs(reopen["results"]) == _shown(ranked(posts="12"))


@_requires_chrome
def test_served_a_query_asked_before_in_other_words_is_named(ask_served):
    assert isinstance(ask_served["similar"], dict), ask_served["similar"]
    est = ask_served["similar"]["est"]

    assert (
        "Ya preguntaste algo casi igual (cambia solo en mayúsculas, puntuación o espacios, y se "
        f"paga aparte): «{AskQuery.of(QUERY).text}»." in est
    )


@_requires_chrome
def test_served_an_ask_with_nothing_to_ask_is_refused_before_confirm(ask_served):
    refused = ask_served["refused"]

    assert refused["go"] is False
    assert refused["error"].startswith("No se puede preguntar: ningún post que preguntar")


@_requires_chrome
def test_served_a_blank_query_is_refused_by_the_server(ask_served):
    blank = ask_served["blank"]

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

    assert seen["estimate"]["title"].startswith("Preguntar: «"), seen
    assert seen["run"]["hidden"] is True
    assert seen["end"]["shown"] is True and seen["end"]["close"] == "Cerrar"
    assert seen["job"]["kind"] == "ask"


@_requires_chrome
def test_served_an_ask_cut_by_the_cap_says_so_and_keeps_what_it_paid(ask_ended):
    seen = ask_ended("tope")

    assert (seen["job"]["state"], seen["job"]["reason"]) == ("interrupted", "tope")
    assert seen["end"]["progress"].startswith(
        "Interrumpido (se alcanzó el tope por trabajo): 1 respuesta guardada · 0 resultados · "
    )


@_requires_chrome
def test_served_an_ask_whose_history_could_not_be_written_keeps_its_paid_answers_in_view(
    ask_ended,
):
    seen = ask_ended("history")
    job = seen["job"]

    assert job["state"] == "done" and job["outcome"]["recorded"] is False
    assert seen["end"]["progress"].startswith("2 respuestas guardadas · ")
    assert (
        f"2 respuestas pagadas y guardadas en {job['outcome']['file']}; el historial no se pudo "
        "escribir: [Errno 28] No space left on device"
    ) in seen["end"]["progress"]


@_requires_chrome
def test_served_an_ask_that_failed_says_why(ask_ended):
    seen = ask_ended("error")

    assert seen["job"]["state"] == "error"
    assert seen["end"]["error"] == "El trabajo falló: TYPESAFE_API_KEY no encontrada"


@_requires_chrome
def test_served_an_asks_failures_unsaved_and_unpriced_answers_are_named(ask_ended):
    end = ask_ended("failures")["end"]

    assert end["progress"].startswith("4 respuestas guardadas · 0 resultados · ")
    assert "1 respuesta cobrada a la media estimada" in end["progress"]
    assert "sin tarifa: fake" in end["progress"]
    assert "1 fallida: 4 (" in end["progress"] and "respuesta ilegible" in end["progress"]
    assert "2 respuestas pagadas sin guardar" in end["progress"]


@_requires_chrome
def test_served_an_ask_that_did_not_reach_the_run_log_says_so(ask_ended):
    end = ask_ended("not-logged")["end"]

    assert end["progress"].startswith("1 respuesta guardada · 0 resultados")
    assert "la pasada no quedó en runs.jsonl" in end["progress"]


@_requires_chrome
def test_served_the_page_can_stop_an_ask_and_what_was_paid_is_kept(ask_ended):
    seen = ask_ended("stop")

    assert (seen["job"]["state"], seen["job"]["reason"]) == ("interrupted", "cancelado")
    kept = seen["job"]["outcome"]["ok"]
    assert 1 <= kept < 3
    assert seen["end"]["progress"].startswith(
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
    await sWait(() => sPanel().shown && /de 5 posts/.test(sPanel().progress || ''), 'el panel de la pregunta en curso');
    return sPanel();
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
    resumed, end = ask_resumed["resumed"], ask_resumed["end"]

    assert resumed["title"] == "Pregunta en curso"
    assert re.match(r"^[0-4] de 5 posts · ", resumed["progress"])
    assert end["panel"]["progress"].startswith("5 respuestas guardadas · 3 resultados")
    assert [h["text"].split(" · ")[0] for h in end["history"]] == [QUERY]


def test_the_fake_answers_what_the_assertions_expect():
    """The results the browser tests read: probability order is not id order, and 1 and 5 tie."""
    assert ranked() == [("3", 0.97), ("1", 0.9), ("5", 0.9)]
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

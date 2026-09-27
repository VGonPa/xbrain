# tests/test_jev_page_panel_browser.py
"""The served page's job panel: what the 10b review left for PR 12 (VGonPa/xbrain#211).

* the probe harness has ONE deadline, so a failing step is named instead of running out
  Chrome's own timeout;
* Confirmar comes back only after a refusal that gives the confirmation back (400, or the
  503 «no se pudo arrancar»), never after the 503 of a server that is stopping;
* a job another tab started that ended badly opens the panel, not just a reload;
* the panel and the Preguntar tab fit a 375 px screen;
* «fuera del registro» going from >0 to 0 hides its line;
* a Confirmar refused with 409 because another tab's job is running follows that job.

Every client is a fake; nothing reaches TypeSafe.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.test_jev_page_browser import (
    _SERVE_JS,
    _page_saw,
    _requires_chrome,
    _served_dump,
)
from xbrain.jev.errors import ServeError

# --------------------------------------------------------------------------- the probe's deadline

_DEADLINE_PROBE = (
    "<script>"
    + _SERVE_JS.replace("/*BUDGET*/75000", "/*BUDGET*/2000")
    + r"""
(async () => {
  for (const name of ['uno', 'dos', 'tres', 'cuatro']) {
    await sStep(name, async () => { await sWait(() => false, 'lo imposible ' + name); return 'nunca'; });
  }
  sDone();
})();
</script>"""
)


@_requires_chrome
def test_a_probe_that_cannot_finish_names_every_step_before_chromes_timeout(tmp_path):
    from tests.test_jev_serve import _Recorder

    started = time.monotonic()
    seen = _served_dump(tmp_path, _DEADLINE_PROBE, client=_Recorder())
    took = time.monotonic() - started

    assert seen["uno"] == "ERROR nunca (se acabó el plazo del probe): lo imposible uno"
    for name in ("dos", "tres", "cuatro"):
        assert seen[name] == f"ERROR nunca (se acabó el plazo del probe): lo imposible {name}"
    # One 2 s budget for the four waits, not 4 × 30 s: well inside Chrome's 120 s.
    assert took < 60


# --------------------------------------------------------------------------- which 503 gives it back

_REFUSED_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('estimate', async () => {
    await sWait(() => seen(sCard('3')), 'la tarjeta 3');
    sPress(sCard('3'), 'Evaluar este post');
    await sWait(() => sPanel().go, 'la estimación');
    return sPanel();
  });
  await sStep('not_started', async () => {
    sId('ask-go').click();
    await sWait(() => (sPanel().error || '').includes('no se pudo arrancar'), 'la negativa al arrancar');
    for (let i = 0; i < 3; i++) await sFetch0.call(window, '/probe-wait');
    return sPanel();
  });
  await sStep('stopping', async () => {
    sId('ask-go').click();
    await sWait(() => (sPanel().error || '').includes('parando'), 'la negativa del servidor que para');
    for (let i = 0; i < 3; i++) await sFetch0.call(window, '/probe-wait');
    return sPanel();
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def refused(tmp_path_factory) -> dict[str, Any]:
    from tests.test_jev_serve import _Recorder
    from xbrain.jev.service import JevService

    class _Refusing(JevService):
        """The first Confirmar: the thread could not start (the confirmation is given back).
        The second: the server is stopping (it never will run)."""

        calls = 0

        def evaluate(self, kind: str, body: Any) -> dict[str, Any]:
            _Refusing.calls += 1
            if _Refusing.calls == 1:
                raise ServeError(503, "no se pudo arrancar el trabajo: sin hilos")
            raise ServeError(503, "el servidor se está parando")

    client = _Recorder()
    seen = _served_dump(
        tmp_path_factory.mktemp("refused"), _REFUSED_PROBE, client=client, base=_Refusing
    )
    seen["asked"] = client.asked
    return seen


@_requires_chrome
def test_confirm_comes_back_after_a_job_that_could_not_start(refused):
    panel = refused["not_started"]

    assert panel["error"] == "No se puede evaluar: no se pudo arrancar el trabajo: sin hilos"
    assert panel["go"] is True


@_requires_chrome
def test_confirm_stays_off_while_the_server_is_stopping(refused):
    panel = refused["stopping"]

    assert panel["error"] == "No se puede evaluar: el servidor se está parando"
    assert panel["go"] is False
    assert refused["asked"] == []


# --------------------------------------------------------------------------- a bad end elsewhere

_ELSEWHERE_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('opened', async () => {
    await sWait(() => sPanel().shown, 'el panel del trabajo que acabó mal en otra pestaña');
    await sWait(() => sRefreshed > 0, 'la recarga');
    return sPanel();
  });
  sDone();
})();
</script>"""
)


#: An ask over the posts with a topics answer (1 and 2): asked twice, the second is free.
_FREE = {"query": "hooks", "only_evaluated": True}
#: What each kind of job asks when another tab launches it.
_BODIES = {"topics": {"ids": ["3"]}, "ask": {"query": "hooks", "limit": 1}}


def _http_job(service: Any, port: int, kind: str, body: dict[str, Any]) -> None:
    """Estimate and confirm a job over HTTP, as another tab would, and wait for its end."""
    import http.client
    import json

    from xbrain.jev.serve import TOKEN_HEADER

    headers = {
        "Content-Type": "application/json",
        TOKEN_HEADER: service.token,
        "Origin": f"http://127.0.0.1:{port}",
    }
    replies: list[dict[str, Any]] = []
    for action in ("estimate", "evaluate"):
        sent = dict(body)
        if replies:
            sent["confirm_token"] = replies[-1]["confirm_token"]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", f"/api/{kind}/{action}", body=json.dumps(sent), headers=headers)
        replies.append(json.loads(conn.getresponse().read()))
        conn.close()
    service.wait(10)


def _failed_elsewhere(kind: str) -> Any:
    """Another tab ran a job of `kind` that failed (no key) — after this page's data was built,
    and without writing a file, so the data the page loads is still the one from before it."""

    def before(service: Any, port: int) -> None:
        service.blob()
        _http_job(service, port, kind, _BODIES[kind])
        assert service.job_view()["state"] == "error", service.job_view()

    return before


@pytest.fixture(scope="module", params=["topics", "ask"])
def elsewhere(request, tmp_path_factory) -> dict[str, Any]:
    from xbrain.jev.client import JevError

    def _no_key() -> Any:
        raise JevError("TYPESAFE_API_KEY no encontrada")

    seen = _served_dump(
        tmp_path_factory.mktemp("elsewhere-" + request.param),
        _ELSEWHERE_PROBE,
        make_client=_no_key,
        before_dump=_failed_elsewhere(request.param),
    )
    seen["kind"] = request.param
    return seen


@_requires_chrome
def test_a_job_that_failed_in_another_tab_opens_the_panel_here(elsewhere):
    panel = elsewhere["opened"]

    assert (
        panel["title"]
        == {
            "topics": "Una evaluación de topics lanzada en otra pestaña",
            "ask": "Una pregunta lanzada en otra pestaña",
        }[elsewhere["kind"]]
    )
    assert panel["error"] == "El trabajo falló: TYPESAFE_API_KEY no encontrada"
    assert panel["close"] == "Cerrar" and panel["stop"] is None and panel["go"] is None


# --------------------------------------------------------------------------- 375 px

_NARROW_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
if (window.top === window) (async () => {
  const frame = document.createElement('iframe');
  frame.style.cssText = 'width:375px;height:760px;border:0;position:absolute;top:0;left:0';
  frame.src = '/#posts?f=all';
  document.body.appendChild(frame);
  // Inside a box that scrolls or clips sideways (the tab bar), an element is that box's
  // business: the box itself is measured like everything else.
  const clipped = (w, e) => {
    for (let a = e.parentElement; a && a !== w.document.body; a = a.parentElement) {
      if (['auto', 'scroll', 'hidden'].includes(w.getComputedStyle(a).overflowX)) return true;
    }
    return false;
  };
  const measure = (w, root) => {
    const doc = w.document;
    const over = [...root.querySelectorAll('*')].filter(e => e.getClientRects().length && !clipped(w, e))
      .filter(e => { const r = e.getBoundingClientRect(); return r.right > 375.5 || r.left < -0.5; })
      .map(e => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (e.className ? '.' + String(e.className).split(' ')[0] : ''));
    return {page: doc.documentElement.scrollWidth, over: over.slice(0, 8)};
  };
  await sStep('panel', async () => {
    await sWait(() => frame.contentWindow && frame.contentWindow.document.getElementById('post-3'), 'la página a 375 px');
    const w = frame.contentWindow;
    const b = [...w.document.querySelectorAll('#post-3 button.evalb')].find(x => x.textContent === 'Evaluar este post');
    b.click();
    await sWait(() => { const g = w.document.getElementById('ask-go'); return g && !g.disabled; }, 'la estimación a 375 px');
    return measure(w, w.document.getElementById('jobp'));
  });
  await sStep('posts', async () => measure(frame.contentWindow, frame.contentWindow.document.body));
  await sStep('ask', async () => {
    const w = frame.contentWindow;
    w.document.getElementById('ask-cancel').click();
    w.location.hash = '#ask';
    await sWait(() => w.document.getElementById('ask-form'), 'el formulario a 375 px');
    w.openAsk('ask', {query: 'hooks'}, 'Preguntar: «' + 'una pregunta larga '.repeat(12) + '»', null);
    await sWait(() => { const g = w.document.getElementById('ask-go'); return g && !g.disabled; }, 'la estimación de la pregunta');
    const main = w.document.querySelector('.askmain').getBoundingClientRect();
    const side = w.document.querySelector('.askside').getBoundingClientRect();
    return Object.assign(measure(w, w.document.body), {width: w.innerWidth,
      main: Math.round(main.width), side_below: side.top >= main.bottom - 1});
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def narrow(tmp_path_factory) -> dict[str, Any]:
    from tests.test_jev_serve import _Recorder

    return _served_dump(tmp_path_factory.mktemp("narrow"), _NARROW_PROBE, client=_Recorder())


@_requires_chrome
@pytest.mark.parametrize("view", ["panel", "posts", "ask"])
def test_the_panel_and_the_tabs_fit_a_375_px_screen(narrow, view):
    measured = narrow[view]

    assert isinstance(measured, dict), measured
    assert measured["page"] <= 375, measured
    assert measured["over"] == [], measured
    if view == "ask":
        # One column: the results use the width, the history goes below them.
        assert measured["main"] >= 300 and measured["side_below"] is True, measured


# --------------------------------------------------------------------------- «fuera del registro»

_OUT_OF_LOG_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('before', async () => {
    await sWait(() => seen(sCard('1')), 'la tarjeta 1');
    return {line: txt(sId('before-log')), n: DATA.cost.out_of_log.assessments};
  });
  await sStep('after', async () => {
    sPress(sCard('1'), 'Re-evaluar');
    await sWait(() => (sPanel().error || '').includes('nada que evaluar'), 'la negativa sin forzar');
    sId('ask-force').click();
    await sWait(() => sPanel().go, 'la estimación forzada');
    sId('ask-go').click();
    await sWait(() => sRefreshed > 0, 'la recarga');
    return {line: txt(sId('before-log')), hidden: sId('before-log').hidden, n: DATA.cost.out_of_log.assessments};
  });
  sDone();
})();
</script>"""
)


def _only_post_1_out_of_the_log(cfg: Any) -> None:
    """Post 1's answer is the only one, and no pass logged it: one answer out of the log."""
    from xbrain.jev.store import load_assessments, save_assessments

    kept = {k: v for k, v in load_assessments(cfg.jev_topics_path).items() if k == "1"}
    save_assessments(kept, cfg.jev_topics_path)
    cfg.jev_runs_path.unlink()


@pytest.fixture(scope="module")
def out_of_log(tmp_path_factory) -> dict[str, Any]:
    from tests.test_jev_serve import _Recorder

    return _served_dump(
        tmp_path_factory.mktemp("out-of-log"),
        _OUT_OF_LOG_PROBE,
        client=_Recorder(),
        prepare=_only_post_1_out_of_the_log,
    )


@_requires_chrome
def test_the_out_of_log_line_goes_when_its_last_answer_is_logged(out_of_log):
    before, after = out_of_log["before"], out_of_log["after"]

    assert before["n"] == 1 and "1 evaluación" in before["line"]
    assert after["n"] == 0
    assert after["line"] is None and after["hidden"] is True


# --------------------------------------------------------------------------- 409: follow it

_FOLLOW_409_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('estimate', async () => {
    await sWait(() => seen(sCard('3')), 'la tarjeta 3');
    // No watch from here on: only Confirmar's own 409 path can find the other job.
    watch = function () {};
    sPress(sCard('3'), 'Evaluar este post');
    await sWait(() => sPanel().go, 'la estimación');
    return sPanel();
  });
  await sStep('other', async () => {
    const h = {'Content-Type': 'application/json', 'X-Xbrain-Token': DATA.serve.token};
    const e = await (await sFetch0('/api/topics/estimate', {method: 'POST', headers: h, body: JSON.stringify({ids: ['4']})})).json();
    const r = await sFetch0('/api/topics/evaluate', {method: 'POST', headers: h,
      body: JSON.stringify({ids: ['4'], confirm_token: e.confirm_token})});
    return r.status;
  });
  await sStep('followed', async () => {
    sId('ask-go').click();
    await sWait(() => sPanel().title === 'Evaluación de topics en curso', 'el trabajo de la otra pestaña');
    const during = sPanel();
    await sWait(() => sRefreshed > 0, 'su final');
    return {during, after: sPanel(), card4: sButtons(sCard('4'))};
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def follow_409(tmp_path_factory) -> dict[str, Any]:
    from tests.test_jev_serve import _Recorder

    watched, saw = _page_saw()

    class _Slow4(_Recorder):
        def ask(self, state, questions):
            if state["post"].startswith("Series A"):
                saw.wait(60)
            return super().ask(state, questions)

    client = _Slow4()
    seen = _served_dump(
        tmp_path_factory.mktemp("follow-409"), _FOLLOW_409_PROBE, client=client, base=watched
    )
    seen["asked"] = client.asked
    return seen


@_requires_chrome
def test_a_confirm_refused_because_another_job_runs_follows_that_job(follow_409):
    followed = follow_409["followed"]

    assert follow_409["other"] == 202
    assert followed["during"]["progress"].startswith("0 de 1 post · ")
    assert followed["after"]["progress"].startswith("1 evaluación guardada")
    assert followed["card4"] == ["Re-evaluar"]
    assert follow_409["asked"] == ["4"]


# --------------------------------------------------------------------------- a clean end elsewhere

_FREE_ELSEWHERE_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('free', async () => {
    await sWait(() => seen(sCard('3')), 'la página');
    const h = {'Content-Type': 'application/json', 'X-Xbrain-Token': DATA.serve.token};
    const body = {query: 'hooks', only_evaluated: true};
    const e = await (await sFetch0('/api/ask/estimate', {method: 'POST', headers: h, body: JSON.stringify(body)})).json();
    await sFetch0('/api/ask/evaluate', {method: 'POST', headers: h,
      body: JSON.stringify(Object.assign({}, body, {confirm_token: e.confirm_token}))});
    for (let i = 0; i < 100; i++) {
      if ((await (await sFetch0('/api/job')).json()).state !== 'running') break;
    }
    // Idle, on the page's own clock: a request always in flight (as `sWait`'s) holds Chrome's
    // virtual time still, and the watch that must see this job runs on it.
    await new Promise(r => setTimeout(r, 2 * WATCH_MS));
    await sWait(() => sRefreshed > 0, 'la recarga');
    for (let i = 0; i < 3; i++) await sFetch0.call(window, '/probe-wait');
    return Object.assign(sPanel(), {posts: e.posts, usd: e.usd});
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def free_elsewhere(tmp_path_factory) -> dict[str, Any]:
    """An ask answered before the page opened, then asked again from another tab: free."""
    from tests.test_jev_serve_ask import _Asker

    return _served_dump(
        tmp_path_factory.mktemp("free-elsewhere"),
        _FREE_ELSEWHERE_PROBE,
        client=_Asker(),
        before_dump=lambda service, port: _http_job(service, port, "ask", _FREE),
    )


@_requires_chrome
def test_a_free_ask_in_another_tab_reloads_here_without_opening_the_panel(free_elsewhere):
    seen = free_elsewhere["free"]

    assert (seen["posts"], seen["usd"]) == (0, 0.0)
    assert seen["shown"] is False
    job = free_elsewhere["job"]
    assert (job["kind"], job["outcome"]["sent"], job["outcome"]["recorded"]) == ("ask", 0, True)


# --------------------------------------------------------------------------- nothing kept

_NOTHING_KEPT_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('stopped', async () => {
    location.hash = '#ask';
    await sWait(() => seen(sId('ask-form')), 'el formulario');
    await sQuery(1);
    sId('ask-go').click();
    await sWait(() => sPanel().stop !== null, 'el trabajo');
    sPress(sId('jobp'), 'Parar (se guarda lo ya pagado)');
    await sWait(() => sRefreshed > 0, 'el final');
    for (let i = 0; i < 3; i++) await sFetch0.call(window, '/probe-wait');
    return Object.assign(sPanel(), {hash: location.hash,
      head: txt(sId('ask-head')), history: [...document.querySelectorAll('#ask-history li')].filter(seen).length});
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def nothing_kept(tmp_path_factory) -> dict[str, Any]:
    """A query already in the history; then an ask whose one call waits for «Parar» and fails:
    interrupted, nothing kept, nothing recorded."""
    import threading

    from tests.test_jev_serve_ask import _Asker
    from xbrain.jev.client import JevError
    from xbrain.jev.service import JevService

    stopped = threading.Event()

    class _Stoppable(JevService):
        def cancel_job(self) -> dict[str, Any]:
            view = super().cancel_job()
            stopped.set()
            return view

    class _FailsOnceStopped(_Asker):
        failing = False

        def ask(self, state, questions):
            if self.failing:
                assert stopped.wait(60)
                raise JevError("respuesta ilegible")
            return super().ask(state, questions)

    client = _FailsOnceStopped()

    def before(service: Any, port: int) -> None:
        _http_job(service, port, "ask", {"query": "una pregunta anterior", "limit": 2})
        client.failing = True

    return _served_dump(
        tmp_path_factory.mktemp("nothing-kept"),
        _NOTHING_KEPT_PROBE,
        client=client,
        base=_Stoppable,
        before_dump=before,
    )


@_requires_chrome
def test_an_ask_that_kept_nothing_stays_on_the_tab_and_says_it_was_not_recorded(nothing_kept):
    seen = nothing_kept["stopped"]
    job = nothing_kept["job"]

    assert (job["state"], job["outcome"]["ok"], job["outcome"]["recorded"]) == (
        "interrupted",
        0,
        False,
    )
    assert "no quedó en el historial: no se guardó ninguna respuesta" in seen["progress"]
    # No jump to a query the history does not have; the tab still shows the one it has.
    assert seen["hash"] == "#ask"
    assert seen["head"].startswith("«una pregunta anterior»") and seen["history"] == 1


# --------------------------------------------------------------------------- an idle page, lost

_IDLE_LOST_PROBE = (
    "<script>"
    + _SERVE_JS
    + r"""
(async () => {
  await sStep('lost', async () => {
    // Five failed looks, WATCH_MS apart, on the page's own clock (see the free-ask probe).
    await new Promise(r => setTimeout(r, (RETRY_MS.length + 0.5) * WATCH_MS));
    await sWait(() => sPanel().shown, 'el aviso de que se perdió el servidor');
    return sPanel();
  });
  await sStep('found', async () => {
    // The server answers again from the seventh look: the message goes.
    await new Promise(r => setTimeout(r, 3 * WATCH_MS));
    await sWait(() => !sPanel().shown, 'el aviso retirado');
    return sPanel();
  });
  sDone();
})();
</script>"""
)


@pytest.fixture(scope="module")
def idle_lost(tmp_path_factory) -> dict[str, Any]:
    from tests.test_jev_serve import _Recorder
    from xbrain.jev.service import JevService

    class _Down(JevService):
        """Down for the page's first six looks at `/api/job`, then back."""

        looks = 0

        def job_view(self) -> dict[str, Any]:
            _Down.looks += 1
            if _Down.looks <= 6:
                raise RuntimeError("se cayó")
            return super().job_view()

    seen = _served_dump(
        tmp_path_factory.mktemp("idle-lost"), _IDLE_LOST_PROBE, client=_Recorder(), base=_Down
    )
    return seen


@_requires_chrome
def test_an_idle_page_says_when_it_lost_the_server(idle_lost):
    panel = idle_lost["lost"]

    assert panel["title"] == "Sin contacto con el servidor"
    assert panel["error"] == "Se perdió el contacto con el servidor; recarga la página."
    assert panel["reload"] == "Recargar la página"
    assert idle_lost["found"]["shown"] is False

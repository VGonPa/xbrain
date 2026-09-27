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
from xbrain.jev.picks import ServeError

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


def _failed_elsewhere(service: Any, port: int) -> None:
    """Another tab ran a job that failed (no key) — after this page's data was built, and
    without writing a file, so the data the page loads is still the one from before it."""
    import http.client
    import json

    from xbrain.jev.serve import TOKEN_HEADER

    service.blob()
    headers = {
        "Content-Type": "application/json",
        TOKEN_HEADER: service.token,
        "Origin": f"http://127.0.0.1:{port}",
    }
    replies: list[dict[str, Any]] = []
    for path in ("/api/topics/estimate", "/api/topics/evaluate"):
        body: dict[str, Any] = {"ids": ["3"]}
        if replies:
            body["confirm_token"] = replies[-1]["confirm_token"]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", path, body=json.dumps(body), headers=headers)
        replies.append(json.loads(conn.getresponse().read()))
        conn.close()
    service.wait(10)
    assert service.job_view()["state"] == "error", service.job_view()


@pytest.fixture(scope="module")
def elsewhere(tmp_path_factory) -> dict[str, Any]:
    from xbrain.jev.client import JevError

    def _no_key() -> Any:
        raise JevError("TYPESAFE_API_KEY no encontrada")

    return _served_dump(
        tmp_path_factory.mktemp("elsewhere"),
        _ELSEWHERE_PROBE,
        make_client=_no_key,
        before_dump=_failed_elsewhere,
    )


@_requires_chrome
def test_a_job_that_failed_in_another_tab_opens_the_panel_here(elsewhere):
    panel = elsewhere["opened"]

    assert panel["title"] == "Una evaluación de topics lanzada en otra pestaña"
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
    return Object.assign(measure(w, w.document.body), {width: w.innerWidth});
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

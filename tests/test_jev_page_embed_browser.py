# tests/test_jev_page_embed_browser.py
"""Every post card shows the post through X's own embed, with the local copy one click away.

The embed is a plain `<iframe>` to X's `Tweet.html` — never `widgets.js`, which would run X's
script in the page's origin, where a served page keeps its spending token. These tests drive
the page in headless Chrome WITH NO NETWORK (`_NO_NETWORK` maps every host but localhost to
NOTFOUND), so X never answers: what X would send is handed to the page's own message handler
(`onXMessage`) with X's origin and the frame's window as the source — the same object a real
`message` event carries.

IN REAL TIME, not under `--virtual-time-budget` like the other page tests: frames are made by
an IntersectionObserver, and virtual time starves the rendering steps that run it (measured:
no frame in 1 s of virtual time after scrolling to the list). So the page is served from a
throwaway local server and the probe POSTs what it saw back to it; the wait for X costs real
seconds (~9 s, once). The real message shape was read off a live embed and is recorded in
ARCHITECTURE.md (jev · The X embed).
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 - runs a local browser binary on a page this test served
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.test_jev_dashboard import _assessment, _data, _item
from tests.test_jev_page_browser import (
    _NO_NETWORK,
    _SEEN,
    CHROME,
    _need_chrome,
    _page,
    _requires_chrome,
)

X_EMBED = "https://platform.twitter.com/embed/Tweet.html"
SANDBOX = "allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"

#: Three posts with X ids (all compared, so the default view lists them) and one whose id X
#: cannot have.
IDS = ("1874382656751771697", "1874800307672420428", "1875187283873206455")
BAD = "no-es-de-x"
#: A post whose frame says it has the post and then nothing (off screen, X lays out later).
QUIET = "1875225613952544936"


def _fixture() -> dict[str, Any]:
    items = [_item(i, text=f"texto local de {i}") for i in (*IDS, QUIET, BAD)]
    return _data(items, {it.id: _assessment(it, membership={"ai-coding": 0.9}) for it in items})


#: Real seconds a probe may take (the one that waits out X's 8 s runs ~10 s).
_DEADLINE_S = 60


def _live(page: Path, suffix: str, height: int = 600) -> dict[str, Any]:
    """Serves `page`'s folder on a free local port, opens `jev.html` + `suffix` in headless
    Chrome in real time with no network (a window `height` px tall), and returns what the
    probe POSTed to /probe-done."""
    assert CHROME is not None
    got: dict[str, str] = {}

    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(page.parent), **kwargs)

        def do_POST(self) -> None:  # noqa: N802 — the stdlib's name
            got["out"] = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = tempfile.mkdtemp(prefix="xbrain-embed-")
    url = f"http://127.0.0.1:{server.server_address[1]}/{page.name}{suffix}"
    chrome = subprocess.Popen(  # nosec B603 - fixed argv, a local server this test made
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--window-size=1280,{height}",
            _NO_NETWORK,
            f"--user-data-dir={profile}",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + _DEADLINE_S
    try:
        while "out" not in got and chrome.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        chrome.kill()
        _, stderr = chrome.communicate()
        server.shutdown()
        server.server_close()
        shutil.rmtree(profile, ignore_errors=True)
    assert "out" in got, f"el probe no terminó (rc={chrome.returncode}): {stderr[-800:]}"
    return json.loads(got["out"])


#: Reads one card: its frame (if any), whether its local copy is on screen, its note and toggle.
_EMBED_JS = r"""
const sleep = ms => new Promise(r => setTimeout(r, ms));
/** Waits (up to `ms`) for `ok()`: the frames are made by an IntersectionObserver, a frame later. */
const until = async (ok, ms) => { for (let t = 0; t < (ms || 3000) && !ok(); t += 50) await sleep(50); return ok(); };
const cardOf = (id, root) => (root || document).querySelector('.card[data-id="' + id + '"]');
const frameOf = (id, root) => { const c = cardOf(id, root); return c && c.querySelector('iframe'); };
const sCard = (id, root) => {
  const c = cardOf(id, root);
  const f = c && c.querySelector('iframe');
  return {
    frame: f ? {src: f.getAttribute('src'), sandbox: f.getAttribute('sandbox'),
      referrerpolicy: f.getAttribute('referrerpolicy'), loading: f.getAttribute('loading'),
      title: f.getAttribute('title'), height: Math.round(f.getBoundingClientRect().height),
      seen: seen(f)} : null,
    local: txt(c.querySelector('.twt')),
    note: txt(c.querySelector('.xnote')),
    toggle: txt(c.querySelector('.vtog')),
    jev: seen(c.querySelector('.jev')),
  };
};
const xSays = (id, method, params) => onXMessage({origin: 'https://platform.twitter.com',
  source: frameOf(id).contentWindow, data: {'twttr.embed': {jsonrpc: '2.0', method: method, id: 'embed-0', params: params}}});
const done = (out) => fetch('/probe-done', {method: 'POST', body: JSON.stringify(out)});
"""

_PROBE = (
    "<script>"
    + _SEEN
    + _EMBED_JS
    + r"""
(async () => {
  const [A, B, C] = IDS_JSON;
  const Q = QUIET_ID;
  const out = {};
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  await sleep(100);
  // The list starts below the header: frames are made only near the viewport.
  document.getElementById('cards').scrollIntoView();
  await until(() => frameOf(A) && frameOf(B) && frameOf(C));
  await step('theme', async () => matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  await step('start', async () => ({a: sCard(A), b: sCard(B), c: sCard(C), bad: sCard(BAD_ID),
    scripts: [...document.scripts].map(s => s.src).filter(Boolean)}));
  await step('foreign', async () => {
    const before = sCard(A).frame.height;
    // A foreign origin, and a message this page posts to itself (origin "null" from a file).
    onXMessage({origin: 'https://evil.example', source: frameOf(A).contentWindow,
      data: {'twttr.embed': {method: 'twttr.private.resize', params: [{height: 777}]}}});
    window.postMessage({'twttr.embed': {method: 'twttr.private.resize', params: [{height: 778}]}}, '*');
    await sleep(30);
    return {before: before, after: sCard(A).frame.height};
  });
  await step('resize', async () => {
    const other = sCard(B).frame.height;
    xSays(A, 'twttr.private.resize', [{width: 550, height: 640, data: {tweet_id: A}}]);
    await sleep(30);
    return {a: sCard(A).frame.height, b_before: other, b: sCard(B).frame.height};
  });
  await step('quiet', async () => { await until(() => frameOf(Q)); xSays(Q, 'twttr.private.initialized', [{data: {tweet_id: Q}}]); return !!frameOf(Q); });
  await step('no_results', async () => { xSays(C, 'twttr.private.no_results', [{data: {tweet_id: C}}]); await sleep(30); return sCard(C); });
  await step('keys', async () => {
    document.dispatchEvent(new KeyboardEvent('keydown', {key: 'j', bubbles: true}));
    await sleep(30);
    const a = document.activeElement;
    return {tag: a.tagName, card: a.classList.contains('card'), id: a.dataset.id, frame: !!a.querySelector('.xembed iframe')};
  });
  // X never answers B: past the wait it falls back to the local copy; A (resized) stays.
  await step('timeout', async () => { await sleep(9000); return {a: sCard(A), b: sCard(B), q: sCard(Q)}; });
  await step('retry', async () => { cardOf(B).querySelector('.vtog').click(); await sleep(30); return sCard(B); });
  await step('to_local', async () => { cardOf(A).querySelector('.vtog').click(); await sleep(30); return sCard(A); });
  await step('to_x', async () => { cardOf(A).querySelector('.vtog').click(); await sleep(30); return sCard(A); });
  await step('switch_local', async () => {
    document.querySelector('#vista button[data-vista="local"]').click();
    await sleep(30);
    let stored; try { stored = localStorage.getItem('xbrain.jev.vista'); } catch (e) { stored = 'ERROR'; }
    return {cards: [A, B, C, BAD_ID].map(id => sCard(id)), stored: stored, read: readViewMode(),
      pressed: document.querySelector('#vista button[data-vista="local"]').getAttribute('aria-pressed')};
  });
  await step('switch_x', async () => {
    document.querySelector('#vista button[data-vista="x"]').click();
    await sleep(30);
    let stored; try { stored = localStorage.getItem('xbrain.jev.vista'); } catch (e) { stored = 'ERROR'; }
    return {a: sCard(A), stored: stored};
  });
  done(out);
})();
</script>"""
)


def _probe(ids: tuple[str, ...], bad: str) -> str:
    return (
        _PROBE.replace("IDS_JSON", json.dumps(list(ids)))
        .replace("BAD_ID", json.dumps(bad))
        .replace("QUIET_ID", json.dumps(QUIET))
    )


@pytest.fixture(scope="module")
def embedded(tmp_path_factory) -> dict[str, Any]:
    _need_chrome()
    page = _page(tmp_path_factory.mktemp("embed"), _fixture(), _probe(IDS, BAD))
    # Tall enough that all five cards are near the viewport once the list is scrolled to.
    return _live(page, "#posts?f=all", height=1600)


def _src(post_id: str, theme: str) -> str:
    return f"{X_EMBED}?id={post_id}&dnt=true&theme={theme}&lang=es"


@_requires_chrome
def test_a_post_with_an_x_id_is_xs_own_embed_in_a_sandboxed_frame(embedded):
    theme = embedded["theme"]
    a = embedded["start"]["a"]

    assert theme in ("light", "dark")
    assert a["frame"]["src"] == _src(IDS[0], theme)
    assert a["frame"]["sandbox"] == SANDBOX
    assert a["frame"]["referrerpolicy"] == "no-referrer"
    assert a["frame"]["loading"] == "lazy"
    assert a["frame"]["title"] == "Post de @alice en X"
    assert a["frame"]["seen"] is True
    # X's frame, never X's script in this page's origin.
    assert embedded["start"]["scripts"] == []
    # The local copy is not drawn beside it; the Jev block stays under it.
    assert a["local"] is None
    assert a["jev"] is True
    assert a["toggle"] == "ver versión local"


@_requires_chrome
def test_a_post_whose_id_x_cannot_have_is_the_local_card_and_says_why(embedded):
    bad = embedded["start"]["bad"]

    assert bad["frame"] is None
    assert bad["local"] == f"texto local de {BAD}"
    assert (
        bad["note"] == "Sin vista de X: el id de este post no es de X. Esta es la copia guardada."
    )
    assert bad["toggle"] is None


@_requires_chrome
def test_a_message_from_another_origin_does_not_resize_a_frame(embedded):
    foreign = embedded["foreign"]

    assert foreign["after"] == foreign["before"]
    assert foreign["before"] == 220  # the reserved height, until X says its own


@_requires_chrome
def test_xs_resize_sets_the_height_of_its_own_frame_only(embedded):
    resize = embedded["resize"]

    assert resize["a"] == 640
    assert resize["b"] == resize["b_before"] == 220


@_requires_chrome
def test_a_post_x_has_not_got_falls_back_to_the_local_card(embedded):
    c = embedded["no_results"]

    assert c["frame"] is None
    assert c["local"] == f"texto local de {IDS[2]}"
    assert c["note"] == (
        "X no tiene este post (borrado, protegido o de una cuenta suspendida). "
        "Esta es la copia guardada."
    )
    assert c["toggle"] == "ver en X"


@_requires_chrome
def test_a_frame_x_never_answers_falls_back_after_the_wait(embedded):
    timeout = embedded["timeout"]

    assert timeout["b"]["frame"] is None
    assert timeout["b"]["local"] == f"texto local de {IDS[1]}"
    assert timeout["b"]["note"] == ("X no respondió en 8 s (¿sin red?). Esta es la copia guardada.")
    # A frame that said anything is X answering: it waits for its height, it does not fall back.
    assert embedded["quiet"] is True
    assert timeout["q"]["frame"]["src"] == _src(QUIET, embedded["theme"])
    assert timeout["q"]["frame"]["height"] == 220
    assert timeout["q"]["note"] is None
    # A frame X did answer is left alone.
    assert timeout["a"]["frame"]["src"] == _src(IDS[0], embedded["theme"])
    assert timeout["a"]["frame"]["height"] == 640


@_requires_chrome
def test_ver_en_x_retries_a_card_that_fell_back(embedded):
    retry = embedded["retry"]

    assert retry["frame"]["src"] == _src(IDS[1], embedded["theme"])
    assert retry["local"] is None
    assert retry["note"] is None


@_requires_chrome
def test_the_per_card_toggle_goes_both_ways(embedded):
    to_local, to_x = embedded["to_local"], embedded["to_x"]

    assert to_local["frame"] is None
    assert to_local["local"] == f"texto local de {IDS[0]}"
    assert to_local["note"] is None  # the reader chose it: nothing to explain
    assert to_local["toggle"] == "ver en X"
    assert to_x["frame"]["src"] == _src(IDS[0], embedded["theme"])
    assert to_x["local"] is None
    assert to_x["toggle"] == "ver versión local"


@_requires_chrome
def test_j_focuses_the_card_not_the_frame(embedded):
    keys = embedded["keys"]

    assert keys == {"tag": "ARTICLE", "card": True, "id": IDS[0], "frame": True}


@_requires_chrome
def test_the_vista_switch_turns_every_card_local_and_is_remembered(embedded):
    local, back = embedded["switch_local"], embedded["switch_x"]

    assert [c["frame"] for c in local["cards"]] == [None] * 4
    assert [c["local"] for c in local["cards"]] == [f"texto local de {i}" for i in (*IDS, BAD)]
    assert local["stored"] == "local"
    assert local["read"] == "local"
    assert local["pressed"] == "true"
    assert back["stored"] == "x"
    assert back["a"]["frame"]["src"] == _src(IDS[0], embedded["theme"])


_OFF_PROBE = (
    "<script>"
    + _SEEN
    + _EMBED_JS
    + r"""
(async () => {
  await sleep(150);
  done({cards: IDS_JSON.map(id => sCard(id)), frames: document.querySelectorAll('iframe').length,
    pressed: document.querySelector('#vista button[data-vista="local"]').getAttribute('aria-pressed')});
})();
</script>"""
)


@_requires_chrome
def test_embed_0_in_the_url_shows_every_card_local(tmp_path):
    _need_chrome()
    probe = _OFF_PROBE.replace("IDS_JSON", json.dumps(list(IDS)))
    page = _page(tmp_path, _fixture(), probe)
    seen = _live(page, "?embed=0#posts?f=all")

    assert seen["frames"] == 0
    assert [c["local"] for c in seen["cards"]] == [f"texto local de {i}" for i in IDS]
    assert [c["toggle"] for c in seen["cards"]] == ["ver en X"] * 3
    assert seen["pressed"] == "true"


_LAZY_PROBE = (
    "<script>"
    + _SEEN
    + _EMBED_JS
    + r"""
(async () => {
  await sleep(150);
  const frames = () => document.querySelectorAll('#cards iframe').length;
  const out = {cards: document.querySelectorAll('#cards .card').length};
  document.getElementById('cards').scrollIntoView();
  await until(() => frames() > 0);
  await sleep(300);
  out.at_top = frames();
  document.querySelector('#cards .card:nth-child(50)').scrollIntoView();
  await until(() => !!document.querySelector('#cards .card:nth-child(50) iframe'));
  await sleep(300);
  out.at_bottom = frames();
  out.last_has_frame = !!document.querySelector('#cards .card:nth-child(50) iframe');
  done(out);
})();
</script>"""
)


@_requires_chrome
def test_only_frames_near_the_viewport_are_created(tmp_path):
    _need_chrome()
    ids = [str(10**18 + n) for n in range(60)]
    items = [_item(i) for i in ids]
    data = _data(items, {it.id: _assessment(it, membership={"ai-coding": 0.9}) for it in items})
    seen = _live(_page(tmp_path, data, _LAZY_PROBE), "#posts?f=all")

    assert seen["cards"] == 50
    assert 1 <= seen["at_top"] <= 8
    assert seen["last_has_frame"] is True
    assert seen["at_bottom"] <= 16


def test_the_page_tests_run_without_network():
    """Fail-closed: every page test maps every host but localhost to NOTFOUND, so a test can
    never pass on something X (or a font server) sent."""
    import inspect

    from tests import test_jev_page_browser as browser

    assert "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1" in browser._NO_NETWORK
    assert "_NO_NETWORK" in inspect.getsource(browser._dump)
    assert "_NO_NETWORK" in inspect.getsource(browser._dump_served)


def test_the_frame_url_is_built_in_one_place_and_widgets_js_is_never_loaded():
    """The frame's `src` is set once, from `xSrc` (X's origin + a validated id); nothing in the
    template loads X's `widgets.js`, which would run in the page's origin next to the token."""
    from tests.test_jev_dashboard import _script_section
    from xbrain.dashboard import _resource

    template = _resource("jev.template.html")
    embed = _script_section(template, "/* embed */", "/* end embed */")

    assert "widgets.js" not in template.replace("NEVER `widgets.js`", "")
    assert "platform.twitter.com/widgets" not in template
    assert embed.count("'src'") == 1
    assert "f.setAttribute('src', xSrc(slot.dataset.id));" in embed
    assert "if (slot.firstChild || !X_ID.test(slot.dataset.id)) return;" in embed
    assert "const X_ID = /^\\d{1,25}$/;" in embed
    assert "if (e.origin !== X_ORIGIN || !e.source) return;" in embed

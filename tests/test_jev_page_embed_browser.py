# tests/test_jev_page_embed_browser.py
"""Every post card shows the post through X's own embed, with the saved copy one click away.

The embed is a plain `<iframe>` to X's `Tweet.html` — never `widgets.js`, which would run X's
script in the page's origin, where a served page keeps its spending token. These tests drive
the page in headless Chrome WITH NO NETWORK (`_NO_NETWORK` maps every host but localhost to
NOTFOUND), so X never answers: what X would send is dispatched as a real `message` event on
the page's window, with X's origin and the frame's window as its source — so it goes through
the page's own listener, exactly as X's `postMessage` would.

IN REAL TIME, not under `--virtual-time-budget` like the other page tests: frames are made by
an IntersectionObserver, and virtual time starves the rendering steps that run it (measured:
no frame in 1 s of virtual time after scrolling to the list). So the page is served from a
throwaway local server and the probe POSTs what it saw back to it. Every wait is a condition
with a deadline; the only waits measured in seconds are X's own 8 s (a frame that never
answers, and a frame that never loads). The real message shape was read off a live embed and
is recorded in ARCHITECTURE.md (jev · The X embed).
"""

from __future__ import annotations

import inspect
import json
import shutil
import socket
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
from xbrain.jev.dashboard import PAGE_ARTICLE_CHARS
from xbrain.models import Content, ContentSourceSuccess, Link

X_EMBED = "https://platform.twitter.com/embed/Tweet.html"
SANDBOX = "allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"
SAVED = "se muestra la copia guardada"

#: Three posts with X ids (all compared, so the default view lists them).
IDS = ("1874382656751771697", "1874800307672420428", "1875187283873206455")
#: Ids X cannot have. Each would pass a check that is unanchored (`123abc`, `12?x=1`,
#: `1/../evil`), unbounded (26 digits) or looser than digits (`no-es-de-x` only fails `\d`).
BAD = "no-es-de-x"
ODD = ("123abc", "12?x=1", "1" * 26, "1/../evil")
#: A post whose frame says it has the post and then nothing (off screen, X lays out later).
QUIET = "1875225613952544936"
#: A post that links an X Article: X's embed of it is the bare link, so it opens saved.
ART = "1876000000000000001"


#: The Article's body as fetched: longer than the page's cap, in paragraphs — the first one
#: carrying markup, which is text from X and must stay text (the served page holds the token).
ART_URL = "http://x.com/i/article/1909295899274039296"
ART_MARKUP = '<img src=x onerror="window.xPwned=1"><script>window.xPwned=2</script> &amp; <b>b</b>'
ART_BODY = "\n\n".join(
    [ART_MARKUP] + [f"Párrafo {n} del artículo, con su frase. " * 12 for n in range(8)]
)


def _fixture() -> dict[str, Any]:
    items = [_item(i, text=f"texto local de {i}") for i in (*IDS, QUIET, ART, BAD, *ODD)]
    items[4].links = [Link(url=ART_URL, domain="x.com")]
    items[4].content = Content(
        fetched_at=items[4].created_at,
        sources=[ContentSourceSuccess(kind="x_article", url=ART_URL, title="Tít", text=ART_BODY)],
    )
    data = _data(items, {it.id: _assessment(it, membership={"ai-coding": 0.9}) for it in items})
    # The vault's notes, so the Article's saved copy can point at its note.
    data["notes_dir"] = "/vault/x"
    next(p for p in data["posts"] if p["id"] == ART)["note"] = "a.md"
    return data


#: Real seconds a probe may take (the one that waits out X's 8 s runs ~12 s; a busy runner
#: has taken four times as long).
_DEADLINE_S = 120

#: A test hook in the page's <head>, before its script: `matchMedia` for the colour scheme
#: answers from an object the probe can flip, so a system theme change can be played.
_THEME_HOOK = r"""<head><script>
(() => {
  const real = window.matchMedia.bind(window);
  const q = '(prefers-color-scheme: light)';
  const fake = new EventTarget();
  fake.media = q;
  fake.matches = real(q).matches;
  window.xTestTheme = fake;
  window.matchMedia = (m) => (m === q ? fake : real(m));
})();
</script>"""


def _live(
    page: Path,
    suffix: str,
    *,
    height: int = 600,
    rules: str = _NO_NETWORK,
) -> dict[str, Any]:
    """Serves `page`'s folder on a free local port, opens `jev.html` + `suffix` in headless
    Chrome in real time with no network (a window `height` px tall; `rules` are the host
    resolver's), and returns what the probe POSTed to /probe-done."""
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
            rules,
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


#: Reads one card: its frame (if any), whether its saved copy is on screen, its note and toggle.
_EMBED_JS = r"""
const sleep = ms => new Promise(r => setTimeout(r, ms));
/** Waits (up to `ms`, 10 s by default) for `ok()`, and returns it: frames are made by an
    IntersectionObserver, a frame later — never a guessed sleep. */
const until = async (ok, ms) => { const end = Date.now() + (ms || 10000); while (!ok() && Date.now() < end) await sleep(25); return ok(); };
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
    slot: !!(c && c.querySelector('.xembed')),
    // The saved copy as the card's view (not the one standing in, in X's slot, until X answers).
    local: txt([...c.querySelectorAll('.twt')].find(t => !t.closest('.xembed')) || null),
    wait: txt(c.querySelector('.xembed > .xwait .twt')),
    busy: c.querySelector('.xembed') ? c.querySelector('.xembed').getAttribute('aria-busy') : null,
    article: txt(c.querySelector('.artb')),
    article_folded: !!(c.querySelector('.artb') && c.querySelector('.artb').classList.contains('clamp')),
    article_links: [...c.querySelectorAll('.artmore a')].map(a => [txt(a), a.getAttribute('href')]),
    article_elements: c.querySelector('.artb') ? c.querySelectorAll('.artb *').length : null,
    pwned: window.xPwned || null,
    note: txt(c.querySelector('.xnote')),
    toggle: txt(c.querySelector('.vtog')),
    jev: seen(c.querySelector('.jev')),
    // The card's own header: who and when, only when X's embed is not showing them.
    who: txt(c.querySelector('.twh .who')),
    head_links: [...c.querySelectorAll('.twh .links > *')].filter(seen).map(e => e.textContent),
  };
};
/** What X's frame would post: a real `message` event on this window, through the page's own
    listener, with X's origin and `source` (the frame's window, unless another is given). */
const xPost = (source, method, params, origin) => dispatchEvent(new MessageEvent('message', {
  origin: origin || 'https://platform.twitter.com', source: source,
  data: {'twttr.embed': {jsonrpc: '2.0', method: method, id: 'embed-0', params: params}}}));
const xSays = (id, method, params) => xPost(frameOf(id).contentWindow, method, params);
const done = (out) => fetch('/probe-done', {method: 'POST', body: JSON.stringify(out)});
"""

_PROBE = (
    "<script>"
    + _SEEN
    + _EMBED_JS
    + r"""
(async () => {
  const [A, B, C] = IDS_JSON;
  const Q = QUIET_ID, ART = ART_ID;
  const out = {};
  const step = async (name, fn) => { try { out[name] = await fn(); } catch (e) { out[name] = 'ERROR ' + e.message; } };
  await until(() => document.querySelector('#cards .card'));
  // The list starts below the header: frames are made only near the viewport.
  cardOf(A).scrollIntoView();
  await until(() => frameOf(A) && frameOf(B) && frameOf(C));
  await step('theme', async () => (xTestTheme.matches ? 'light' : 'dark'));
  // B's saved copy, as it stands in for X, and B's height: what X never answering changes.
  const bWait = cardOf(B).querySelector('.xembed > .xwait > .local');
  if (bWait) bWait.__probe = 'b';
  const bHeight = cardOf(B).getBoundingClientRect().height;
  await step('start', async () => ({a: sCard(A), b: sCard(B), c: sCard(C), bad: sCard(BAD_ID),
    // A slot showing nothing: neither the saved copy nor X's frame on screen.
    empty: [...document.querySelectorAll('.xembed')].filter(sl => !(seen(sl.querySelector(':scope > .xwait')) &&
      txt(sl.querySelector(':scope > .xwait .twt'))) && !seen(sl.querySelector('iframe'))).length,
    slots: document.querySelectorAll('.xembed').length,
    odd: ODD_JSON.map(id => sCard(id)), art: sCard(ART), frames: document.querySelectorAll('iframe').length,
    scripts: [...document.scripts].map(s => s.src).filter(Boolean)}));
  await step('net', async () => {
    // The harness's own block, measured: X, and a host given as a bare IP, never answer.
    const tryFetch = (u) => fetch(u, {mode: 'no-cors', cache: 'no-store'}).then(() => 'REACHED', () => 'BLOCKED');
    return {x: await tryFetch('https://platform.twitter.com/embed/Tweet.html?id=' + A), ip: await tryFetch('https://1.1.1.1/')};
  });
  await step('map', async () => ({same: postById() === postById(), size: Object.keys(postById()).length}));
  // A second card of A and of B (what Posts plus a Preguntar result draw): each follows its post.
  const twins = document.createElement('div');
  twins.id = 'twins';
  twins.append(card(postById()[A]), card(postById()[B]));
  document.getElementById('cards').after(twins);
  // X says «initialized» from both twins' frames, so neither falls back on its own: whatever
  // they show later comes from their post.
  twins.scrollIntoView();
  await until(() => frameOf(A, twins) && frameOf(B, twins));
  [A, B].forEach(id => xPost(frameOf(id, twins).contentWindow, 'twttr.private.initialized', [{data: {tweet_id: id}}]));
  cardOf(A).scrollIntoView();
  await step('foreign', async () => {
    const before = [sCard(A).frame.height, sCard(B).frame.height];
    const stray = document.body.appendChild(document.createElement('iframe'));
    const big = [{height: 777}];
    xPost(frameOf(A).contentWindow, 'twttr.private.resize', big, 'https://evil.example');  // not X
    xPost(window, 'twttr.private.resize', big);  // X's origin, but this window is no frame of ours
    xPost(stray.contentWindow, 'twttr.private.resize', big);  // a frame, but not an X card's
    // A message this page posts to itself: it arrives later, so wait for it to have been read.
    const read = new Promise(r => addEventListener('message', r, {once: true}));
    window.postMessage({'twttr.embed': {method: 'twttr.private.resize', params: [{height: 778}]}}, '*');
    await read;
    stray.remove();
    return {before: before, after: [sCard(A).frame.height, sCard(B).frame.height]};
  });
  await step('bounds', async () => {
    const seenHeights = [];
    for (const h of [0, -5, 1e9, 'abc', null]) {
      xSays(A, 'twttr.private.resize', [{height: h}]);
      seenHeights.push(sCard(A).frame.height);
    }
    return seenHeights;
  });
  await step('resize', async () => {
    const other = sCard(B).frame.height;
    // Every height A's slot takes from X's first height on: it goes once, one way.
    const slot = cardOf(A).querySelector('.xembed');
    const heights = [Math.round(slot.getBoundingClientRect().height)];
    const watch = new ResizeObserver(() => heights.push(Math.round(slot.getBoundingClientRect().height)));
    watch.observe(slot);
    xSays(A, 'twttr.private.resize', [{width: 550, height: 640, data: {tweet_id: A}}]);
    const now = sCard(A);
    await until(() => !slot.classList.contains('xgrow') && !slot.style.height, 3000);
    await sleep(100);
    watch.disconnect();
    return {a: now, b_before: other, b: sCard(B).frame.height, heights: heights,
      settled: {height: Math.round(slot.getBoundingClientRect().height), grow: slot.classList.contains('xgrow'),
        inline: slot.style.height}};
  });
  let quietAt = null;
  await step('quiet', async () => {
    cardOf(Q).scrollIntoView();
    await until(() => frameOf(Q));
    quietAt = Date.now();
    xSays(Q, 'twttr.private.initialized', [{data: {tweet_id: Q}}]);
    cardOf(A).scrollIntoView();
    return !!frameOf(Q);
  });
  await step('no_results', async () => { xSays(C, 'twttr.private.no_results', [{data: {tweet_id: C}}]); return sCard(C); });
  await step('keys', async () => {
    // j walks the cards from the top; at A the focus is A's card, with X's frame inside it.
    for (let n = 0; n < 12 && document.activeElement.dataset.id !== A; n++) {
      document.dispatchEvent(new KeyboardEvent('keydown', {key: 'j', bubbles: true}));
    }
    const a = document.activeElement;
    return {tag: a.tagName, card: a.classList.contains('card'), id: a.dataset.id, frame: !!a.querySelector('.xembed iframe')};
  });
  // X never answers B: past the wait it falls back to the saved copy, in both its cards. Q
  // said «initialized» and nothing more: it is still waiting for its height when a frame
  // that loaded with it would long have timed out (its own 8 s, plus 1 s).
  await step('timeout', async () => {
    await until(() => sCard(B).local !== null && Date.now() - quietAt >= 9000, 30000);
    const local = cardOf(B).querySelector('.local');
    const note = cardOf(B).querySelector('.xnote');
    return {a: sCard(A), b: sCard(B), b_twin: sCard(B, twins), q: sCard(Q),
      b_same_copy: !!local && local.__probe === 'b' && local === bWait,
      b_grew: Math.round(cardOf(B).getBoundingClientRect().height - bHeight),
      b_note: note ? Math.round(note.getBoundingClientRect().height + parseFloat(getComputedStyle(note).marginTop)) : null};
  });
  await step('retry', async () => {
    cardOf(B).querySelector('.vtog').click();
    await until(() => frameOf(B));
    return {b: sCard(B), b_twin: sCard(B, twins)};
  });
  await step('to_local', async () => { cardOf(A).querySelector('.vtog').click(); return {a: sCard(A), a_twin: sCard(A, twins)}; });
  await step('to_x', async () => {
    cardOf(A).querySelector('.vtog').click();
    await until(() => frameOf(A));
    return {a: sCard(A), a_twin: sCard(A, twins)};
  });
  await step('switch_local', async () => {
    document.querySelector('#vista button[data-vista="local"]').click();
    let stored; try { stored = localStorage.getItem('xbrain.jev.vista'); } catch (e) { stored = 'ERROR'; }
    return {cards: [A, B, C, BAD_ID].map(id => sCard(id)), art: sCard(ART), stored: stored, read: readViewMode(),
      label: txt(document.querySelector('#vista button[data-vista="local"]')),
      pressed: document.querySelector('#vista button[data-vista="local"]').getAttribute('aria-pressed')};
  });
  await step('switch_x', async () => {
    document.querySelector('#vista button[data-vista="x"]').click();
    await until(() => frameOf(A));
    let stored; try { stored = localStorage.getItem('xbrain.jev.vista'); } catch (e) { stored = 'ERROR'; }
    return {a: sCard(A), art: sCard(ART), stored: stored};
  });
  await step('theme_flip', async () => {
    // The system theme changes: every drawn frame is made again in the new theme.
    xTestTheme.matches = !xTestTheme.matches;
    xTestTheme.dispatchEvent(new Event('change'));
    const want = 'theme=' + (xTestTheme.matches ? 'light' : 'dark');
    await until(() => frameOf(A) && frameOf(A).getAttribute('src').includes(want));
    return sCard(A);
  });
  await step('article', async () => {
    cardOf(ART).querySelector('.vtog').click();
    cardOf(ART).scrollIntoView();
    await until(() => frameOf(ART));
    return sCard(ART);
  });
  done(out);
})();
</script>"""
)


def _probe() -> str:
    return (
        _PROBE.replace("IDS_JSON", json.dumps(list(IDS)))
        .replace("ODD_JSON", json.dumps(list(ODD)))
        .replace("BAD_ID", json.dumps(BAD))
        .replace("QUIET_ID", json.dumps(QUIET))
        .replace("ART_ID", json.dumps(ART))
    )


def _hooked(page: Path) -> Path:
    """The page with `_THEME_HOOK` at the top of its <head>."""
    html = page.read_text(encoding="utf-8")
    assert html.count("<head>") == 1
    page.write_text(html.replace("<head>", _THEME_HOOK), encoding="utf-8")
    return page


@pytest.fixture(scope="module")
def embedded(tmp_path_factory) -> dict[str, Any]:
    _need_chrome()
    page = _hooked(_page(tmp_path_factory.mktemp("embed"), _fixture(), _probe()))
    # Tall enough that A, B and C are near the viewport together once the list is reached.
    return _live(page, "#posts?f=all", height=2400)


def _src(post_id: str, theme: str) -> str:
    return f"{X_EMBED}?id={post_id}&dnt=true&theme={theme}&lang=es"


def _other(theme: str) -> str:
    return "light" if theme == "dark" else "dark"


def _note(why: str) -> str:
    return f"{why} · {SAVED}"


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
    # Until X says its height the frame loads out of sight, and the saved copy stands in.
    assert a["frame"]["seen"] is False
    assert a["wait"] == f"texto local de {IDS[0]}" and a["busy"] == "true"
    # X's frame, never X's script in this page's origin.
    assert embedded["start"]["scripts"] == []
    # The saved copy is not drawn as the card's view beside it; the Jev block stays under it.
    assert a["local"] is None
    assert a["jev"] is True
    assert a["toggle"] == "ver copia guardada"
    # X's embed shows the author and the date: the card's header does not say them twice, and
    # keeps what is ours — the links and the toggle.
    assert a["who"] is None
    assert a["head_links"][-1] == "ver copia guardada" and "X ↗" in a["head_links"]


@_requires_chrome
def test_a_post_whose_id_x_cannot_have_is_the_saved_copy_and_says_why(embedded):
    """The id is the one value from the data that goes into the frame's URL: anything but
    1–25 digits, whole, never gets a frame — the unanchored, the unbounded, the URL-shaped."""
    start = embedded["start"]
    why = _note("Sin vista de X: el id de este post no es de X")

    for post_id, card in zip((BAD, *ODD), (start["bad"], *start["odd"]), strict=True):
        assert card["frame"] is None, post_id
        assert card["slot"] is False, post_id
        # The saved copy has no author line of its own: the card's header carries it.
        assert card["who"] is not None and "@" in card["who"], post_id
        assert card["local"] == f"texto local de {post_id}"
        assert card["note"] == why
        assert card["toggle"] is None
    # Every frame on the page is one of the posts with an X id.
    assert all(embedded["start"][k]["frame"]["src"].startswith(X_EMBED + "?id=18") for k in "abc")


@_requires_chrome
def test_an_x_article_opens_on_the_saved_copy_and_ver_en_x_shows_xs_view(embedded):
    art, flipped = embedded["start"]["art"], embedded["article"]

    assert art["frame"] is None
    assert art["slot"] is False
    assert art["local"] == f"texto local de {ART}"
    assert art["note"] == _note("Artículo de X: la vista de X solo enseña su enlace")
    assert art["toggle"] == "ver en X"
    # The saved copy carries the Article's body, cut to the page's cap at a paragraph end,
    # folded under «ver todo», and says where the rest is.
    assert art["article"] is not None
    assert art["article"].endswith(" …")
    assert ART_BODY.startswith(art["article"][: -len(" …")])
    assert len(art["article"]) <= PAGE_ARTICLE_CHARS
    assert art["article_folded"] is True
    # Its markup is text: shown as typed, no element made from it, nothing of it ran.
    assert art["article"].startswith(ART_MARKUP)
    assert (art["article_elements"], art["pwned"]) == (0, None)
    assert art["article_links"] == [
        ["sigue en X ↗", ART_URL.replace("http://", "https://")],
        ["nota ↗", "obsidian://open?path=%2Fvault%2Fx%2Fa.md"],
    ]
    assert flipped["frame"]["src"] == _src(ART, _other(embedded["theme"]))
    assert flipped["local"] is None
    assert flipped["note"] is None
    assert flipped["toggle"] == "ver copia guardada"


@_requires_chrome
def test_the_page_tests_really_have_no_network(embedded):
    """Fail-closed, measured in the browser this file launches: X, and a bare IP, never
    answer — so no test here can pass on something X sent."""
    assert embedded["net"] == {"x": "BLOCKED", "ip": "BLOCKED"}


def test_every_page_launcher_uses_the_no_network_rule():
    """The rule `test_the_page_tests_really_have_no_network` measures is the one every page
    launcher passes to Chrome."""
    from tests import test_jev_page_browser as browser

    assert "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1" == browser._NO_NETWORK
    assert inspect.signature(_live).parameters["rules"].default == _NO_NETWORK
    assert "_NO_NETWORK" in inspect.getsource(browser._dump)
    assert "_NO_NETWORK" in inspect.getsource(browser._dump_served)


@_requires_chrome
def test_the_post_map_is_built_once_per_data_load(embedded):
    assert embedded["map"] == {"same": True, "size": 10}


@_requires_chrome
def test_a_message_not_from_one_of_the_cards_frames_at_xs_origin_changes_nothing(embedded):
    """Another origin with a card's frame as source; X's origin with this window as source;
    X's origin from a frame that is not a card's; a message the page posts to itself."""
    foreign = embedded["foreign"]

    assert foreign["before"] == [220, 220]  # the reserved height, until X says its own
    assert foreign["after"] == foreign["before"]


@_requires_chrome
def test_a_resize_outside_one_to_twenty_thousand_px_is_ignored(embedded):
    assert embedded["bounds"] == [220] * 5


@_requires_chrome
def test_xs_resize_sets_the_height_of_its_own_frame_only(embedded):
    resize = embedded["resize"]

    assert resize["a"]["frame"]["height"] == 640
    assert resize["b"] == resize["b_before"] == 220


@_requires_chrome
def test_no_slot_is_ever_an_empty_box_the_saved_copy_stands_in_until_x_answers(embedded):
    """PR 16 UX C1: a card with X's view shows the saved copy at once, in X's place (no
    network needed), however long X takes — never a blank slot."""
    start = embedded["start"]

    assert start["slots"] >= 3 and start["empty"] == 0
    for key, post in zip("bc", IDS[1:], strict=True):
        assert start[key]["wait"] == f"texto local de {post}"
        assert start[key]["frame"]["seen"] is False and start[key]["local"] is None


@_requires_chrome
def test_xs_first_height_swaps_its_frame_in_for_the_saved_copy_once(embedded):
    """The frame takes the saved copy's place on X's first height: the copy goes, the frame is
    shown at X's height, and the slot's height moves once, one way, then is left to the frame."""
    resize = embedded["resize"]
    a, heights = resize["a"], resize["heights"]

    assert a["frame"]["seen"] is True and a["frame"]["height"] == 640
    assert a["wait"] is None and a["local"] is None and a["busy"] is None
    assert heights[-1] == 640 and heights[0] != 640
    step = 1 if heights[-1] > heights[0] else -1
    assert all((b - h) * step >= 0 for h, b in zip(heights, heights[1:], strict=False)), heights
    assert resize["settled"] == {"height": 640, "grow": False, "inline": ""}


@_requires_chrome
def test_a_post_x_has_not_got_falls_back_to_the_saved_copy(embedded):
    c = embedded["no_results"]

    assert c["frame"] is None
    assert c["local"] == f"texto local de {IDS[2]}"
    assert c["note"] == _note(
        "X no tiene este post (borrado, protegido o de una cuenta suspendida)"
    )
    assert c["toggle"] == "ver en X"


@_requires_chrome
def test_a_frame_x_never_answers_falls_back_after_the_wait_in_every_card_of_the_post(embedded):
    timeout = embedded["timeout"]

    for b in (timeout["b"], timeout["b_twin"]):
        assert b["frame"] is None
        assert b["slot"] is False
        assert b["local"] == f"texto local de {IDS[1]}"
        assert b["note"] == _note("X no respondió en 8 s (¿sin red?)")
        assert b["toggle"] == "ver en X"
    # PR 16 UX C1: the saved copy that stood in stays — the same nodes, nothing drawn again —
    # and the card grows by no more than the note (the slot's own margin goes with it).
    assert timeout["b_same_copy"] is True
    assert timeout["b_note"] and 0 < timeout["b_grew"] <= timeout["b_note"] + 1
    # A frame that said anything is X answering: it waits for its height, it does not fall back.
    assert embedded["quiet"] is True
    assert timeout["q"]["frame"]["src"] == _src(QUIET, embedded["theme"])
    assert timeout["q"]["frame"]["height"] == 220
    assert timeout["q"]["note"] is None
    # A frame X did answer is left alone.
    assert timeout["a"]["frame"]["src"] == _src(IDS[0], embedded["theme"])
    assert timeout["a"]["frame"]["height"] == 640


@_requires_chrome
def test_ver_en_x_retries_a_post_that_fell_back_in_every_card(embedded):
    retry = embedded["retry"]

    assert retry["b"]["frame"]["src"] == _src(IDS[1], embedded["theme"])
    assert retry["b"]["local"] is None
    assert retry["b"]["note"] is None
    assert retry["b_twin"]["slot"] is True
    assert retry["b_twin"]["local"] is None
    assert retry["b_twin"]["toggle"] == "ver copia guardada"


@_requires_chrome
def test_the_toggle_goes_both_ways_for_every_card_of_the_post(embedded):
    to_local, to_x = embedded["to_local"], embedded["to_x"]

    for a in (to_local["a"], to_local["a_twin"]):
        assert a["frame"] is None
        assert a["slot"] is False
        assert a["local"] == f"texto local de {IDS[0]}"
        assert a["note"] is None  # the reader chose it: nothing to explain
        assert a["toggle"] == "ver en X"
    assert to_x["a"]["frame"]["src"] == _src(IDS[0], embedded["theme"])
    assert to_x["a"]["local"] is None
    assert to_x["a"]["toggle"] == "ver copia guardada"
    assert to_x["a_twin"]["slot"] is True
    assert to_x["a_twin"]["local"] is None
    assert to_x["a_twin"]["toggle"] == "ver copia guardada"


@_requires_chrome
def test_j_focuses_the_card_not_the_frame(embedded):
    keys = embedded["keys"]

    assert keys == {"tag": "ARTICLE", "card": True, "id": IDS[0], "frame": True}


@_requires_chrome
def test_the_vista_switch_turns_every_card_saved_and_is_remembered(embedded):
    local, back = embedded["switch_local"], embedded["switch_x"]

    assert [c["frame"] for c in local["cards"]] == [None] * 4
    assert [c["local"] for c in local["cards"]] == [f"texto local de {i}" for i in (*IDS, BAD)]
    assert local["label"] == "copia guardada"
    assert local["stored"] == "local"
    assert local["read"] == "local"
    assert local["pressed"] == "true"
    # The reader chose it for the whole page: an Article says nothing of its own either.
    assert local["art"]["note"] is None
    assert back["stored"] == "x"
    assert back["a"]["frame"]["src"] == _src(IDS[0], embedded["theme"])
    # Back on X, an Article opens on its saved copy again, and says why.
    assert back["art"]["frame"] is None
    assert back["art"]["note"] == _note("Artículo de X: la vista de X solo enseña su enlace")


@_requires_chrome
def test_a_system_theme_change_makes_the_frames_again_in_the_new_theme(embedded):
    flipped = embedded["theme_flip"]

    assert flipped["frame"]["src"] == _src(IDS[0], _other(embedded["theme"]))


_OFF_PROBE = (
    "<script>"
    + _SEEN
    + _EMBED_JS
    + r"""
(async () => {
  await until(() => document.querySelector('#cards .card'));
  done({cards: IDS_JSON.map(id => sCard(id)), frames: document.querySelectorAll('iframe').length,
    pressed: document.querySelector('#vista button[data-vista="local"]').getAttribute('aria-pressed')});
})();
</script>"""
)


@_requires_chrome
def test_embed_0_in_the_url_shows_every_card_saved(tmp_path):
    _need_chrome()
    probe = _OFF_PROBE.replace("IDS_JSON", json.dumps(list(IDS)))
    page = _page(tmp_path, _fixture(), probe)
    seen = _live(page, "?embed=0#posts?f=all")

    assert seen["frames"] == 0
    assert [c["local"] for c in seen["cards"]] == [f"texto local de {i}" for i in IDS]
    assert [c["toggle"] for c in seen["cards"]] == ["ver en X"] * 3
    assert seen["pressed"] == "true"


#: Headless Chrome keeps a window at least 500 px wide, so the page is measured in a 375 px
#: frame of itself (same origin; the copy in the frame does nothing).
_PHONE_PROBE = (
    "<script>"
    + _SEEN
    + _EMBED_JS
    + r"""
(async () => {
  if (window !== window.top) return;
  const phone = document.createElement('iframe');
  phone.style.cssText = 'width:375px;height:800px;border:0';
  phone.src = location.pathname + location.hash;
  document.body.prepend(phone);
  const doc = () => phone.contentDocument;
  await until(() => doc() && doc().querySelector('#cards .card'));
  const win = phone.contentWindow;
  const boxes = [...doc().querySelectorAll('#vista button')].map(b => {
    const r = b.getBoundingClientRect();
    const st = win.getComputedStyle(b);
    return {left: Math.round(r.left), right: Math.round(r.right),
      seen: r.width > 0 && st.visibility !== 'hidden' && st.display !== 'none'};
  });
  const w = (sel) => Math.round(doc().querySelector(sel).getBoundingClientRect().width);
  done({width: win.innerWidth, page: doc().documentElement.scrollWidth, buttons: boxes,
    tabs: w('.tabs.sub'), bar: w('.tabbar')});
})();
</script>"""
)


@_requires_chrome
def test_on_a_phone_the_vista_switch_is_on_screen_without_scrolling_the_tabs(tmp_path):
    _need_chrome()
    seen = _live(_page(tmp_path, _fixture(), _PHONE_PROBE), "#posts?f=all")

    assert seen["width"] == 375
    assert seen["page"] <= seen["width"]  # no sideways scroll of the page either
    assert len(seen["buttons"]) == 2
    for button in seen["buttons"]:
        assert button["seen"] is True
        assert 0 <= button["left"] < button["right"] <= seen["width"], button
    # The switch takes a row of its own: the section tabs keep the whole width to scroll in.
    assert seen["tabs"] == seen["bar"]


_LAZY_PROBE = (
    "<script>"
    + _SEEN
    + _EMBED_JS
    + r"""
(async () => {
  // Every slot the page's observer lets go, from here on.
  const let_go = new Set();
  const unobserve0 = IntersectionObserver.prototype.unobserve;
  IntersectionObserver.prototype.unobserve = function (el) { let_go.add(el); return unobserve0.call(this, el); };
  await until(() => document.querySelector('#cards .card'));
  const frames = () => document.querySelectorAll('#cards iframe').length;
  const out = {cards: document.querySelectorAll('#cards .card').length};
  document.getElementById('cards').scrollIntoView();
  await until(() => frames() > 0);
  await sleep(300);  // what else the observer makes near the top, if it makes more
  out.at_top = frames();
  document.querySelector('#cards .card:nth-child(50)').scrollIntoView();
  await until(() => !!document.querySelector('#cards .card:nth-child(50) iframe'));
  await sleep(300);
  out.at_bottom = frames();
  out.last_has_frame = !!document.querySelector('#cards .card:nth-child(50) iframe');
  // The slots jumped over are still watched; turning the page to the saved copy drops them.
  const waiting = [...document.querySelectorAll('#cards .xembed')].filter(s => !s.querySelector('iframe'));
  out.waiting = waiting.length;
  document.querySelector('#vista button[data-vista="local"]').click();
  await until(() => waiting.every(s => let_go.has(s)), 3000);
  out.let_go = waiting.filter(s => let_go.has(s)).length;
  out.slots_after = document.querySelectorAll('.xembed').length;
  done(out);
})();
</script>"""
)


@_requires_chrome
def test_only_frames_near_the_viewport_are_made_and_dropped_slots_are_let_go(tmp_path):
    _need_chrome()
    ids = [str(10**18 + n) for n in range(60)]
    items = [_item(i) for i in ids]
    data = _data(items, {it.id: _assessment(it, membership={"ai-coding": 0.9}) for it in items})
    seen = _live(_page(tmp_path, data, _LAZY_PROBE), "#posts?f=all")

    assert seen["cards"] == 50
    assert 1 <= seen["at_top"] <= 8
    assert seen["last_has_frame"] is True
    assert seen["at_bottom"] <= 16
    assert seen["waiting"] >= 20
    assert seen["let_go"] == seen["waiting"]
    assert seen["slots_after"] == 0


#: X's host sent to a local socket that takes the connection and never answers: the frame
#: never LOADS. The rest of the network stays blocked.
_HANGING_X = (
    "--host-resolver-rules=MAP platform.twitter.com 127.0.0.1:{port} , "
    + (_NO_NETWORK.split("=", 1)[1])
)

_UNLOADED_PROBE = (
    "<script>"
    + _SEEN
    + _EMBED_JS
    + r"""
(async () => {
  const A = A_ID;
  await until(() => document.querySelector('#cards .card'));
  cardOf(A).scrollIntoView();
  await until(() => frameOf(A));
  const f = frameOf(A);
  let loaded = false;
  f.addEventListener('load', () => { loaded = true; });
  const made = Date.now();
  // X's 8 s, and 1.5 s more, counted from the frame's creation: a clock started there fails it.
  await until(() => sCard(A).local !== null || Date.now() - made >= 9500, 20000);
  done({loaded: loaded, same_frame: frameOf(A) === f, a: sCard(A)});
})();
</script>"""
)


@_requires_chrome
def test_the_wait_for_x_starts_when_the_frame_has_loaded_not_when_it_is_made(tmp_path):
    """`loading="lazy"` may hold a frame back after it is made; one that has not loaded has
    not asked X anything yet, so it must not fall back."""
    _need_chrome()
    hole = socket.socket()
    hole.bind(("127.0.0.1", 0))
    hole.listen(16)
    held: list[socket.socket] = []

    def _hold() -> None:
        while True:
            try:
                held.append(hole.accept()[0])
            except OSError:
                return

    threading.Thread(target=_hold, daemon=True).start()
    try:
        probe = _UNLOADED_PROBE.replace("A_ID", json.dumps(IDS[0]))
        rules = _HANGING_X.format(port=hole.getsockname()[1])
        seen = _live(_page(tmp_path, _fixture(), probe), "#posts?f=all", rules=rules)
    finally:
        hole.close()
        for conn in held:
            conn.close()

    assert seen["loaded"] is False
    assert seen["same_frame"] is True
    assert seen["a"]["note"] is None
    assert seen["a"]["local"] is None


def test_the_frame_url_is_built_in_one_place_and_widgets_js_is_never_loaded():
    """The frame's `src` is set once, from `xSrc`; nothing in the template loads X's
    `widgets.js`, which would run in the page's origin next to the token. (What reaches the
    URL, and which messages are read, is behaviour, tested above.)"""
    from tests.test_jev_dashboard import _script_section
    from xbrain.dashboard import _resource

    template = _resource("jev.template.html")
    embed = _script_section(template, "/* embed */", "/* end embed */")

    assert "widgets.js" not in template.replace("NEVER `widgets.js`", "")
    assert "platform.twitter.com/widgets" not in template
    assert embed.count("'src'") == 1
    assert "f.setAttribute('src', xSrc(slot.dataset.id));" in embed

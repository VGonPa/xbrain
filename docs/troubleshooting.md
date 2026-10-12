# Troubleshooting & FAQ

Common failures and how to fix them. Most are environment issues (auth, PATH,
external tools), not bugs.

## X session expired / auth fails

Symptoms: `extract`/`sync` scrapes 0 posts, or `status` says it can't
authenticate. X sessions are short-lived.

Fix — re-import cookies from a browser you're logged in to:

```bash
# Chrome — log in to x.com in Chrome first, then:
.venv/bin/python scripts/import_chrome_session.py
# → "auth_token: OK"

# Safari — log in in Safari, grant your terminal "Full Disk Access"
# (System Settings → Privacy & Security), then:
.venv/bin/python scripts/import_safari_session.py
```

`xbrain login` (in-app Playwright login) exists but is unreliable with
Google/SSO accounts — the automated browser gets blocked. Cookie import is the
recommended path.

## "Re-saw 0 known items on a non-empty store" — the run aborts without saving

A safety tripwire: extraction saw none of the items it already has, which almost
always means an **expired session** or an X GraphQL change, not that your
bookmarks vanished. It aborts rather than overwrite good data. Re-authenticate
(above) and re-run. If you're sure the store is stale, `--force` overrides it.

## Getting rate-limited / the browser stalls

`extract` runs **headful** (visible Chromium) by default to look human, paces
itself, and backs off on `429`. If you still hit limits, wait and re-run — the
store is incremental, so you lose nothing. Don't run many extracts back-to-back.

## `digest-video` reports `sin transcript`

XBrain found a video bookmark, but it could not obtain usable transcript text.
Typical causes: X did not expose a caption/text-track URL, the video is not a
progressive MP4 that the temporary fallback can fetch, or the external ASR
transcriber returned no speech. The item stays in the library and can be retried
later.

## `digest-video` is slow or times out

The expensive steps are temporary ASR fallback downloads/transcription and the
text LLM executive summary. If the run is slow, process fewer videos with
`--limit`, cap fallback size with `--max-size`, or run by topic/ids:

```bash
uv run xbrain digest-video --all-pending --limit 5
uv run xbrain digest-video --all-pending --max-size 500MB
uv run xbrain digest-video --topic ai-coding
```

## Every video comes back `fallidos`

`fallidos` means the caption URL existed but could not be fetched/parsed, the
temporary MP4 fallback or external transcriber failed, or the configured text LLM
failed to produce a valid summary. Re-run once; signed text/video URLs can
expire. If it repeats, refresh X metadata first:

```bash
uv run xbrain refresh-media --source bookmarks --headless
uv run xbrain digest-video --all-pending
```

If the failures mention `transcriber`, install the configured ASR binary or set a
working `[transcribe].command` in `config.toml`.

## `generate` hangs or takes very long

If your vault is on **iCloud** with "Optimize Mac Storage" on, files can be
evicted to the cloud (dataless), and reading/writing them blocks on
re-download — worst at night with no activity. Run `generate` while the machine
is active, or keep the vault folder materialized (turn off Optimize Storage for
it). `data/items.json` already holds every digest, so a slow `generate` never
loses data — just re-run it.

## Do I need an API key?

No. The default execution mode (`vocab`/`enrich`/`topics`/`describe`) uses a
**Claude Code session** — no key, no cost. `ANTHROPIC_API_KEY` is only for
`--executor api` when `[llm].provider = "anthropic"` for unattended LLM runs.
`FIRECRAWL_API_KEY` is an optional fallback fetcher for JavaScript-heavy pages.

## Where's the source of truth? Can I delete the vault notes?

`data/items.json` is the hub — the markdown is **derived and disposable**.
Delete `items/`, `topics/`, `_index.md` and re-run `generate` any time. Every
destructive command auto-snapshots `items.json` first (see
[Snapshots & safety](../README.md#snapshots--safety)); restore from
`data/snapshots/` if needed.

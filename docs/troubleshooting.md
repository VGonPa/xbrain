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

## `parakeet-mlx` / `ffmpeg` not found (digest-video)

```
transcriber '.../xbrain-transcribe' exited 1: FileNotFoundError: 'parakeet-mlx'
```

The external tools aren't on `PATH`. Two cases:

- **Interactive shell:** install them (`brew install ffmpeg`,
  `uv tool install parakeet-mlx mlx-vlm`) and make sure `~/.local/bin` +
  `/opt/homebrew/bin` are on your `PATH`.
- **cron / launchd / a scheduled job:** these run with a **minimal PATH** that
  excludes `~/.local/bin` and `/opt/homebrew/bin`. Set the job's environment
  explicitly — e.g. in a launchd plist:

  ```xml
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/Users/you/.local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  ```

  When testing a job, reproduce its env (`env -i HOME=$HOME PATH=... your-cmd`),
  not your shell — your shell's full PATH hides the bug.

## `digest-video` is slow or times out

Local vision (`--frames`) is the bottleneck: a slide-heavy talk can have up to
40 key-frames, and a local VLM reloads the model per frame. On a 16 GB Mac,
`qwen-7b` is ~2 min/frame → a long talk takes over an hour.

- **First run of a large model** can exceed the 300 s per-frame timeout while it
  *downloads* — pre-pull once: `~/.local/share/uv/tools/mlx-vlm/bin/python -c
  "from mlx_vlm import load; load('mlx-community/Qwen2.5-VL-7B-Instruct-4bit')"`.
- **Too slow overall?** Use a smaller model (`--vision-model qwen-3b`), or
  transcript-only (drop `--frames`), or cloud (`--vision-model opus`, needs
  `ANTHROPIC_API_KEY`).
- Frame extraction never hangs the run — ffmpeg is bounded by its own timeout.

## Every video comes back `fallidos` / `sin voz`

- `sin voz` (silent): the video has **no audio track** at the source (GIFs,
  muted screencasts). This is expected — it attaches as `has_speech=false`
  ("silent video"), not an error. Verify with `yt-dlp -f bestaudio <tweet-url>`
  (errors = no audio exists).
- `fallidos` (real failures): usually `parakeet-mlx` not found (see the PATH
  section above) — the fix is almost always the environment, not the video.

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
`--executor api` (unattended LLM runs) and cloud vision (`--vision-model opus`).
`FIRECRAWL_API_KEY` is an optional fallback fetcher for JavaScript-heavy pages.

## Where's the source of truth? Can I delete the vault notes?

`data/items.json` is the hub — the markdown is **derived and disposable**.
Delete `items/`, `topics/`, `_index.md` and re-run `generate` any time. Every
destructive command auto-snapshots `items.json` first (see
[Snapshots & safety](../README.md#snapshots--safety)); restore from
`data/snapshots/` if needed.

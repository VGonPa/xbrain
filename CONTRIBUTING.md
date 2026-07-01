# Contributing to XBrain

Thanks for your interest. XBrain is a small, focused tool — contributions that
keep it small and focused are the most welcome.

## Development setup

```bash
uv venv
uv pip install -e ".[dev]" --index-url https://pypi.org/simple
uv run playwright install chromium
```

The `[dev]` extra above also installs the quality-gate tools (`poe`, `ruff`,
`mypy`, and the rest).

Copy `config.toml.example` to `config.toml` and fill in your own values. That
file is not tracked by git.

## Before opening a pull request

- Run the full quality gate: `uv run poe check`. It must be green — warnings are
  OK, failures are not.
- Individual checks are available too: `uv run poe test`, `uv run poe lint`,
  `uv run poe types`, and the rest of the `poe` tasks.
- CI runs the same gate on every PR.
- Every new module needs a matching `tests/test_*.py`. This project is built
  test-first.
- Keep the PR focused on one change. No drive-by refactors.
- Describe what changed and why, in your own words.

## Safety: destructive operations auto-snapshot

Destructive commands (`vocab --regenerate`, `topics --resynth`, `fetch --force`,
`refresh-media`, `media`, `describe`, `download-videos`, `digest-video`) copy the
full `data/` directory to
`data/snapshots/<UTC-ts>-pre-<command>/` before they write anything. (`download-videos`
takes its snapshot *after* the interactive size-gate confirmation, so a declined
run leaves no stray snapshot — but always before the first byte is written.) If your change introduces or modifies a destructive
operation, **wire the auto-snapshot** — see `_auto_snapshot` in `src/xbrain/cli.py`
and the unit + integration tests under `tests/test_snapshot*.py`. A snapshot
failure must propagate and abort the destructive op; never `try/except`-swallow
it. Manual snapshots are available via `xbrain snapshot create`; restore via
`xbrain snapshot restore <name>`.

`digest-video` is destructive because it attaches each video's transcript to the
item as an `x_video` content source and rewrites `items.json`. It snapshots *only
when it is about to write* (a pure already-digested / no-fetchable-video run
attaches nothing, so it takes no snapshot), but always before the first store
write; a snapshot failure propagates and aborts before any change lands.

Not every command that writes bytes is destructive. `xbrain list-videos` is
read-only, and `xbrain fetch-video` is **intentionally NOT** in the
auto-snapshot set: it fetches a video ephemerally to the caller-supplied `--to`
directory and never mutates `items.json`, never writes `data/media/`, and takes
no snapshot — there is nothing in `data/` to protect, so a snapshot would be
noise. Keep it that way: `fetch-video` must stay store-non-mutating (a test
asserts `items.json` is byte-identical before/after a fetch).

## The external transcriber (`digest-video`)

`xbrain digest-video` keeps xbrain **mechanical**: it fetches a video ephemerally
and shells out to an **external local transcriber** (ASR), then attaches the
transcript and discards the bytes. The heavy ML lives **outside** xbrain core —
there is no MLX/CoreML/torch dependency in the CLI; a test asserts
`transcribe.py` imports no ML library.

Configure the transcriber in `config.toml`:

```toml
[transcribe]
command = "parakeet-mlx"          # default; whisper / faster-whisper is a portable fallback
# model  = "parakeet-tdt-0.6b-v2" # optional; omit for the tool default
```

xbrain invokes `<command> [--model M] --output-format json --output-dir <TMPDIR>
<mediapath>` (the `command` is shlex-split, so a multi-token wrapper works, and
it runs **without** a shell). The real `parakeet-mlx` writes its transcript to a
**file** at `<TMPDIR>/<stem>.json`, so xbrain reads that produced file:

```json
{"text": "…", "language": "en", "segments": [{"start": 0.0, "end": 3.2, "text": "…"}]}
```

`--language` is **not** passed — parakeet auto-detects and rejects it (the
`--language` CLI flag only records a fallback language on the result). A
wrapper that emits the same JSON on **stdout** instead of a file is also
supported (stdout is the fallback source).

The **no-audio / no-speech** case is graceful ONLY via a real JSON signal: a
valid document with empty text (`{"text": ""}`), empty segments, or
`has_speech: false` yields `has_speech=False` + empty text (attached as a "silent
video" marker). A transcriber that exits 0 but produces **no usable output** (no
file / empty file AND empty stdout) is a hard error (`TranscriberFailed`) — never
inferred as no-speech, because that would silently lose the transcript. A
**missing / non-executable binary** surfaces a clear operator error (install it
or fix `[transcribe].command`), never a crash. If your transcriber's native CLI
differs, point `command` at a thin wrapper script that adapts it to this
contract.

## Pull requests written with AI agents

XBrain is built with AI coding agents, and PRs written that way are welcome — under
the same bar as hand-written code:

- **You are accountable for every line your agent writes.** "An agent did it" is
  not a description and not an excuse.
- **Read your agent's diff before you open the PR.** Do not submit code you have
  not reviewed yourself.
- **A green pipeline is the floor, not the goal.** Run the tests and the linter
  locally; passing CI does not mean the change is correct.
- **Keep agent PRs small.** Large agent-generated diffs are hard to review and
  usually mix unrelated changes.
- **Disclose it.** One line in the PR description ("substantially written with an
  AI agent") is enough and appreciated.
- **Do not point agents at this repo's dependencies** or open agent-generated PRs
  to upstream projects from this work.

## Adding an output language

XBrain's output language (LLM summaries/overviews + wiki section headers) is
parameterised via `[output].language` in `config.toml`. Today's supported
values are `English` (default) and `Spanish`. To add a third language:

1. Append a new entry to `_STRINGS` in `src/xbrain/i18n.py` with the
   translated wiki headers (`topics_label`, `content_header`, etc.).
2. Update `config.toml.example` and the README's Configuration table to list
   the new value.
3. That is it — `SUPPORTED_LANGUAGES` is derived from the dict, and the
   `{language}` placeholder in `rubric-summary.md` / `rubric-topic-page.md` /
   `rubric-vocab.md` is substituted verbatim. The LLM does the translation.

## Scope and responsible use

XBrain reads X through X's internal endpoints, for personal use, with your own
account and your own data. Keep contributions within that scope. See the README's
"Responsible use" section.

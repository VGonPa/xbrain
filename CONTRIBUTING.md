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
`refresh-media`, `media`, `describe`, `download-videos`) copy the full `data/`
directory to
`data/snapshots/<UTC-ts>-pre-<command>/` before they write anything. (`download-videos`
takes its snapshot *after* the interactive size-gate confirmation, so a declined
run leaves no stray snapshot — but always before the first byte is written.) If your change introduces or modifies a destructive
operation, **wire the auto-snapshot** — see `_auto_snapshot` in `src/xbrain/cli.py`
and the unit + integration tests under `tests/test_snapshot*.py`. A snapshot
failure must propagate and abort the destructive op; never `try/except`-swallow
it. Manual snapshots are available via `xbrain snapshot create`; restore via
`xbrain snapshot restore <name>`.

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

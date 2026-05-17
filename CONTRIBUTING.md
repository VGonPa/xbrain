# Contributing to XBrain

Thanks for your interest. XBrain is a small, focused tool — contributions that
keep it small and focused are the most welcome.

## Development setup

```bash
uv venv
uv pip install -e ".[dev]" --index-url https://pypi.org/simple
uv run playwright install chromium
```

Copy `config.toml.example` to `config.toml` and fill in your own values. That
file is not tracked by git.

## Before opening a pull request

- Run the full test suite: `uv run pytest -v`. It must be green.
- Run the linter: `uv run ruff check .`.
- Every new module needs a matching `tests/test_*.py`. This project is built
  test-first.
- Keep the PR focused on one change. No drive-by refactors.
- Describe what changed and why, in your own words.

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

## Scope and responsible use

XBrain reads X through X's internal endpoints, for personal use, with your own
account and your own data. Keep contributions within that scope. See the README's
"Responsible use" section.

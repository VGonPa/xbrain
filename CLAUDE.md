# CLAUDE.md — xbrain

Python CLI (`xbrain`) that extracts X bookmarks/tweets into a JSON store and
generates an Obsidian wiki.

## Stack
- Python 3.12+ (venv currently runs 3.13), `uv`, `pydantic` v2, `typer`, `playwright`, `trafilatura`, `pytest`.
- `uv pip install` needs `--index-url https://pypi.org/simple` to bypass the
  machine-wide private FITIZENS pip index.

## Architecture
- Pipeline: `extract → import-archive → fetch → [enrich] → generate`.
- Media side-pipeline: `media` (download photos) → `describe` (vision LLM);
  `refresh-media` re-captures X to backfill the playable video URL + bitrate +
  duration onto already-stored items (video-only, preserves photos/enrichment;
  destructive → auto-snapshot; no video download yet).
- `data/items.json` (dict keyed by tweet id) is the source of truth; markdown
  is derived. All stages are idempotent and incremental.
- `enrich` is a stub — the LLM executor is intentionally in pause (spec §9).

## Conventions
- TDD: every module has a `tests/test_*.py`. Run `uv run pytest -v`.
- The X GraphQL parser anchors on key names, not paths — X's private API drifts.
- Never commit personal data: `auth/storage_state.json`, `data/`, `config.toml`.
  All are gitignored.

## Git workflow
- This repo has only `main`: `feature-branch → PR → main`.

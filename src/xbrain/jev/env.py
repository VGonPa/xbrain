"""Where the TypeSafe API key comes from: the environment, else `<repo>/.env`.

The SDK already reads `TYPESAFE_API_KEY` from the environment itself, so this module exists
to add the `.env` FILE source it lacks — and to make "no key" a local, Spanish failure
instead of a remote authentication error.

`.env` is gitignored. Only `TYPESAFE_API_KEY` is read, from a line that may carry an
`export` prefix, single or double quotes and a trailing `#` comment; the LAST assignment
wins, as `source` would. See `_value_from` for the exact grammar and `test_dotenv_line_forms`
for the cases. No python-dotenv: one key, one file, and a parser small enough to read whole.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

ENV_VAR = "TYPESAFE_API_KEY"
#: A comment starts at a `#` that follows whitespace (space or tab), which is how people
#: actually align them. `KEY=ts-abc#1` is NOT a comment: `#` is legal inside a key.
_COMMENT = re.compile(r"\s#")
_EXPORT = re.compile(r"^export\s+")


def dotenv_path(repo_root: Path) -> Path:
    """The `.env` this module reads — a seam the test suite redirects.

    `tests/conftest.py` points it at an empty directory for EVERY test, so "no key
    configured" can never mean "no key on the machine running the tests".
    """
    return repo_root / ".env"


def _value_from(raw: str) -> str | None:
    """The value of a `KEY=value` line, or None when it is blank or only a comment.

    A value opened with a quote ends at the MATCHING quote, so a `#` inside it is literal
    and everything after it is dropped, comment or not; an unmatched quote is left in place
    rather than half-stripped. An unquoted value ends at its first whitespace-`#`.
    """
    value = raw.strip()
    if value[:1] in ("'", '"'):
        closing = value.find(value[0], 1)
        if closing != -1:
            return value[1:closing].strip() or None
        return value or None
    if value.startswith("#"):
        return None
    return _COMMENT.split(value, maxsplit=1)[0].strip() or None


def typesafe_api_key(repo_root: Path, environ: Mapping[str, str] | None = None) -> str | None:
    """The key from `environ[TYPESAFE_API_KEY]`, else from the `.env`, else None.

    A blank value counts as absent, so a key copied from `.env.example` and left unfilled
    reads as "no key" rather than as an empty string that fails later at the API.
    """
    env = os.environ if environ is None else environ
    value = env.get(ENV_VAR, "").strip()
    if value:
        return value
    dotenv = dotenv_path(repo_root)
    if not dotenv.exists():
        return None
    found: str | None = None
    # `utf-8-sig` drops a BOM, which is not whitespace and would otherwise make the key name
    # never match. Every matching line is read: the LAST wins, like `source`, because
    # rotating a key means pasting the new one at the bottom.
    for line in dotenv.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        # `export FOO=bar` is a valid line in a file people also `source`.
        if _EXPORT.sub("", key.strip()).strip() == ENV_VAR:
            found = _value_from(raw)
    return found

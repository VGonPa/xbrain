"""Where the TypeSafe API key comes from: the environment, else `<repo>/.env`.

`.env` is gitignored and holds one `KEY=value` per line; only `TYPESAFE_API_KEY` is read.
No python-dotenv: the parser is a few lines and the file has one purpose.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ENV_VAR = "TYPESAFE_API_KEY"


def _value_from(raw: str) -> str | None:
    """The value of a `KEY=value` line, or None when it is blank.

    A value opened with a quote ends at the MATCHING quote, so `#` inside it is literal
    and anything after it (a trailing comment) is dropped; an unmatched quote is left in
    place rather than half-stripped. An unquoted value ends at a ` #` comment.
    """
    value = raw.strip()
    if value[:1] in ("'", '"'):
        closing = value.find(value[0], 1)
        if closing != -1:
            return value[1:closing].strip() or None
        return value or None
    return value.split(" #", 1)[0].strip() or None


def typesafe_api_key(repo_root: Path, environ: Mapping[str, str] | None = None) -> str | None:
    """The key from `environ[TYPESAFE_API_KEY]`, else from `<repo_root>/.env`, else None.

    A blank value counts as absent, so the empty `.env.example` line never yields ''.
    """
    env = os.environ if environ is None else environ
    value = env.get(ENV_VAR, "").strip()
    if value:
        return value
    dotenv = repo_root / ".env"
    if not dotenv.exists():
        return None
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        # `export FOO=bar` is a valid line in a file people also `source`.
        if key.strip().removeprefix("export ").strip() == ENV_VAR:
            return _value_from(raw)
    return None

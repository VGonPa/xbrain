# tests/test_transcribe_wrapper_runtime.py — the wrappers must RUN, not just parse.
"""`[transcribe].command` is executed by whatever `python3` the PATH resolves to.

That is not this project's interpreter. On this machine `#!/usr/bin/env python3`
resolved to 3.7.9, and every wrapper died at import with
`TypeError: 'type' object is not subscriptable` — PEP 585 generics (`list[str]`)
in a signature are EVALUATED when the function is defined, and 3.7 cannot. The
scripts still compiled cleanly, so `py_compile` and ruff both stayed green while
`digest-video` failed on every single video and reported it only as
`fallidos: N` with exit code 0.

`from __future__ import annotations` makes annotations strings, so they are never
evaluated and the wrappers load on any Python the shebang may find.
"""

from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_WRAPPERS = sorted(_SCRIPTS.glob("xbrain-transcribe-*"))


def test_there_are_wrappers_to_check():
    """A glob that silently matches nothing would make every test below vacuous."""
    assert _WRAPPERS, f"no transcribe wrappers found under {_SCRIPTS}"


@pytest.mark.parametrize("script", _WRAPPERS, ids=lambda p: p.name)
def test_wrapper_postpones_annotation_evaluation(script: Path):
    """Every wrapper declares `from __future__ import annotations`.

    Asserted on ALL of them, not only the ones that currently use a generic: the
    next `list[str]` someone adds must not reintroduce the failure, and the import
    costs nothing.
    """
    source = script.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source, (
        f"{script.name} must postpone annotation evaluation — its shebang is "
        "`/usr/bin/env python3`, which is not this project's interpreter."
    )


@pytest.mark.parametrize("script", _WRAPPERS, ids=lambda p: p.name)
def test_wrapper_future_import_precedes_every_other_statement(script: Path):
    """`from __future__` is only legal at the top; otherwise it is a SyntaxError."""
    import ast

    tree = ast.parse(script.read_text(encoding="utf-8"))
    body = [n for n in tree.body if not isinstance(n, ast.Expr)]  # drop the docstring
    assert body, f"{script.name} has no statements"
    first = body[0]
    assert isinstance(first, ast.ImportFrom) and first.module == "__future__", (
        f"{script.name}: the __future__ import must come first, got {ast.dump(first)[:60]}"
    )

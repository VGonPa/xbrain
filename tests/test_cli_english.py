"""The CLI speaks English: no Spanish in `--help`, for the app and every command.

The command list is not hand-written: it is walked off the Click object the public
`app` builds, so a command added tomorrow is covered without touching this file
(a hand-written list would go stale silently and keep passing).
"""

from __future__ import annotations

import re

import click
import pytest
import typer
from typer.testing import CliRunner

from xbrain.cli import app

runner = CliRunner()

# Characters English help text never needs and Spanish prose almost always has, plus
# Spanish function words that never occur in English prose. The words close the gap
# the characters leave: "Filtra por estado" carries no accent at all.
SPANISH_MARKERS = re.compile(
    r"[áéíóúÁÉÍÓÚñÑ¿¡]|\b(?:los|las|del|para|que|una|por|sin|con|pero|desde)\b",
    re.IGNORECASE,
)


def _command_paths() -> list[tuple[str, ...]]:
    """Every invocable path under the root app, e.g. ("index", "build")."""
    root = typer.main.get_command(app)
    paths: list[tuple[str, ...]] = [()]

    def walk(cmd: click.Command, prefix: tuple[str, ...]) -> None:
        if isinstance(cmd, click.Group):
            for name, sub in cmd.commands.items():
                path = (*prefix, name)
                paths.append(path)
                walk(sub, path)

    walk(root, ())
    return paths


PATHS = _command_paths()


def test_the_walk_finds_the_command_tree() -> None:
    # Guards the guard: an empty walk would make every parametrized case vacuous.
    assert ("search",) in PATHS
    assert ("index", "build") in PATHS
    assert len(PATHS) > 30


@pytest.mark.parametrize("path", PATHS, ids=lambda p: " ".join(p) or "<root>")
def test_help_has_no_spanish(path: tuple[str, ...]) -> None:
    result = runner.invoke(app, [*path, "--help"], terminal_width=200)
    assert result.exit_code == 0, result.output
    found = sorted(set(SPANISH_MARKERS.findall(result.output)))
    assert not found, (
        f"`xbrain {' '.join(path)} --help` has Spanish markers {found}:\n{result.output}"
    )

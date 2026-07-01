"""Shell out to an external local vision model to describe a video key frame.

The thin, ML-free wrapper for `digest-video --frames`' visual layer (#44 PR4),
mirroring `xbrain.transcribe`. xbrain stays **mechanical**: the heavy vision (a
local VLM) runs as an EXTERNAL subprocess located via config/PATH, so the CLI
carries **no** vision/ML dependency. This module imports no vision library — a
test asserts it.

The vision contract xbrain expects:

- It is invoked as ``<command> [--model M] <image-path>`` where ``<command>``
  comes from ``[vision].command`` and may itself be a multi-token command (a
  wrapper script), split with ``shlex`` and run WITHOUT a shell. The optional
  ``[vision].model`` is passed through as ``--model M``.
- The description is printed on **stdout** as plain text; xbrain strips it and
  uses it as the frame's caption. There is **no bundled default** vision model,
  so an unset / empty ``[vision].command`` is an operator error, not a silent
  no-op — `describe_image` raises `VisionNotFound`.
- **An exit-0 run with no output is a failure, never a silent empty
  description.** Dropping a slide's content invisibly would defeat the whole
  point of the visual layer, so empty stdout raises `VisionFailed`.

Failures surface as clear operator errors (subclasses of `RuntimeError`, which
the CLI's `_handle_cli_errors` turns into a clean exit-1): a **missing /
non-executable / unconfigured** binary (`VisionNotFound`, which — like a missing
transcriber — aborts the run), a **non-zero exit / timeout / empty output**
(`VisionFailed`, which the digest treats as a per-video visual-layer failure and
records). The `runner` (a `subprocess.run` stand-in) is injectable so tests run
offline against a fake — no real vision model, ever.
"""

from __future__ import annotations

import shlex
import subprocess  # nosec B404 - the vision model is an external subprocess by design (#44)
from collections.abc import Callable
from pathlib import Path

# A generous wall-clock cap: describing a single downscaled frame is quick, but a
# wedged VLM process must not hang the run forever.
_DEFAULT_TIMEOUT_SECONDS = 300

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class VisionError(RuntimeError):
    """Base class for every vision failure (a clean CLI exit-1)."""


class VisionNotFound(VisionError):
    """The configured vision command is missing / not executable / unconfigured."""


class VisionFailed(VisionError):
    """The vision command ran but failed: non-zero exit, timeout, or no output."""


def _build_argv(command: str, model: str | None, path: Path) -> list[str]:
    """Assemble the subprocess argv: shlex-split command + optional model + image."""
    argv = shlex.split(command)
    if model:
        argv += ["--model", model]
    argv.append(str(path))
    return argv


def _run_vision(argv: list[str], runner: Runner, timeout_seconds: int) -> str:
    """Run the vision command; return its stdout, or raise a clear operator error."""
    try:
        completed = runner(argv, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise VisionNotFound(
            f"vision command {argv[0]!r} not found — install it or set a valid "
            f"[vision].command in config.toml ({exc})"
        ) from exc
    except OSError as exc:  # not-executable, permission denied, etc.
        raise VisionNotFound(
            f"vision command {argv[0]!r} could not be executed — check "
            f"[vision].command in config.toml ({exc})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VisionFailed(
            f"vision command {argv[0]!r} timed out after {timeout_seconds}s"
        ) from exc
    except UnicodeDecodeError as exc:  # subprocess.run(text=True) on non-UTF-8 stdout
        raise VisionFailed(f"vision command {argv[0]!r} produced non-UTF-8 stdout: {exc}") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise VisionFailed(
            f"vision command {argv[0]!r} exited {completed.returncode}: {stderr or '(no stderr)'}"
        )
    return completed.stdout or ""


def describe_image(
    path: Path | str,
    *,
    command: str,
    model: str | None = None,
    runner: Runner | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Describe one frame image via the external vision subprocess; return the text.

    Shells out to `command` (a multi-token wrapper is supported and run WITHOUT a
    shell) as ``<command> [--model M] <path>`` and returns its stripped stdout. An
    empty / unconfigured `command` raises `VisionNotFound` (there is no bundled
    default); a missing / non-executable binary raises `VisionNotFound` too — both
    abort the run. A non-zero exit, timeout, or **empty output** raises
    `VisionFailed` (never a silent empty description). `runner` defaults to
    `subprocess.run`, resolved at call time so it stays monkeypatchable.
    """
    if not command.strip():
        raise VisionNotFound(
            "no [vision].command configured — set it in config.toml to use "
            "`digest-video --frames` (there is no bundled default vision model)"
        )
    active_runner: Runner = runner if runner is not None else subprocess.run
    argv = _build_argv(command, model, Path(path))
    stdout = _run_vision(argv, active_runner, timeout_seconds)
    description = stdout.strip()
    if not description:
        raise VisionFailed(
            f"vision command {argv[0]!r} exited 0 but produced no description — "
            "a slide's content would be lost"
        )
    return description

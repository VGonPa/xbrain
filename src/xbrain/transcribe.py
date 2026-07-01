"""Shell out to an external local transcriber and parse its output.

The thin, ML-free wrapper at the heart of the `digest-video` stage (#44). xbrain
stays **mechanical**: the heavy ASR (Parakeet TDT via `parakeet-mlx`, or a
whisper fallback) runs as an EXTERNAL subprocess located via config/PATH, so the
CLI carries **no** MLX / CoreML / torch dependency. This module imports no ML
library — a test asserts it.

The transcriber contract xbrain expects:

- It is invoked as ``<command> [--model M] [--language L] --output-format json
  <media-path>`` where ``<command>`` comes from ``[transcribe].command`` (default
  ``parakeet-mlx``) and may itself be a multi-token command (a wrapper script),
  split with ``shlex`` and run WITHOUT a shell.
- On success it writes a single JSON document to **stdout**::

      {"text": "...", "language": "en",
       "segments": [{"start": 0.0, "end": 3.2, "text": "..."}]}

  ``has_speech`` may be reported explicitly; otherwise it is derived (non-empty
  text or any segment ⇒ speech).
- The **no-audio / no-speech** case is graceful, never an error: an empty JSON
  (``{"text": ""}``) OR empty stdout on a zero exit both yield
  ``has_speech=False`` + empty text. Many X videos are silent/screen-only, so a
  no-speech outcome is recorded data, not a failure.

Failures surface as clear operator errors (subclasses of `RuntimeError`, which
the CLI's `_handle_cli_errors` turns into a clean exit-1): a **missing binary**
(`TranscriberNotFound`), a **non-zero exit** or **malformed output**
(`TranscriberFailed`). The `runner` (a `subprocess.run` stand-in) is injectable
so tests run offline against a fake — no real transcriber, ever.
"""

from __future__ import annotations

import json
import shlex
import subprocess  # nosec B404 - the transcriber is an external subprocess by design (#44)
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# The default external transcriber (Parakeet TDT 0.6b via MLX/Metal). Override
# with `[transcribe].command`; whisper / faster-whisper is the portable fallback.
DEFAULT_TRANSCRIBE_COMMAND = "parakeet-mlx"

# A generous default wall-clock cap: a 72-minute talk transcribes in minutes on
# Apple Silicon, but a wedged process must not hang the run forever.
_DEFAULT_TIMEOUT_SECONDS = 1800

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class TranscribeError(RuntimeError):
    """Base class for every transcriber failure (a clean CLI exit-1)."""


class TranscriberNotFound(TranscribeError):
    """The configured transcriber binary is missing / not executable."""


class TranscriberFailed(TranscribeError):
    """The transcriber ran but failed: non-zero exit or unparseable output."""


@dataclass(frozen=True)
class Segment:
    """One timestamped transcript segment (seconds)."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """The parsed result of a transcription run.

    `has_speech=False` (with empty `text` + no `segments`) is the graceful
    no-audio / no-speech outcome — carried, not raised. `language` is the
    detected language when the transcriber reports it, else the requested one.
    """

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    has_speech: bool = False


def _build_argv(command: str, model: str | None, language: str | None, path: Path) -> list[str]:
    """Assemble the subprocess argv from config + the media path.

    `command` is `shlex`-split so a multi-token wrapper (`python -m my_asr`)
    works, and the whole thing runs WITHOUT a shell (no injection surface).
    """
    argv = shlex.split(command)
    if model:
        argv += ["--model", model]
    if language:
        argv += ["--language", language]
    argv += ["--output-format", "json", str(path)]
    return argv


def _run_transcriber(argv: list[str], runner: Runner, timeout_seconds: int) -> str:
    """Run the transcriber; return its stdout, or raise a clear operator error."""
    try:
        completed = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise TranscriberNotFound(
            f"transcriber {argv[0]!r} not found — install it or set a valid "
            f"[transcribe].command in config.toml ({exc})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriberFailed(
            f"transcriber {argv[0]!r} timed out after {timeout_seconds}s"
        ) from exc
    except OSError as exc:  # not-executable, permission denied, etc.
        raise TranscriberNotFound(
            f"transcriber {argv[0]!r} could not be executed — check "
            f"[transcribe].command in config.toml ({exc})"
        ) from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise TranscriberFailed(
            f"transcriber {argv[0]!r} exited {completed.returncode}: {stderr or '(no stderr)'}"
        )
    return completed.stdout or ""


def _parse_segment(raw: object) -> Segment:
    """Parse one segment dict into a `Segment`, or raise on a malformed shape."""
    if not isinstance(raw, dict):
        raise TranscriberFailed(f"transcriber segment is not an object: {raw!r}")
    try:
        return Segment(
            start=float(raw.get("start", 0.0)),
            end=float(raw.get("end", 0.0)),
            text=str(raw.get("text", "")),
        )
    except (TypeError, ValueError) as exc:
        raise TranscriberFailed(f"transcriber segment has non-numeric bounds: {raw!r}") from exc


def _parse_output(stdout: str, requested_language: str | None) -> Transcript:
    """Parse the transcriber stdout into a `Transcript`.

    Empty/whitespace stdout is the graceful no-speech path (many silent clips
    print nothing); otherwise the payload must be a JSON object, else it is
    malformed output (`TranscriberFailed`).
    """
    stripped = stdout.strip()
    if not stripped:
        return Transcript(text="", segments=[], language=requested_language, has_speech=False)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise TranscriberFailed(f"transcriber output was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TranscriberFailed(f"transcriber output was not a JSON object: {type(data).__name__}")

    text = str(data.get("text") or "")
    segments = [_parse_segment(s) for s in data.get("segments", [])]
    language = data.get("language") or requested_language
    reported = data.get("has_speech")
    has_speech = bool(reported) if reported is not None else bool(text.strip() or segments)
    return Transcript(text=text, segments=segments, language=language, has_speech=has_speech)


def transcribe_media(
    path: Path | str,
    *,
    command: str = DEFAULT_TRANSCRIBE_COMMAND,
    model: str | None = None,
    language: str | None = None,
    runner: Runner | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> Transcript:
    """Transcribe one audio/video file via the external transcriber subprocess.

    Shells out to `command` (default `parakeet-mlx`; a multi-token wrapper is
    supported and run WITHOUT a shell), passing the optional `model` / `language`
    and the media `path`, then parses the JSON stdout into a `Transcript`. The
    no-audio / no-speech case is graceful (`has_speech=False`, empty `text`,
    never raises). A missing binary raises `TranscriberNotFound`; a non-zero exit
    or malformed output raises `TranscriberFailed` — both clean CLI exit-1s.

    `runner` (a `subprocess.run` stand-in) is injectable for tests; it defaults
    to `subprocess.run`, resolved at call time so it stays monkeypatchable.
    """
    argv = _build_argv(command, model, language, Path(path))
    active_runner: Runner = runner if runner is not None else subprocess.run
    stdout = _run_transcriber(argv, active_runner, timeout_seconds)
    return _parse_output(stdout, language)

"""Tests for `xbrain.transcribe` — the external-transcriber subprocess wrapper.

`transcribe_media` shells out to an EXTERNAL local transcriber (config
`[transcribe].command`, default `parakeet-mlx`) and parses its JSON output into
a `Transcript`. It imports NO ML library — the heavy ASR lives outside xbrain
core (the locked #44 architecture). Every test injects a fake `runner` (a stand
-in for `subprocess.run`) so NO real subprocess runs; the no-audio/no-speech
path yields `has_speech=False` + empty text and never raises, and a missing
binary surfaces a clear operator error.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from xbrain.transcribe import (
    DEFAULT_TRANSCRIBE_COMMAND,
    Segment,
    Transcript,
    TranscriberFailed,
    TranscriberNotFound,
    transcribe_media,
)


def _completed(
    stdout: str, *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess:
    """A `CompletedProcess` stand-in for the injected fake runner."""
    return subprocess.CompletedProcess(
        args=["x"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _runner_returning(stdout: str, *, returncode: int = 0, stderr: str = ""):
    """A fake `subprocess.run` that records its argv and returns canned output."""
    calls: list[list[str]] = []

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        return _completed(stdout, returncode=returncode, stderr=stderr)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


# ------------------------------------------------------------ happy path


def test_parses_transcriber_json_into_transcript(tmp_path: Path):
    payload = {
        "text": "hello world",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello"},
            {"start": 1.5, "end": 2.7, "text": "world"},
        ],
    }
    result = transcribe_media(tmp_path / "a.mp4", runner=_runner_returning(json.dumps(payload)))
    assert isinstance(result, Transcript)
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.has_speech is True
    assert result.segments == [
        Segment(start=0.0, end=1.5, text="hello"),
        Segment(start=1.5, end=2.7, text="world"),
    ]


def test_builds_argv_from_command_model_language_and_path(tmp_path: Path):
    """The subprocess argv carries the configured command + model + language + the
    media path — the operator's `[transcribe]` config drives the external call."""
    runner = _runner_returning(json.dumps({"text": "hi", "segments": []}))
    media = tmp_path / "clip.mp4"
    transcribe_media(
        media, command="my-asr", model="parakeet-tdt-0.6b", language="es", runner=runner
    )
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[0] == "my-asr"
    assert "parakeet-tdt-0.6b" in argv
    assert "es" in argv
    assert str(media) == argv[-1]


def test_multi_token_command_is_split(tmp_path: Path):
    """A `[transcribe].command` with args (e.g. a wrapper `python -m foo`) is
    split into argv tokens, not run through a shell."""
    runner = _runner_returning(json.dumps({"text": "hi", "segments": []}))
    transcribe_media(tmp_path / "a.mp4", command="python -m my_asr", runner=runner)
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[:3] == ["python", "-m", "my_asr"]


def test_default_command_is_parakeet_mlx(tmp_path: Path):
    runner = _runner_returning(json.dumps({"text": "hi", "segments": []}))
    transcribe_media(tmp_path / "a.mp4", runner=runner)
    assert runner.calls[0][0] == DEFAULT_TRANSCRIBE_COMMAND  # type: ignore[attr-defined]
    assert DEFAULT_TRANSCRIBE_COMMAND == "parakeet-mlx"


# ------------------------------------------------------------ no-speech / no-audio


def test_empty_json_yields_no_speech_never_raises(tmp_path: Path):
    """A silent clip: transcriber emits `{"text": ""}` → has_speech=False, empty
    text, no exception (the corpus has many no-audio / screen-only videos)."""
    result = transcribe_media(
        tmp_path / "silent.mp4", runner=_runner_returning(json.dumps({"text": "", "segments": []}))
    )
    assert result.has_speech is False
    assert result.text == ""
    assert result.segments == []


def test_empty_stdout_yields_no_speech(tmp_path: Path):
    """A transcriber that prints NOTHING for a silent clip (exit 0, empty stdout)
    is treated as no-speech, not a parse error — graceful by contract."""
    result = transcribe_media(tmp_path / "silent.mp4", runner=_runner_returning("   \n"))
    assert result.has_speech is False
    assert result.text == ""


def test_whitespace_only_text_is_no_speech(tmp_path: Path):
    result = transcribe_media(
        tmp_path / "s.mp4", runner=_runner_returning(json.dumps({"text": "   ", "segments": []}))
    )
    assert result.has_speech is False


def test_explicit_has_speech_flag_is_honored(tmp_path: Path):
    """If the transcriber reports `has_speech` explicitly, xbrain trusts it over
    the empty-text heuristic."""
    result = transcribe_media(
        tmp_path / "s.mp4",
        runner=_runner_returning(json.dumps({"text": "", "has_speech": True, "segments": []})),
    )
    assert result.has_speech is True


# ------------------------------------------------------------ errors


def test_missing_binary_raises_clear_operator_error(tmp_path: Path):
    """A missing transcriber binary is a clear operator error (install/configure
    `[transcribe].command`), not a raw crash."""

    def _run(_argv, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "parakeet-mlx")

    with pytest.raises(TranscriberNotFound) as excinfo:
        transcribe_media(tmp_path / "a.mp4", runner=_run)
    assert "parakeet-mlx" in str(excinfo.value)
    assert "transcribe" in str(excinfo.value).lower()


def test_nonzero_exit_raises_with_stderr(tmp_path: Path):
    runner = _runner_returning("", returncode=3, stderr="model weights not found")
    with pytest.raises(TranscriberFailed) as excinfo:
        transcribe_media(tmp_path / "a.mp4", runner=runner)
    assert "model weights not found" in str(excinfo.value)


def test_malformed_json_raises(tmp_path: Path):
    runner = _runner_returning("this is not json {")
    with pytest.raises(TranscriberFailed):
        transcribe_media(tmp_path / "a.mp4", runner=runner)


def test_malformed_segment_raises(tmp_path: Path):
    """A segment missing/holding a non-numeric bound is malformed output → error
    (not a silently-dropped segment)."""
    runner = _runner_returning(
        json.dumps({"text": "hi", "segments": [{"start": "nope", "end": 1.0, "text": "x"}]})
    )
    with pytest.raises(TranscriberFailed):
        transcribe_media(tmp_path / "a.mp4", runner=runner)


def test_timeout_raises_transcriber_failed(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="parakeet-mlx", timeout=1)

    with pytest.raises(TranscriberFailed):
        transcribe_media(tmp_path / "a.mp4", runner=_run)


def test_transcribe_imports_no_ml_library():
    """The locked #44 architecture: the CLI carries NO MLX/CoreML/torch/whisper
    dependency — the transcriber is an external subprocess. Guard it so a future
    edit can't quietly `import mlx`."""
    import xbrain.transcribe as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import mlx",
        "import torch",
        "import whisper",
        "parakeet_mlx",
        "coremltools",
    ):
        assert forbidden not in source

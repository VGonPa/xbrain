# tests/test_xbrain_transcribe_mlx.py — the scripts/xbrain-transcribe-mlx GPU backend.
"""The GPU multilingual backend.

Its reason to exist is a measurement, not a preference: the CPU whisper CLI took
7 min 25 s on a 68-second Spanish clip from the store (six times slower than
realtime) where mlx-whisper took 17.7 s for a character-identical transcript. The
behaviour these tests pin is what makes that swap safe — when the accelerator is
not available the script must fail LOUDLY and write NOTHING, so the router falls
back instead of the pipeline recording a transcription failure.
"""

import importlib.util
import json
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-transcribe-mlx"
_LOADER = SourceFileLoader("xbrain_transcribe_mlx", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_transcribe_mlx", _LOADER)
xtm = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xtm)

_RESULT = {
    "text": " Hola, esto es una prueba. ",
    "language": "es",
    "segments": [{"start": 0.0, "end": 2.0, "text": " Hola, esto es una prueba. ", "tokens": [1]}],
}


def _drive(monkeypatch, tmp_path, *, result=_RESULT, uv=True, rc=0, no_audio=False, env=None):
    """Run main() with the uv/mlx subprocess mocked; return (code, argv, out_path)."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    seen: dict = {}
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    monkeypatch.setattr(xtm.shutil, "which", lambda n: "/bin/uv" if uv else None)
    monkeypatch.setattr(xtm, "confirmed_no_audio", lambda m: no_audio)

    def fake_run(cmd, *a, **k):
        seen["argv"] = cmd
        stdout = json.dumps(result) if (rc == 0 and result is not None) else ""
        return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr="boom")

    monkeypatch.setattr(xtm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xbrain-transcribe-mlx",
            "--output-format",
            "json",
            "--output-dir",
            str(tmp_path),
            str(media),
        ],
    )
    return xtm.main(), seen.get("argv"), tmp_path / "clip.json"


def test_output_is_normalised_to_the_shape_xbrain_reads(monkeypatch, tmp_path):
    code, _argv, out = _drive(monkeypatch, tmp_path)
    assert code == 0
    written = json.loads(out.read_text())
    assert written["text"] == "Hola, esto es una prueba."
    assert written["language"] == "es"
    assert written["has_speech"] is True
    # Extra model fields (tokens, avg_logprob…) are dropped; the shape is fixed.
    assert written["segments"] == [{"start": 0.0, "end": 2.0, "text": "Hola, esto es una prueba."}]


@pytest.mark.parametrize(
    "reason,kwargs",
    [
        ("no uv on PATH", {"uv": False}),
        ("uv/mlx exited non-zero", {"rc": 1}),
        ("no output at all", {"result": None}),
    ],
)
def test_unusable_backend_fails_without_writing_anything(monkeypatch, tmp_path, reason, kwargs):
    """The fallback contract. A written file would make the router think it worked."""
    code, _argv, out = _drive(monkeypatch, tmp_path, **kwargs)
    assert code == 1, reason
    assert not out.exists(), reason


def test_unreadable_output_is_a_failure_not_a_crash(monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    monkeypatch.setattr(xtm.shutil, "which", lambda n: "/bin/uv")
    monkeypatch.setattr(xtm, "confirmed_no_audio", lambda m: False)
    monkeypatch.setattr(
        xtm.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="{not json", stderr=""),
    )
    monkeypatch.setattr(sys, "argv", ["x", "--output-dir", str(tmp_path), str(media)])
    assert xtm.main() == 1
    assert not (tmp_path / "clip.json").exists()


@pytest.mark.parametrize(
    "model,expected",
    [
        (None, "mlx-community/whisper-small-mlx"),
        ("small", "mlx-community/whisper-small-mlx"),
        ("large-v3-turbo", "mlx-community/whisper-large-v3-turbo"),
        ("someone/custom-mlx-repo", "someone/custom-mlx-repo"),  # explicit repo passes through
        ("mlx-community/parakeet-tdt-0.6b-v2", "mlx-community/parakeet-tdt-0.6b-v2"),
        ("not-a-model", "mlx-community/whisper-small-mlx"),  # unknown name → the default
    ],
)
def test_resolve_repo_maps_names_and_passes_repos_through(monkeypatch, model, expected):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert xtm.resolve_repo(model) == expected


def test_language_is_forwarded_to_the_model(monkeypatch, tmp_path):
    _code, argv, _out = _drive(monkeypatch, tmp_path, env={"WHISPER_LANGUAGE": "es"})
    assert argv is not None and argv[-1] == "es"


@pytest.mark.parametrize("value", ["auto", "AUTO", ""])
def test_auto_language_lets_the_model_detect(monkeypatch, tmp_path, value):
    """An empty language is the detection request — never a guessed default."""
    _code, argv, _out = _drive(monkeypatch, tmp_path, env={"WHISPER_LANGUAGE": value})
    assert argv is not None and argv[-1] == ""


def test_public_index_is_pinned_on_the_uv_resolve(monkeypatch, tmp_path):
    """The machine's pip.conf points at a private index that needs auth and would
    hang the resolve waiting on stdin."""
    _code, argv, _out = _drive(monkeypatch, tmp_path)
    assert argv is not None
    assert argv[argv.index("--index-url") + 1] == "https://pypi.org/simple"


def test_confirmed_silent_video_yields_empty_speech_without_running_the_model(
    monkeypatch, tmp_path
):
    code, argv, out = _drive(monkeypatch, tmp_path, no_audio=True)
    assert code == 0
    assert argv is None  # the model never ran
    assert json.loads(out.read_text())["has_speech"] is False


@pytest.mark.parametrize(
    "artefact",
    [
        "Subtítulos realizados por la comunidad de Amara.org",
        "Thanks for watching!",
        "[Música]",
    ],
)
def test_no_speech_hallucinations_are_discarded_here_too(monkeypatch, tmp_path, artefact):
    """Both whisper backends wrap the same model family, so both inherit the
    artefact — a guard on only one would make the vault's protection depend on
    which machine ran the nightly."""
    _code, _argv, out = _drive(
        monkeypatch,
        tmp_path,
        result={
            "text": artefact,
            "language": "es",
            "segments": [{"start": 0.0, "end": 1.0, "text": artefact}],
        },
    )
    written = json.loads(out.read_text())
    assert written["text"] == ""
    assert written["segments"] == []
    assert written["has_speech"] is False


def test_real_speech_mentioning_an_artefact_phrase_survives(monkeypatch, tmp_path):
    real = "Thanks for watching! Now here is the actual demo."
    _code, _argv, out = _drive(
        monkeypatch, tmp_path, result={"text": real, "language": "en", "segments": []}
    )
    assert json.loads(out.read_text())["text"] == real


def test_missing_media_is_a_clean_usage_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["xbrain-transcribe-mlx", "--output-dir", "/tmp"])
    assert xtm.main() == 2


def test_hallucination_lists_are_identical_across_both_whisper_backends():
    """One drifting list would silently reopen the hole on one backend only."""
    other = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-transcribe-whisper"
    loader = SourceFileLoader("xbrain_transcribe_whisper_cmp", str(other))
    spec = importlib.util.spec_from_loader("xbrain_transcribe_whisper_cmp", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    assert xtm.NO_SPEECH_ARTEFACTS == module._NO_SPEECH_ARTEFACTS

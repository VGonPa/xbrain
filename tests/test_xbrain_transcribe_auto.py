# tests/test_xbrain_transcribe_auto.py — the scripts/xbrain-transcribe-auto ASR router.
"""The routing decision is the whole point, so these tests pin the DECISION.

parakeet does not fail on Spanish audio, it fabricates: exit 0, fluent broken
English, nothing that was actually said. That failure is invisible downstream, so
"which backend" must be settled before transcription and must fail toward whisper
on every uncertainty.
"""

import importlib.util
import json
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-transcribe-auto"
_LOADER = SourceFileLoader("xbrain_transcribe_auto", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_transcribe_auto", _LOADER)
xta = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xta)


@pytest.mark.parametrize(
    "language,expected",
    [
        ("en", "parakeet"),
        ("english", "parakeet"),
        ("EN", "parakeet"),  # normalised by the caller, but the set is lowercase
        ("es", "whisper"),
        ("fr", "whisper"),
        ("", "whisper"),
        (None, "whisper"),  # UNDETECTED must never reach parakeet
    ],
)
def test_pick_backend_routes_non_english_and_unknown_to_whisper(language, expected):
    assert xta.pick_backend(language.lower() if isinstance(language, str) else language) == expected


def _stub_tools(monkeypatch, *, ffmpeg=True, whisper=True):
    monkeypatch.setattr(
        xta.shutil,
        "which",
        lambda name: {
            "ffmpeg": "/bin/ffmpeg" if ffmpeg else None,
            "whisper": "/bin/whisper" if whisper else None,
            "ffprobe": "/bin/ffprobe",
        }.get(name),
    )


def test_detect_language_reads_whispers_reported_language(monkeypatch, tmp_path):
    """Detection is whisper run WITHOUT `--language` — the omission is the request."""
    _stub_tools(monkeypatch)
    monkeypatch.setattr(xta.tempfile, "mkdtemp", lambda **k: str(tmp_path))
    seen: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        seen.append(cmd)
        if cmd[0] == "/bin/ffmpeg":
            (tmp_path / "probe.wav").write_bytes(b"x")
        else:
            (tmp_path / "probe.json").write_text(json.dumps({"language": "Spanish"}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(xta.subprocess, "run", fake_run)
    monkeypatch.setattr(xta.shutil, "rmtree", lambda *a, **k: None)

    assert xta.detect_language("/clips/a.mp4") == "spanish"
    whisper_cmd = next(c for c in seen if c[0] == "/bin/whisper")
    assert "--language" not in whisper_cmd


@pytest.mark.parametrize(
    "scenario",
    ["no_ffmpeg", "no_whisper", "ffmpeg_fails", "whisper_fails", "no_output", "bad_json"],
)
def test_detect_language_returns_none_on_every_failure(monkeypatch, tmp_path, scenario):
    """Every uncertainty collapses to None — and None routes to whisper."""
    _stub_tools(monkeypatch, ffmpeg=scenario != "no_ffmpeg", whisper=scenario != "no_whisper")
    monkeypatch.setattr(xta.tempfile, "mkdtemp", lambda **k: str(tmp_path))
    monkeypatch.setattr(xta.shutil, "rmtree", lambda *a, **k: None)

    def fake_run(cmd, *a, **k):
        if cmd[0] == "/bin/ffmpeg":
            if scenario == "ffmpeg_fails":
                return types.SimpleNamespace(returncode=1, stdout="", stderr="")
            (tmp_path / "probe.wav").write_bytes(b"x")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if scenario == "whisper_fails":
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        if scenario == "bad_json":
            (tmp_path / "probe.json").write_text("{not json")
        elif scenario != "no_output":
            (tmp_path / "probe.json").write_text(json.dumps({"language": "es"}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(xta.subprocess, "run", fake_run)

    assert xta.detect_language("/clips/a.mp4") is None
    assert xta.pick_backend(xta.detect_language("/clips/a.mp4")) == "whisper"


def _drive(monkeypatch, tmp_path, *, language, no_audio=False, force=None, codes=None):
    """Run main() with detection stubbed; return (rc, all_delegated_argvs, last_env).

    `codes` maps a backend-script suffix to the exit code it should return, so a
    test can make the GPU backend unavailable and watch the fallback.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    calls: list[list[str]] = []
    envs: list[dict] = []

    monkeypatch.setattr(xta, "_confirmed_no_audio", lambda m: no_audio)
    monkeypatch.setattr(xta, "detect_language", lambda m: language)
    monkeypatch.delenv("XBRAIN_ASR_FORCE", raising=False)
    if force:
        monkeypatch.setenv("XBRAIN_ASR_FORCE", force)

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        envs.append(k.get("env", {}))
        code = 0
        for suffix, value in (codes or {}).items():
            if cmd[1].endswith(suffix):
                code = value
        return types.SimpleNamespace(returncode=code, stdout="", stderr="")

    monkeypatch.setattr(xta.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xbrain-transcribe-auto",
            "--output-format",
            "json",
            "--output-dir",
            str(tmp_path),
            str(media),
        ],
    )
    return xta.main(), calls, (envs[-1] if envs else {})


def _last_backend(calls):
    return calls[-1][1] if calls else None


def test_english_is_delegated_to_parakeet(monkeypatch, tmp_path):
    rc, calls, _env = _drive(monkeypatch, tmp_path, language="en")
    assert rc == 0
    assert len(calls) == 1
    assert _last_backend(calls).endswith("xbrain-transcribe")


def test_spanish_is_delegated_to_a_whisper_backend_with_its_language(monkeypatch, tmp_path):
    """The measured bug: this exact clip used to reach parakeet and come back
    as fabricated English."""
    rc, calls, env = _drive(monkeypatch, tmp_path, language="es")
    assert rc == 0
    assert _last_backend(calls).endswith("xbrain-transcribe-mlx")  # GPU first
    assert env["WHISPER_LANGUAGE"] == "es"


def test_gpu_backend_failure_falls_back_to_the_cpu_one(monkeypatch, tmp_path):
    """A missing accelerator must degrade to SLOW, never to wrong.

    `xbrain-transcribe-mlx` exits non-zero without writing when it cannot run, so
    the CPU wrapper gets a clean output dir and the same arguments.
    """
    rc, calls, env = _drive(
        monkeypatch, tmp_path, language="es", codes={"xbrain-transcribe-mlx": 1}
    )
    assert rc == 0
    assert [Path(c[1]).name for c in calls] == [
        "xbrain-transcribe-mlx",
        "xbrain-transcribe-whisper",
    ]
    assert env["WHISPER_LANGUAGE"] == "es"


def test_both_whisper_backends_failing_surfaces_a_failure(monkeypatch, tmp_path):
    """Nothing left to try is a real failure — never a silent success."""
    rc, calls, _env = _drive(
        monkeypatch,
        tmp_path,
        language="es",
        codes={"xbrain-transcribe-mlx": 1, "xbrain-transcribe-whisper": 3},
    )
    assert rc == 3
    assert len(calls) == 2


def test_undetected_language_goes_to_whisper_on_autodetect(monkeypatch, tmp_path):
    """Not Spanish-by-default: forcing a guessed language is the same class of bug."""
    _rc, calls, env = _drive(monkeypatch, tmp_path, language=None)
    assert _last_backend(calls).endswith("xbrain-transcribe-mlx")
    assert env["WHISPER_LANGUAGE"] == "auto"


def test_original_arguments_are_passed_through_untouched(monkeypatch, tmp_path):
    _rc, calls, _env = _drive(monkeypatch, tmp_path, language="en")
    assert calls[-1][2:] == [
        "--output-format",
        "json",
        "--output-dir",
        str(tmp_path),
        str(tmp_path / "clip.mp4"),
    ]


@pytest.mark.parametrize(
    "force,name", [("parakeet", "xbrain-transcribe"), ("whisper", "xbrain-transcribe-mlx")]
)
def test_force_env_skips_detection(monkeypatch, tmp_path, force, name):
    """An operator override must not pay for a detection pass it is overriding."""
    called: list[str] = []
    monkeypatch.setattr(xta, "detect_language", lambda m: called.append(m) or "es")
    _rc, calls, _env = _drive(monkeypatch, tmp_path, language="es", force=force)
    assert Path(_last_backend(calls)).name == name
    assert called == []  # detection never ran


def test_silent_video_short_circuits_before_any_detection(monkeypatch, tmp_path):
    """A clip with no audio track needs no ASR and no language — just the signal."""
    rc, calls, _env = _drive(monkeypatch, tmp_path, language="en", no_audio=True)
    assert rc == 0
    assert calls == []  # nothing was delegated
    written = json.loads((tmp_path / "clip.json").read_text())
    assert written == {"text": "", "language": None, "segments": [], "has_speech": False}


def test_missing_media_is_a_clean_usage_error(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["xbrain-transcribe-auto"])
    assert xta.main() == 2

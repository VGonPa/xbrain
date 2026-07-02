# tests/test_xbrain_transcribe.py — the scripts/xbrain-transcribe parakeet wrapper.
import importlib.util
import json
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-transcribe"
_LOADER = SourceFileLoader("xbrain_transcribe", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_transcribe", _LOADER)
xt = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xt)


def _run(monkeypatch, tmp_path, *, parakeet_rc, writes, confirmed_no_audio):
    """Drive main() with parakeet + ffprobe mocked.

    `writes` is what parakeet "produces": None (nothing), or a (filename, content)
    tuple written to `tmp_path` (content "" simulates an empty stub file).
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def fake_run(cmd, *a, **k):
        if writes is not None and parakeet_rc == 0:
            name, content = writes
            (tmp_path / name).write_text(content)
        return types.SimpleNamespace(returncode=parakeet_rc, stdout="", stderr="")

    monkeypatch.setattr(xt.subprocess, "run", fake_run)
    monkeypatch.setattr(xt, "_confirmed_no_audio", lambda m: confirmed_no_audio)
    monkeypatch.setattr(
        sys,
        "argv",
        ["xbrain-transcribe", "--output-format", "json", "--output-dir", str(tmp_path), str(media)],
    )
    return xt.main(), tmp_path


def _only_json(d: Path) -> dict:
    return json.loads(
        next(p for p in sorted(d.glob("*.json")) if p.read_text().strip()).read_text()
    )


def test_speech_passes_parakeet_transcript_through(monkeypatch, tmp_path):
    rc, d = _run(
        monkeypatch,
        tmp_path,
        parakeet_rc=0,
        writes=("clip.json", '{"text": "hola", "segments": []}'),
        confirmed_no_audio=False,
    )
    assert rc == 0
    assert _only_json(d)["text"] == "hola"  # not overwritten


def test_transcript_detected_regardless_of_output_filename(monkeypatch, tmp_path):
    # xbrain globs *.json by content, not <stem>.json — a differently-named non-empty
    # file must still count as "parakeet produced a transcript" (no false failure).
    rc, d = _run(
        monkeypatch,
        tmp_path,
        parakeet_rc=0,
        writes=("weird-name.json", '{"text": "hi"}'),
        confirmed_no_audio=False,
    )
    assert rc == 0


def test_confirmed_silent_gets_empty_speech_json(monkeypatch, tmp_path):
    rc, d = _run(monkeypatch, tmp_path, parakeet_rc=0, writes=None, confirmed_no_audio=True)
    assert rc == 0
    data = _only_json(d)
    assert data["has_speech"] is False and data["text"] == ""


def test_empty_stub_json_is_not_treated_as_output(monkeypatch, tmp_path):
    # parakeet writes a 0-content stub → must NOT count as a transcript; the silent
    # path runs and replaces it with the empty-speech JSON (finding #1).
    rc, d = _run(
        monkeypatch, tmp_path, parakeet_rc=0, writes=("clip.json", "   "), confirmed_no_audio=True
    )
    assert rc == 0
    assert _only_json(d)["has_speech"] is False


def test_audio_or_unverifiable_is_a_real_failure(monkeypatch, tmp_path):
    # exit 0, no transcript, ffprobe did NOT confirm silence → real failure, not masked.
    rc, d = _run(monkeypatch, tmp_path, parakeet_rc=0, writes=None, confirmed_no_audio=False)
    assert rc == 1
    assert not any(p.read_text().strip() for p in d.glob("*.json"))  # nothing written


def test_parakeet_nonzero_exit_propagates(monkeypatch, tmp_path):
    rc, d = _run(monkeypatch, tmp_path, parakeet_rc=2, writes=None, confirmed_no_audio=True)
    assert rc == 2
    assert not list(d.glob("*.json"))


def _fake_ffprobe(monkeypatch, *, returncode, stdout, has_ffprobe=True):
    monkeypatch.setattr(xt.shutil, "which", lambda _: "/usr/bin/ffprobe" if has_ffprobe else None)
    monkeypatch.setattr(
        xt.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=""),
    )


def test_confirmed_no_audio_true_when_ffprobe_finds_no_audio(monkeypatch):
    _fake_ffprobe(monkeypatch, returncode=0, stdout="")
    assert xt._confirmed_no_audio("/x/clip.mp4") is True


def test_confirmed_no_audio_false_when_audio_present(monkeypatch):
    _fake_ffprobe(monkeypatch, returncode=0, stdout="audio\n")
    assert xt._confirmed_no_audio("/x/clip.mp4") is False


def test_confirmed_no_audio_false_when_ffprobe_errors(monkeypatch):
    _fake_ffprobe(monkeypatch, returncode=1, stdout="")  # can't trust → not silent
    assert xt._confirmed_no_audio("/x/clip.mp4") is False


def test_confirmed_no_audio_false_without_ffprobe_or_path(monkeypatch):
    _fake_ffprobe(monkeypatch, returncode=0, stdout="", has_ffprobe=False)
    assert xt._confirmed_no_audio("/x/clip.mp4") is False
    assert xt._confirmed_no_audio("") is False

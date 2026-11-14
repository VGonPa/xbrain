# tests/test_xbrain_vision.py — the scripts/xbrain-vision model-selector wrapper.
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

# The wrapper is a bare script (no .py suffix), so give importlib an explicit
# source loader. Top-level imports are stdlib only → safe without mlx/anthropic.
_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-vision"
_LOADER = SourceFileLoader("xbrain_vision", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_vision", _LOADER)
xv = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xv)


def test_resolve_local_aliases():
    assert xv._resolve("qwen-3b") == ("local", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    assert xv._resolve("qwen-7b") == ("local", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    assert xv._resolve("qwen-32b")[0] == "local"


def test_resolve_cloud_aliases_use_current_model_ids():
    assert xv._resolve("opus") == ("cloud", "claude-opus-4-8")
    assert xv._resolve("sonnet") == ("cloud", "claude-sonnet-4-6")
    assert xv._resolve("haiku") == ("cloud", "claude-haiku-4-5")


def test_resolve_claude_prefix_passthrough():
    assert xv._resolve("claude-opus-4-8") == ("cloud", "claude-opus-4-8")


def test_resolve_hf_repo_is_local():
    assert xv._resolve("mlx-community/Some-VLM-4bit") == ("local", "mlx-community/Some-VLM-4bit")


def test_resolve_unknown_model_exits():
    with pytest.raises(SystemExit):
        xv._resolve("gpt-9")


def test_default_model_is_local_qwen3b():
    assert xv.DEFAULT_MODEL == "qwen-3b"
    assert xv._resolve(xv.DEFAULT_MODEL)[0] == "local"


def test_main_returns_1_on_empty_description(monkeypatch, tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(xv, "_describe_local", lambda model, image: "")
    monkeypatch.setattr(sys, "argv", ["xbrain-vision", "--model", "qwen-3b", str(img)])
    assert xv.main() == 1  # empty output is a failure, per the vision contract


def test_main_prints_description_and_returns_0(monkeypatch, tmp_path, capsys):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(xv, "_describe_cloud", lambda model, image: "Un gráfico de barras.")
    monkeypatch.setattr(sys, "argv", ["xbrain-vision", "--model", "opus", str(img)])
    assert xv.main() == 0
    assert "Un gráfico de barras." in capsys.readouterr().out


def test_prompt_prefers_the_injected_rubric(monkeypatch):
    """xbrain injects the frame rubric as XBRAIN_VISION_PROMPT; the wrapper is the
    reference implementation of that contract, so it must USE it rather than its
    own constant. Read at CALL time, so a value set after import still applies."""
    monkeypatch.setenv("XBRAIN_VISION_PROMPT", "transcribe verbatim, injected")
    assert xv._prompt() == "transcribe verbatim, injected"


def test_prompt_falls_back_when_the_env_var_is_absent(monkeypatch):
    """Run bare, outside xbrain, the script still needs a working prompt — an empty
    one would produce a useless caption instead of an error."""
    monkeypatch.delenv("XBRAIN_VISION_PROMPT", raising=False)
    assert xv._prompt() == xv._FALLBACK_PROMPT


def test_prompt_treats_a_blank_env_var_as_absent(monkeypatch):
    """A variable set to whitespace is a misconfiguration, not an instruction to
    send the model nothing."""
    monkeypatch.setenv("XBRAIN_VISION_PROMPT", "   \n  ")
    assert xv._prompt() == xv._FALLBACK_PROMPT


def test_fallback_prompt_still_demands_verbatim_transcription():
    """The fallback is a real prompt, not a stub: a bare run must still get the
    discipline #90 exists to enforce, or the caption silently paraphrases again."""
    assert "VERBATIM" in xv._FALLBACK_PROMPT
    assert xv._FALLBACK_PROMPT.strip()


def test_wrapper_and_xbrain_agree_on_the_env_var_name():
    """The two halves of the injection contract spell this string INDEPENDENTLY —
    the wrapper cannot import xbrain (it runs under the system python), so the
    duplication is deliberate. Nothing else pins them equal: rename either side and
    every test still passes while production silently falls back to the weaker
    built-in prompt, reverting this whole change with no failure anywhere.
    """
    from xbrain.vision import PROMPT_ENV_VAR

    assert xv._PROMPT_ENV_VAR == PROMPT_ENV_VAR

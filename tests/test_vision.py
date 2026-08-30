"""Tests for `xbrain.vision` — the external-vision subprocess wrapper.

`describe_image` shells out to an EXTERNAL local vision command (config
`[vision].command`) and reads a text description of a frame image. It mirrors
`transcribe.py`: it imports NO vision/ML library — the heavy vision lives OUTSIDE
xbrain core (the locked #44 architecture), invoked as a subprocess located via
config/PATH.

The contract xbrain expects: `<command> [--model M] <image-path>`, the
description printed on **stdout** (plain text). A **missing / unconfigured**
binary is a clear operator error (`VisionNotFound`) that ABORTS the run — like a
missing transcriber; a **non-zero exit / timeout / empty output** is a per-image
`VisionFailed` (an exit-0-with-no-output is a FAILURE, never a silent empty
description). Every test injects a fake `runner` (a `subprocess.run` stand-in) so
NO real subprocess runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from xbrain.vision import VisionFailed, VisionNotFound, describe_image


def _completed(
    stdout: str, *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["v"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _runner(stdout: str, *, returncode: int = 0, stderr: str = ""):
    calls: list[list[str]] = []

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        return _completed(stdout, returncode=returncode, stderr=stderr)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def test_parses_stdout_into_description(tmp_path: Path):
    result = describe_image(
        tmp_path / "f.png",
        command="vlm-describe",
        language="English",
        runner=_runner("A slide titled 'Loops'.\n"),
    )
    assert result == "A slide titled 'Loops'."


def test_argv_carries_model_and_image_path(tmp_path: Path):
    runner = _runner("desc")
    image = tmp_path / "f.png"
    describe_image(
        image, command="vlm-describe", model="qwen2-vl", language="English", runner=runner
    )
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[0] == "vlm-describe"
    assert "--model" in argv and "qwen2-vl" in argv
    assert argv[-1] == str(image)


def test_multi_token_command_is_split(tmp_path: Path):
    runner = _runner("desc")
    describe_image(
        tmp_path / "f.png", command="python -m my_vlm", language="English", runner=runner
    )
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[:3] == ["python", "-m", "my_vlm"]


def test_unconfigured_command_raises_vision_not_found(tmp_path: Path):
    """An empty `[vision].command` is a clear operator error (abort), not a crash —
    there is NO bundled default vision model."""
    with pytest.raises(VisionNotFound):
        describe_image(tmp_path / "f.png", command="   ", language="English", runner=_runner("x"))


def test_missing_binary_raises_vision_not_found(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "vlm-describe")

    with pytest.raises(VisionNotFound) as excinfo:
        describe_image(tmp_path / "f.png", command="vlm-describe", language="English", runner=_run)
    assert "vlm-describe" in str(excinfo.value)


def test_permission_denied_raises_vision_not_found(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise PermissionError(13, "Permission denied", "vlm-describe")

    with pytest.raises(VisionNotFound):
        describe_image(tmp_path / "f.png", command="vlm-describe", language="English", runner=_run)


def test_empty_output_is_failure_not_silent_empty(tmp_path: Path):
    """Exit 0 with NO output is a `VisionFailed` — never a silent empty
    description (that would drop the slide's content invisibly)."""
    with pytest.raises(VisionFailed) as excinfo:
        describe_image(
            tmp_path / "f.png", command="vlm-describe", language="English", runner=_runner("   \n")
        )
    assert "no" in str(excinfo.value).lower()


def test_nonzero_exit_raises_with_stderr(tmp_path: Path):
    runner = _runner("", returncode=2, stderr="model weights not found")
    with pytest.raises(VisionFailed) as excinfo:
        describe_image(
            tmp_path / "f.png", command="vlm-describe", language="English", runner=runner
        )
    assert "model weights not found" in str(excinfo.value)


def test_timeout_raises_vision_failed(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="vlm-describe", timeout=1)

    with pytest.raises(VisionFailed):
        describe_image(tmp_path / "f.png", command="vlm-describe", language="English", runner=_run)


def test_non_utf8_stdout_raises_vision_failed(tmp_path: Path):
    """`subprocess.run(text=True)` raises `UnicodeDecodeError` on non-UTF-8 stdout;
    the wrapper surfaces it as a clear `VisionFailed`, never a crash or silent drop."""

    def _run(_argv, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with pytest.raises(VisionFailed) as excinfo:
        describe_image(tmp_path / "f.png", command="vlm-describe", language="English", runner=_run)
    assert "non-UTF-8" in str(excinfo.value)


def test_vision_imports_no_ml_or_vision_library():
    """The locked #44 architecture: xbrain core carries NO vision/ML dependency —
    the vision step is an external subprocess. Guard the module."""
    import xbrain.vision as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import torch",
        "import mlx",
        "import transformers",
        "import cv2",
        "coremltools",
    ):
        assert forbidden not in source


def _env_runner(stdout: str = "desc"):
    """A fake runner that records the `env=` kwarg as well as the argv."""
    calls: list[dict] = []

    def _run(argv, **kwargs):
        calls.append({"argv": list(argv), "env": kwargs.get("env")})
        return _completed(stdout)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def test_rubric_reaches_the_subprocess_through_the_env_var(tmp_path: Path):
    """The caption discipline travels by environment variable, so the argv
    contract stays frozen and a third-party command keeps working."""
    from xbrain.vision import PROMPT_ENV_VAR

    runner = _env_runner()
    describe_image(tmp_path / "f.png", command="vlm", language="English", runner=runner)
    env = runner.calls[0]["env"]  # type: ignore[attr-defined]
    assert env is not None
    prompt = env[PROMPT_ENV_VAR]
    assert "VERBATIM" in prompt
    assert "{language}" not in prompt
    assert "English" in prompt


def test_env_var_is_added_to_the_inherited_environment_not_replacing_it(tmp_path: Path):
    """PATH and ANTHROPIC_API_KEY must survive — the cloud backend of the bundled
    wrapper needs the key, and every backend needs PATH."""
    import os

    runner = _env_runner()
    describe_image(tmp_path / "f.png", command="vlm", language="English", runner=runner)
    env = runner.calls[0]["env"]  # type: ignore[attr-defined]
    for key in os.environ:
        assert key in env


def test_argv_contract_is_unchanged_by_the_prompt_injection(tmp_path: Path):
    """The backward-compatibility guarantee, pinned: adding the rule must NOT add
    an argument, or every third-party `[vision].command` breaks at once."""
    runner = _env_runner()
    image = tmp_path / "f.png"
    describe_image(image, command="vlm", model="qwen-3b", language="English", runner=runner)
    argv = runner.calls[0]["argv"]  # type: ignore[attr-defined]
    assert argv == ["vlm", "--model", "qwen-3b", str(image)]


def test_a_runner_that_ignores_env_still_works(tmp_path: Path):
    """Simulates a third-party command that never reads the variable: it must
    still produce a description, not an error."""
    result = describe_image(
        tmp_path / "f.png", command="vlm", language="English", runner=_runner("plain caption")
    )
    assert result == "plain caption"


def test_bundled_wrapper_reads_the_prompt_from_the_environment():
    """`scripts/xbrain-vision` is the reference implementation of the contract, so
    it must prefer the injected rubric over its own fallback constant. It is
    stdlib-only by design (it runs under the system python, which has no xbrain),
    so this is asserted on the source text rather than by importing it."""
    from pathlib import Path

    from xbrain.vision import PROMPT_ENV_VAR

    source = Path(__file__).resolve().parents[1].joinpath("scripts/xbrain-vision").read_text()
    assert PROMPT_ENV_VAR in source
    assert "import xbrain" not in source


def test_bundled_wrapper_keeps_a_fallback_prompt():
    """Run outside xbrain (a bare `xbrain-vision photo.png`), the wrapper must
    still have a prompt rather than sending the model an empty string."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath("scripts/xbrain-vision").read_text()
    assert "_FALLBACK_PROMPT" in source

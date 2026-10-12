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

from xbrain.vision import VisionFailed, VisionNotFound, describe_image, describe_image_with_llm


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
        tmp_path / "f.png", command="vlm-describe", runner=_runner("A slide titled 'Loops'.\n")
    )
    assert result == "A slide titled 'Loops'."


def test_argv_carries_model_and_image_path(tmp_path: Path):
    runner = _runner("desc")
    image = tmp_path / "f.png"
    describe_image(image, command="vlm-describe", model="qwen2-vl", runner=runner)
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[0] == "vlm-describe"
    assert "--model" in argv and "qwen2-vl" in argv
    assert argv[-1] == str(image)


def test_runner_receives_llm_environment(tmp_path: Path):
    seen = {}

    def runner(_argv, **kwargs):
        seen.update(kwargs)
        return _completed("desc")

    env = {
        "XBRAIN_LLM_PROVIDER": "nanogpt",
        "XBRAIN_LLM_MODEL": "zai-org/glm-5.2",
        "XBRAIN_LLM_VISION_MODEL": "xiaomi/mimo-v2.5",
    }
    describe_image(
        tmp_path / "f.png",
        command="vlm-describe",
        model="zai-org/glm-5.2",
        env=env,
        runner=runner,
    )
    assert seen["env"] == env


def test_multi_token_command_is_split(tmp_path: Path):
    runner = _runner("desc")
    describe_image(tmp_path / "f.png", command="python -m my_vlm", runner=runner)
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[:3] == ["python", "-m", "my_vlm"]


def test_unconfigured_command_raises_vision_not_found(tmp_path: Path):
    """An empty `[vision].command` is a clear operator error (abort), not a crash —
    there is NO bundled default vision model."""
    with pytest.raises(VisionNotFound):
        describe_image(tmp_path / "f.png", command="   ", runner=_runner("x"))


def test_missing_binary_raises_vision_not_found(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "vlm-describe")

    with pytest.raises(VisionNotFound) as excinfo:
        describe_image(tmp_path / "f.png", command="vlm-describe", runner=_run)
    assert "vlm-describe" in str(excinfo.value)


def test_permission_denied_raises_vision_not_found(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise PermissionError(13, "Permission denied", "vlm-describe")

    with pytest.raises(VisionNotFound):
        describe_image(tmp_path / "f.png", command="vlm-describe", runner=_run)


def test_empty_output_is_failure_not_silent_empty(tmp_path: Path):
    """Exit 0 with NO output is a `VisionFailed` — never a silent empty
    description (that would drop the slide's content invisibly)."""
    with pytest.raises(VisionFailed) as excinfo:
        describe_image(tmp_path / "f.png", command="vlm-describe", runner=_runner("   \n"))
    assert "no" in str(excinfo.value).lower()


def test_nonzero_exit_raises_with_stderr(tmp_path: Path):
    runner = _runner("", returncode=2, stderr="model weights not found")
    with pytest.raises(VisionFailed) as excinfo:
        describe_image(tmp_path / "f.png", command="vlm-describe", runner=runner)
    assert "model weights not found" in str(excinfo.value)


def test_timeout_raises_vision_failed(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="vlm-describe", timeout=1)

    with pytest.raises(VisionFailed):
        describe_image(tmp_path / "f.png", command="vlm-describe", runner=_run)


def test_non_utf8_stdout_raises_vision_failed(tmp_path: Path):
    """`subprocess.run(text=True)` raises `UnicodeDecodeError` on non-UTF-8 stdout;
    the wrapper surfaces it as a clear `VisionFailed`, never a crash or silent drop."""

    def _run(_argv, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with pytest.raises(VisionFailed) as excinfo:
        describe_image(tmp_path / "f.png", command="vlm-describe", runner=_run)
    assert "non-UTF-8" in str(excinfo.value)


class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Response:
    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class _Messages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response("A slide about agent handoffs.")


class _Client:
    def __init__(self):
        self.messages = _Messages()


def test_cloud_vision_posts_frame_to_configured_llm(tmp_path: Path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _Client()

    result = describe_image_with_llm(
        image,
        provider="nanogpt",
        model="xiaomi/mimo-v2.5",
        output_language="English",
        base_url="https://nano-gpt.com/api/v1",
        client=client,
    )

    assert result == "A slide about agent handoffs."
    call = client.messages.calls[0]
    assert call["model"] == "xiaomi/mimo-v2.5"
    assert "English" in call["system"]
    content = call["messages"][0]["content"]
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1]["type"] == "text"


def test_cloud_vision_empty_response_is_failure(tmp_path: Path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    class _EmptyMessages:
        def create(self, **_kwargs):
            return _Response("   ")

    class _EmptyClient:
        messages = _EmptyMessages()

    with pytest.raises(VisionFailed):
        describe_image_with_llm(
            image,
            provider="nanogpt",
            model="xiaomi/mimo-v2.5",
            output_language="English",
            client=_EmptyClient(),
        )


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

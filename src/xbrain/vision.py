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

import base64
import shlex
import subprocess  # nosec B404 - the vision model is an external subprocess by design (#44)
from collections.abc import Callable
from pathlib import Path
from typing import Mapping, Protocol

from xbrain.llm_client import LlmProvider, build_llm_client, recoverable_llm_errors

# A generous wall-clock cap: describing a single downscaled frame is quick, but a
# wedged VLM process must not hang the run forever.
_DEFAULT_TIMEOUT_SECONDS = 300

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_CLOUD_MAX_TOKENS = 500


class MessagesClient(Protocol):
    def create(self, **kwargs: object) -> object:
        """Send one provider API call; return the raw response."""
        ...


class VisionClient(Protocol):
    messages: MessagesClient


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


def _run_vision(
    argv: list[str],
    runner: Runner,
    timeout_seconds: int,
    env: Mapping[str, str] | None,
) -> str:
    """Run the vision command; return its stdout, or raise a clear operator error."""
    try:
        completed = runner(argv, capture_output=True, text=True, timeout=timeout_seconds, env=env)
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
    env: Mapping[str, str] | None = None,
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
    stdout = _run_vision(argv, active_runner, timeout_seconds, env)
    description = stdout.strip()
    if not description:
        raise VisionFailed(
            f"vision command {argv[0]!r} exited 0 but produced no description — "
            "a slide's content would be lost"
        )
    return description


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")


def _image_block(path: Path) -> dict[str, object]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VisionFailed(f"could not read frame image {path}: {exc}") from exc
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _media_type(path),
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def _cloud_system_prompt(language: str) -> str:
    return (
        "You describe key frames extracted from saved X videos for a personal "
        f"knowledge base. Respond in {language}. Focus on visible text, slide "
        "titles, diagrams, UI states, code, named tools, claims and process steps. "
        "Ignore decorative styling. Be concise but specific; 2-4 sentences."
    )


def _response_text(response: object) -> str:
    blocks = [
        block
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    ]
    text = "".join(str(block.text) for block in blocks).strip()
    if not text:
        raise VisionFailed("cloud vision response produced no description")
    return text


def describe_image_with_llm(
    path: Path | str,
    *,
    provider: LlmProvider,
    model: str,
    output_language: str,
    base_url: str | None = None,
    client: VisionClient | None = None,
    max_tokens: int = _CLOUD_MAX_TOKENS,
) -> str:
    """Describe one video frame through the configured LLM API.

    This is the cloud/API counterpart to `describe_image`. It keeps
    `digest-video --frames` usable on a VPS where a local VLM wrapper is not
    configured: the selected `[llm].provider` handles the image directly, with
    `[llm].vision_model` or `--vision-model`.
    """
    active_client = client or build_llm_client(provider, base_url=base_url)
    frame = Path(path)
    try:
        response = active_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_cloud_system_prompt(output_language),
            messages=[
                {
                    "role": "user",
                    "content": [
                        _image_block(frame),
                        {
                            "type": "text",
                            "text": (
                                "Describe this video key frame as knowledge-base evidence. "
                                "Extract any visible text or structured steps."
                            ),
                        },
                    ],
                }
            ],
        )
    except recoverable_llm_errors() as exc:
        raise VisionFailed(f"cloud vision failed for {frame}: {exc}") from exc
    return _response_text(response)

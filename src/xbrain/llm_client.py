"""LLM client adapters used by API-backed XBrain stages."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import requests

LlmProvider = Literal["nanogpt", "anthropic"]

DEFAULT_LLM_PROVIDER: LlmProvider = "nanogpt"
DEFAULT_NANOGPT_MODEL = "zai-org/glm-5.2"
DEFAULT_NANOGPT_VISION_MODEL = "xiaomi/mimo-v2.5"
DEFAULT_ANTHROPIC_TEXT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ANTHROPIC_VISION_MODEL = "claude-sonnet-4-6"
DEFAULT_NANOGPT_BASE_URL = "https://nano-gpt.com/api/v1"
DEFAULT_CHAT_TIMEOUT_SECONDS = 120
ANTHROPIC_MODEL_ALIASES = frozenset({"opus", "sonnet", "haiku"})


class NanoGPTAPIError(Exception):
    """A recoverable NanoGPT API or response-shape failure."""


@dataclass(frozen=True)
class TextBlock:
    """Anthropic-shaped text block returned by text adapters.

    XBrain's JSON parser already consumes `.content[*].type == "text"` blocks.
    Keeping that tiny response shape avoids provider-specific parsing in every
    enrichment stage.
    """

    text: str
    type: str = "text"


@dataclass(frozen=True)
class TextResponse:
    """Provider-neutral response with the Anthropic text-block shape."""

    content: list[TextBlock]


def normalize_llm_provider(provider: str) -> LlmProvider:
    """Validate and normalize a configured text LLM provider name."""
    normalized = provider.strip().lower()
    if normalized in ("nanogpt", "nano-gpt", "nano_gpt"):
        return "nanogpt"
    if normalized == "anthropic":
        return "anthropic"
    raise ValueError(f"config.toml: [llm].provider must be nanogpt|anthropic, got {provider!r}")


def validate_llm_model(provider: LlmProvider | str, model: str, *, setting: str) -> None:
    """Reject model IDs that clearly belong to a different configured provider."""
    normalized = normalize_llm_provider(provider)
    stripped = model.strip()
    if not stripped:
        raise ValueError(f"LLM config: {setting} must not be empty")

    if normalized == "nanogpt":
        if stripped.startswith("claude-") or stripped in ANTHROPIC_MODEL_ALIASES:
            raise ValueError(
                f"LLM config: {setting}={model!r} is an Anthropic-native model id, "
                'but [llm].provider is "nanogpt". Use a NanoGPT model id or set '
                '[llm].provider = "anthropic".'
            )
        return

    if not stripped.startswith("claude-"):
        raise ValueError(
            f"LLM config: {setting}={model!r} does not look like an Anthropic model id, "
            'but [llm].provider is "anthropic". Use a claude-* model id or set '
            '[llm].provider = "nanogpt".'
        )


def _message_text(content: Any) -> str:
    """Extract text from OpenAI/Anthropic-ish content blocks when needed."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        if parts:
            return "\n".join(parts)
    raise NanoGPTAPIError("NanoGPT response message has no text content")


def _convert_content_block(block: Any) -> Any:
    """Convert Anthropic-style content blocks to NanoGPT/OpenAI-compatible blocks."""
    if not isinstance(block, dict):
        return block
    block_type = block.get("type")
    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block_type == "image_url":
        return block
    if block_type == "image":
        source = block.get("source")
        if not isinstance(source, dict):
            raise NanoGPTAPIError("image block has no source object")
        if source.get("type") != "base64":
            raise NanoGPTAPIError("NanoGPT adapter only supports base64 image blocks")
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise NanoGPTAPIError("base64 image block must include media_type and data")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    return block


def _convert_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert one chat message into NanoGPT/OpenAI-compatible content."""
    content = message.get("content")
    if isinstance(content, list):
        converted = {**message}
        converted["content"] = [_convert_content_block(block) for block in content]
        return converted
    return message


def _has_image_content(messages: list[dict[str, Any]]) -> bool:
    """Return True when a chat payload contains any image block."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"image", "image_url"}:
                return True
    return False


class NanoGPTMessages:
    """Adapter exposing `messages.create(...)` on top of NanoGPT chat completions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_NANOGPT_BASE_URL,
        timeout: int = DEFAULT_CHAT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("NANOGPT_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> TextResponse:
        """Create one non-streaming JSON-oriented chat completion."""
        if not self._api_key:
            raise NanoGPTAPIError('NANOGPT_API_KEY is required for [llm].provider = "nanogpt"')

        chat_messages: list[dict[str, Any]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(_convert_message(message) for message in messages)
        has_image_content = _has_image_content(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "stream": False,
            "reasoning": {"exclude": True},
        }
        if not has_image_content:
            payload["response_format"] = {"type": "json_object"}
        payload.update({key: value for key, value in kwargs.items() if value is not None})

        try:
            response = self._session.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise NanoGPTAPIError(f"NanoGPT API request failed: {exc}") from exc

        if response.status_code >= 400:
            body = response.text[:500]
            raise NanoGPTAPIError(f"NanoGPT API HTTP {response.status_code}: {body}")

        try:
            data = response.json()
        except ValueError as exc:
            raise NanoGPTAPIError("NanoGPT API returned non-JSON response") from exc

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise NanoGPTAPIError("NanoGPT API response has no first choice message") from exc

        content = _message_text(message.get("content"))
        return TextResponse(content=[TextBlock(content)])


class NanoGPTClient:
    """Small client object with the same `.messages` entrypoint as Anthropic."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_NANOGPT_BASE_URL,
        timeout: int = DEFAULT_CHAT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self.messages = NanoGPTMessages(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            session=session,
        )


def build_llm_client(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Build the configured LLM client."""
    normalized = normalize_llm_provider(provider)
    if normalized == "nanogpt":
        return NanoGPTClient(
            api_key=api_key,
            base_url=base_url or DEFAULT_NANOGPT_BASE_URL,
        )

    from anthropic import Anthropic  # lazy: tests inject fakes elsewhere

    if api_key:
        return Anthropic(api_key=api_key)
    return Anthropic()


def build_text_client(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Compatibility alias for older call sites; prefer `build_llm_client`."""
    return build_llm_client(provider, api_key=api_key, base_url=base_url)


def recoverable_llm_errors() -> tuple[type[Exception], ...]:
    """Exception classes a per-item/per-topic LLM failure can swallow."""
    errors: list[type[Exception]] = [
        NanoGPTAPIError,
        ValueError,
        json.JSONDecodeError,
        KeyError,
    ]
    try:
        from anthropic import APIError

        errors.insert(0, APIError)
    except ImportError:
        pass
    return tuple(errors)

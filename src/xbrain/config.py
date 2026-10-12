"""Configuration loading for XBrain."""

from __future__ import annotations

import os
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from xbrain.i18n import strings_for
from xbrain.llm_client import (
    DEFAULT_ANTHROPIC_TEXT_MODEL,
    DEFAULT_ANTHROPIC_VISION_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_NANOGPT_BASE_URL,
    DEFAULT_NANOGPT_MODEL,
    DEFAULT_NANOGPT_VISION_MODEL,
    LlmProvider,
    normalize_llm_provider,
    validate_llm_model,
)
from xbrain.models import ExecutorName

# In-body `**Topics:**` line styles. `wikilink` (default) keeps the current
# navigation-first behaviour; `hashtag` emits Obsidian tags so the line pivots
# into the tag pane. Frontmatter `tags:` are unaffected by this toggle.
SUPPORTED_TOPIC_STYLES: tuple[str, ...] = ("wikilink", "hashtag")
DEFAULT_DRIVE_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
)


@dataclass(frozen=True)
class Config:
    repo_root: Path
    vault: Path
    output_dir: Path
    data_dir: Path
    drive_enabled: bool
    drive_root_folder_id: str
    drive_cache_dir: Path
    drive_credentials_path: Path
    drive_token_path: Path
    drive_scopes: tuple[str, ...]
    email_enabled: bool
    email_recipient: str
    email_sender: str
    email_smtp_host: str
    email_smtp_port: int
    email_smtp_username: str
    email_smtp_password: str
    email_smtp_starttls: bool
    email_smtp_ssl: bool
    snapshot_auto_prune_keep_last: int
    x_handle: str
    llm_provider: LlmProvider
    llm_model: str
    llm_vision_model: str
    llm_base_url: str
    enrich_executor: ExecutorName
    enrich_model: str
    vocab_target_count: int
    topics_resynth_threshold: int
    output_language: str  # one of xbrain.i18n.SUPPORTED_LANGUAGES
    topic_style: str  # one of xbrain.config.SUPPORTED_TOPIC_STYLES
    # Kept as a compatibility alias for call sites/tests that still read the old
    # stage-specific field. It is always equal to `llm_vision_model`.
    describe_model: str
    # `describe_version` tags every produced description so a prompt
    # evolution can be rolled out incrementally: bumping the value here
    # makes the next `xbrain describe` run re-describe stale entries
    # automatically (no `--force` needed). The string is exact-match —
    # there is no ordering relation, only equality.
    describe_version: str
    # `transcribe_command` is the EXTERNAL transcriber `xbrain digest-video`
    # shells out to (#44) — the heavy ASR lives outside xbrain core, invoked as
    # a subprocess located via PATH/config. Defaults to `parakeet-mlx`; may be a
    # multi-token wrapper command (split with shlex, no shell). `transcribe_model`
    # is the optional model id passed through (`None` → the transcriber's own
    # default). `transcribe_timeout_seconds` caps the external ASR subprocess so
    # long videos can be allowed explicitly without changing code.
    transcribe_command: str
    transcribe_model: str | None
    transcribe_timeout_seconds: int
    # `vision_command` is an optional external/custom vision command for
    # `xbrain digest-video --frames` (#44 PR4). When unset, the CLI describes
    # frames directly through `[llm].provider` + `[llm].vision_model`; when set,
    # it shells out to this command and passes `vision_model` / the LLM vision
    # model through as `--model`.
    vision_command: str
    vision_model: str | None

    @property
    def items_path(self) -> Path:
        return self.data_dir / "items.json"

    @property
    def media_dir(self) -> Path:
        """Root directory for downloaded photo bytes.

        In local mode, photos live under `data/media/`. In Google Drive mode,
        the generated vault is the synchronized, user-facing library, so media
        lives directly under `<output_dir>/_media/` and is not duplicated in
        `data/media/`.
        """
        if self.drive_enabled:
            return self.output_dir / "_media"
        return self.data_dir / "media"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def topics_path(self) -> Path:
        return self.data_dir / "topics.json"

    @property
    def storage_state_path(self) -> Path:
        return self.repo_root / "auth" / "storage_state.json"


@dataclass(frozen=True)
class _DriveSettings:
    enabled: bool
    root_folder_id: str
    cache_dir: Path
    credentials_path: Path
    token_path: Path
    scopes: tuple[str, ...]
    vault: Path
    data_dir: Path


@dataclass(frozen=True)
class _EmailSettings:
    enabled: bool
    recipient: str
    sender: str
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_starttls: bool
    smtp_ssl: bool


def _resolve_repo_path(repo_root: Path, value: str, *, setting: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config.toml: {setting} must be a non-empty string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_dotenv(repo_root: Path) -> None:
    """Load simple KEY=VALUE pairs from `.env` without overriding the environment."""
    dotenv = repo_root / ".env"
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _drive_selected_root(repo_root: Path) -> str:
    path = repo_root / "auth" / "google_drive_selection.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    value = payload.get("root_folder_id") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else ""


def _bool_setting(value: object, *, setting: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"config.toml: {setting} must be boolean")


def _configured_text_model(provider: LlmProvider, settings: dict) -> str:
    """Resolve the text API model: env > [llm].model > legacy [enrich].model."""
    provider_default = (
        DEFAULT_NANOGPT_MODEL if provider == "nanogpt" else DEFAULT_ANTHROPIC_TEXT_MODEL
    )
    env_model = os.environ.get("XBRAIN_LLM_MODEL")
    env_setting = "XBRAIN_LLM_MODEL"
    if provider == "nanogpt":
        if not env_model:
            env_model = os.environ.get("NANOGPT_MODEL")
            env_setting = "NANOGPT_MODEL"
    else:
        if not env_model:
            env_model = os.environ.get("ANTHROPIC_MODEL")
            env_setting = "ANTHROPIC_MODEL"
    if env_model:
        validate_llm_model(provider, env_model, setting=env_setting)
        return env_model

    llm = settings.get("llm", {})
    enrich = settings.get("enrich", {})
    configured: list[tuple[str, str]] = []
    for location, section in (
        ("[llm].model", llm),
        ("[enrich].model", enrich),
    ):
        value = section.get("model")
        if isinstance(value, str) and value:
            configured.append((location, value))
    values = {value for _, value in configured}
    if len(values) > 1:
        locations = ", ".join(f"{location}={value!r}" for location, value in configured)
        raise ValueError(
            "config.toml: only one text API LLM model can be configured at a time; "
            f"got {locations}. Move the chosen model to [llm].model."
        )
    model = configured[0][1] if configured else provider_default
    setting = configured[0][0] if configured else "[llm].model"
    validate_llm_model(provider, model, setting=setting)
    return model


def _configured_vision_model(provider: LlmProvider, settings: dict) -> str:
    """Resolve the image/vision API model: env > [llm].vision_model > [describe].model."""
    provider_default = (
        DEFAULT_NANOGPT_VISION_MODEL if provider == "nanogpt" else DEFAULT_ANTHROPIC_VISION_MODEL
    )
    env_model = os.environ.get("XBRAIN_LLM_VISION_MODEL")
    env_setting = "XBRAIN_LLM_VISION_MODEL"
    if provider == "nanogpt":
        if not env_model:
            env_model = os.environ.get("NANOGPT_VISION_MODEL")
            env_setting = "NANOGPT_VISION_MODEL"
    else:
        if not env_model:
            env_model = os.environ.get("ANTHROPIC_VISION_MODEL")
            env_setting = "ANTHROPIC_VISION_MODEL"
    if env_model:
        validate_llm_model(provider, env_model, setting=env_setting)
        return env_model

    llm = settings.get("llm", {})
    describe = settings.get("describe", {})
    configured: list[tuple[str, str]] = []
    for location, section, key in (
        ("[llm].vision_model", llm, "vision_model"),
        ("[describe].model", describe, "model"),
    ):
        value = section.get(key)
        if isinstance(value, str) and value:
            configured.append((location, value))
    values = {value for _, value in configured}
    if len(values) > 1:
        locations = ", ".join(f"{location}={value!r}" for location, value in configured)
        raise ValueError(
            "config.toml: only one vision API LLM model can be configured at a time; "
            f"got {locations}. Move the chosen model to [llm].vision_model."
        )
    model = configured[0][1] if configured else provider_default
    setting = configured[0][0] if configured else "[llm].vision_model"
    validate_llm_model(provider, model, setting=setting)
    return model


def _configured_base_url(provider: LlmProvider, settings: dict) -> str:
    """Resolve provider-specific API base URL."""
    if provider != "nanogpt":
        return ""
    llm = settings.get("llm", {})
    return (
        os.environ.get("XBRAIN_LLM_BASE_URL")
        or os.environ.get("NANOGPT_BASE_URL")
        or llm.get("base_url")
        or DEFAULT_NANOGPT_BASE_URL
    )


def _drive_settings(repo_root: Path, paths: dict, settings: dict) -> _DriveSettings:
    drive = settings.get("drive", {})
    enabled = bool(drive.get("enabled", False))
    cache_dir = _resolve_repo_path(
        repo_root,
        drive.get("cache_dir", ".xbrain-cache/google-drive"),
        setting="[drive].cache_dir",
    )
    credentials_path = _resolve_repo_path(
        repo_root,
        drive.get("credentials_path", "auth/google_drive_credentials.json"),
        setting="[drive].credentials_path",
    )
    token_path = _resolve_repo_path(
        repo_root,
        drive.get("token_path", "auth/google_drive_token.json"),
        setting="[drive].token_path",
    )
    scopes_raw = drive.get("scopes", list(DEFAULT_DRIVE_SCOPES))
    if not isinstance(scopes_raw, list) or not all(
        isinstance(scope, str) and scope for scope in scopes_raw
    ):
        raise ValueError("config.toml: [drive].scopes must be a list of non-empty strings")
    root_folder_id = (
        os.environ.get("XBRAIN_DRIVE_ROOT_FOLDER_ID")
        or str(drive.get("root_folder_id", ""))
        or _drive_selected_root(repo_root)
    )
    if enabled and not root_folder_id:
        raise ValueError("config.toml: [drive].root_folder_id is required when drive.enabled=true")
    vault = Path(paths["vault"]).expanduser()
    data_dir = repo_root / paths["data_dir"]
    if enabled:
        vault = cache_dir / "vault"
        data_dir = cache_dir / "data"
    return _DriveSettings(
        enabled=enabled,
        root_folder_id=root_folder_id,
        cache_dir=cache_dir,
        credentials_path=credentials_path,
        token_path=token_path,
        scopes=tuple(scopes_raw),
        vault=vault,
        data_dir=data_dir,
    )


def _email_settings(settings: dict) -> _EmailSettings:
    email = settings.get("email", {})
    enabled = _bool_setting(
        os.environ.get("XBRAIN_EMAIL_ENABLED", email.get("enabled", False)),
        setting="[email].enabled",
    )
    smtp_username = os.environ.get("XBRAIN_SMTP_USERNAME") or str(email.get("smtp_username", ""))
    sender = os.environ.get("XBRAIN_EMAIL_FROM") or str(email.get("sender", "")) or smtp_username
    out = _EmailSettings(
        enabled=enabled,
        recipient=os.environ.get("XBRAIN_EMAIL_RECIPIENT") or str(email.get("recipient", "")),
        sender=sender,
        smtp_host=os.environ.get("XBRAIN_SMTP_HOST") or str(email.get("smtp_host", "")),
        smtp_port=int(os.environ.get("XBRAIN_SMTP_PORT") or email.get("smtp_port", 587)),
        smtp_username=smtp_username,
        smtp_password=os.environ.get("XBRAIN_SMTP_PASSWORD") or str(email.get("smtp_password", "")),
        smtp_starttls=_bool_setting(
            os.environ.get("XBRAIN_SMTP_STARTTLS", email.get("smtp_starttls", True)),
            setting="[email].smtp_starttls",
        ),
        smtp_ssl=_bool_setting(
            os.environ.get("XBRAIN_SMTP_SSL", email.get("smtp_ssl", False)),
            setting="[email].smtp_ssl",
        ),
    )
    if out.smtp_port < 1:
        raise ValueError("config.toml: [email].smtp_port must be >= 1")
    if out.enabled:
        missing = [
            name
            for name, value in (
                ("recipient", out.recipient),
                ("sender", out.sender),
                ("smtp_host", out.smtp_host),
                ("smtp_username", out.smtp_username),
                ("smtp_password", out.smtp_password),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"config.toml: [email] missing required settings: {missing}")
    return out


def load_config(repo_root: Path) -> Config:
    """Load config.toml from a repo root into a Config."""
    _load_dotenv(repo_root)
    settings = tomllib.loads((repo_root / "config.toml").read_text(encoding="utf-8"))
    paths = settings["paths"]
    x_settings = settings["x"]
    if not x_settings.get("handle"):
        raise ValueError("config.toml: [x].handle is empty — set your X handle")
    drive = _drive_settings(repo_root, paths, settings)
    email = _email_settings(settings)
    snapshots = settings.get("snapshots", {})
    snapshot_auto_prune_keep_last = int(snapshots.get("auto_prune_keep_last", 25))
    if snapshot_auto_prune_keep_last < 0:
        raise ValueError("config.toml: [snapshots].auto_prune_keep_last must be >= 0")
    enrich = settings.get("enrich", {})
    vocab = settings.get("vocab", {})
    executor = enrich.get("executor", "claude-code")
    valid_executors = get_args(ExecutorName)
    if executor not in valid_executors:
        raise ValueError(
            f"config.toml: [enrich].executor must be manual|api|claude-code, got {executor!r}"
        )
    target_count = int(vocab.get("target_count", 30))
    if target_count < 1:
        raise ValueError("config.toml: [vocab].target_count must be >= 1")
    topics = settings.get("topics", {})
    resynth_threshold = int(topics.get("resynth_threshold", 25))
    if resynth_threshold < 1:
        raise ValueError("config.toml: [topics].resynth_threshold must be >= 1")
    output = settings.get("output", {})
    output_language = output.get("language", "English")
    # Validate via strings_for: it already raises ValueError listing supported
    # languages on an unknown value. Single source of truth for the check.
    strings_for(output_language)
    topic_style = output.get("topic_style", "wikilink")
    if topic_style not in SUPPORTED_TOPIC_STYLES:
        raise ValueError(
            f"config.toml: [output].topic_style must be one of "
            f"{list(SUPPORTED_TOPIC_STYLES)}, got {topic_style!r}"
        )
    describe = settings.get("describe", {})
    transcribe = settings.get("transcribe", {})
    transcribe_timeout_seconds = int(transcribe.get("timeout_seconds", 1800))
    if transcribe_timeout_seconds < 1:
        raise ValueError("config.toml: [transcribe].timeout_seconds must be >= 1")
    vision = settings.get("vision", {})
    llm = settings.get("llm", {})
    llm_provider = normalize_llm_provider(
        os.environ.get("XBRAIN_LLM_PROVIDER") or llm.get("provider", DEFAULT_LLM_PROVIDER)
    )
    llm_base_url = _configured_base_url(llm_provider, settings)
    llm_model = _configured_text_model(llm_provider, settings)
    llm_vision_model = _configured_vision_model(llm_provider, settings)
    return Config(
        repo_root=repo_root,
        vault=drive.vault,
        output_dir=drive.vault / paths["output_subdir"],
        data_dir=drive.data_dir,
        drive_enabled=drive.enabled,
        drive_root_folder_id=drive.root_folder_id,
        drive_cache_dir=drive.cache_dir,
        drive_credentials_path=drive.credentials_path,
        drive_token_path=drive.token_path,
        drive_scopes=drive.scopes,
        email_enabled=email.enabled,
        email_recipient=email.recipient,
        email_sender=email.sender,
        email_smtp_host=email.smtp_host,
        email_smtp_port=email.smtp_port,
        email_smtp_username=email.smtp_username,
        email_smtp_password=email.smtp_password,
        email_smtp_starttls=email.smtp_starttls,
        email_smtp_ssl=email.smtp_ssl,
        snapshot_auto_prune_keep_last=snapshot_auto_prune_keep_last,
        x_handle=x_settings["handle"],
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_vision_model=llm_vision_model,
        llm_base_url=llm_base_url,
        enrich_executor=executor,
        enrich_model=llm_model,
        vocab_target_count=target_count,
        topics_resynth_threshold=resynth_threshold,
        output_language=output_language,
        topic_style=topic_style,
        describe_model=llm_vision_model,
        describe_version=describe.get("version", "v1"),
        transcribe_command=transcribe.get("command", "parakeet-mlx"),
        transcribe_model=transcribe.get("model"),
        transcribe_timeout_seconds=transcribe_timeout_seconds,
        vision_command=vision.get("command", ""),
        vision_model=vision.get("model"),
    )

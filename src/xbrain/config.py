"""Configuration loading for XBrain."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from xbrain.i18n import strings_for
from xbrain.models import ExecutorName

# In-body `**Topics:**` line styles. `wikilink` (default) keeps the current
# navigation-first behaviour; `hashtag` emits Obsidian tags so the line pivots
# into the tag pane. Frontmatter `tags:` are unaffected by this toggle.
SUPPORTED_TOPIC_STYLES: tuple[str, ...] = ("wikilink", "hashtag")


@dataclass(frozen=True)
class Config:
    repo_root: Path
    vault: Path
    output_dir: Path
    data_dir: Path
    x_handle: str
    enrich_executor: ExecutorName
    enrich_model: str
    vocab_target_count: int
    topics_resynth_threshold: int
    output_language: str  # one of xbrain.i18n.SUPPORTED_LANGUAGES
    topic_style: str  # one of xbrain.config.SUPPORTED_TOPIC_STYLES

    @property
    def items_path(self) -> Path:
        return self.data_dir / "items.json"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def topics_path(self) -> Path:
        return self.data_dir / "topics.json"

    @property
    def storage_state_path(self) -> Path:
        return self.repo_root / "auth" / "storage_state.json"


def load_config(repo_root: Path) -> Config:
    """Load config.toml from a repo root into a Config."""
    settings = tomllib.loads((repo_root / "config.toml").read_text(encoding="utf-8"))
    paths = settings["paths"]
    x_settings = settings["x"]
    if not x_settings.get("handle"):
        raise ValueError("config.toml: [x].handle is empty — set your X handle")
    vault = Path(paths["vault"]).expanduser()
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
    return Config(
        repo_root=repo_root,
        vault=vault,
        output_dir=vault / paths["output_subdir"],
        data_dir=repo_root / paths["data_dir"],
        x_handle=x_settings["handle"],
        enrich_executor=executor,
        enrich_model=enrich.get("model", "claude-haiku-4-5-20251001"),
        vocab_target_count=target_count,
        topics_resynth_threshold=resynth_threshold,
        output_language=output_language,
        topic_style=topic_style,
    )

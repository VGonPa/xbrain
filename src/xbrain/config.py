"""Configuration loading for XBrain."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from xbrain.i18n import strings_for
from xbrain.models import ExecutorName
from xbrain.video_frames import (
    DEFAULT_DEDUPE_DISTANCE,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_MAX_FRAMES,
    DEFAULT_SCENE_THRESHOLD,
)

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
    # `describe_model` defaults to Sonnet 4.6 — the spec settled on it as the
    # quality / cost sweet spot for vision (~$3-5 for a 2k-image corpus).
    # Override per run via `xbrain describe --model ...` when iterating on
    # prompt or budget; the CLI flag wins over the config value.
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
    # default).
    transcribe_command: str
    transcribe_model: str | None
    # `vision_command` is the EXTERNAL vision model `xbrain digest-video --frames`
    # shells out to (#44 PR4) to describe key-frame slides — the heavy vision lives
    # outside xbrain core, invoked as a subprocess located via PATH/config. There
    # is NO bundled default: it defaults to `""` (unset), and `--frames` errors
    # clearly until it is configured. May be a multi-token wrapper (split with
    # shlex, no shell). `vision_model` is the optional model id passed through
    # (`None` → the vision tool's own default).
    vision_command: str
    vision_model: str | None
    # `[frames]` — the `digest-video --frames` visual layer. Defaults live in
    # `xbrain.video_frames`. Pipeline: extract → dedupe (perceptual hash) → cap.
    frames_max_frames: int
    frames_scene_threshold: float
    frames_interval_seconds: float
    frames_dedupe: bool
    frames_dedupe_distance: int
    # `[index]` — the persisted knowledge index (Plan 02). Names are FLAT, like every other
    # field here: `Config` is a plain dataclass and a nested class per TOML section would be
    # a second convention for the same job (m19). `index_dir` is a NAME under `data/`, never
    # a path, so `resolve_index_dir` can prove containment instead of trusting a string.
    index_dir: str
    index_max_matches_per_item: int
    index_get_char_budget: int

    @property
    def index_path(self) -> Path:
        """Where `data/index/` lives, PROVEN to stay inside `data/` (§12.6, m8).

        Resolved through `resolve_index_dir` rather than joined here, because rejecting `..`
        is not the same check as proving containment: a symlink under `data/` passes the
        first and fails the second.
        """
        from xbrain.knowledge.index_schema import resolve_index_dir

        return resolve_index_dir(self.data_dir, self.index_dir)

    @property
    def payload_dir(self) -> Path:
        """Where raw X payloads live, so `extract` is re-runnable offline (PR-J)."""
        return self.data_dir / "payloads"

    @property
    def items_path(self) -> Path:
        return self.data_dir / "items.json"

    @property
    def media_dir(self) -> Path:
        """Root directory for downloaded photo bytes.

        Photos are stored at ``<media_dir>/<item-id>/<index>.<ext>``. Lives
        under `data/` so it shares the gitignore with the rest of the
        artifact tree. The snapshot lifecycle in `xbrain.snapshot`
        currently covers only the JSON store (`items.json`, `state.json`,
        `vocab.yaml`, `topics.json`) — the binary photo bytes are NOT
        snapshotted today. A re-download via `xbrain media` is the
        recovery path if `data/media/` is lost.
        """
        return self.data_dir / "media"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def topics_path(self) -> Path:
        return self.data_dir / "topics.json"

    @property
    def vocab_path(self) -> Path:
        """`data/vocab.yaml` — the third input the knowledge index derives from (P1a).

        A property beside `items_path` and `topics_path` so the three files the index
        fingerprints and signals on are named in ONE place; the CLI used to spell this one
        as `cfg.data_dir / "vocab.yaml"` at every call site.
        """
        return self.data_dir / "vocab.yaml"

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
    describe = settings.get("describe", {})
    transcribe = settings.get("transcribe", {})
    vision = settings.get("vision", {})
    frames = settings.get("frames", {})
    frames_max_frames = int(frames.get("max_frames", DEFAULT_MAX_FRAMES))
    if frames_max_frames < 1:
        raise ValueError("config.toml: [frames].max_frames must be >= 1")
    frames_dedupe_distance = int(frames.get("dedupe_distance", DEFAULT_DEDUPE_DISTANCE))
    if frames_dedupe_distance < 0:
        raise ValueError("config.toml: [frames].dedupe_distance must be >= 0")
    frames_scene_threshold = float(frames.get("scene_threshold", DEFAULT_SCENE_THRESHOLD))
    if not 0.0 <= frames_scene_threshold <= 1.0:
        raise ValueError("config.toml: [frames].scene_threshold must be in [0.0, 1.0]")
    frames_interval_seconds = float(frames.get("interval_seconds", DEFAULT_INTERVAL_SECONDS))
    if frames_interval_seconds <= 0:
        raise ValueError(
            "config.toml: [frames].interval_seconds must be > 0 (0 selects every frame)"
        )
    index = _index_settings(settings.get("index", {}))
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
        describe_model=describe.get("model", "claude-sonnet-4-6"),
        describe_version=describe.get("version", "v1"),
        transcribe_command=transcribe.get("command", "parakeet-mlx"),
        transcribe_model=transcribe.get("model"),
        vision_command=vision.get("command", ""),
        vision_model=vision.get("model"),
        frames_max_frames=frames_max_frames,
        frames_scene_threshold=frames_scene_threshold,
        frames_interval_seconds=frames_interval_seconds,
        frames_dedupe=bool(frames.get("dedupe", True)),
        frames_dedupe_distance=frames_dedupe_distance,
        index_dir=index["dir"],
        index_max_matches_per_item=index["max_matches_per_item"],
        index_get_char_budget=index["get_char_budget"],
    )


def _index_settings(raw: dict) -> dict:
    """The `[index]` section, defaulted and range-checked with actionable messages.

    Extracted rather than inlined for the reason `[frames]` should have been: `load_config` is
    one branch per setting, and every section added to it costs a point of cyclomatic
    complexity that belongs to the section, not to the loader.

    The two knowledge imports are deferred to HERE, not module scope: `config` is imported by
    almost everything and `knowledge.get_service` pulls in the whole knowledge stack, so a
    module-level edge would make a future `knowledge -> config` import a cycle.
    """
    from xbrain.knowledge.get_service import DEFAULT_CHAR_BUDGET
    from xbrain.knowledge.index_schema import DEFAULT_INDEX_DIR_NAME

    max_matches = int(raw.get("max_matches_per_item", 3))
    if max_matches < 1:
        raise ValueError("config.toml: [index].max_matches_per_item must be >= 1")
    budget = int(raw.get("get_char_budget", DEFAULT_CHAR_BUDGET))
    if budget < 1:
        raise ValueError("config.toml: [index].get_char_budget must be >= 1")
    return {
        # A NAME under `data/`, never a path: `Config.index_path` resolves it and PROVES
        # containment, which rejecting `..` alone would not (a symlink passes that check).
        "dir": raw.get("dir", DEFAULT_INDEX_DIR_NAME),
        "max_matches_per_item": max_matches,
        "get_char_budget": budget,
    }

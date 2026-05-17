"""Configuration loading for XBrain."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    repo_root: Path
    vault: Path
    output_dir: Path
    data_dir: Path
    x_handle: str

    @property
    def items_path(self) -> Path:
        return self.data_dir / "items.json"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

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
    return Config(
        repo_root=repo_root,
        vault=vault,
        output_dir=vault / paths["output_subdir"],
        data_dir=repo_root / paths["data_dir"],
        x_handle=x_settings["handle"],
    )

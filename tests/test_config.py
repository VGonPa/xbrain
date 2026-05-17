# tests/test_config.py
from pathlib import Path

import pytest

from xbrain.config import load_config


def _write_repo(root: Path, handle: str = "vgonpa") -> None:
    (root / "config.toml").write_text(
        '[paths]\n'
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        '[x]\n'
        f'handle = "{handle}"\n',
        encoding="utf-8",
    )
    (root / "courses.yaml").write_text(
        "courses:\n"
        "  - id: TechEntre\n"
        "    name: Tech-Powered Entrepreneurship\n"
        "    institution: IE University\n"
        "    themes: [lean-startup]\n",
        encoding="utf-8",
    )


def test_load_config_resolves_paths_and_courses(tmp_path: Path):
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.x_handle == "vgonpa"
    assert cfg.output_dir == Path("/tmp/vault/learnings/x-knowledge")
    assert cfg.items_path == tmp_path / "data" / "items.json"
    assert cfg.courses[0].id == "TechEntre"
    assert cfg.courses[0].themes == ["lean-startup"]


def test_load_config_rejects_empty_handle(tmp_path: Path):
    _write_repo(tmp_path, handle="")
    with pytest.raises(ValueError, match="handle"):
        load_config(tmp_path)

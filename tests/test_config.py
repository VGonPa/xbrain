# tests/test_config.py
from pathlib import Path

import pytest

from xbrain.config import load_config


def _write_repo(root: Path, handle: str = "vgonpa") -> None:
    (root / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        f'handle = "{handle}"\n',
        encoding="utf-8",
    )


def test_load_config_resolves_paths(tmp_path: Path):
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.x_handle == "vgonpa"
    assert cfg.output_dir == Path("/tmp/vault/learnings/x-knowledge")
    assert cfg.items_path == tmp_path / "data" / "items.json"


def test_load_config_rejects_empty_handle(tmp_path: Path):
    _write_repo(tmp_path, handle="")
    with pytest.raises(ValueError, match="handle"):
        load_config(tmp_path)


def test_load_config_reads_pipeline_settings(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[enrich]\n"
        'executor = "api"\n'
        'model = "claude-haiku-4-5-20251001"\n'
        "[vocab]\n"
        "target_count = 25\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.enrich_executor == "api"
    assert cfg.enrich_model == "claude-haiku-4-5-20251001"
    assert cfg.vocab_target_count == 25


def test_load_config_pipeline_settings_have_defaults(tmp_path: Path):
    _write_repo(tmp_path)  # config.toml WITHOUT [enrich]/[vocab]
    cfg = load_config(tmp_path)
    assert cfg.enrich_executor == "claude-code"  # subscription is the default
    assert cfg.enrich_model == "claude-haiku-4-5-20251001"
    assert cfg.vocab_target_count == 30


def test_load_config_rejects_unknown_executor(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[enrich]\n"
        'executor = "gpt"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="executor must be"):
        load_config(tmp_path)


def test_load_config_rejects_zero_target_count(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[vocab]\n"
        "target_count = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_count must be >= 1"):
        load_config(tmp_path)


def test_config_topics_threshold_defaults_to_25(tmp_path):
    from xbrain.config import load_config

    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/v"\noutput_subdir = "o"\ndata_dir = "data"\n[x]\nhandle = "h"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.topics_resynth_threshold == 25
    assert cfg.topics_path == tmp_path / "data" / "topics.json"


def test_config_topics_threshold_is_configurable(tmp_path):
    from xbrain.config import load_config

    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/v"\noutput_subdir = "o"\ndata_dir = "data"\n'
        '[x]\nhandle = "h"\n'
        "[topics]\nresynth_threshold = 50\n",
        encoding="utf-8",
    )
    assert load_config(tmp_path).topics_resynth_threshold == 50

# tests/test_config.py
import re
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


def test_load_config_defaults_transcribe_command_to_parakeet(tmp_path: Path):
    """No [transcribe] section → the external transcriber defaults to
    `parakeet-mlx`, model unset (the transcriber's own default)."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.transcribe_command == "parakeet-mlx"
    assert cfg.transcribe_model is None


def test_load_config_round_trips_transcribe_command_and_model(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[transcribe]\n"
        'command = "my-asr --quiet"\n'
        'model = "parakeet-tdt-0.6b-v2"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.transcribe_command == "my-asr --quiet"
    assert cfg.transcribe_model == "parakeet-tdt-0.6b-v2"


def test_load_config_defaults_vision_command_to_unset(tmp_path: Path):
    """No [vision] section → the external vision command is unset (`""`) and the
    model is None. `digest-video --frames` errors clearly until it is configured —
    there is NO bundled default vision model (#44 PR4)."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.vision_command == ""
    assert cfg.vision_model is None


def test_load_config_round_trips_vision_command_and_model(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[vision]\n"
        'command = "vlm-describe --fast"\n'
        'model = "qwen2-vl-7b"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.vision_command == "vlm-describe --fast"
    assert cfg.vision_model == "qwen2-vl-7b"


def test_load_config_defaults_output_language_to_english(tmp_path: Path):
    """No [output] section → English default."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.output_language == "English"


def test_load_config_round_trips_spanish_language(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[output]\n"
        'language = "Spanish"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.output_language == "Spanish"


def test_load_config_rejects_unknown_language(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[output]\n"
        'language = "Klingon"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Klingon"):
        load_config(tmp_path)


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


def test_load_config_defaults_topic_style_to_wikilink(tmp_path: Path):
    """No `[output] topic_style` key → wikilink default (backwards-compat)."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.topic_style == "wikilink"


def test_load_config_round_trips_hashtag_topic_style(tmp_path: Path):
    """Explicit `topic_style = "hashtag"` round-trips."""
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[output]\n"
        'topic_style = "hashtag"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.topic_style == "hashtag"


def test_load_config_rejects_unknown_topic_style(tmp_path: Path):
    """Unknown topic_style fails fast with the supported list in the message."""
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[output]\n"
        'topic_style = "bogus"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="topic_style"):
        load_config(tmp_path)


def test_load_config_describe_settings_have_defaults(tmp_path: Path):
    """No [describe] section → Sonnet 4.6 + version v1 (the spec defaults)."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.describe_model == "claude-sonnet-4-6"
    assert cfg.describe_version == "v1"


def test_load_config_round_trips_describe_overrides(tmp_path: Path):
    """[describe] section overrides — operators can pin a different model + version."""
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[describe]\n"
        'model = "claude-opus-4-1"\n'
        'version = "v3"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.describe_model == "claude-opus-4-1"
    assert cfg.describe_version == "v3"


def test_load_config_frames_defaults(tmp_path: Path):
    """No [frames] section → the visual-layer knobs take their video_frames
    defaults: safety-ceiling cap, dedup on."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.frames_max_frames == 60
    assert cfg.frames_footage_max_frames == 6
    assert cfg.frames_scene_threshold == 0.4
    assert cfg.frames_interval_seconds == 15.0
    assert cfg.frames_dedupe is True
    assert cfg.frames_dedupe_distance == 6


def test_load_config_frames_overrides(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[frames]\n"
        "max_frames = 120\n"
        "footage_max_frames = 3\n"
        "scene_threshold = 0.5\n"
        "interval_seconds = 20\n"
        "dedupe = false\n"
        "dedupe_distance = 10\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.frames_max_frames == 120
    assert cfg.frames_footage_max_frames == 3
    assert cfg.frames_scene_threshold == 0.5
    assert cfg.frames_interval_seconds == 20.0
    assert cfg.frames_dedupe is False
    assert cfg.frames_dedupe_distance == 10


def test_load_config_frames_rejects_non_positive_max(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[frames]\n"
        "max_frames = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_frames"):
        load_config(tmp_path)


def test_load_config_frames_rejects_non_positive_footage_max(tmp_path: Path):
    """`footage_max_frames = 0` would describe NO frame of a silent video — the
    hollow entry the footage path exists to prevent — so it is refused on load.
    The full message is pinned: `match="max_frames"` alone would also be satisfied
    by the `[frames].max_frames` error, so it could pass on the wrong check."""
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/tmp/vault"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n[frames]\nfootage_max_frames = 0\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=re.escape("config.toml: [frames].footage_max_frames must be >= 1"),
    ):
        load_config(tmp_path)


def test_load_config_frames_rejects_bad_scene_threshold(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/tmp/vault"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n[frames]\nscene_threshold = 1.5\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scene_threshold"):
        load_config(tmp_path)


def test_load_config_frames_rejects_non_positive_interval(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/tmp/vault"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n[frames]\ninterval_seconds = 0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="interval_seconds"):
        load_config(tmp_path)


# ---------------------------------------------------------------------------
# [index] — Plan 02 §9. The three settings `index build/update/status`, `search`
# and `get` read. Flat names, like every other section (`frames_max_frames`).
# ---------------------------------------------------------------------------


def test_load_config_index_defaults_are_the_owning_modules_constants(tmp_path: Path, monkeypatch):
    """The defaults are IMPORTED, never retyped — rule 5, asserted so it can FAIL.

    `40000` typed into `config.py` would be a second definition of a budget
    `get_service` already owns, and `"index"` a second definition of the directory
    name `index_schema` owns; two copies drift the day one of them moves, and nothing
    goes red because each file is internally consistent.

    COMPARING AGAINST THE IMPORTED CONSTANT IS NOT ENOUGH TO CATCH THAT, and this is
    rule 2 on a test rather than on a metric: `assert cfg.index_get_char_budget ==
    DEFAULT_CHAR_BUDGET` passes just as happily on a hand-typed `40_000`, because both
    sides are 40000 either way — the assertion could not come out differently. So the
    owning constants are MOVED here, and the configured defaults must move with them.
    A literal in `config.py` fails this; an import passes it.
    """
    monkeypatch.setattr("xbrain.knowledge.get_service.DEFAULT_CHAR_BUDGET", 12345)
    monkeypatch.setattr("xbrain.knowledge.index_schema.DEFAULT_INDEX_DIR_NAME", "moved-index")
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.index_dir == tmp_path / "data" / "moved-index"
    assert cfg.index_get_char_budget == 12345


def test_load_config_index_defaults_are_the_documented_values(tmp_path: Path):
    """And the values themselves, once, so `config.toml.example` stays honest."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.index_dir == tmp_path / "data" / "index"
    assert cfg.index_get_char_budget == 40_000
    assert cfg.index_max_matches_per_item == 3


def test_load_config_index_round_trips_every_setting(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/tmp/vault"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n'
        '[index]\ndir = "idx"\nmax_matches_per_item = 7\nget_char_budget = 1234\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.index_dir == tmp_path / "data" / "idx"
    assert cfg.index_max_matches_per_item == 7
    assert cfg.index_get_char_budget == 1234


@pytest.mark.parametrize(
    ("setting", "value", "needle"),
    [
        ("max_matches_per_item", "0", "max_matches_per_item"),
        ("max_matches_per_item", "-1", "max_matches_per_item"),
        ("get_char_budget", "0", "get_char_budget"),
        ("get_char_budget", "-5", "get_char_budget"),
    ],
)
def test_load_config_index_rejects_out_of_range(
    tmp_path: Path, setting: str, value: str, needle: str
):
    """An out-of-range value fails with an actionable message, never a `KeyError`.

    `max_matches_per_item = 0` would cap every item at zero matches, so `search`
    would return results with no citable fragment and no warning; `get_char_budget
    = 0` would truncate every bundle to nothing and page forever. Both are refused
    where the value is read, not where it is used.
    """
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/tmp/vault"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n'
        f"[index]\n{setting} = {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=needle):
        load_config(tmp_path)


@pytest.mark.parametrize("escape", ["../outside", "/etc", "sub/../../outside"])
def test_load_config_index_dir_must_stay_inside_data_dir(tmp_path: Path, escape: str):
    """Plan 02 §12.6: rejecting `..` is not the same check as proving containment.

    The index directory is created, unlinked and rebuilt by `index build --force`,
    so a configured value that resolves outside `data/` hands those operations a
    path the operator never meant to give them. The check is
    `is_relative_to(data_dir.resolve())`, which an absolute path fails too.
    """
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/tmp/vault"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n'
        f'[index]\ndir = "{escape}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index"):
        load_config(tmp_path)


def test_load_config_index_dir_rejects_a_symlink_out_of_data_dir(tmp_path: Path):
    """A symlink passes every `..` test and still lands outside `data/`.

    This is the case Plan 02 §12.6 names explicitly: the containment check must
    run on the RESOLVED path, so a link is followed before the question is asked.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "linked").symlink_to(outside, target_is_directory=True)
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/tmp/vault"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n'
        '[index]\ndir = "linked"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index"):
        load_config(tmp_path)

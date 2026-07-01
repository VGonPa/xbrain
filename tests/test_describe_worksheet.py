# tests/test_describe_worksheet.py
import json
from datetime import datetime, timezone

from xbrain.describe import apply_describe_worksheet, export_describe_worksheet
from xbrain.generate import generate
from xbrain.models import (
    Author,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
)

DT = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _photo(local_path="1/0.png"):
    return MediaPhotoDownloaded(
        url="https://p/" + local_path,
        local_path=local_path,
        width=4,
        height=4,
        bytes_size=9,
        downloaded_at=DT,
    )


def _item(item_id="1", media=None):
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"text {item_id}",
        created_at=DT,
        captured_at=DT,
        media=media or [],
    )


def test_export_lists_eligible_photos_with_image_paths(tmp_path):
    store = {
        "1": _item("1", [_photo("1/0.png")]),
        "2": _item("2", [_photo("2/0.png"), _photo("2/1.png")]),
    }
    ws_path = tmp_path / "ws.json"
    n = export_describe_worksheet(
        store, tmp_path / "media", ws_path, version="v1", output_language="Spanish"
    )
    ws = json.loads(ws_path.read_text(encoding="utf-8"))
    assert n == 3
    assert {(p["item_id"], p["index"]) for p in ws["photos"]} == {("1", 0), ("2", 0), ("2", 1)}
    assert ws["photos"][0]["image_path"].endswith("1/0.png")
    assert ws["rubric"] and ws["judgments"] == []


def test_apply_transitions_to_described_and_enforces_decorative_empty(tmp_path):
    store = {"1": _item("1", [_photo("1/0.png")]), "2": _item("2", [_photo("2/0.png")])}
    ws_path = tmp_path / "ws.json"
    export_describe_worksheet(
        store, tmp_path / "media", ws_path, version="v1", output_language="Spanish"
    )
    ws = json.loads(ws_path.read_text(encoding="utf-8"))
    ws["judgments"] = [
        {
            "item_id": "1",
            "index": 0,
            "is_decorative": False,
            "description": "Un gráfico de barras.",
        },
        {"item_id": "2", "index": 0, "is_decorative": True, "description": "ignored by contract"},
    ]
    ws_path.write_text(json.dumps(ws), encoding="utf-8")

    applied, invalid = apply_describe_worksheet(store, ws_path)
    assert applied == 2
    assert invalid == []
    d1, d2 = store["1"].media[0], store["2"].media[0]
    assert isinstance(d1, MediaPhotoDescribed) and d1.description == "Un gráfico de barras."
    assert not d1.is_decorative
    assert isinstance(d2, MediaPhotoDescribed) and d2.is_decorative and d2.description == ""


def test_apply_skips_unknown_id_and_index(tmp_path):
    store = {"1": _item("1", [_photo("1/0.png")])}
    ws_path = tmp_path / "ws.json"
    ws_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "language": "Spanish",
                "judgments": [
                    {"item_id": "1", "index": 9, "is_decorative": False, "description": "x"},
                    {"item_id": "nope", "index": 0, "is_decorative": False, "description": "y"},
                ],
            }
        ),
        encoding="utf-8",
    )
    applied, invalid = apply_describe_worksheet(store, ws_path)
    assert applied == 0
    assert isinstance(store["1"].media[0], MediaPhotoDownloaded)  # unchanged
    # Both judgments are well-formed but address no downloaded photo → reported,
    # not silently dropped.
    assert {label for label, _ in invalid} == {"1#9", "nope#0"}


def test_apply_reports_malformed_judgments_but_applies_valid(tmp_path):
    # A hand/agent-authored worksheet mixes one good judgment with malformed
    # ones. The good one must apply; each bad one is isolated + reported —
    # never a whole-run abort or a raw TypeError traceback.
    store = {"1": _item("1", [_photo("1/0.png")]), "2": _item("2", [_photo("2/0.png")])}
    ws_path = tmp_path / "ws.json"
    ws_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "language": "Spanish",
                "judgments": [
                    {"item_id": "1", "index": 0, "is_decorative": False, "description": "Bien."},
                    {"item_id": "2", "index": None, "is_decorative": False, "description": "x"},
                    {"item_id": "2", "is_decorative": False, "description": "sin index"},
                    {"item_id": "2", "index": True, "is_decorative": False, "description": "bool"},
                    "no soy un objeto",
                ],
            }
        ),
        encoding="utf-8",
    )
    applied, invalid = apply_describe_worksheet(store, ws_path)
    assert applied == 1
    assert isinstance(store["1"].media[0], MediaPhotoDescribed)
    assert isinstance(store["2"].media[0], MediaPhotoDownloaded)  # untouched
    # Four malformed judgments, each reported with its position label.
    assert len(invalid) == 4
    assert {label for label, _ in invalid} == {
        "judgment[1]",
        "judgment[2]",
        "judgment[3]",
        "judgment[4]",
    }


def test_apply_reports_duplicate_keys(tmp_path):
    store = {"1": _item("1", [_photo("1/0.png")])}
    ws_path = tmp_path / "ws.json"
    ws_path.write_text(
        json.dumps(
            {
                "judgments": [
                    {"item_id": "1", "index": 0, "is_decorative": False, "description": "first"},
                    {"item_id": "1", "index": 0, "is_decorative": False, "description": "second"},
                ]
            }
        ),
        encoding="utf-8",
    )
    applied, invalid = apply_describe_worksheet(store, ws_path)
    assert applied == 1  # first-wins; duplicate is rejected, not last-wins
    assert store["1"].media[0].description == "first"
    assert [label for label, _ in invalid] == ["1#0"]


def test_apply_raises_on_non_object_worksheet(tmp_path):
    import pytest

    ws_path = tmp_path / "ws.json"
    ws_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        apply_describe_worksheet({}, ws_path)


def test_apply_raises_on_non_list_judgments(tmp_path):
    import pytest

    ws_path = tmp_path / "ws.json"
    ws_path.write_text(json.dumps({"judgments": {"item_id": "1"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        apply_describe_worksheet({}, ws_path)


def test_generate_multiline_description_quotes_every_line(tmp_path):
    # A multi-line vision description must be quoted line-by-line: an unquoted
    # trailing line starting with `##`/`-` would inject structure into the note.
    desc = "Primera línea.\n## No es un heading\n- tampoco un bullet"
    store = {"1": _item("1", [_described("1/0.png", desc)])}
    generate(store, tmp_path, output_language="Spanish")
    note = next((tmp_path / "items").glob("*-1.md")).read_text(encoding="utf-8")
    assert "> Primera línea." in note
    assert "> ## No es un heading" in note
    assert "> - tampoco un bullet" in note
    # No description line escaped the blockquote into raw note body.
    assert "\n## No es un heading" not in note
    assert "\n- tampoco un bullet" not in note


def _described(local_path, description, *, decorative=False):
    return MediaPhotoDescribed(
        url="https://p/" + local_path,
        local_path=local_path,
        width=4,
        height=4,
        bytes_size=9,
        downloaded_at=DT,
        is_decorative=decorative,
        description=description,
        description_lang="Spanish",
        description_version="v1",
        described_at=DT,
    )


def test_generate_renders_photo_description_as_caption(tmp_path):
    store = {
        "1": _item("1", [_described("1/0.png", "Un diagrama de flujo.")]),
        "2": _item("2", [_described("2/0.png", "", decorative=True)]),
    }
    generate(store, tmp_path, output_language="Spanish")

    note1 = next((tmp_path / "items").glob("*-1.md")).read_text(encoding="utf-8")
    assert "> Un diagrama de flujo." in note1  # described photo → searchable caption

    note2 = next((tmp_path / "items").glob("*-2.md")).read_text(encoding="utf-8")
    assert "\n> " not in note2  # decorative photo → no caption line

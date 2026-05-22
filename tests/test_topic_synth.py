# tests/test_topic_synth.py
from conftest import FakeAnthropic

from xbrain.topic_synth import OverviewJudgment, TopicInput, synthesize_overviews_api


def test_synthesize_overviews_api_returns_one_judgment_per_topic():
    client = FakeAnthropic(
        [
            {"overview": "Resumen de ai-coding.", "notes": ["Nota."]},
            {"overview": "Resumen de career.", "notes": []},
        ]
    )
    inputs = [
        TopicInput(slug="ai-coding", description="d", summaries=["s1", "s2"]),
        TopicInput(slug="career", description="d", summaries=["s3"]),
    ]
    results = synthesize_overviews_api(inputs, model="m", output_language="English", client=client)
    assert [r.slug for r in results] == ["ai-coding", "career"]
    assert results[0].overview == "Resumen de ai-coding."


def test_synthesize_overviews_api_substitutes_language_in_system_prompt():
    """The topic-page rubric has two {language} placeholders. Both must be
    substituted before the prompt is sent to the LLM.
    """
    client = FakeAnthropic([{"overview": "ok", "notes": []}])
    synthesize_overviews_api(
        [TopicInput(slug="x", description="d", summaries=["s"])],
        model="m",
        output_language="Spanish",
        client=client,
    )
    system = client.messages.calls[0]["system"]
    assert "{language}" not in system
    # Two placeholders in the rubric → two "in Spanish" mentions after substitution
    assert system.count("in Spanish") == 2


def test_synthesize_overviews_api_skips_an_invalid_judgment():
    # A judgment containing a wikilink fails validate_overview and is dropped;
    # the batch continues.
    client = FakeAnthropic(
        [
            {"overview": "Mira [[items/x]].", "notes": []},
            {"overview": "Resumen válido.", "notes": ["n"]},
        ]
    )
    inputs = [
        TopicInput(slug="bad", description="d", summaries=["s"]),
        TopicInput(slug="good", description="d", summaries=["s"]),
    ]
    results = synthesize_overviews_api(inputs, model="m", output_language="English", client=client)
    assert [r.slug for r in results] == ["good"]


def test_synthesize_overviews_api_isolates_an_api_error():
    client = FakeAnthropic([RuntimeError("API caída"), {"overview": "ok", "notes": []}])
    inputs = [
        TopicInput(slug="first", description="d", summaries=["s"]),
        TopicInput(slug="second", description="d", summaries=["s"]),
    ]
    results = synthesize_overviews_api(inputs, model="m", output_language="English", client=client)
    assert [r.slug for r in results] == ["second"]


def test_overview_judgment_is_a_model():
    judgment = OverviewJudgment(slug="ai-coding", overview="o", notes=["n"])
    assert judgment.slug == "ai-coding"


def test_export_and_import_topic_worksheet_round_trip(tmp_path):
    import json

    from xbrain.topic_synth import export_topic_worksheet, import_topic_worksheet

    inputs = [TopicInput(slug="ai-coding", description="d", summaries=["s1", "s2"])]
    path = tmp_path / "topic-worksheet.json"
    export_topic_worksheet(inputs, path, output_language="English")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["topics"][0]["slug"] == "ai-coding"
    assert payload["topics"][0]["summaries"] == ["s1", "s2"]
    assert "rubric" in payload
    assert import_topic_worksheet(path) == []


def test_import_topic_worksheet_rejects_non_list_judgments(tmp_path):
    import json

    import pytest

    from xbrain.topic_synth import import_topic_worksheet

    path = tmp_path / "ws.json"
    path.write_text(json.dumps({"judgments": "no soy lista"}), encoding="utf-8")
    with pytest.raises(ValueError):
        import_topic_worksheet(path)


def test_apply_overview_judgments_splits_valid_and_invalid():
    from xbrain.topic_synth import apply_overview_judgments

    judgments = [
        {"slug": "good", "overview": "Resumen.", "notes": ["n"]},
        {"slug": "bad", "overview": "", "notes": []},
    ]
    valid, invalid = apply_overview_judgments(judgments)
    assert [j.slug for j in valid] == ["good"]
    assert invalid[0][0] == "bad"


def test_apply_overview_judgments_rejects_wikilink_in_overview():
    from xbrain.topic_synth import apply_overview_judgments

    valid, invalid = apply_overview_judgments(
        [{"slug": "x", "overview": "Mira [[items/y]].", "notes": []}]
    )
    assert valid == []
    assert any("wikilink" in e for e in invalid[0][1])


def test_apply_overview_judgments_rejects_a_non_dict_entry():
    from xbrain.topic_synth import apply_overview_judgments

    judgments = [
        "oops",
        {"slug": "good", "overview": "Resumen.", "notes": ["n"]},
    ]
    valid, invalid = apply_overview_judgments(judgments)
    assert [j.slug for j in valid] == ["good"]
    assert any("not a JSON object" in e for e in invalid[0][1])

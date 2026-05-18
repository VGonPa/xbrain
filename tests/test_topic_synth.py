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
    results = synthesize_overviews_api(inputs, model="m", client=client)
    assert [r.slug for r in results] == ["ai-coding", "career"]
    assert results[0].overview == "Resumen de ai-coding."


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
    results = synthesize_overviews_api(inputs, model="m", client=client)
    assert [r.slug for r in results] == ["good"]


def test_synthesize_overviews_api_isolates_an_api_error():
    client = FakeAnthropic([RuntimeError("API caída"), {"overview": "ok", "notes": []}])
    inputs = [
        TopicInput(slug="first", description="d", summaries=["s"]),
        TopicInput(slug="second", description="d", summaries=["s"]),
    ]
    results = synthesize_overviews_api(inputs, model="m", client=client)
    assert [r.slug for r in results] == ["second"]


def test_overview_judgment_is_a_model():
    judgment = OverviewJudgment(slug="ai-coding", overview="o", notes=["n"])
    assert judgment.slug == "ai-coding"

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


def test_synthesize_overviews_api_isolates_an_api_error(capsys):
    from anthropic import APIError

    client = FakeAnthropic(
        [
            APIError("API caída", request=None, body=None),
            {"overview": "ok", "notes": []},
        ]
    )
    inputs = [
        TopicInput(slug="first", description="d", summaries=["s"]),
        TopicInput(slug="second", description="d", summaries=["s"]),
    ]
    results = synthesize_overviews_api(inputs, model="m", output_language="English", client=client)
    assert [r.slug for r in results] == ["second"]
    # Partial-failure summary line is visible on stderr
    assert "SUMMARY: synthesized: 1, failed: 1" in capsys.readouterr().err


def test_synthesize_overviews_api_raises_when_all_topics_fail():
    """Total failure (every API call raises) must surface non-zero exit, not
    silent empty result. The CLI's _handle_cli_errors catches RuntimeError."""
    import pytest
    from anthropic import APIError

    client = FakeAnthropic(
        [
            APIError("401 unauthorized", request=None, body=None),
            APIError("401 unauthorized", request=None, body=None),
        ]
    )
    inputs = [
        TopicInput(slug="a", description="d", summaries=["s"]),
        TopicInput(slug="b", description="d", summaries=["s"]),
    ]
    with pytest.raises(RuntimeError, match="All 2 topics failed synthesis"):
        synthesize_overviews_api(inputs, model="m", output_language="English", client=client)


def test_synthesize_overviews_api_propagates_programmer_bugs():
    """`AttributeError` and other non-recoverable exceptions must NOT be
    swallowed — the developer needs to see the traceback.
    """
    import pytest

    class _Boom:
        class messages:  # noqa: N801
            @staticmethod
            def create(*_args, **_kwargs):
                raise AttributeError("programmer bug — undefined attribute")

    with pytest.raises(AttributeError, match="programmer bug"):
        synthesize_overviews_api(
            [TopicInput(slug="x", description="d", summaries=["s"])],
            model="m",
            output_language="English",
            client=_Boom(),
        )


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


def test_synthesize_overviews_api_propagates_keyboard_interrupt():
    """Ctrl-C must propagate — falls through the narrow catch because
    KeyboardInterrupt inherits from BaseException, not Exception. Regression
    test against a future refactor that widens the catch.
    """
    import pytest

    class _CtrlC:
        class messages:  # noqa: N801
            @staticmethod
            def create(*_args, **_kwargs):
                raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        synthesize_overviews_api(
            [TopicInput(slug="x", description="d", summaries=["s"])],
            model="m",
            output_language="English",
            client=_CtrlC(),
        )


def test_synthesize_overviews_api_emits_no_summary_on_total_failure(capsys):
    """The all-failed branch raises before printing the summary."""
    import pytest
    from anthropic import APIError

    client = FakeAnthropic([APIError("503", request=None, body=None)])
    with pytest.raises(RuntimeError):
        synthesize_overviews_api(
            [TopicInput(slug="x", description="d", summaries=["s"])],
            model="m",
            output_language="English",
            client=client,
        )
    assert "SUMMARY:" not in capsys.readouterr().err


def test_synthesize_overviews_api_emits_no_summary_when_all_succeed(capsys):
    """No failures, no noise: a clean batch stays silent on stderr."""
    client = FakeAnthropic([{"overview": "ok", "notes": []}])
    synthesize_overviews_api(
        [TopicInput(slug="x", description="d", summaries=["s"])],
        model="m",
        output_language="English",
        client=client,
    )
    assert "SUMMARY:" not in capsys.readouterr().err


def test_user_prompt_includes_image_descriptions_when_provided():
    """When `image_descriptions` is non-empty, the prompt carries an `Images` section."""
    from xbrain.topic_synth import _user_prompt

    topic_input = TopicInput(
        slug="ai-coding",
        description="LLMs writing software",
        summaries=["s1"],
        image_descriptions=["A flowchart of a feedback loop.", "Code in a terminal."],
    )
    prompt = _user_prompt(topic_input)
    assert "Images across the 2 content-bearing photos" in prompt
    assert "A flowchart of a feedback loop." in prompt
    assert "Code in a terminal." in prompt


def test_user_prompt_omits_image_section_when_empty():
    """Default empty `image_descriptions` produces no image section — regression guard."""
    from xbrain.topic_synth import _user_prompt

    topic_input = TopicInput(slug="x", description="d", summaries=["s"])
    prompt = _user_prompt(topic_input)
    assert "Images across" not in prompt


def test_user_prompt_includes_video_transcripts_when_provided():
    """When `video_transcripts` is non-empty, the topic prompt carries a video block."""
    from xbrain.topic_synth import _user_prompt

    topic_input = TopicInput(
        slug="ai-coding",
        description="LLMs writing software",
        summaries=["s1"],
        video_transcripts=["A talk on retrieval-augmented agents.", "A live coding demo."],
    )
    prompt = _user_prompt(topic_input)
    assert "Video transcripts across the 2 videos" in prompt
    assert "A talk on retrieval-augmented agents." in prompt
    assert "A live coding demo." in prompt


def test_user_prompt_omits_video_section_when_empty():
    """Default empty `video_transcripts` produces no video section — regression guard."""
    from xbrain.topic_synth import _user_prompt

    prompt = _user_prompt(TopicInput(slug="x", description="d", summaries=["s"]))
    assert "Video transcripts across" not in prompt

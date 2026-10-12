from datetime import datetime, timezone

from tests.conftest import FakeLLMClient

from xbrain.models import Author, Item, MediaVideoPending
from xbrain.transcribe import Transcript
from xbrain.video_fetch import FetchReport, FetchResult
from xbrain.video_transcript import (
    VideoTranscript,
    VideoTranscriptFetchSkipped,
    fetch_video_transcript,
    fetch_or_transcribe_video_transcript,
    parse_transcript_text,
    summarize_video_transcript,
)


def _video_item() -> Item:
    item = Item(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="alice", name="Alice"),
        text="Bookmarked video",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    item.media = [
        MediaVideoPending(
            url="https://video.twimg.com/amplify_video/900/vid/720/a.mp4",
            thumbnail_url="https://pbs.twimg.com/poster.jpg",
        )
    ]
    return item


def test_parse_transcript_text_cleans_vtt_cues_and_duplicates():
    raw = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Speaker>Hello &amp; welcome</v>

00:00:03.000 --> 00:00:05.000
Hello &amp; welcome

00:00:05.000 --> 00:00:07.000
Second idea.
"""

    assert parse_transcript_text(raw, "vtt") == "Hello & welcome\nSecond idea."


def test_parse_transcript_text_reads_srt():
    raw = """1
00:00:01,000 --> 00:00:03,000
Primera idea.

2
00:00:03,000 --> 00:00:05,000
Segunda idea.
"""

    assert parse_transcript_text(raw, "srt") == "Primera idea.\nSegunda idea."


def test_parse_transcript_text_reads_nested_json_text():
    raw = '{"events":[{"segs":[{"utf8":"Hello "},{"utf8":"world"}]},{"text":"Next"}]}'

    assert parse_transcript_text(raw, "json") == "Hello\nworld\nNext"


def test_fetch_video_transcript_requests_caption_url_only():
    class _Response:
        status_code = 200
        headers = {"content-type": "text/vtt"}
        text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nCaption line"

    class _Session:
        def __init__(self):
            self.calls: list[tuple[str, int]] = []

        def get(self, url, *, timeout):
            self.calls.append((url, timeout))
            return _Response()

    entry = MediaVideoPending(
        url="https://video.twimg.com/amplify_video/900/vid/720/a.mp4",
        thumbnail_url="https://pbs.twimg.com/poster.jpg",
        transcript_url="https://video.twimg.com/amplify_video/900/captions/en.vtt",
        transcript_language="en",
    )
    session = _Session()

    transcript = fetch_video_transcript(entry, session=session, timeout_seconds=12)

    assert session.calls == [("https://video.twimg.com/amplify_video/900/captions/en.vtt", 12)]
    assert transcript.text == "Caption line"
    assert transcript.language == "en"
    assert transcript.format == "vtt"


def test_fetch_or_transcribe_uses_captions_before_temporary_download():
    class _Response:
        status_code = 200
        headers = {"content-type": "text/vtt"}
        text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nCaption first"

    class _Session:
        def get(self, url, *, timeout):
            assert url == "https://video.twimg.com/captions/en.vtt"
            return _Response()

    item = _video_item()
    entry = item.media[0]
    entry.transcript_url = "https://video.twimg.com/captions/en.vtt"
    called = False

    def _fetch_fallback(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("temporary video fallback should not run when captions exist")

    transcript = fetch_or_transcribe_video_transcript(
        item, entry, session=_Session(), fetch_fn=_fetch_fallback
    )

    assert transcript.text == "Caption first"
    assert transcript.source_url == "https://video.twimg.com/captions/en.vtt"
    assert called is False


def test_fetch_or_transcribe_downloads_temporarily_and_deletes_video():
    item = _video_item()
    entry = item.media[0]
    seen: dict[str, object] = {}

    def _fetch(store, ids, dest_dir, **kwargs):
        dest = dest_dir
        seen["dest"] = dest
        seen["max_size_bytes"] = kwargs["max_size_bytes"]
        path = dest / f"{ids[0]}.mp4"
        path.write_bytes(b"fake mp4 bytes")
        assert store[item.id] is item
        return FetchReport([FetchResult(item.id, "fetched", path=str(path), size_bytes=14)])

    def _transcribe(path, **kwargs):
        seen["path_exists_during_asr"] = path.exists()
        assert kwargs["command"] == "fake-asr"
        assert kwargs["model"] == "asr-model"
        assert kwargs["timeout_seconds"] == 7200
        return Transcript(text="Spoken ideas", language="es", has_speech=True)

    transcript = fetch_or_transcribe_video_transcript(
        item,
        entry,
        language="en",
        transcribe_command="fake-asr",
        transcribe_model="asr-model",
        transcribe_timeout_seconds=7200,
        max_size_bytes=123,
        fetch_fn=_fetch,
        transcribe_fn=_transcribe,
    )

    assert transcript.text == "Spoken ideas"
    assert transcript.language == "es"
    assert transcript.source_url == item.url
    assert transcript.format == "text"
    assert seen["path_exists_during_asr"] is True
    assert seen["max_size_bytes"] == 123
    assert not seen["dest"].exists()


def test_fetch_or_transcribe_surfaces_temporary_download_skips():
    item = _video_item()
    entry = item.media[0]

    def _fetch(_store, ids, _dest_dir, **_kwargs):
        return FetchReport([FetchResult(ids[0], "skipped", reason="too_large")])

    try:
        fetch_or_transcribe_video_transcript(item, entry, fetch_fn=_fetch)
    except VideoTranscriptFetchSkipped as exc:
        assert exc.reason == "too_large"
    else:
        raise AssertionError("expected VideoTranscriptFetchSkipped")


def test_summarize_video_transcript_uses_configured_text_llm_and_formats_markdown():
    client = FakeLLMClient(
        [
            {
                "title": "Video Strategy",
                "summary": "A useful executive summary.",
                "main_ideas": ["Idea one", "Idea two"],
                "first_order_conclusions": ["Direct conclusion"],
                "second_order_conclusions": ["Deeper implication"],
                "didactic_use": ["Teach it as a workflow"],
                "practical_applications": ["Apply it to the vault"],
            }
        ]
    )
    transcript = VideoTranscript(
        text="Original transcript content.",
        language="en",
        source_url="https://video.twimg.com/captions/en.vtt",
        format="vtt",
    )

    summary = summarize_video_transcript(
        "bookmark text",
        "alice",
        transcript,
        provider="nanogpt",
        model="zai-org/glm-5.2",
        output_language="Spanish",
        client=client,
    )

    assert summary.title == "Video Strategy"
    assert "### Executive Summary" in summary.markdown
    assert "A useful executive summary." in summary.markdown
    assert "- Idea one" in summary.markdown
    call = client.messages.calls[0]
    assert call["model"] == "zai-org/glm-5.2"
    assert "Write in Spanish" in call["system"]
    assert "Original video transcript:" in call["messages"][0]["content"]

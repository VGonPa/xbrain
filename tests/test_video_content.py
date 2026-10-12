from xbrain.models import ContentSourceSuccess, VideoFrame
from xbrain.video_content import video_content_text


def test_video_content_uses_executive_summary_text_only():
    source = ContentSourceSuccess(
        kind="x_video",
        url="https://x.com/v",
        text="Executive summary.",
        raw_transcript="Full original transcript that must not feed analysis.",
        has_speech=True,
        frames=[
            VideoFrame(timestamp=12.0, local_path="1/frames/0.png", description="A title slide.")
        ],
    )

    text = video_content_text(source)

    assert text == "Executive summary."
    assert "Full original transcript" not in text
    assert "title slide" not in text


def test_video_content_ignores_visual_only_legacy_frames():
    source = ContentSourceSuccess(
        kind="x_video",
        url="https://x.com/v",
        text="",
        has_speech=False,
        frames=[VideoFrame(timestamp=3.0, local_path="1/frames/0.png", description="A workflow.")],
    )

    assert video_content_text(source) is None


def test_video_content_keeps_summary_even_when_has_speech_false():
    source = ContentSourceSuccess(
        kind="x_video",
        url="https://x.com/v",
        text="Executive summary from captions.",
        has_speech=False,
    )

    assert video_content_text(source) == "Executive summary from captions."

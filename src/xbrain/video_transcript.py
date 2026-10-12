"""Video transcript ingestion and executive summaries.

The preferred path fetches a small caption/text-track URL already captured on a
`MediaVideo*` entry. When X does not expose captions, the fallback may download a
progressive MP4 into a private temporary directory, shell out to the configured
external transcriber, convert the ASR output to plain transcript text, and delete
the media bytes before returning. The persistent store/vault receives only the
raw text transcript and the LLM executive summary.
"""

from __future__ import annotations

import html
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from xbrain.llm_client import LlmProvider, build_llm_client, recoverable_llm_errors
from xbrain.llm_json import json_from_response
from xbrain.models import (
    Item,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
    TranscriptFormat,
)
from xbrain.transcribe import (
    DEFAULT_TRANSCRIBE_COMMAND,
    Transcript,
    TranscribeError,
    transcribe_media,
)
from xbrain.video_fetch import FetchReport, fetch_videos

_VIDEO_TYPES = (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)
_TIMESTAMP_RE = re.compile(
    r"^\s*\d{0,2}:?\d{1,2}:\d{2}[,.]\d{1,3}\s+-->\s+"
    r"\d{0,2}:?\d{1,2}:\d{2}[,.]\d{1,3}"
)
_TAG_RE = re.compile(r"<[^>]+>")
_TRANSCRIPT_PROMPT_CHAR_LIMIT = 60000
_VIDEO_SUMMARY_MAX_TOKENS = 2400
VIDEO_SUMMARY_VERSION = "v1"


class VideoTranscriptUnavailable(RuntimeError):
    """No usable transcript can be obtained for a video."""


class VideoTranscriptFetchSkipped(VideoTranscriptUnavailable):
    """Temporary video fallback was intentionally skipped before downloading."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class VideoTranscriptFetchFailed(RuntimeError):
    """Caption fetch, temporary video fetch, or ASR failed."""


class VideoSummaryFailed(RuntimeError):
    """The configured LLM did not produce a usable executive summary."""


@dataclass(frozen=True)
class VideoTranscript:
    """Plain transcript text from captions or temporary ASR."""

    text: str
    language: str | None
    source_url: str
    format: TranscriptFormat


@dataclass(frozen=True)
class VideoExecutiveSummary:
    """Structured executive summary produced from a raw transcript."""

    title: str
    markdown: str


def _format_from_response(
    declared: TranscriptFormat | None,
    url: str,
    content_type: str,
) -> TranscriptFormat:
    if declared and declared != "unknown":
        return declared
    lowered = content_type.lower()
    if "webvtt" in lowered or "vtt" in lowered:
        return "vtt"
    if "subrip" in lowered or "srt" in lowered:
        return "srt"
    if "json" in lowered:
        return "json"
    if lowered.startswith("text/"):
        return "text"
    path = urlparse(url).path.lower()
    if path.endswith(".vtt"):
        return "vtt"
    if path.endswith(".srt"):
        return "srt"
    if path.endswith(".json"):
        return "json"
    if path.endswith(".txt"):
        return "text"
    return "unknown"


def _clean_caption_line(line: str) -> str:
    cleaned = html.unescape(_TAG_RE.sub("", line)).strip()
    return " ".join(cleaned.split())


def _parse_timed_text(text: str) -> str:
    lines: list[str] = []
    in_note_block = False
    for raw_line in text.replace("\ufeff", "").splitlines():
        line = raw_line.strip()
        if not line:
            in_note_block = False
            continue
        upper = line.upper()
        if upper.startswith(("WEBVTT", "STYLE", "REGION")):
            continue
        if upper.startswith("NOTE"):
            in_note_block = True
            continue
        if in_note_block:
            continue
        if line.isdigit() or _TIMESTAMP_RE.match(line):
            continue
        cleaned = _clean_caption_line(line)
        if cleaned:
            lines.append(cleaned)
    return _dedupe_adjacent(lines)


def _dedupe_adjacent(lines: list[str]) -> str:
    deduped: list[str] = []
    last = ""
    for line in lines:
        if line == last:
            continue
        deduped.append(line)
        last = line
    return "\n".join(deduped).strip()


def _collect_json_text(obj: Any) -> list[str]:
    chunks: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = key.lower()
            if lowered in {"text", "utf8", "caption", "transcript"} and isinstance(value, str):
                cleaned = _clean_caption_line(value)
                if cleaned:
                    chunks.append(cleaned)
            elif isinstance(value, (dict, list)):
                chunks.extend(_collect_json_text(value))
    elif isinstance(obj, list):
        for value in obj:
            chunks.extend(_collect_json_text(value))
    return chunks


def parse_transcript_text(raw: str, fmt: TranscriptFormat) -> str:
    """Parse a fetched caption/text body into plain transcript text."""
    if fmt in ("vtt", "srt", "unknown"):
        parsed = _parse_timed_text(raw)
        if parsed:
            return parsed
    if fmt == "json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VideoTranscriptFetchFailed(f"transcript JSON is malformed: {exc}") from exc
        parsed = _dedupe_adjacent(_collect_json_text(data))
        if parsed:
            return parsed
    if fmt in ("text", "unknown"):
        lines = [_clean_caption_line(line) for line in raw.splitlines()]
        parsed = _dedupe_adjacent([line for line in lines if line])
        if parsed:
            return parsed
    raise VideoTranscriptFetchFailed("transcript text track contained no usable text")


def fetch_video_transcript(
    entry: MediaVideoPending | MediaVideoDownloaded | MediaVideoFailed,
    *,
    session: requests.Session | None = None,
    timeout_seconds: int = 30,
) -> VideoTranscript:
    """Fetch and parse a caption/text-track URL without downloading media bytes."""
    if not entry.transcript_url:
        raise VideoTranscriptUnavailable("video has no transcript/caption URL")
    active_session = session or requests.Session()
    try:
        response = active_session.get(entry.transcript_url, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise VideoTranscriptFetchFailed(f"transcript request failed: {exc}") from exc
    if response.status_code >= 400:
        raise VideoTranscriptFetchFailed(f"transcript HTTP {response.status_code}")
    fmt = _format_from_response(
        entry.transcript_format,
        entry.transcript_url,
        response.headers.get("content-type", ""),
    )
    try:
        raw = response.text
    except UnicodeDecodeError as exc:
        raise VideoTranscriptFetchFailed(f"transcript is not decodable text: {exc}") from exc
    text = parse_transcript_text(raw, fmt)
    return VideoTranscript(
        text=text,
        language=entry.transcript_language,
        source_url=entry.transcript_url,
        format=fmt,
    )


def _with_language(transcript: VideoTranscript, language: str | None) -> VideoTranscript:
    if language is None or transcript.language is not None:
        return transcript
    return VideoTranscript(
        text=transcript.text,
        language=language,
        source_url=transcript.source_url,
        format=transcript.format,
    )


def _fetched_path(report: FetchReport, item_id: str) -> Path:
    result = next((entry for entry in report.results if entry.id == item_id), None)
    if result is None:
        raise VideoTranscriptFetchFailed("temporary video fetch produced no result")
    if result.outcome == "skipped":
        reason = result.reason or "skipped"
        raise VideoTranscriptFetchSkipped(reason, f"temporary video fetch skipped ({reason})")
    if result.outcome == "failed":
        detail = result.error or result.reason or "unknown error"
        raise VideoTranscriptFetchFailed(f"temporary video fetch failed: {detail}")
    if not result.path:
        raise VideoTranscriptFetchFailed("temporary video fetch returned no path")
    return Path(result.path)


def _transcript_from_asr(
    item: Item,
    transcript: Transcript,
    *,
    language: str | None,
) -> VideoTranscript:
    text = transcript.text.strip()
    if not transcript.has_speech or not text:
        raise VideoTranscriptUnavailable("temporary transcription produced no speech")
    return VideoTranscript(
        text=text,
        language=transcript.language or language,
        source_url=item.url,
        format="text",
    )


def transcribe_video_temporarily(
    item: Item,
    *,
    transcribe_command: str = DEFAULT_TRANSCRIBE_COMMAND,
    transcribe_model: str | None = None,
    transcribe_timeout_seconds: int = 1800,
    language: str | None = None,
    max_size_bytes: int | None = None,
    fetch_fn=fetch_videos,
    transcribe_fn=transcribe_media,
    session: requests.Session | None = None,
    download_timeout_seconds: int = 30,
) -> VideoTranscript:
    """Download one video to a private temp dir, transcribe it, and remove bytes.

    `fetch_fn` is expected to be `xbrain.video_fetch.fetch_videos` compatible and
    store-non-mutating: it writes only under the supplied temporary directory.
    The temp directory is always removed by `TemporaryDirectory`, including when
    the fetch or ASR step fails.
    """
    with tempfile.TemporaryDirectory(prefix="xbrain-video-asr-") as tmp:
        report = fetch_fn(
            {item.id: item},
            [item.id],
            Path(tmp),
            max_size_bytes=max_size_bytes,
            limit=1,
            throttle_seconds=0,
            session=session,
            timeout_seconds=download_timeout_seconds,
        )
        media_path = _fetched_path(report, item.id)
        try:
            transcript = transcribe_fn(
                media_path,
                command=transcribe_command,
                model=transcribe_model,
                language=language,
                timeout_seconds=transcribe_timeout_seconds,
            )
        except TranscribeError as exc:
            raise VideoTranscriptFetchFailed(f"temporary transcription failed: {exc}") from exc
        return _transcript_from_asr(item, transcript, language=language)


def fetch_or_transcribe_video_transcript(
    item: Item,
    entry: MediaVideoPending | MediaVideoDownloaded | MediaVideoFailed,
    *,
    language: str | None = None,
    session: requests.Session | None = None,
    caption_timeout_seconds: int = 30,
    transcribe_command: str = DEFAULT_TRANSCRIBE_COMMAND,
    transcribe_model: str | None = None,
    transcribe_timeout_seconds: int = 1800,
    max_size_bytes: int | None = None,
    fetch_fn=fetch_videos,
    transcribe_fn=transcribe_media,
    download_timeout_seconds: int = 30,
) -> VideoTranscript:
    """Use X captions first; if absent, fall back to temporary MP4 transcription."""
    try:
        transcript = fetch_video_transcript(
            entry, session=session, timeout_seconds=caption_timeout_seconds
        )
        return _with_language(transcript, language)
    except VideoTranscriptUnavailable:
        return transcribe_video_temporarily(
            item,
            transcribe_command=transcribe_command,
            transcribe_model=transcribe_model,
            transcribe_timeout_seconds=transcribe_timeout_seconds,
            language=language,
            max_size_bytes=max_size_bytes,
            fetch_fn=fetch_fn,
            transcribe_fn=transcribe_fn,
            session=session,
            download_timeout_seconds=download_timeout_seconds,
        )


def video_entry_with_transcript(item: object):
    """Return the first video media entry with a transcript URL, else None."""
    media = getattr(item, "media", [])
    for entry in media:
        if isinstance(entry, _VIDEO_TYPES) and entry.transcript_url:
            return entry
    return None


def _transcript_for_prompt(text: str) -> str:
    """Keep long transcripts representative without silently dropping the ending."""
    if len(text) <= _TRANSCRIPT_PROMPT_CHAR_LIMIT:
        return text
    head = text[:24000]
    midpoint = len(text) // 2
    middle = text[max(0, midpoint - 9000) : midpoint + 9000]
    tail = text[-18000:]
    return (
        head
        + "\n\n[... transcript middle excerpt ...]\n\n"
        + middle
        + "\n\n[... transcript ending excerpt ...]\n\n"
        + tail
    )


def _system_prompt(language: str) -> str:
    return (
        "You summarize X bookmarked video transcripts for a personal knowledge base.\n"
        f"Write in {language}.\n"
        "Return a single JSON object with these keys only:\n"
        '{"title":"short specific title",'
        '"summary":"executive summary, 2-4 substantial paragraphs",'
        '"main_ideas":["5-8 key ideas"],'
        '"first_order_conclusions":["3-5 direct conclusions"],'
        '"second_order_conclusions":["3-5 deeper implications"],'
        '"didactic_use":["3-5 ways to teach or apply this"],'
        '"practical_applications":["3-5 concrete actions or workflows"]}\n'
        "Do not mention that you received a transcript. Do not invent facts absent from it."
    )


def _user_prompt(item_text: str, author: str, transcript: VideoTranscript) -> str:
    return "\n".join(
        [
            f"Post author: @{author}",
            "Post text:",
            item_text,
            "",
            "Original video transcript:",
            _transcript_for_prompt(transcript.text),
        ]
    )


def _list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(entry).strip() for entry in value if str(entry).strip()]


def _markdown_from_summary(data: dict[str, Any]) -> VideoExecutiveSummary:
    title = str(data.get("title") or "Video executive summary").strip()
    summary = str(data.get("summary") or "").strip()
    if not summary:
        raise VideoSummaryFailed("video summary response has no summary")
    sections = [
        ("Executive Summary", [summary]),
        ("Main Ideas", _list(data.get("main_ideas"))),
        ("First-Order Conclusions", _list(data.get("first_order_conclusions"))),
        ("Second-Order Conclusions", _list(data.get("second_order_conclusions"))),
        ("Didactic Use", _list(data.get("didactic_use"))),
        ("Practical Applications", _list(data.get("practical_applications"))),
    ]
    lines: list[str] = []
    for heading, entries in sections:
        if not entries:
            continue
        lines += [f"### {heading}", ""]
        if len(entries) == 1 and heading == "Executive Summary":
            lines += [entries[0], ""]
        else:
            lines += [f"- {entry}" for entry in entries]
            lines.append("")
    return VideoExecutiveSummary(title=title, markdown="\n".join(lines).strip())


def summarize_video_transcript(
    item_text: str,
    author: str,
    transcript: VideoTranscript,
    *,
    provider: LlmProvider,
    model: str,
    output_language: str,
    base_url: str | None = None,
    client: Any = None,
) -> VideoExecutiveSummary:
    """Summarize a raw transcript with the configured text LLM."""
    active_client = client or build_llm_client(provider, base_url=base_url)
    try:
        response = active_client.messages.create(
            model=model,
            max_tokens=_VIDEO_SUMMARY_MAX_TOKENS,
            system=_system_prompt(output_language),
            messages=[
                {
                    "role": "user",
                    "content": _user_prompt(item_text, author, transcript),
                }
            ],
        )
        return _markdown_from_summary(json_from_response(response, context="video summary"))
    except recoverable_llm_errors() as exc:
        raise VideoSummaryFailed(f"video summary failed: {exc}") from exc

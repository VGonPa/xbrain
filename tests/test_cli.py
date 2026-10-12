# tests/test_cli.py
import http.client
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from xbrain import snapshot
from xbrain.cli import (
    _copy_bootstrap_tree,
    _drive_cache_vault_output_dir,
    _render_note_page,
    _resolve_served_media,
    _resolve_served_note,
    _run_delete_bookmark,
    _run_reprocess_note,
    _run_topic_action,
    app,
)
from xbrain.config import load_config
from xbrain.models import Author, Enrichment, Item, Link, Topic
from xbrain.notes_io import GEN_END, GEN_START
from xbrain.rubrics import load_vocab, save_vocab
from xbrain.store import load_store, save_store

runner = CliRunner()


def _setup_repo(tmp_path: Path, monkeypatch) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        f'vault = "{vault}"\n'
        'output_subdir = "x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n',
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    return vault


def _linked_item(item_id: str = "1") -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/p", domain="example.com")],
    )


def test_status_runs_on_empty_store(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Items: 0" in result.stdout


def test_serve_dashboard_supports_head_smoke_checks(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    dashboard = tmp_path / "vault" / "x-knowledge" / "dashboard.html"
    dashboard.parent.mkdir(parents=True)
    dashboard.write_text("<!doctype html><title>XBrain</title>", encoding="utf-8")
    notes = dashboard.parent / "items"
    notes.mkdir()
    (notes / "2026-01-01-search-1.md").write_text(
        "# Searchable note\n\nA distinctive library retrieval phrase.", encoding="utf-8"
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    env = os.environ.copy()
    env["XBRAIN_REPO_ROOT"] = str(tmp_path)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "xbrain.cli",
            "serve-dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                conn.request("GET", "/api/status")
                response = conn.getresponse()
                response.read()
                conn.close()
                if response.status == 200:
                    break
            except OSError:
                time.sleep(0.1)
        else:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"serve-dashboard did not start: {stderr}")

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("HEAD", "/dashboard.html")
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("content-type") == "text/html; charset=utf-8"
        assert int(response.getheader("content-length") or "0") == len(dashboard.read_bytes())
        assert response.getheader("cache-control") == "no-store"
        assert response.getheader("x-content-type-options") == "nosniff"
        assert response.getheader("x-frame-options") == "DENY"
        assert "frame-ancestors 'none'" in (response.getheader("content-security-policy") or "")
        assert response.read() == b""
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("HEAD", "/api/status")
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("content-type") == "application/json"
        assert response.read() == b""
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/api/search?q=distinctive+retrieval")
        response = conn.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["results"][0]["title"] == "Searchable note"
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("POST", "/api/refresh-all")
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request(
            "POST",
            "/api/chat",
            body=json.dumps({"question": "What is saved?", "item_ids": []}),
            headers={
                "Content-Type": "application/json",
                "X-XBrain-Action": "1",
                "Origin": f"http://127.0.0.1:{port}",
            },
        )
        response = conn.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["sources"] == []
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request(
            "POST",
            "/api/chat",
            body=json.dumps({"question": "What is saved?"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request(
            "POST",
            "/api/retry-failed",
            headers={"X-XBrain-Action": "1", "Origin": "https://evil.example"},
        )
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_drive_select_root_writes_local_selection(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)

    result = runner.invoke(app, ["drive", "select-root", "folder-123"])

    assert result.exit_code == 0
    selection = tmp_path / "auth" / "google_drive_selection.json"
    assert "folder-123" in selection.read_text(encoding="utf-8")
    cfg = load_config(tmp_path)
    assert cfg.drive_root_folder_id == "folder-123"


def test_drive_status_reports_media_dir(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)

    result = runner.invoke(app, ["drive", "status"])

    assert result.exit_code == 0
    assert "media_dir:" in result.stdout


def test_copy_bootstrap_tree_copies_files(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "file.txt").write_text("body", encoding="utf-8")

    copied = _copy_bootstrap_tree(source, destination, force=False)

    assert copied == 1
    assert (destination / "nested" / "file.txt").read_text(encoding="utf-8") == "body"


def test_drive_cache_vault_output_dir_preserves_nested_output_subdir(tmp_path: Path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        f'vault = "{vault}"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))

    cfg = load_config(tmp_path)

    assert _drive_cache_vault_output_dir(cfg) == (
        tmp_path / ".xbrain-cache" / "google-drive" / "vault" / "learnings" / "x-knowledge"
    )


def test_generate_creates_output_dir(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 0
    assert (vault / "x-knowledge" / "_index.md").exists()


def test_status_reports_non_empty_store(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Items: 1" in result.stdout
    assert "con enlace: 1" in result.stdout


def test_generate_writes_item_note_for_linked_item(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 0
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    assert notes


def test_served_note_resolver_allows_only_generated_markdown(tmp_path: Path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    notes_dir = output_dir / "items"
    notes_dir.mkdir(parents=True)
    note = notes_dir / "source.md"
    note.write_text("# Source\n\nBody", encoding="utf-8")

    assert _resolve_served_note(output_dir, str(note)) == note.resolve()
    assert _resolve_served_note(output_dir, "items/source.md") == note.resolve()

    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(PermissionError):
        _resolve_served_note(output_dir, str(outside))

    text_file = notes_dir / "source.txt"
    text_file.write_text("not markdown", encoding="utf-8")
    with pytest.raises(ValueError):
        _resolve_served_note(output_dir, str(text_file))


def test_served_note_resolver_redirects_stale_slug_by_item_id(tmp_path: Path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    notes_dir = output_dir / "items"
    notes_dir.mkdir(parents=True)
    current = notes_dir / "2026-01-01-new-title-123.md"
    stale = notes_dir / "2026-01-01-old-title-123.md"
    current.write_text(f"{GEN_START}\n---\nid: 123\n---\n# New\n{GEN_END}", encoding="utf-8")

    assert _resolve_served_note(output_dir, str(stale)) == current.resolve()


def test_served_media_resolver_allows_only_generated_images(tmp_path: Path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    media_dir = output_dir / "_media" / "1" / "article"
    media_dir.mkdir(parents=True)
    image = media_dir / "0.jpg"
    image.write_bytes(b"jpg")

    assert _resolve_served_media(output_dir, "_media/1/article/0.jpg") == image.resolve()
    assert _resolve_served_media(output_dir, "1/article/0.jpg") == image.resolve()

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    with pytest.raises(PermissionError):
        _resolve_served_media(output_dir, str(outside))

    text_file = media_dir / "note.txt"
    text_file.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError):
        _resolve_served_media(output_dir, "_media/1/article/note.txt")


def test_render_note_page_escapes_markdown_for_mobile_view(tmp_path: Path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    notes_dir = output_dir / "items"
    notes_dir.mkdir(parents=True)
    note = notes_dir / "source.md"
    note.write_text(
        f"{GEN_START}\n"
        "---\nid: 1\n---\n"
        "# Mobile Source\n\n- **Insight**\n\n<script>alert(1)</script>\n"
        f"{GEN_END}\n\nPersonal tail",
        encoding="utf-8",
    )

    html = _render_note_page(note, output_dir).decode()

    assert "<h1>Mobile Source</h1>" in html
    assert "<li><strong>Insight</strong></li>" in html
    assert "# Mobile Source" not in html
    assert "xbrain:generated" not in html
    assert "id: 1" not in html
    assert "Personal tail" in html
    assert "items/source.md" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "obsidian://open?path=" in html
    assert '<html lang="en" data-theme="dark">' in html
    assert "xbrain.theme" in html
    assert 'id="note-theme"' in html
    assert "note-glow" in html
    assert "--color-bg" in html
    assert "Forest" not in html  # note titles stay source-driven, not dashboard-branded
    assert "Sugerencias" not in html
    assert "selecciona" not in html
    assert 'id="topics-note"' in html
    assert 'id="topic-modal"' in html
    assert 'id="topic-suggestions"' in html
    assert 'id="topic-prioritize"' in html
    assert 'id="topic-promote"' in html
    assert 'aria-describedby="topic-intro"' in html
    assert "Review topics" in html
    assert "Confirm current topics" in html
    assert "Move to Misc" in html
    assert "Reclassify automatically" in html
    assert "Try selected suggestions" in html
    assert "Add to vocabulary &amp; reclassify" in html
    assert "Close without changes" in html
    assert 'id="topic-action-status"' in html
    assert 'id="topic-promote-confirm"' in html
    assert ">Accept</button>" not in html
    assert ">Reject</button>" not in html
    assert "selectedSuggestions" in html
    assert "/api/topic-state" in html
    assert "/api/topic-action" in html
    assert 'id="reprocess-note"' in html
    assert 'class="note-icon"' in html
    assert ">↻</button>" in html
    assert 'id="delete-note"' in html
    assert ">×</button>" in html
    assert 'id="delete-modal"' in html
    assert "/api/reprocess-note" in html
    assert "'reprocess-note:' + itemId" in html
    assert "/api/delete-bookmark" in html
    assert "connect-src 'self'" in html


def test_render_note_page_topic_review_modal_is_explanatory_and_keyboard_safe(tmp_path: Path):
    from playwright.sync_api import sync_playwright

    output_dir = tmp_path / "vault" / "x-knowledge"
    notes_dir = output_dir / "items"
    notes_dir.mkdir(parents=True)
    note = notes_dir / "source.md"
    note.write_text(
        f"{GEN_START}\n---\nid: 42\n---\n# Topic review\n\nBody\n{GEN_END}",
        encoding="utf-8",
    )
    html = _render_note_page(note, output_dir).decode()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content(html)
        page.evaluate(
            """
            () => {
              window.fetch = async url => {
                if (String(url).startsWith('/api/topic-state')) {
                  return {
                    ok: true,
                    json: async () => ({
                      enriched: true,
                      primary_topic: 'ai-coding',
                      topics: ['ai-coding', 'developer-tools'],
                      topic_confidence: 'medium',
                      suggested_new_topics: ['agentic-workflows', 'coding-agents']
                    })
                  };
                }
                throw new Error('Unexpected request: ' + url);
              };
            }
            """
        )

        page.click("#topics-note")
        page.locator("#topic-state .topic-facts").wait_for()

        assert page.locator("#topic-modal").is_visible()
        assert page.locator(".topic-modal-card").bounding_box()["width"] > 650
        assert page.evaluate("document.activeElement.id") == "topic-close"
        assert page.locator("#topic-accept").is_enabled()
        assert page.locator("#topic-prioritize").is_enabled()
        assert page.locator("#topic-promote").is_enabled()
        assert page.locator(".topic-action-help").count() == 5

        page.locator("#topic-suggestion-list input").evaluate_all(
            """inputs => inputs.forEach(input => {
              input.checked = false;
              input.dispatchEvent(new Event('change', {bubbles: true}));
            })"""
        )
        assert page.locator("#topic-prioritize").is_disabled()
        assert page.locator("#topic-promote").is_disabled()

        page.locator("#topic-suggestion-list input").first.check()
        page.click("#topic-promote")
        assert page.locator("#topic-promote-confirm").is_visible()
        assert page.evaluate("document.activeElement.id") == "topic-promote-confirm-button"
        page.click("#topic-promote-cancel")
        assert page.locator("#topic-promote-confirm").is_hidden()
        assert page.evaluate("document.activeElement.id") == "topic-promote"

        page.keyboard.press("Escape")
        assert page.locator("#topic-modal").is_hidden()
        assert page.evaluate("document.activeElement.id") == "topics-note"
        assert errors == []
        browser.close()


def test_render_note_page_opens_video_artifacts_in_modal_and_copies_transcript(
    tmp_path: Path,
):
    from playwright.sync_api import sync_playwright

    output_dir = tmp_path / "vault" / "x-knowledge"
    notes_dir = output_dir / "items"
    video_dir = output_dir / "videos" / "video-42"
    notes_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    note = notes_dir / "source.md"
    note.write_text(
        f"{GEN_START}\n---\nid: 42\n---\n# Video source\n\n"
        "[Resumen ejecutivo](../videos/video-42/summary.md) · "
        "[Transcripción original](../videos/video-42/transcript.md)\n\n"
        "### Executive Summary\n\nInline summary.\n"
        f"{GEN_END}",
        encoding="utf-8",
    )
    (video_dir / "summary.md").write_text(
        f"{GEN_START}\n---\nkind: video_summary\n---\n# Summary\n\n"
        "[Transcripción original](transcript.md)\n\n"
        "### Executive Summary\n\nUseful **summary**.\n\n### Main Ideas\n\n- One\n"
        f"{GEN_END}",
        encoding="utf-8",
    )
    (video_dir / "transcript.md").write_text(
        f"{GEN_START}\n---\nkind: video_transcript\n---\n# Original Transcript\n\n"
        "[Video digest](summary.md)\n\n## Transcript\n\n"
        "First original line.\n\nSecond original line.\n"
        f"{GEN_END}",
        encoding="utf-8",
    )

    rendered = _render_note_page(note, output_dir).decode()

    assert 'id="video-artifact-modal"' in rendered
    assert 'data-video-artifact="summary"' in rendered
    assert 'data-video-artifact="transcript"' in rendered
    assert 'id="video-copy-transcript"' in rendered
    assert "../videos/video-42/summary.md" not in rendered
    assert "../videos/video-42/transcript.md" not in rendered

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content(rendered)
        page.evaluate(
            """
            Object.defineProperty(navigator, 'clipboard', {
              configurable: true,
              value: {writeText: async text => { window.__copiedTranscript = text; }}
            })
            """
        )

        summary_link = page.locator(
            '.note-body .video-artifact-link[data-video-artifact="summary"]'
        ).first
        summary_link.click()
        assert page.locator("#video-artifact-modal").is_visible()
        assert page.locator("#video-artifact-title").text_content() == "Executive summary"
        assert (
            page.locator("#video-artifact-panel-summary")
            .get_by_text("Useful summary.")
            .is_visible()
        )
        assert page.evaluate("document.activeElement.id") == "video-artifact-close"

        page.click("#video-artifact-tab-transcript")
        assert page.locator("#video-artifact-title").text_content() == "Original transcript"
        assert (
            page.locator("#video-artifact-panel-transcript")
            .get_by_text("First original line.")
            .is_visible()
        )
        copy_button = page.locator("#video-copy-transcript")
        assert copy_button.is_visible()
        assert copy_button.bounding_box()["height"] <= 32
        copy_button.click()
        assert page.evaluate("window.__copiedTranscript") == (
            "First original line.\n\nSecond original line."
        )
        assert page.locator("#video-copy-status").text_content() == "Copied"

        page.keyboard.press("Escape")
        assert page.locator("#video-artifact-modal").is_hidden()
        assert summary_link.evaluate("element => document.activeElement === element")
        assert errors == []
        browser.close()


def test_render_note_page_renders_obsidian_image_embeds(tmp_path: Path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    notes_dir = output_dir / "items"
    media_dir = output_dir / "_media" / "1" / "article"
    notes_dir.mkdir(parents=True)
    media_dir.mkdir(parents=True)
    (media_dir / "4.jpg").write_bytes(b"jpg")
    note = notes_dir / "source.md"
    note.write_text(
        "# Source\n\nBefore\n\n![[_media/1/article/4.jpg]]\n\nAfter",
        encoding="utf-8",
    )

    html = _render_note_page(note, output_dir).decode()

    assert '<img src="/media?path=_media%2F1%2Farticle%2F4.jpg" alt="4.jpg"' in html
    assert "![[_media/1/article/4.jpg]]" not in html


def test_render_note_page_omits_reprocess_without_item_id(tmp_path: Path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    notes_dir = output_dir / "items"
    notes_dir.mkdir(parents=True)
    note = notes_dir / "source.md"
    note.write_text("# Source\n\nBody", encoding="utf-8")

    html = _render_note_page(note, output_dir).decode()

    assert 'id="reprocess-note"' not in html
    assert "/api/reprocess-note" not in html
    assert 'id="delete-note"' not in html
    assert "/api/delete-bookmark" not in html
    assert 'id="topics-note"' not in html
    assert "/api/topic-state" not in html
    assert "/api/topic-action" not in html
    assert 'id="note-theme"' in html
    assert "xbrain.theme" in html


def test_run_topic_action_accepts_and_rejects_topic_assignment(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    cfg = load_config(tmp_path)
    save_vocab(
        [
            Topic(slug="misc", description="Miscellaneous"),
            Topic(slug="ai-coding", description="AI coding"),
        ],
        cfg.data_dir / "vocab.yaml",
    )
    item = _linked_item("42")
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="api",
        summary="summary",
        primary_topic="misc",
        topics=["misc"],
        topic_confidence="low",
        suggested_new_topics=["frontier-ai"],
    )
    save_store({"42": item}, cfg.items_path)

    accepted = _run_topic_action(cfg, "42", "accept")
    store = load_store(cfg.items_path)

    assert accepted.action == "accept"
    assert store["42"].enriched.topic_confidence == "high"
    assert store["42"].enriched.suggested_new_topics == []

    rejected = _run_topic_action(cfg, "42", "reject")
    store = load_store(cfg.items_path)

    assert rejected.action == "reject"
    assert store["42"].enriched.primary_topic == "misc"
    assert store["42"].enriched.topics == ["misc"]
    assert store["42"].enriched.topic_confidence == "high"
    assert (cfg.output_dir / "dashboard.html").exists()


def test_run_topic_action_regenerate_reenriches_item_and_refreshes_outputs(
    tmp_path: Path, monkeypatch
):
    from xbrain.executors.base import EnrichmentJudgment
    import xbrain.cli as cli

    _setup_repo(tmp_path, monkeypatch)
    cfg = load_config(tmp_path)
    save_vocab(
        [
            Topic(slug="misc", description="Miscellaneous"),
            Topic(slug="ai-coding", description="AI coding"),
        ],
        cfg.data_dir / "vocab.yaml",
    )
    item = _linked_item("42")
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="api",
        summary="old summary",
        primary_topic="misc",
        topics=["misc"],
        topic_confidence="low",
    )
    save_store({"42": item}, cfg.items_path)
    calls: list[str] = []

    class _FakeApiExecutor:
        def __init__(self, *args, **kwargs):
            calls.append("executor")

        def enrich_items(self, items, vocab):
            calls.append(f"enrich:{[item.id for item in items]}")
            return [
                EnrichmentJudgment(
                    item_id="42",
                    summary="new summary",
                    primary_topic="ai-coding",
                    topics=["ai-coding"],
                    topic_confidence="high",
                )
            ]

    monkeypatch.setattr(cli, "ApiExecutor", _FakeApiExecutor)
    monkeypatch.setattr(
        cli, "_run_topics_executor", lambda *_args, **_kwargs: calls.append("topics")
    )
    monkeypatch.setattr(cli, "_run_generate", lambda *_args, **_kwargs: calls.append("generate"))

    report = _run_topic_action(cfg, "42", "regenerate")
    store = load_store(cfg.items_path)

    assert report.action == "regenerate"
    assert report.primary_topic == "ai-coding"
    assert store["42"].enriched.summary == "new summary"
    assert calls == ["executor", "enrich:['42']", "topics", "generate"]
    assert any(
        path.name.endswith("-pre-topic-regenerate-42")
        for path, _ in snapshot.snapshot_list(cfg.data_dir)
    )


def test_run_topic_action_regenerate_prioritizes_selected_suggestions(tmp_path: Path, monkeypatch):
    from xbrain.executors.base import EnrichmentJudgment
    import xbrain.cli as cli

    _setup_repo(tmp_path, monkeypatch)
    cfg = load_config(tmp_path)
    save_vocab(
        [
            Topic(slug="misc", description="Miscellaneous"),
            Topic(slug="ai-coding", description="AI coding"),
        ],
        cfg.data_dir / "vocab.yaml",
    )
    item = _linked_item("42")
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="api",
        summary="old summary",
        primary_topic="misc",
        topics=["misc"],
        topic_confidence="low",
        suggested_new_topics=["ai-productivity-tools", "llm-workflows"],
    )
    save_store({"42": item}, cfg.items_path)
    calls: list[object] = []

    class _FakeApiExecutor:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("topic_hints"))

        def enrich_items(self, items, vocab):
            return [
                EnrichmentJudgment(
                    item_id="42",
                    summary="new summary",
                    primary_topic="ai-coding",
                    topics=["ai-coding"],
                    topic_confidence="medium",
                    suggested_new_topics=["ai-productivity-tools"],
                )
            ]

    monkeypatch.setattr(cli, "ApiExecutor", _FakeApiExecutor)
    monkeypatch.setattr(cli, "_run_topics_executor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_run_generate", lambda *_args, **_kwargs: None)

    report = _run_topic_action(
        cfg,
        "42",
        "regenerate",
        prioritize_suggestions=True,
        suggested_topics=["ai-productivity-tools"],
    )

    assert report.action == "regenerate"
    assert calls == [{"42": ["ai-productivity-tools"]}]
    assert [topic.slug for topic in load_vocab(cfg.data_dir / "vocab.yaml")] == [
        "misc",
        "ai-coding",
    ]


def test_run_topic_action_promotes_selected_suggestions_before_regenerate(
    tmp_path: Path, monkeypatch
):
    from xbrain.executors.base import EnrichmentJudgment
    import xbrain.cli as cli

    _setup_repo(tmp_path, monkeypatch)
    cfg = load_config(tmp_path)
    save_vocab([Topic(slug="misc", description="Miscellaneous")], cfg.data_dir / "vocab.yaml")
    item = _linked_item("42")
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="api",
        summary="old summary",
        primary_topic="misc",
        topics=["misc"],
        topic_confidence="low",
        suggested_new_topics=["ai-productivity-tools"],
    )
    save_store({"42": item}, cfg.items_path)
    calls: list[object] = []

    class _FakeApiExecutor:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("topic_hints"))

        def enrich_items(self, items, vocab):
            assert {topic.slug for topic in vocab} == {"misc", "ai-productivity-tools"}
            return [
                EnrichmentJudgment(
                    item_id="42",
                    summary="new summary",
                    primary_topic="ai-productivity-tools",
                    topics=["ai-productivity-tools"],
                    topic_confidence="high",
                )
            ]

    monkeypatch.setattr(cli, "ApiExecutor", _FakeApiExecutor)
    monkeypatch.setattr(cli, "_run_topics_executor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_run_generate", lambda *_args, **_kwargs: None)

    report = _run_topic_action(
        cfg,
        "42",
        "regenerate",
        promote_suggestions=True,
        suggested_topics=["ai-productivity-tools"],
    )
    store = load_store(cfg.items_path)

    assert report.promoted_topics == ["ai-productivity-tools"]
    assert store["42"].enriched.primary_topic == "ai-productivity-tools"
    assert calls == [{"42": ["ai-productivity-tools"]}]
    assert [topic.slug for topic in load_vocab(cfg.data_dir / "vocab.yaml")] == [
        "misc",
        "ai-productivity-tools",
    ]


def test_run_topic_action_rejects_unavailable_topic_suggestions(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    cfg = load_config(tmp_path)
    save_vocab([Topic(slug="misc", description="Miscellaneous")], cfg.data_dir / "vocab.yaml")
    item = _linked_item("42")
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="api",
        summary="old summary",
        primary_topic="misc",
        topics=["misc"],
        topic_confidence="low",
        suggested_new_topics=["ai-productivity-tools"],
    )
    save_store({"42": item}, cfg.items_path)

    with pytest.raises(ValueError, match="not available"):
        _run_topic_action(
            cfg,
            "42",
            "regenerate",
            prioritize_suggestions=True,
            suggested_topics=["invented-topic"],
        )


def test_run_reprocess_note_refreshes_single_item_pipeline(tmp_path: Path, monkeypatch):
    from xbrain.executors.base import EnrichmentJudgment
    import xbrain.cli as cli

    _setup_repo(tmp_path, monkeypatch)
    cfg = load_config(tmp_path)
    save_vocab([Topic(slug="misc", description="Miscellaneous")], cfg.data_dir / "vocab.yaml")
    save_store({"42": _linked_item("42"), "43": _linked_item("43")}, cfg.items_path)
    calls: list[str] = []

    def _fake_fetch_pending(selected, *, force):
        calls.append(f"fetch:{sorted(selected)}:{force}")
        return 1

    def _fake_fetch_x_articles(selected, _storage_state, force, *, headless, linked_store):
        assert sorted(linked_store) == ["42", "43"]
        calls.append(f"x_fetch:{sorted(selected)}:{force}:{headless}")
        return 2

    class _FakeApiExecutor:
        def __init__(self, *args, **kwargs):
            calls.append("executor")

        def enrich_items(self, items, vocab):
            calls.append(f"enrich:{[item.id for item in items]}")
            return [
                EnrichmentJudgment(
                    item_id="42",
                    summary="reprocessed",
                    primary_topic="misc",
                    topics=["misc"],
                    topic_confidence="high",
                )
            ]

    monkeypatch.setattr(cli, "fetch_pending", _fake_fetch_pending)
    monkeypatch.setattr(cli, "fetch_x_articles", _fake_fetch_x_articles)
    monkeypatch.setattr(cli, "_run_digest_video", lambda *_args, **_kwargs: calls.append("digest"))
    monkeypatch.setattr(cli, "_run_media", lambda *_args, **_kwargs: calls.append("media"))
    monkeypatch.setattr(cli, "_run_describe", lambda *_args, **_kwargs: calls.append("describe"))
    monkeypatch.setattr(cli, "ApiExecutor", _FakeApiExecutor)
    monkeypatch.setattr(
        cli, "_run_topics_executor", lambda *_args, **_kwargs: calls.append("topics")
    )
    monkeypatch.setattr(cli, "_run_generate", lambda *_args, **_kwargs: calls.append("generate"))

    report = _run_reprocess_note(cfg, "42", headless=True)
    store = load_store(cfg.items_path)

    assert report.item_id == "42"
    assert report.articles == 1
    assert report.x_articles == 2
    assert store["42"].enriched.summary == "reprocessed"
    assert store["43"].enriched is None
    assert calls == [
        "fetch:['42']:True",
        "x_fetch:['42']:True:True",
        "digest",
        "media",
        "describe",
        "executor",
        "enrich:['42']",
        "topics",
        "generate",
    ]
    assert any(
        path.name.endswith("-pre-reprocess-note-42")
        for path, _ in snapshot.snapshot_list(cfg.data_dir)
    )


def test_run_delete_bookmark_removes_store_note_media_and_regenerates(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    output_dir = vault / "x-knowledge"
    save_store(
        {"42": _linked_item("42"), "43": _linked_item("43")},
        tmp_path / "data" / "items.json",
    )
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 0, result.output
    assert list((output_dir / "items").glob("*-42.md"))

    data_media = tmp_path / "data" / "media" / "42"
    vault_media = output_dir / "_media" / "42"
    video_artifact = output_dir / "videos" / "clip-42"
    data_media.mkdir(parents=True)
    vault_media.mkdir(parents=True)
    video_artifact.mkdir(parents=True)
    (data_media / "0.jpg").write_bytes(b"jpg")
    (vault_media / "0.jpg").write_bytes(b"jpg")
    (video_artifact / "summary.md").write_text("---\nid: 42\n---\n# Clip", encoding="utf-8")

    report = _run_delete_bookmark(load_config(tmp_path), "42")

    assert report.item_id == "42"
    assert "42" not in load_store(tmp_path / "data" / "items.json")
    assert "43" in load_store(tmp_path / "data" / "items.json")
    assert not list((output_dir / "items").glob("*-42.md"))
    assert list((output_dir / "items").glob("*-43.md"))
    assert not data_media.exists()
    assert not vault_media.exists()
    assert not video_artifact.exists()
    assert (output_dir / "dashboard.html").exists()


def test_cli_reports_missing_config_cleanly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    result = runner.invoke(app, ["status"])
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_media_command_runs_on_empty_store(tmp_path: Path, monkeypatch):
    """`xbrain media` with no media is a no-op (exit 0, items.json unchanged)."""
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"1": _linked_item("1")}, items_path)
    before_bytes = items_path.read_bytes()

    result = runner.invoke(app, ["media"])

    assert result.exit_code == 0
    # The pre-media snapshot must have been created — the destructive-op
    # contract: every command that writes items.json snapshots first.
    snapshots = list((tmp_path / "data" / "snapshots").iterdir())
    assert any("pre-media" in p.name for p in snapshots)
    # items.json content is byte-identical: no media to advance, no
    # transitions, no spurious rewrites that would dirty timestamps.
    assert items_path.read_bytes() == before_bytes


def test_media_command_creates_media_dir_and_downloads_pending_photos(tmp_path: Path, monkeypatch):
    """End-to-end through the CLI with a fake session.

    The test patches `xbrain.media.requests.Session` to inject canned
    responses so the CLI hits the real `download_all` orchestrator
    without any real network call.
    """
    import io as _io

    from PIL import Image

    from xbrain.models import MediaPhotoDownloaded, MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("42")
    item.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/Z.png")]
    save_store({"42": item}, tmp_path / "data" / "items.json")

    # Build a valid PNG byte payload Pillow can decode.
    buffer = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buffer, format="PNG")
    bytes_data = buffer.getvalue()

    class _FakeSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 200
                content = bytes_data

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _FakeSession)
    result = runner.invoke(app, ["media"])
    assert result.exit_code == 0, result.output

    # The downloaded photo landed in the variant + the file is on disk.
    from xbrain.store import load_store

    reloaded = load_store(tmp_path / "data" / "items.json")
    entry = reloaded["42"].media[0]
    assert isinstance(entry, MediaPhotoDownloaded)
    assert entry.local_path == "42/0.png"
    assert (tmp_path / "data" / "media" / "42" / "0.png").exists()


def test_media_command_resume_after_interrupt_completes_remaining(tmp_path: Path, monkeypatch):
    """Ctrl-C mid-batch leaves items.json valid; the next run completes the rest.

    Setup: an item with 3 pending photos. Run 1 raises KeyboardInterrupt
    after the 2nd photo's GET, so:
      - photo 0 was downloaded and persisted by `on_progress`,
      - photo 1's session.get raises, propagating out of `download_all`
        (mid-photo: no transition persisted for it, stays Pending),
      - photo 2 never starts.
    Run 2 uses a fresh session that succeeds for every photo and verifies
    that all three end up Downloaded.
    """
    import io as _io

    from PIL import Image

    from xbrain.models import MediaPhotoDownloaded, MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("99")
    item.media = [
        MediaPhotoPending(url="https://pbs.twimg.com/media/R0.png"),
        MediaPhotoPending(url="https://pbs.twimg.com/media/R1.png"),
        MediaPhotoPending(url="https://pbs.twimg.com/media/R2.png"),
    ]
    save_store({"99": item}, tmp_path / "data" / "items.json")

    buffer = _io.BytesIO()
    Image.new("RGB", (4, 3), color=(50, 60, 70)).save(buffer, format="PNG")
    png = buffer.getvalue()

    class _InterruptingSession:
        """Returns 200 for the 1st photo, raises KeyboardInterrupt on the 2nd."""

        def __init__(self):
            self.headers: dict[str, str] = {}
            self.call_count = 0

        def get(self, _url, *, timeout):
            self.call_count += 1
            if self.call_count == 2:
                raise KeyboardInterrupt

            class _Resp:
                status_code = 200
                content = png

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _InterruptingSession)
    result1 = runner.invoke(app, ["media"])
    # Typer surfaces KeyboardInterrupt as a non-zero exit code.
    assert result1.exit_code != 0

    from xbrain.store import load_store

    reloaded = load_store(tmp_path / "data" / "items.json")
    media = reloaded["99"].media
    assert isinstance(media[0], MediaPhotoDownloaded)
    # Photo 1 stays Pending because the KeyboardInterrupt fired BEFORE
    # the transition was recorded (the seam: `on_progress` is what
    # persists per-photo state; we never reached it).
    assert isinstance(media[1], MediaPhotoPending)
    assert isinstance(media[2], MediaPhotoPending)

    # Run 2: fresh session that succeeds for every GET.
    class _OkSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 200
                content = png

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _OkSession)
    result2 = runner.invoke(app, ["media"])
    assert result2.exit_code == 0

    final = load_store(tmp_path / "data" / "items.json")
    for entry in final["99"].media:
        assert isinstance(entry, MediaPhotoDownloaded)


def test_media_command_propagates_total_failure_as_exit_1(tmp_path: Path, monkeypatch):
    """A run where every download fails surfaces as exit-1 at the CLI boundary."""
    from xbrain.models import MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("99")
    item.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/X.png")]
    save_store({"99": item}, tmp_path / "data" / "items.json")

    class _FakeSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 404
                content = b""

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _FakeSession)
    result = runner.invoke(app, ["media"])
    assert result.exit_code == 1
    # Even on total failure, the failure record is persisted: the CLI's
    # finally block writes the store before the RuntimeError propagates.
    from xbrain.store import load_store

    from xbrain.models import MediaPhotoFailed as _MPF

    reloaded = load_store(tmp_path / "data" / "items.json")
    assert isinstance(reloaded["99"].media[0], _MPF)


def _article_item_with_pending_image(item_id: str, image_url: str) -> Item:
    """An item whose `x_article` source carries one pending inline image (#39 PR4)."""
    from xbrain.models import (
        ArticleImageBlock,
        ArticleTextBlock,
        Content,
        ContentSourceSuccess,
        MediaPhotoPending,
    )

    source = ContentSourceSuccess(
        kind="x_article",
        url=f"https://x.com/i/article/{item_id}",
        title="An Article",
        text="Body text.",
        blocks=[
            ArticleTextBlock(text="Body text."),
            ArticleImageBlock(media=MediaPhotoPending(url=image_url)),
        ],
        http_status=200,
        attempts=1,
    )
    item = _linked_item(item_id)
    item.content = Content(
        fetched_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        sources=[source],
    )
    return item


def test_media_command_downloads_article_inline_images(tmp_path: Path, monkeypatch):
    """End-to-end: `xbrain media` downloads an Article's inline image to `<id>/article/`."""
    import io as _io

    from PIL import Image

    from xbrain.models import MediaPhotoDownloaded

    _setup_repo(tmp_path, monkeypatch)
    item = _article_item_with_pending_image("77", "https://pbs.twimg.com/media/ART.png")
    save_store({"77": item}, tmp_path / "data" / "items.json")

    buffer = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buffer, format="PNG")
    bytes_data = buffer.getvalue()

    class _FakeSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 200
                content = bytes_data

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _FakeSession)
    result = runner.invoke(app, ["media"])
    assert result.exit_code == 0, result.output

    from xbrain.store import load_store

    reloaded = load_store(tmp_path / "data" / "items.json")
    block = reloaded["77"].content.sources[0].blocks[1]
    assert isinstance(block.media, MediaPhotoDownloaded)
    assert block.media.local_path == "77/article/0.png"
    assert (tmp_path / "data" / "media" / "77" / "article" / "0.png").exists()
    # The pre-media snapshot fired (article images ride the SAME snapshot boundary).
    snapshots = list((tmp_path / "data" / "snapshots").iterdir())
    assert any("pre-media" in p.name for p in snapshots)


def test_media_command_snapshot_failure_aborts_before_any_write(tmp_path: Path, monkeypatch):
    """A snapshot failure propagates and aborts — the Article image stays pending."""
    from xbrain.models import MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item = _article_item_with_pending_image("88", "https://pbs.twimg.com/media/ARTX.png")
    save_store({"88": item}, tmp_path / "data" / "items.json")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("snapshot disk full")

    # The download session must never be reached: if the snapshot aborts first,
    # no HTTP call and no store write happen.
    monkeypatch.setattr("xbrain.cli.snapshot.snapshot_create", _boom)
    result = runner.invoke(app, ["media"])
    assert result.exit_code != 0

    from xbrain.store import load_store

    reloaded = load_store(tmp_path / "data" / "items.json")
    block = reloaded["88"].content.sources[0].blocks[1]
    assert isinstance(block.media, MediaPhotoPending)  # untouched
    assert not (tmp_path / "data" / "media" / "88").exists()


def test_media_command_warns_when_items_filter_matches_nothing(tmp_path: Path, monkeypatch):
    """`--items` with IDs absent from the store prints an AVISO to stderr."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["media", "--items", "ghost-id-1,ghost-id-2"])
    assert result.exit_code == 0
    # The CLI mixes stdout and stderr in CliRunner output; the AVISO is on stderr
    # — combined output is fine for the substring check.
    assert "AVISO" in result.output
    assert "ghost-id-1" in result.output


def test_media_command_verbose_lists_failed_urls(tmp_path: Path, monkeypatch):
    """`--verbose` prints `<item_id> <reason> <url>` per failure on stderr."""
    from xbrain.models import MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    items = {
        "1": _linked_item("1"),
        "2": _linked_item("2"),
    }
    items["1"].media = [MediaPhotoPending(url="https://pbs.twimg.com/media/V1.png")]
    items["2"].media = [MediaPhotoPending(url="https://pbs.twimg.com/media/V2.png")]
    save_store(items, tmp_path / "data" / "items.json")

    import io as _io
    from PIL import Image

    buffer = _io.BytesIO()
    Image.new("RGB", (4, 3)).save(buffer, format="PNG")
    png = buffer.getvalue()

    class _MixedSession:
        """Photo for item 1 succeeds; photo for item 2 returns 404 three times."""

        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, url, *, timeout):
            if "V1" in url:

                class _Ok:
                    status_code = 200
                    content = png

                return _Ok()

            class _Fail:
                status_code = 404
                content = b""

            return _Fail()

    monkeypatch.setattr("xbrain.media.requests.Session", _MixedSession)
    result = runner.invoke(app, ["media", "--verbose"])
    assert result.exit_code == 0
    assert "Failed media" in result.output
    assert "http_4xx" in result.output
    assert "V2" in result.output


def test_media_command_respects_items_filter(tmp_path: Path, monkeypatch):
    """`--items` limits the run to specific item IDs."""
    import io as _io

    from PIL import Image

    from xbrain.models import MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item_a = _linked_item("a")
    item_b = _linked_item("b")
    item_a.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/A.png")]
    item_b.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/B.png")]
    save_store({"a": item_a, "b": item_b}, tmp_path / "data" / "items.json")

    buffer = _io.BytesIO()
    Image.new("RGB", (4, 3)).save(buffer, format="PNG")
    bytes_data = buffer.getvalue()

    class _FakeSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 200
                content = bytes_data

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _FakeSession)
    result = runner.invoke(app, ["media", "--items", "b"])
    assert result.exit_code == 0

    from xbrain.store import load_store

    from xbrain.models import MediaPhotoDownloaded, MediaPhotoPending as _MPP

    reloaded = load_store(tmp_path / "data" / "items.json")
    # `a` stayed pending; only `b` was downloaded.
    assert isinstance(reloaded["a"].media[0], _MPP)
    assert isinstance(reloaded["b"].media[0], MediaPhotoDownloaded)


# ------------------------------------------------------------ download-videos


_MP4_URL = "https://video.twimg.com/ext_tw_video/1/vid/720/A.mp4?tag=12"
_HLS_URL = "https://video.twimg.com/ext_tw_video/1/pl/B.m3u8?c=fmp4"
_POSTER = "https://pbs.twimg.com/ext_tw_video_thumb/1/img/P.jpg"


def _video_item(
    item_id: str,
    url: str = _MP4_URL,
    source: str = "bookmark",
    *,
    transcript_url: str | None = None,
    transcript_language: str | None = "en",
    transcript_format: str | None = "vtt",
):
    from xbrain.models import MediaVideoPending

    item = Item(
        id=item_id,
        source=source,
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    item.media = [
        MediaVideoPending(
            url=url,
            thumbnail_url=_POSTER,
            bitrate=2_176_000,
            duration_millis=30_000,
            transcript_url=transcript_url,
            transcript_language=transcript_language,
            transcript_format=transcript_format,
        )
    ]
    return item


def test_download_videos_command_is_disabled_and_does_not_write(tmp_path: Path, monkeypatch):
    """The CLI must not provide any MP4 persistence path."""
    from xbrain.models import MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42")}, items_path)
    before = items_path.read_bytes()

    result = runner.invoke(app, ["download-videos", "--max-size", "banana", "--yes"])

    assert result.exit_code == 1
    assert "deshabilitado" in result.output
    assert "no guarda MP4" in result.output
    assert items_path.read_bytes() == before
    assert isinstance(load_store(items_path)["42"].media[0], MediaVideoPending)
    assert not (tmp_path / "data" / "media").exists()
    snapshots = tmp_path / "data" / "snapshots"
    assert not snapshots.exists()


def test_download_videos_command_is_disabled_before_confirmation(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["download-videos"], input="y\n")

    assert result.exit_code == 1
    assert "deshabilitado" in result.output
    assert "Continuar" not in result.output


def test_parse_date_returns_utc_aware():
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-01-01")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert _parse_date(None) is None


def test_parse_date_since_keeps_midnight():
    """A date-only `since` stays at the day's start (00:00:00) — no snapping."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31")
    assert parsed == datetime(2025, 12, 31, 0, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_date_until_snaps_date_only_to_end_of_day():
    """A date-only `until` snaps to the last microsecond so the whole day is included."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31", end_of_day=True)
    assert parsed == datetime(2025, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)


def test_parse_date_until_respects_explicit_time():
    """An explicit time on `until` is respected as-is — never snapped to end of day."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31T09:00:00", end_of_day=True)
    assert parsed == datetime(2025, 12, 31, 9, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_date_until_space_basic_time_is_respected():
    """A space-separated basic-format time (no colon) is a time — must NOT snap."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31 120000", end_of_day=True)
    assert parsed == datetime(2025, 12, 31, 12, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_date_until_bare_date_with_offset_snaps():
    """A bare date carrying only a tz offset (no time-of-day) still snaps to end of day."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31+00:00", end_of_day=True)
    assert parsed == datetime(2025, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)


def test_cli_generate_until_date_only_includes_whole_final_day(tmp_path: Path, monkeypatch):
    """`--until <date>` (date-only) includes items up to 23:59:59.999999 of that day.

    Pins the microsecond boundary: an item at the last microsecond of the day is
    INCLUDED, the next-day midnight item is EXCLUDED. Catches a `>`→`>=` flip or any
    coarsening of the snap value.
    """
    vault = _setup_repo(tmp_path, monkeypatch)
    midday = _linked_item("1")
    midday.created_at = datetime(2026, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
    last_us = _linked_item("2")
    last_us.created_at = datetime(2026, 1, 15, 23, 59, 59, 999999, tzinfo=timezone.utc)
    next_day = _linked_item("3")
    next_day.created_at = datetime(2026, 1, 16, 0, 0, 0, tzinfo=timezone.utc)
    save_store({"1": midday, "2": last_us, "3": next_day}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate", "--until", "2026-01-15"])
    assert result.exit_code == 0
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    names = sorted(n.name for n in notes)
    assert len(names) == 2
    assert names[0].endswith("-1.md") and names[1].endswith("-2.md")
    assert not any(n.endswith("-3.md") for n in names)


def test_cli_generate_until_explicit_time_is_respected(tmp_path: Path, monkeypatch):
    """`--until <date>T09:00:00` excludes an item at 10:00 and includes one at 08:00."""
    vault = _setup_repo(tmp_path, monkeypatch)
    after = _linked_item("1")
    after.created_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    before = _linked_item("2")
    before.created_at = datetime(2026, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
    save_store({"1": after, "2": before}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate", "--until", "2026-01-15T09:00:00"])
    assert result.exit_code == 0
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    assert len(notes) == 1
    assert notes[0].name.endswith("-2.md")


def test_cli_generate_since_date_only_includes_midnight_item(tmp_path: Path, monkeypatch):
    """`--since <date>` (date-only) still includes an item created at that day's midnight."""
    vault = _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("1")
    item.created_at = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    save_store({"1": item}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate", "--since", "2026-01-15"])
    assert result.exit_code == 0
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    assert len(notes) == 1


def test_cli_enrich_until_date_only_includes_whole_final_day(tmp_path, monkeypatch):
    """Pins the `end_of_day=True` wiring on a SECOND command (`enrich`), not just `generate`.

    An item created on the `--until` day at 15:00 must count as pending — a mutation
    dropping `end_of_day=True` from the enrich site would exclude it (0 pending).
    """
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.models import Topic
    from xbrain.rubrics import save_vocab

    item = _linked_item("1")
    item.created_at = datetime(2026, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
    save_store({"1": item}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["enrich", "--executor", "manual", "--until", "2026-01-15"])
    assert result.exit_code == 0
    assert "1 items exportados" in result.output
    assert (tmp_path / "data" / "enrich-worksheet.json").exists()


def test_cli_fetch_with_since_does_not_crash(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    old_item = _linked_item("1")
    old_item.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    save_store({"1": old_item}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["fetch", "--since", "2025-01-01"])
    assert result.exit_code == 0


def test_cli_generate_with_since_filters_notes(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    old_item = _linked_item("1")
    old_item.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new_item = _linked_item("2")
    new_item.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    save_store({"1": old_item, "2": new_item}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate", "--since", "2025-01-01"])
    assert result.exit_code == 0
    assert (vault / "x-knowledge" / "_index.md").exists()
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    assert len(notes) == 1


def test_vocab_command_persists_induced_topics(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    from xbrain.models import Topic
    import xbrain.cli as cli

    monkeypatch.setattr(
        cli, "induce_vocab", lambda *a, **k: [Topic(slug="misc", description="Noise.")]
    )
    result = runner.invoke(app, ["vocab", "--executor", "api"])
    assert result.exit_code == 0
    from xbrain.rubrics import load_vocab

    assert [t.slug for t in load_vocab(tmp_path / "data" / "vocab.yaml")] == ["misc"]


def test_enrich_manual_exports_a_worksheet(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Topic

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["enrich", "--executor", "manual"])
    assert result.exit_code == 0
    assert (tmp_path / "data" / "enrich-worksheet.json").exists()


def test_enrich_apply_imports_a_filled_worksheet(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Topic

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps(
            {
                "judgments": [
                    {"item_id": "1", "summary": "s", "primary_topic": "misc", "topics": ["misc"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["enrich", "--apply", str(ws)])
    assert result.exit_code == 0
    from xbrain.store import load_store

    store = load_store(tmp_path / "data" / "items.json")
    assert store["1"].enriched is not None


def test_enrich_api_executor_enriches_the_store(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import load_store, save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Topic
    from xbrain.executors.base import EnrichmentJudgment
    import xbrain.cli as cli

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")

    class _FakeApiExecutor:
        def __init__(self, *a, **k):
            pass

        def enrich_items(self, items, vocab):
            return [
                EnrichmentJudgment(
                    item_id=i.id, summary="resumen", primary_topic="misc", topics=["misc"]
                )
                for i in items
            ]

    monkeypatch.setattr(cli, "ApiExecutor", _FakeApiExecutor)
    result = runner.invoke(app, ["enrich", "--executor", "api"])
    assert result.exit_code == 0
    store = load_store(tmp_path / "data" / "items.json")
    assert store["1"].enriched is not None


def test_enrich_taxonomy_risk_reenriches_existing_misc_item(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import load_store, save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Enrichment, Topic
    from xbrain.executors.base import EnrichmentJudgment
    import xbrain.cli as cli

    item = _linked_item("1")
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="api",
        summary="old",
        primary_topic="misc",
        topics=["misc"],
        topic_confidence="high",
    )
    save_store({"1": item}, tmp_path / "data" / "items.json")
    save_vocab(
        [Topic(slug="ai-coding", description="d"), Topic(slug="misc", description="d")],
        tmp_path / "data" / "vocab.yaml",
    )

    class _FakeApiExecutor:
        def __init__(self, *a, **k):
            pass

        def enrich_items(self, items, vocab):
            return [
                EnrichmentJudgment(
                    item_id=i.id,
                    summary="new",
                    primary_topic="ai-coding",
                    topics=["ai-coding"],
                    topic_confidence="high",
                )
                for i in items
            ]

    monkeypatch.setattr(cli, "ApiExecutor", _FakeApiExecutor)
    result = runner.invoke(app, ["enrich", "--executor", "api", "--taxonomy-risk"])
    assert result.exit_code == 0, result.output
    store = load_store(tmp_path / "data" / "items.json")
    assert store["1"].enriched.summary == "new"
    assert store["1"].enriched.primary_topic == "ai-coding"


def test_enrich_rejects_unknown_executor(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Topic

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["enrich", "--executor", "bogus"])
    assert result.exit_code == 1
    assert "bogus" in result.output
    assert "desconocido" in result.output


def _photo_item(item_id: str = "1") -> Item:
    from xbrain.models import MediaPhotoDownloaded

    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        media=[
            MediaPhotoDownloaded(
                url="https://p/0.png",
                local_path=f"{item_id}/0.png",
                width=4,
                height=4,
                bytes_size=9,
                downloaded_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
            )
        ],
    )


def test_describe_rejects_unknown_executor(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["describe", "--executor", "bogus"])
    assert result.exit_code == 1
    assert "bogus" in result.output
    assert "desconocido" in result.output


def test_describe_claude_code_exports_a_worksheet(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _photo_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["describe", "--executor", "claude-code"])
    assert result.exit_code == 0
    ws = tmp_path / "data" / "describe-worksheet.json"
    assert ws.exists()
    payload = json.loads(ws.read_text(encoding="utf-8"))
    assert len(payload["photos"]) == 1
    assert payload["judgments"] == []


def test_describe_apply_dispatches_and_reports_unmatched(tmp_path, monkeypatch):
    import json

    from xbrain.models import MediaPhotoDescribed
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _photo_item("1")}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps(
            {
                "version": "v1",
                "language": "English",
                "judgments": [
                    {"item_id": "1", "index": 0, "is_decorative": False, "description": "A chart."},
                    {"item_id": "ghost", "index": 0, "is_decorative": False, "description": "x"},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["describe", "--apply", str(ws)])
    assert result.exit_code == 0
    assert "1 fotos descritas" in result.output
    assert "ghost#post:0" in result.output  # unmatched judgment surfaced, not dropped silently
    store = load_store(tmp_path / "data" / "items.json")
    assert isinstance(store["1"].media[0], MediaPhotoDescribed)


def test_enrich_manual_without_vocab_fails(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["enrich", "--executor", "manual"])
    assert result.exit_code == 1
    assert "vocabulario" in result.output


def test_enrich_apply_without_vocab_fails(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"judgments": []}), encoding="utf-8")
    result = runner.invoke(app, ["enrich", "--apply", str(ws)])
    assert result.exit_code == 1
    assert "vocabulario" in result.output


def test_enrich_manual_with_no_pending_items_writes_no_worksheet(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Enrichment, Topic

    item = _linked_item("1")
    item.enriched = Enrichment(
        enriched_at=datetime.now(timezone.utc),
        executor="manual",
        summary="s",
        primary_topic="misc",
        topics=["misc"],
    )
    save_store({"1": item}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["enrich", "--executor", "manual"])
    assert result.exit_code == 0
    assert "No hay items pendientes" in result.output
    assert not (tmp_path / "data" / "enrich-worksheet.json").exists()


def test_cli_fetch_reports_x_articles(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr(cli, "fetch_pending", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "fetch_x_articles", lambda *a, **k: 3)
    monkeypatch.setattr(cli, "expand_threads", lambda *a, **k: 0)
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 0
    assert "3 de X" in result.output


def test_cli_fetch_prunes_non_library_items(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Media
    from xbrain.store import load_store

    article = _linked_item("article")
    video = _linked_item("video")
    video.links = []
    video.media = [Media(type="video", url="https://video.twimg.com/v/video.mp4")]
    plain = _linked_item("plain")
    plain.links = []
    own = _linked_item("own")
    own.source = "own_tweet"
    save_store(
        {item.id: item for item in [article, video, plain, own]},
        tmp_path / "data" / "items.json",
    )
    monkeypatch.setattr(cli, "fetch_pending", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "fetch_x_articles", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "expand_threads", lambda *a, **k: 0)

    result = runner.invoke(app, ["fetch"])

    assert result.exit_code == 0, result.output
    assert "2 descartados por política" in result.output
    assert set(load_store(tmp_path / "data" / "items.json")) == {"article", "video"}


def test_cli_fetch_persists_partial_work_when_a_stage_raises(tmp_path, monkeypatch):
    # `_run_fetch` wraps the three fetch stages in `try:` and `save_store` in
    # `finally:` — a stage error must not discard the in-memory work the
    # earlier stages already produced. This proves that `finally` contract.
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Content, ContentSourceSuccess
    from xbrain.store import load_store

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")

    def _fake_fetch_pending(store, *a, **k):
        # Mutate the SAME store object _run_fetch passed in — that is what
        # the `finally` block later persists to disk.
        store["1"].content = Content(
            fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.com/p",
                    text="cuerpo",
                )
            ],
        )
        return 1

    def _fake_fetch_x_articles(*a, **k):
        raise RuntimeError("Sesión de X caducada")

    monkeypatch.setattr(cli, "fetch_pending", _fake_fetch_pending)
    monkeypatch.setattr(cli, "fetch_x_articles", _fake_fetch_x_articles)

    result = runner.invoke(app, ["fetch"])
    # The stage error still surfaces via _handle_cli_errors.
    assert result.exit_code == 1

    # ...but the partial work from fetch_pending survived: save_store ran in
    # the `finally` despite the raise.
    store = load_store(tmp_path / "data" / "items.json")
    assert store["1"].content is not None
    assert store["1"].content.sources[0].text == "cuerpo"


def test_retry_failed_retries_only_failed_bookmarks(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Content, ContentSourceFailure, ContentSourceSuccess

    failed_external = _linked_item("ext")
    failed_external.content = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSourceFailure(
                kind="external_article",
                url="https://example.com/p",
                failure_reason="js_required",
            )
        ],
    )
    failed_x = _linked_item("x")
    failed_x.links = [Link(url="https://x.com/i/article/1", domain="x.com")]
    failed_x.content = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSourceFailure(
                kind="x_article",
                url="https://x.com/i/article/1",
                failure_reason="empty_content",
            )
        ],
    )
    own_failed = _linked_item("own")
    own_failed.source = "own_tweet"
    own_failed.content = failed_external.content
    ok = _linked_item("ok")
    ok.content = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSourceSuccess(kind="external_article", url="https://example.com/p", text="ok")
        ],
    )
    save_store(
        {"ext": failed_external, "x": failed_x, "own": own_failed, "ok": ok},
        tmp_path / "data" / "items.json",
    )

    calls: list[tuple] = []
    monkeypatch.setattr(
        cli, "_auto_snapshot", lambda _cfg, command: calls.append(("snapshot", command))
    )
    monkeypatch.setattr(
        cli,
        "fetch_pending",
        lambda store, *args, **kwargs: (
            calls.append(("fetch_pending", sorted(store), kwargs["force"])) or len(store)
        ),
    )

    def _fake_fetch_x_articles(store, _state, force, *args, headless=False, **kwargs):
        calls.append(("fetch_x", sorted(store), force, headless))
        return len(store)

    monkeypatch.setattr(cli, "fetch_x_articles", _fake_fetch_x_articles)
    monkeypatch.setattr(
        cli, "_run_generate", lambda _cfg, since, until: calls.append(("generate", since, until))
    )

    result = runner.invoke(app, ["retry-failed"])

    assert result.exit_code == 0, result.output
    assert ("snapshot", "retry-failed-bookmarks") in calls
    assert ("fetch_pending", ["ext"], True) in calls
    assert ("fetch_x", ["x"], True, True) in calls
    assert ("generate", None, None) in calls
    assert "Fallidos relanzados: 2 items" in result.output


def test_retry_failed_noops_without_failed_bookmarks(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr(cli, "_auto_snapshot", lambda *_args, **_kwargs: pytest.fail("no snapshot"))
    monkeypatch.setattr(cli, "fetch_pending", lambda *_args, **_kwargs: pytest.fail("no fetch"))
    monkeypatch.setattr(
        cli, "fetch_x_articles", lambda *_args, **_kwargs: pytest.fail("no x fetch")
    )
    monkeypatch.setattr(cli, "_run_generate", lambda *_args, **_kwargs: pytest.fail("no generate"))

    result = runner.invoke(app, ["retry-failed"])

    assert result.exit_code == 0, result.output
    assert "No hay bookmarks fallidos que relanzar." in result.output


def _enriched_item(item_id: str = "1"):
    from xbrain.models import Enrichment

    item = _linked_item(item_id)
    item.enriched = Enrichment(
        enriched_at=datetime.now(timezone.utc),
        executor="api",
        summary="resumen",
        primary_topic="misc",
        topics=["misc"],
    )
    return item


def test_topics_claude_code_exports_a_worksheet(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.models import Topic
    from xbrain.rubrics import save_vocab

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["topics", "--executor", "claude-code"])
    assert result.exit_code == 0
    assert (tmp_path / "data" / "topic-worksheet.json").exists()
    assert (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").exists()


def test_topics_apply_writes_the_overview(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    from xbrain.models import Topic
    from xbrain.rubrics import save_vocab

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps(
            {"judgments": [{"slug": "misc", "overview": "Resumen del cajón.", "notes": []}]}
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["topics", "--apply", str(ws)])
    assert result.exit_code == 0
    page = (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").read_text(encoding="utf-8")
    assert "Resumen del cajón." in page


def test_topics_api_executor_synthesizes(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Topic
    from xbrain.rubrics import save_vocab
    from xbrain.topic_synth import OverviewJudgment

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")

    def _fake_synth(inputs, model, output_language="English", **kwargs):
        return [OverviewJudgment(slug="misc", overview="Sintetizado por API.", notes=[])]

    monkeypatch.setattr(cli, "synthesize_overviews_api", _fake_synth)
    result = runner.invoke(app, ["topics", "--executor", "api"])
    assert result.exit_code == 0
    page = (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").read_text(encoding="utf-8")
    assert "Sintetizado por API." in page


def test_topics_without_vocab_fails(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["topics"])
    assert result.exit_code == 1
    assert "vocabulario" in result.output


def test_topics_run_with_no_stale_overviews(tmp_path, monkeypatch):
    # Store enriched + an up-to-date TopicPage (count matches the live posts):
    # nothing is stale, so the run only refreshes the lists.
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.models import Topic, TopicPage
    from xbrain.rubrics import save_vocab
    from xbrain.store import save_topic_pages

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    save_topic_pages(
        {
            "misc": TopicPage(
                slug="misc",
                overview="Overview ya sintetizado.",
                notes=[],
                synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
                post_count_at_synth=1,
            )
        },
        tmp_path / "data" / "topics.json",
    )
    result = runner.invoke(app, ["topics"])
    assert result.exit_code == 0
    assert "sin overviews pendientes" in result.output


def test_topics_resynth_succeeds(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Topic, TopicPage
    from xbrain.rubrics import save_vocab
    from xbrain.store import save_topic_pages
    from xbrain.topic_synth import OverviewJudgment

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    # Page synthesized when the topic had 0 posts — a resynth picks up the change.
    save_topic_pages(
        {
            "misc": TopicPage(
                slug="misc",
                overview="Overview viejo.",
                notes=[],
                synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
                post_count_at_synth=0,
            )
        },
        tmp_path / "data" / "topics.json",
    )

    def _fake_synth(inputs, model, output_language="English", **kwargs):
        return [OverviewJudgment(slug="misc", overview="Re-sintetizado.", notes=[])]

    monkeypatch.setattr(cli, "synthesize_overviews_api", _fake_synth)
    result = runner.invoke(app, ["topics", "--resynth", "--executor", "api"])
    assert result.exit_code == 0
    page = (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").read_text(encoding="utf-8")
    assert "Re-sintetizado." in page


def test_vocab_claude_code_exports_a_worksheet(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["vocab", "--executor", "claude-code"])
    assert result.exit_code == 0
    assert (tmp_path / "data" / "vocab-worksheet.json").exists()


def test_vocab_apply_writes_vocab_yaml(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps({"topics": [{"slug": "misc", "description": "Ruido."}]}), encoding="utf-8"
    )
    result = runner.invoke(app, ["vocab", "--apply", str(ws)])
    assert result.exit_code == 0
    from xbrain.rubrics import load_vocab

    assert [t.slug for t in load_vocab(tmp_path / "data" / "vocab.yaml")] == ["misc"]


def test_vocab_apply_regenerate_marks_items(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone

    from xbrain.models import Enrichment

    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("1")
    item.enriched = Enrichment(
        enriched_at=datetime.now(timezone.utc),
        executor="manual",
        summary="s",
        primary_topic="misc",
        topics=["misc"],
    )
    save_store({"1": item}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps({"topics": [{"slug": "misc", "description": "Ruido."}]}), encoding="utf-8"
    )
    result = runner.invoke(app, ["vocab", "--apply", str(ws), "--regenerate"])
    assert result.exit_code == 0
    from xbrain.store import load_store as _ls

    assert _ls(tmp_path / "data" / "items.json")["1"].enriched is None


def test_vocab_apply_with_no_valid_topics_fails(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"topics": [{"slug": "BAD SLUG"}]}), encoding="utf-8")
    result = runner.invoke(app, ["vocab", "--apply", str(ws)])
    assert result.exit_code == 1
    # `_report_invalid` must run BEFORE the raise — the user needs to see WHY
    # the topic was rejected, so the bad slug shows up in the output.
    assert "BAD SLUG" in result.output


def test_vocab_api_executor_still_works(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Topic

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr(cli, "induce_vocab", lambda *a, **k: [Topic(slug="misc", description="d")])
    result = runner.invoke(app, ["vocab", "--executor", "api"])
    assert result.exit_code == 0
    from xbrain.rubrics import load_vocab

    assert [t.slug for t in load_vocab(tmp_path / "data" / "vocab.yaml")] == ["misc"]


def test_vocab_rejects_unknown_executor(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["vocab", "--executor", "bogus"])
    assert result.exit_code == 1
    assert "bogus" in result.output


# ----------------------------------------------------------------------- describe


def _setup_describe_repo(tmp_path: Path, monkeypatch) -> Path:
    """Like `_setup_repo` but with a pre-populated photo on disk + Downloaded variant."""
    import io as _io
    from datetime import datetime, timezone

    from PIL import Image

    from xbrain.models import Author, Item, MediaPhotoDownloaded

    _setup_repo(tmp_path, monkeypatch)
    media_root = tmp_path / "data" / "media"
    (media_root / "42").mkdir(parents=True, exist_ok=True)
    buffer = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buffer, format="JPEG")
    (media_root / "42" / "0.jpg").write_bytes(buffer.getvalue())

    item = Item(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="a", name="A"),
        text="text",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=[
            MediaPhotoDownloaded(
                url="https://pbs.twimg.com/media/42-0.jpg",
                local_path="42/0.jpg",
                width=8,
                height=6,
                bytes_size=200,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            )
        ],
    )
    save_store({"42": item}, tmp_path / "data" / "items.json")
    return tmp_path


def test_describe_command_transitions_downloaded_to_described(tmp_path: Path, monkeypatch):
    """End-to-end: a Downloaded photo becomes Described via the CLI command."""
    import json as _json

    from xbrain.models import MediaPhotoDescribed
    from xbrain.store import load_store

    _setup_describe_repo(tmp_path, monkeypatch)

    class _FakeBlock:
        type = "text"

        def __init__(self, text: str):
            self.text = text

    class _FakeResp:
        def __init__(self, text: str):
            self.content = [_FakeBlock(text)]

    class _FakeMessages:
        def __init__(self):
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            payload = _json.dumps([{"index": 0, "is_decorative": False, "description": "A chart."}])
            return _FakeResp(payload)

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    import xbrain.cli as cli

    fake_client = _FakeClient()

    def _patched_run(store, media_root, **kwargs):
        kwargs["client"] = fake_client
        return _orig(store, media_root, **kwargs)

    _orig = cli.run_describe_all
    monkeypatch.setattr(cli, "run_describe_all", _patched_run)

    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 0, result.output
    reloaded = load_store(tmp_path / "data" / "items.json")
    entry = reloaded["42"].media[0]
    assert isinstance(entry, MediaPhotoDescribed)
    assert entry.description == "A chart."
    assert entry.description_lang == "English"
    assert entry.description_version == "v1"
    assert fake_client.messages.calls[0]["model"] == "xiaomi/mimo-v2.5"


def test_describe_command_runs_on_empty_store(tmp_path: Path, monkeypatch):
    """No items → describe is a no-op exit-0, with a snapshot still taken."""
    _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 0
    # A snapshot was created — recovery boundary mirrors media.
    snapshots = list((tmp_path / "data" / "snapshots").glob("*-describe"))
    assert snapshots, "describe must auto-snapshot data/ before running"


def test_describe_command_warns_when_items_filter_matches_nothing(tmp_path, monkeypatch):
    """`--items` with no matches surfaces an AVISO line — same pattern as media."""
    _setup_describe_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["describe", "--items", "no-such-id"])
    assert result.exit_code == 0
    assert "AVISO" in result.output or "AVISO" in (result.stderr or "")


def test_describe_command_rejects_provider_mismatched_model_override(tmp_path, monkeypatch):
    """`--model` is still an API model, so it must match `[llm].provider`."""
    _setup_describe_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["describe", "--model", "claude-sonnet-4-6"])
    assert result.exit_code == 1
    assert "Anthropic-native" in result.output


def test_describe_command_propagates_total_failure_as_exit_1(tmp_path, monkeypatch):
    """A total-failure RuntimeError surfaces as exit code 1 via `_handle_cli_errors`."""
    from anthropic import APIError

    _setup_describe_repo(tmp_path, monkeypatch)

    class _AlwaysFailingMessages:
        def __init__(self):
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise APIError("401 unauthorized", request=None, body=None)

    class _FakeClient:
        def __init__(self):
            self.messages = _AlwaysFailingMessages()

    import xbrain.cli as cli

    def _patched(store, media_root, **kwargs):
        kwargs["client"] = _FakeClient()
        return _orig(store, media_root, **kwargs)

    _orig = cli.run_describe_all
    monkeypatch.setattr(cli, "run_describe_all", _patched)

    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_describe_command_emits_exactly_one_summary_on_partial_failure(tmp_path, monkeypatch):
    """The CLI is the single source of truth for the SUMMARY line.

    A partial-failure run must emit EXACTLY one `SUMMARY:` on stderr —
    if the orchestrator regrows a second emitter the count goes to 2
    and this pins it. Mirrors the dedup test on the orchestrator
    side (`test_describe_all_does_not_emit_summary_on_partial_failure`).
    """
    import io as _io
    import json as _json
    from datetime import datetime, timezone

    from anthropic import APIError
    from PIL import Image

    from xbrain.models import Author, Item, MediaPhotoDownloaded
    from xbrain.store import save_store as _save_store

    _setup_repo(tmp_path, monkeypatch)
    media_root = tmp_path / "data" / "media"
    (media_root / "1").mkdir(parents=True, exist_ok=True)
    (media_root / "2").mkdir(parents=True, exist_ok=True)
    buf = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buf, format="JPEG")
    (media_root / "1" / "0.jpg").write_bytes(buf.getvalue())
    (media_root / "2" / "0.jpg").write_bytes(buf.getvalue())

    def _build_item(item_id: str) -> Item:
        return Item(
            id=item_id,
            source="bookmark",
            url=f"https://x.com/a/status/{item_id}",
            author=Author(handle="a", name="A"),
            text="text",
            created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
            captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            media=[
                MediaPhotoDownloaded(
                    url=f"https://pbs.twimg.com/media/{item_id}-0.jpg",
                    local_path=f"{item_id}/0.jpg",
                    width=8,
                    height=6,
                    bytes_size=200,
                    downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
                )
            ],
        )

    _save_store(
        {"1": _build_item("1"), "2": _build_item("2")},
        tmp_path / "data" / "items.json",
    )

    class _PartialFailMessages:
        def __init__(self):
            self.calls: list[dict] = []
            self._counter = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            self._counter += 1
            if self._counter == 1:
                # First batch fails.
                raise APIError("503", request=None, body=None)

            # Second batch returns one judgment.
            class _Block:
                type = "text"

                def __init__(self, text: str):
                    self.text = text

            class _Resp:
                def __init__(self, text: str):
                    self.content = [_Block(text)]
                    self.stop_reason = "end_turn"

            return _Resp(_json.dumps([{"index": 0, "is_decorative": False, "description": "ok"}]))

    class _FakeClient:
        def __init__(self):
            self.messages = _PartialFailMessages()

    import xbrain.cli as cli

    fake = _FakeClient()

    def _patched(store, media_root, **kwargs):
        kwargs["client"] = fake
        kwargs["batch_size"] = 1
        return _orig(store, media_root, **kwargs)

    _orig = cli.run_describe_all
    monkeypatch.setattr(cli, "run_describe_all", _patched)

    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 0
    # Exactly one SUMMARY emission — the CLI's `emit_summary_line` is
    # the single source of truth; the orchestrator stays silent.
    assert result.output.count("SUMMARY:") == 1


def test_describe_command_verbose_lists_failed_photos(tmp_path, monkeypatch):
    """`--verbose` prints `Failed photos:` plus per-failure rows on partial failure.

    Pins the diagnostic branch in `_run_describe` so a future refactor
    that drops the verbose output is caught.
    """
    import io as _io
    import json as _json
    from datetime import datetime, timezone

    from anthropic import APIError
    from PIL import Image

    from xbrain.models import Author, Item, MediaPhotoDownloaded
    from xbrain.store import save_store as _save_store

    _setup_repo(tmp_path, monkeypatch)
    media_root = tmp_path / "data" / "media"
    (media_root / "1").mkdir(parents=True, exist_ok=True)
    (media_root / "2").mkdir(parents=True, exist_ok=True)
    buf = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buf, format="JPEG")
    (media_root / "1" / "0.jpg").write_bytes(buf.getvalue())
    (media_root / "2" / "0.jpg").write_bytes(buf.getvalue())

    def _build_item(item_id: str) -> Item:
        return Item(
            id=item_id,
            source="bookmark",
            url=f"https://x.com/a/status/{item_id}",
            author=Author(handle="a", name="A"),
            text="text",
            created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
            captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            media=[
                MediaPhotoDownloaded(
                    url=f"https://pbs.twimg.com/media/{item_id}-0.jpg",
                    local_path=f"{item_id}/0.jpg",
                    width=8,
                    height=6,
                    bytes_size=200,
                    downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
                )
            ],
        )

    _save_store(
        {"1": _build_item("1"), "2": _build_item("2")},
        tmp_path / "data" / "items.json",
    )

    class _PartialFailMessages:
        def __init__(self):
            self.calls: list[dict] = []
            self._counter = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            self._counter += 1
            if self._counter == 1:
                raise APIError("503", request=None, body=None)

            class _Block:
                type = "text"

                def __init__(self, text: str):
                    self.text = text

            class _Resp:
                def __init__(self, text: str):
                    self.content = [_Block(text)]
                    self.stop_reason = "end_turn"

            return _Resp(_json.dumps([{"index": 0, "is_decorative": False, "description": "ok"}]))

    class _FakeClient:
        def __init__(self):
            self.messages = _PartialFailMessages()

    import xbrain.cli as cli

    fake = _FakeClient()

    def _patched(store, media_root, **kwargs):
        kwargs["client"] = fake
        kwargs["batch_size"] = 1
        return _orig(store, media_root, **kwargs)

    _orig = cli.run_describe_all
    monkeypatch.setattr(cli, "run_describe_all", _patched)

    result = runner.invoke(app, ["describe", "--verbose"])
    assert result.exit_code == 0
    assert "Failed photos:" in result.output
    # The failed item id (1 or 2 — order is filesystem-dependent) and the URL
    # both appear on the verbose row.
    assert "pbs.twimg.com" in result.output


# ------------------------------------------------------------------ refresh-media


def _poster_era_item(item_id: str = "42"):
    """A store item with a downloaded photo + a poster-era video (url = poster)."""
    from xbrain.models import Author, Item, MediaPhotoDownloaded, MediaVideoPending

    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=[
            MediaPhotoDownloaded(
                url="https://pbs.twimg.com/media/42-0.jpg",
                local_path=f"{item_id}/0.jpg",
                width=8,
                height=6,
                bytes_size=200,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            ),
            MediaVideoPending(url="https://pbs.twimg.com/poster.jpg"),
        ],
    )


def _playable_fresh_item(item_id: str = "42"):
    """A fresh capture of the same id whose video carries the playable stream."""
    from xbrain.models import Author, Item, MediaVideoPending

    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=[
            MediaVideoPending(
                url="https://v/high.mp4",
                thumbnail_url="https://pbs.twimg.com/poster.jpg",
                bitrate=2_176_000,
                duration_millis=30_000,
            )
        ],
    )


def _mock_browser(monkeypatch, extract_impl):
    """Patch `cli.x_context` (a no-op CM) + `cli.extract_source` (the impl)."""
    import contextlib

    import xbrain.cli as cli

    @contextlib.contextmanager
    def _ctx(_path, *, headless=False):
        yield object()

    monkeypatch.setattr(cli, "x_context", _ctx)
    monkeypatch.setattr(cli, "extract_source", extract_impl)


def test_refresh_media_backfills_video_and_preserves_photo(tmp_path: Path, monkeypatch):
    """End-to-end: the poster-era video gains the playable URL; the photo survives."""
    from xbrain.models import MediaPhotoDownloaded, MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    _mock_browser(monkeypatch, lambda *a, **k: [_playable_fresh_item("42")])

    result = runner.invoke(app, ["refresh-media"])
    assert result.exit_code == 0, result.output

    reloaded = load_store(tmp_path / "data" / "items.json")
    media = reloaded["42"].media
    # The downloaded photo is untouched (still Downloaded, same local_path).
    assert isinstance(media[0], MediaPhotoDownloaded)
    assert media[0].local_path == "42/0.jpg"
    # The video now carries the playable stream + bitrate + duration.
    video = media[1]
    assert isinstance(video, MediaVideoPending)
    assert video.url == "https://v/high.mp4"
    assert video.bitrate == 2_176_000
    assert video.duration_millis == 30_000
    # The END summary prints the report counts + a size estimate (with its
    # full "...; N with unknown size." tail — the refreshed mp4 is estimable).
    assert "1 refreshed" in result.output
    assert "Estimated video download" in result.output
    assert "0 with unknown size." in result.output


def test_refresh_media_creates_pre_snapshot(tmp_path: Path, monkeypatch):
    """The destructive op snapshots `data/` before writing (label pre-refresh-media)."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    _mock_browser(monkeypatch, lambda *a, **k: [_playable_fresh_item("42")])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])
    assert result.exit_code == 0, result.output
    snapshots = list((tmp_path / "data" / "snapshots").glob("*-pre-refresh-media"))
    assert snapshots, "refresh-media must auto-snapshot data/ before writing"


def test_refresh_media_disables_skip_known_with_empty_known_ids(tmp_path: Path, monkeypatch):
    """Backfill must pass an EMPTY known_ids set so the FULL history is scrolled."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    captured: dict[str, object] = {}

    def _spy(context, src, url, known_ids, *a, **k):
        captured["known_ids"] = known_ids
        # Return a re-seen item so the run takes the non-failure path (≥1 known
        # item seen) — the empty-capture guard is exercised separately below.
        return [_playable_fresh_item("42")]

    _mock_browser(monkeypatch, _spy)

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])
    assert result.exit_code == 0, result.output
    assert captured["known_ids"] == set()


def test_refresh_media_blocks_on_empty_capture_against_nonempty_store(tmp_path: Path, monkeypatch):
    """A logged-in-but-empty capture (GraphQL drift / interrupted scroll) must NOT
    report success: 0 known items re-seen on a non-empty store exits non-zero,
    warns loudly, and saves nothing."""
    from xbrain.models import MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    _mock_browser(monkeypatch, lambda *a, **k: [])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])

    assert result.exit_code != 0
    assert "no se actualizó nada" in result.output
    # No success summary was printed.
    assert "videos updated" not in result.output
    # The store is intact and still poster-era (nothing was overwritten).
    reloaded = load_store(tmp_path / "data" / "items.json")
    video = reloaded["42"].media[1]
    assert isinstance(video, MediaVideoPending)
    assert video.url == "https://pbs.twimg.com/poster.jpg"


def test_refresh_media_force_proceeds_on_empty_capture(tmp_path: Path, monkeypatch):
    """`--force` downgrades the empty-capture guard to a warning and proceeds."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    _mock_browser(monkeypatch, lambda *a, **k: [])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks", "--force"])

    assert result.exit_code == 0, result.output
    assert "AVISO" in result.output
    # It proceeds: the END summary is printed (0 re-seen).
    assert "0 known items re-seen" in result.output


def test_refresh_media_empty_store_does_not_warn(tmp_path: Path, monkeypatch):
    """A fresh project (empty store) with an empty capture is a clean no-op."""
    _setup_repo(tmp_path, monkeypatch)  # no items.json saved → empty store
    _mock_browser(monkeypatch, lambda *a, **k: [])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])

    assert result.exit_code == 0, result.output
    assert "AVISO" not in result.output
    assert "no se actualizó nada" not in result.output


def test_refresh_media_reports_unknown_size_without_zero_gb(tmp_path: Path, monkeypatch):
    """When no video is estimable, the summary must NOT misread as '~0.0 GB'."""
    from xbrain.models import Author, Item, MediaVideoPending

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")

    # A fresh HLS stream is a real upgrade (url != thumbnail) but carries no
    # bitrate/duration, so the size estimate has 0 estimable, 1 unknown.
    hls = Item(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=[
            MediaVideoPending(
                url="https://v/play.m3u8",
                thumbnail_url="https://pbs.twimg.com/poster.jpg",
                bitrate=None,
                duration_millis=None,
            )
        ],
    )
    _mock_browser(monkeypatch, lambda *a, **k: [hls])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])

    assert result.exit_code == 0, result.output
    assert "size unknown for 1 videos" in result.output
    assert "GB" not in result.output


def test_refresh_media_surfaces_logged_out_runtimeerror(tmp_path: Path, monkeypatch):
    """A logged-out session (extract_source raises RuntimeError) exits 1 cleanly."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")

    def _raise(*a, **k):
        raise RuntimeError("Sesión de X caducada. Ejecuta `xbrain login`.")

    _mock_browser(monkeypatch, _raise)

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])
    assert result.exit_code == 1
    assert "Sesión de X caducada" in result.output


def test_extract_advances_cursor_to_integer_max_id(tmp_path: Path, monkeypatch):
    """The cursor must track the integer-max id, not list order or string compare."""
    from xbrain.store import load_state

    _setup_repo(tmp_path, monkeypatch)
    # Out-of-order, and "98" > "100" lexicographically — only an int max gives 100.
    _mock_browser(monkeypatch, lambda *a, **k: [_linked_item("98"), _linked_item("100")])

    result = runner.invoke(app, ["extract", "--source", "bookmarks"])

    assert result.exit_code == 0
    assert load_state(tmp_path / "data" / "state.json").bookmarks.last_seen_id == "100"


def test_extract_reports_seen_existing_and_batch_duplicates(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    _mock_browser(
        monkeypatch,
        lambda *a, **k: [_linked_item("1"), _linked_item("2"), _linked_item("2")],
    )

    result = runner.invoke(app, ["extract", "--source", "bookmarks"])

    assert result.exit_code == 0, result.output
    assert "bookmark: 1 nuevos, 1 ya existentes, 1 duplicados en lote (3 vistos)" in result.output


def test_extract_discards_own_tweets_and_plain_bookmarks(tmp_path: Path, monkeypatch):
    from xbrain.models import Media
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    article = _linked_item("1")
    video = _linked_item("2")
    video.links = []
    video.media = [Media(type="video", url="https://video.twimg.com/v/2.mp4")]
    plain = _linked_item("3")
    plain.links = []

    def _extract(_context, source, *_args, **_kwargs):
        return [article, video, plain] if source == "bookmark" else []

    _mock_browser(monkeypatch, _extract)

    result = runner.invoke(app, ["extract", "--source", "all"])

    assert result.exit_code == 0, result.output
    assert "bookmark: 2 nuevos" in result.output
    assert "1 descartados por política" in result.output
    store = load_store(tmp_path / "data" / "items.json")
    assert set(store) == {"1", "2"}


def test_extract_tweets_source_is_disabled_by_content_policy(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)

    def _extract(*_args, **_kwargs):
        raise AssertionError("own tweet extraction should not run")

    _mock_browser(monkeypatch, _extract)

    result = runner.invoke(app, ["extract", "--source", "tweets"])

    assert result.exit_code == 0, result.output
    assert "tweets propios desactivada" in result.output


def test_extract_truncation_persists_nothing_and_exits_nonzero(tmp_path: Path, monkeypatch):
    """A RateLimitTruncated source must not merge items nor advance the cursor."""
    from xbrain.extract.extractor import RateLimitTruncated
    from xbrain.store import load_state, load_store

    _setup_repo(tmp_path, monkeypatch)

    def _truncate(*_a, **_k):
        raise RateLimitTruncated("bookmark: truncado a media timeline")

    _mock_browser(monkeypatch, _truncate)

    result = runner.invoke(app, ["extract", "--source", "bookmarks"])

    assert result.exit_code == 1
    assert load_store(tmp_path / "data" / "items.json") == {}
    assert load_state(tmp_path / "data" / "state.json").bookmarks.last_seen_id is None


def test_refresh_all_runs_daily_pipeline_in_order(tmp_path: Path, monkeypatch):
    import xbrain.cli as cli

    _setup_repo(tmp_path, monkeypatch)
    calls: list[tuple] = []

    monkeypatch.setattr(
        cli,
        "_run_extract",
        lambda cfg, source, since, until, *, headless: calls.append(("extract", source, headless)),
    )
    monkeypatch.setattr(
        cli,
        "_run_fetch",
        lambda cfg, since, until, force, *, headless: calls.append(("fetch", force, headless)),
    )
    monkeypatch.setattr(
        cli,
        "_run_media",
        lambda cfg, *, force, limit, items_filter, verbose=False: calls.append(
            ("media", force, limit, items_filter)
        ),
    )
    monkeypatch.setattr(
        cli,
        "_run_describe",
        lambda cfg, **kwargs: calls.append(
            ("describe", kwargs["model"], kwargs["batch_size"], kwargs["limit"])
        ),
    )
    monkeypatch.setattr(
        cli,
        "_run_enrich_api",
        lambda cfg, since, until: calls.append(("enrich", since, until)),
    )
    monkeypatch.setattr(
        cli,
        "_run_topics_executor",
        lambda cfg, executor, *, resynth=False: calls.append(("topics", executor, resynth)),
    )
    monkeypatch.setattr(
        cli,
        "_run_generate",
        lambda cfg, since, until: calls.append(("generate", since, until)),
    )

    result = runner.invoke(app, ["refresh-all", "--describe-limit", "3"])

    assert result.exit_code == 0, result.output
    assert calls == [
        ("extract", "bookmarks", True),
        ("fetch", False, True),
        ("media", False, None, None),
        ("describe", "xiaomi/mimo-v2.5", 5, 3),
        ("enrich", None, None),
        ("topics", "api", False),
        ("generate", None, None),
    ]
    assert "refresh-all: completado" in result.output


# ------------------------------------------------------ list-videos / fetch-video


def _enriched_video_item(
    item_id: str,
    topic: str,
    url: str = _MP4_URL,
    source: str = "bookmark",
    *,
    transcript_url: str | None = None,
):
    from xbrain.models import Enrichment

    item = _video_item(item_id, url=url, source=source, transcript_url=transcript_url)
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="manual",
        primary_topic=topic,
    )
    return item


def test_list_videos_json_is_parseable_array(tmp_path: Path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _enriched_video_item("42", "ai")}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["list-videos", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, list) and len(payload) == 1
    row = payload[0]
    assert set(row) == {"id", "url", "state", "digest", "topic", "size_bytes", "mp4_url", "text"}
    assert row["id"] == "42"
    assert row["state"] == "pending"
    assert row["digest"] == "pending"
    assert row["topic"] == "ai"
    assert row["mp4_url"] == _MP4_URL


def test_list_videos_human_table(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _enriched_video_item("42", "ai")}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["list-videos"])
    assert result.exit_code == 0, result.output
    assert "STATE" in result.stdout
    assert "DIGEST" in result.stdout
    assert "pending" in result.stdout
    assert "ai" in result.stdout


def test_list_videos_is_read_only(tmp_path: Path, monkeypatch):
    """list-videos performs zero writes and takes no snapshot."""
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _enriched_video_item("42", "ai")}, items_path)
    before = items_path.read_bytes()

    result = runner.invoke(app, ["list-videos", "--json"])
    assert result.exit_code == 0
    assert items_path.read_bytes() == before
    snapshots = tmp_path / "data" / "snapshots"
    assert not snapshots.exists()


def test_list_videos_topic_filter(tmp_path: Path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store(
        {
            "1": _enriched_video_item("1", "ai"),
            "2": _enriched_video_item("2", "climate"),
        },
        tmp_path / "data" / "items.json",
    )

    result = runner.invoke(app, ["list-videos", "--topic", "climate", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [row["id"] for row in payload] == ["2"]


def test_list_videos_empty_store(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["list-videos"])
    assert result.exit_code == 0
    assert "No hay" in result.stdout


def test_list_videos_rejects_bad_status(tmp_path: Path, monkeypatch):
    """`--status` is a typer Enum: a bad value is a usage error (exit 2), like
    `--source`."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["list-videos", "--status", "bogus"])
    assert result.exit_code == 2


def test_list_videos_json_null_topic(tmp_path: Path, monkeypatch):
    """An un-enriched video's topic is JSON null in --json (human table shows —)."""
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["list-videos", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload[0]["topic"] is None


def test_list_videos_status_poster_era_enum(tmp_path: Path, monkeypatch):
    """The `poster-era` status value (hyphenated enum) is accepted and filters."""
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"9": _video_item("9", url=_POSTER)}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["list-videos", "--status", "poster-era", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [row["id"] for row in payload] == ["9"]


def test_fetch_video_command_is_disabled_even_without_to(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["fetch-video", "--ids", "42"])

    assert result.exit_code == 1
    assert "deshabilitado" in result.output
    assert "no deja MP4 en disco" in result.output
    assert not (tmp_path / "data" / "media").exists()


def test_fetch_video_command_is_disabled_and_does_not_write_to_dir(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    dest = tmp_path / "out"
    save_store({"42": _video_item("42")}, items_path)
    before = items_path.read_bytes()

    result = runner.invoke(app, ["fetch-video", "--ids", "42", "--to", str(dest), "--json"])

    assert result.exit_code == 1
    assert "deshabilitado" in result.output
    assert items_path.read_bytes() == before
    assert not dest.exists()


# ------------------------------------------------------ digest-video

# A distinct signed URL for the SAME amplify_video id as _AMPLIFY_URL_1 — proves
# dedup keys on the stable path id, not the rotating URL.
_AMPLIFY_URL_1 = "https://video.twimg.com/amplify_video/900/vid/720/a.mp4?tag=16"
_AMPLIFY_URL_2 = "https://video.twimg.com/amplify_video/900/vid/1080/b.mp4?tag=21"
_TRANSCRIPT_URL = "https://video.twimg.com/amplify_video/900/captions/en.vtt"


def _wire_caption_digest(
    monkeypatch,
    *,
    transcript_text: str = "original transcript line",
    summary_markdown: str = "### Executive Summary\n\nExecutive summary from captions.",
    calls: list[str] | None = None,
):
    from xbrain.video_transcript import VideoExecutiveSummary, VideoTranscript

    def _fake_fetch(_item, entry, **_kwargs):
        if calls is not None:
            calls.append(entry.url)
        return VideoTranscript(
            text=transcript_text,
            language=entry.transcript_language,
            source_url=entry.transcript_url,
            format=entry.transcript_format or "vtt",
        )

    def _fake_summary_fn(**_kwargs):
        def _summarize(_item, _transcript):
            return VideoExecutiveSummary(title="Executive video", markdown=summary_markdown)

        return _summarize

    monkeypatch.setattr("xbrain.cli.fetch_or_transcribe_video_transcript", _fake_fetch)
    monkeypatch.setattr("xbrain.cli.configured_summary_fn", _fake_summary_fn)


def test_digest_video_captions_summarizes_and_attaches(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {"42": _video_item("42", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL)},
        items_path,
    )
    _wire_caption_digest(
        monkeypatch,
        transcript_text="hello from the original captions",
        summary_markdown="### Executive Summary\n\nUseful executive summary.",
    )

    result = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    sources = store["42"].content.sources
    assert sources[0].kind == "x_video"
    assert sources[0].text == "### Executive Summary\n\nUseful executive summary."
    assert sources[0].raw_transcript == "hello from the original captions"
    assert sources[0].raw_transcript_url == _TRANSCRIPT_URL
    assert sources[0].raw_transcript_format == "vtt"
    assert sources[0].executive_summary_version == "v1"
    assert sources[0].has_speech is True
    assert sources[0].frames == []
    assert "resumidos 1" in result.stdout


def test_digest_video_dedups_same_video(tmp_path: Path, monkeypatch):
    """Two items bookmarking the same video → transcript fetched once, both carry summary."""
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {
            "a": _video_item("a", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL),
            "b": _video_item("b", url=_AMPLIFY_URL_2, transcript_url=_TRANSCRIPT_URL),
        },
        items_path,
    )
    calls: list[str] = []
    _wire_caption_digest(
        monkeypatch, summary_markdown="### Executive Summary\n\nShared.", calls=calls
    )

    result = runner.invoke(app, ["digest-video", "--ids", "a,b"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    store = load_store(items_path)
    assert store["a"].content.sources[0].text == "### Executive Summary\n\nShared."
    assert store["b"].content.sources[0].text == "### Executive Summary\n\nShared."
    assert "1 items ← 1 vídeos" not in result.stdout  # 2 items, 1 video
    assert "2 items ← 1 vídeos" in result.stdout


def test_digest_video_without_captions_uses_temporary_transcript_fallback(
    tmp_path: Path, monkeypatch
):
    from xbrain.store import load_store
    from xbrain.video_transcript import VideoExecutiveSummary, VideoTranscript

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)

    def _fake_transcript(item, entry, **kwargs):
        assert item.id == "42"
        assert entry.transcript_url is None
        assert kwargs["transcribe_command"] == "parakeet-mlx"
        return VideoTranscript(
            text="transcript generated from temporary ASR",
            language="en",
            source_url=item.url,
            format="text",
        )

    def _factory(**_kwargs):
        def _summarize(_item, transcript):
            assert transcript.text == "transcript generated from temporary ASR"
            return VideoExecutiveSummary(
                title="Temporary ASR",
                markdown="### Executive Summary\n\nSummary from temporary transcript.",
            )

        return _summarize

    monkeypatch.setattr("xbrain.cli.fetch_or_transcribe_video_transcript", _fake_transcript)
    monkeypatch.setattr("xbrain.cli.configured_summary_fn", _factory)

    result = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    source = store["42"].content.sources[0]
    assert source.kind == "x_video"
    assert source.raw_transcript == "transcript generated from temporary ASR"
    assert source.raw_transcript_url == "https://x.com/a/status/42"
    assert source.raw_transcript_format == "text"
    assert not (tmp_path / "data" / "media").exists()


def test_digest_video_idempotent(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {"42": _video_item("42", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL)},
        items_path,
    )
    calls: list[str] = []
    _wire_caption_digest(monkeypatch, calls=calls)

    first = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert second.exit_code == 0, second.output
    assert "ya digeridos 1" in second.stdout
    assert len(calls) == 1


def test_digest_video_force_redigests(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {"42": _video_item("42", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL)},
        items_path,
    )

    _wire_caption_digest(monkeypatch, summary_markdown="### Executive Summary\n\nv1")
    runner.invoke(app, ["digest-video", "--ids", "42"])
    _wire_caption_digest(monkeypatch, summary_markdown="### Executive Summary\n\nv2")
    result = runner.invoke(app, ["digest-video", "--ids", "42", "--force"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    assert store["42"].content.sources[0].text == "### Executive Summary\n\nv2"


def test_digest_video_all_pending_selects_pending(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {"42": _video_item("42", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL)},
        items_path,
    )
    _wire_caption_digest(monkeypatch)

    result = runner.invoke(app, ["digest-video", "--all-pending"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    assert store["42"].content.sources[0].kind == "x_video"


def test_digest_video_requires_selection(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["digest-video"])
    assert result.exit_code == 1
    assert "ids" in result.output or "all-pending" in result.output


def test_digest_video_transcript_fetch_failure_exits_zero_and_does_not_persist(
    tmp_path: Path, monkeypatch
):
    from xbrain.store import load_store
    from xbrain.video_transcript import VideoTranscriptFetchFailed

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {"42": _video_item("42", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL)},
        items_path,
    )

    def _boom(_item, _entry, **_kwargs):
        raise VideoTranscriptFetchFailed("caption HTTP 403")

    _wire_caption_digest(monkeypatch)
    monkeypatch.setattr("xbrain.cli.fetch_or_transcribe_video_transcript", _boom)
    result = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert result.exit_code == 0, result.output
    assert "fallidos 1" in result.output
    assert load_store(items_path)["42"].content is None


# A SECOND, DISTINCT video (different ext_tw_video id) for limit/selection tests.
_DISTINCT_VIDEO_URL = "https://video.twimg.com/ext_tw_video/7700/vid/720/z.mp4?tag=3"
_DISTINCT_TRANSCRIPT_URL = "https://video.twimg.com/ext_tw_video/7700/captions/en.vtt"


def test_digest_video_limit_caps_items(tmp_path: Path, monkeypatch):
    """`--limit 1` over a 2-distinct-video store summarizes exactly one item."""
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {
            "a": _video_item("a", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL),
            "b": _video_item("b", url=_DISTINCT_VIDEO_URL, transcript_url=_DISTINCT_TRANSCRIPT_URL),
        },
        items_path,
    )
    _wire_caption_digest(monkeypatch)

    result = runner.invoke(app, ["digest-video", "--ids", "a,b", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "resumidos 1" in result.stdout
    store = load_store(items_path)
    digested = [i for i in ("a", "b") if store[i].content is not None]
    assert len(digested) == 1  # exactly one, not both


def test_digest_video_max_size_caps_temporary_video_fallback(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    seen: list[int | None] = []

    def _fake_transcript(_item, _entry, **kwargs):
        from xbrain.video_transcript import VideoTranscriptUnavailable

        seen.append(kwargs["max_size_bytes"])
        raise VideoTranscriptUnavailable("no transcript in this test")

    monkeypatch.setattr("xbrain.cli.fetch_or_transcribe_video_transcript", _fake_transcript)

    result = runner.invoke(app, ["digest-video", "--ids", "42", "--max-size", "500MB"])
    assert result.exit_code == 0, result.output
    assert seen == [500_000_000]
    assert load_store(items_path)["42"].content is None


def test_digest_video_invalid_max_size_is_rejected(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["digest-video", "--ids", "42", "--max-size", "banana"])

    assert result.exit_code == 1
    assert "invalid --max-size" in result.output


def test_digest_video_reports_temporary_video_too_large(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store
    from xbrain.video_transcript import VideoTranscriptFetchSkipped

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)

    def _fake_transcript(_item, _entry, **_kwargs):
        raise VideoTranscriptFetchSkipped("too_large", "temporary video exceeds cap")

    monkeypatch.setattr("xbrain.cli.fetch_or_transcribe_video_transcript", _fake_transcript)

    result = runner.invoke(app, ["digest-video", "--ids", "42", "--max-size", "10MB"])

    assert result.exit_code == 0, result.output
    assert "> --max-size 1" in result.stdout
    assert load_store(items_path)["42"].content is None


def test_digest_video_topic_selects_only_matching(tmp_path: Path, monkeypatch):
    """`--topic ai` digests only the ai-topic video, honoring the catalog filter."""
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {
            "1": _enriched_video_item(
                "1", "ai", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL
            ),
            "2": _enriched_video_item(
                "2",
                "climate",
                url=_DISTINCT_VIDEO_URL,
                transcript_url=_DISTINCT_TRANSCRIPT_URL,
            ),
        },
        items_path,
    )
    _wire_caption_digest(monkeypatch)

    result = runner.invoke(app, ["digest-video", "--topic", "ai"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    assert store["1"].content is not None  # ai video digested
    assert store["2"].content is None  # climate video untouched


def test_digest_video_frames_are_disabled(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {"42": _video_item("42", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL)},
        items_path,
    )
    _wire_caption_digest(monkeypatch)

    result = runner.invoke(app, ["digest-video", "--ids", "42", "--frames"])

    assert result.exit_code == 1
    assert "--frames está deshabilitado" in result.output
    assert load_store(items_path)["42"].content is None
    assert not (tmp_path / "data" / "media").exists()


def test_digest_video_vision_model_is_disabled(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store(
        {"42": _video_item("42", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL)},
        tmp_path / "data" / "items.json",
    )

    result = runner.invoke(app, ["digest-video", "--ids", "42", "--vision-model", "opus"])

    assert result.exit_code == 1
    assert "--vision-model" in result.output
    assert "deshabilitado" in result.output


def test_digest_video_then_generate_writes_summary_and_transcript_artifacts(
    tmp_path: Path, monkeypatch
):
    vault = _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {
            "42": _enriched_video_item(
                "42", "ai", url=_AMPLIFY_URL_1, transcript_url=_TRANSCRIPT_URL
            )
        },
        items_path,
    )
    _wire_caption_digest(
        monkeypatch,
        transcript_text="raw caption transcript kept for audit only",
        summary_markdown="### Executive Summary\n\nDashboard-ready video summary.",
    )

    assert runner.invoke(app, ["digest-video", "--ids", "42"]).exit_code == 0
    assert runner.invoke(app, ["generate"]).exit_code == 0

    output = vault / "x-knowledge"
    note = next((output / "items").glob("*42*.md")).read_text(encoding="utf-8")
    assert "Dashboard-ready video summary" in note
    assert "raw caption transcript kept for audit only" not in note
    assert "../videos/" in note
    assert "![[_media/" not in note
    assert "_media/42/frames" not in note

    summary = next((output / "videos").glob("*/summary.md")).read_text(encoding="utf-8")
    transcript = next((output / "videos").glob("*/transcript.md")).read_text(encoding="utf-8")
    assert "Dashboard-ready video summary" in summary
    assert "raw caption transcript kept for audit only" in transcript
    assert "xbrain_exclude: true" in transcript

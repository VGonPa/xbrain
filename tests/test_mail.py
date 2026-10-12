from datetime import datetime, timezone
from pathlib import Path

from xbrain.config import load_config
from xbrain.mail import send_bookmark_update_email
from xbrain.models import Author, Enrichment, Item
from xbrain.notes_io import note_filename


class FakeSMTP:
    sent_messages = []
    logins = []
    starttls_calls = 0

    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def starttls(self, *, context):
        assert context is not None
        FakeSMTP.starttls_calls += 1

    def login(self, username: str, password: str):
        FakeSMTP.logins.append((username, password))

    def send_message(self, message):
        FakeSMTP.sent_messages.append(message)


def test_send_bookmark_update_email_uses_configured_smtp(tmp_path: Path, monkeypatch):
    vault = tmp_path / "vault"
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        f'vault = "{vault}"\n'
        'output_subdir = "x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[drive]\n"
        "enabled = true\n"
        'root_folder_id = "folder-123"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XBRAIN_EMAIL_ENABLED", "true")
    monkeypatch.setenv("XBRAIN_EMAIL_FROM", "sender@example.com")
    monkeypatch.setenv("XBRAIN_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("XBRAIN_SMTP_USERNAME", "smtp-user")
    monkeypatch.setenv("XBRAIN_SMTP_PASSWORD", "smtp-pass")
    monkeypatch.setenv("XBRAIN_EMAIL_RECIPIENT", "recipient@example.com")
    monkeypatch.setattr("xbrain.mail.smtplib.SMTP", FakeSMTP)
    FakeSMTP.sent_messages = []
    FakeSMTP.logins = []
    FakeSMTP.starttls_calls = 0

    cfg = load_config(tmp_path)
    item = Item(
        id="57",
        source="bookmark",
        url="https://example.com/articles/deep-dive?ref=x",
        author=Author(handle="alice", name="Alice Example"),
        text="Fallback text",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        enriched=Enrichment(
            enriched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            executor="api",
            summary="Resumen sintetico del bookmark.",
        ),
    )
    note_path = cfg.output_dir / "items" / note_filename(item)
    note_path.parent.mkdir(parents=True)
    note_path.write_text("generated note body", encoding="utf-8")
    (cfg.data_dir / "items.json").parent.mkdir(parents=True)
    (cfg.data_dir / "items.json").write_text("{}", encoding="utf-8")

    send_bookmark_update_email(
        cfg,
        command="refresh-all",
        store={"57": item},
        updated_item_ids=["57"],
    )

    assert FakeSMTP.logins == [("smtp-user", "smtp-pass")]
    assert FakeSMTP.starttls_calls == 1
    message = FakeSMTP.sent_messages[0]
    assert message["To"] == "recipient@example.com"
    assert message["From"] == "sender@example.com"
    text_body = message.get_body(preferencelist=("plain",)).get_content()
    html_body = message.get_body(preferencelist=("html",)).get_content()
    assert "refresh-all" in text_body
    assert "Items totales en biblioteca: 1" in text_body
    assert "Alice Example (@alice)" in text_body
    assert "Resumen sintetico del bookmark." in text_body
    assert "Peso:" in text_body
    assert "KB" in text_body
    assert "MB" in text_body
    assert '<a href="https://example.com/articles/deep-dive?ref=x">' in html_body

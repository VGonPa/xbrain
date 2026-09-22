# tests/test_jev_cli.py
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.jev_fakes import FakeJevClient
from xbrain import cli
from xbrain.cli import app
from xbrain.config import Config
from xbrain.jev.client import JevClient, JevResult, Question
from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.store import load_assessments, save_assessments
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.rubrics import save_vocab
from xbrain.store import save_store

runner = CliRunner()
DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _refusing_client(reason: str) -> Callable[[Config], JevClient]:
    """A `_jev_client` stand-in that fails the test if the CLI builds a client at all.

    The factory is the seam where money starts being spent, so "did not call Jev" is
    asserted by making the CALL impossible, not by counting calls afterwards.
    """

    def _build(cfg: Config) -> JevClient:
        raise AssertionError(reason)

    return _build


class _PerItemProviderClient(FakeJevClient):
    """A `FakeJevClient` that attributes each answer to a different provider AND model.

    `INPUT_USD_PER_MTOK` prices `typesafe` and nothing else. A run priced at ONE flat rate
    instead of per record would bill the unpriced half too, and no single-provider test can
    see that: with one provider the per-record and flat formulas agree. The second model is
    the same argument for the summary's `modelo(s)` — `[jev].model` defaults to a moving
    alias, so a long run legitimately spans two.
    """

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        answered = super().ask(state, questions)
        if "Claude" in state["post"]:
            return replace(answered, provider="typesafe", model="jev-1.13.0")
        return replace(answered, provider="otro-juez", model="jev-1.14.0")


class _ClosingFailsClient(FakeJevClient):
    """A client whose pool coughs on teardown — the realistic vendor socket failure.

    `FakeJevClient.close` cannot fail, which is exactly what made the ordering defect
    invisible: a no-op close makes "save before releasing" untestable.
    """

    def close(self) -> None:
        super().close()
        raise OSError("transport teardown: [Errno 54] Connection reset by peer")


def _stored_assessment(item_id: str) -> TopicAssessment:
    """A record from some EARLIER run, for an item the current corpus no longer holds."""
    return TopicAssessment(
        item_id=item_id,
        provider="typesafe",
        model="jev-1.13.0",
        asked_at=DT,
        contract="b" * 64,
        state_chars=3,
        membership={"ai-coding": 0.9},
        primary=PrimaryChoice(choice="ai-coding", confidence=0.8, probabilities={"ai-coding": 0.8}),
    )


def _setup_repo(tmp_path: Path, monkeypatch, jev: str = "") -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        f'vault = "{vault}"\n'
        'output_subdir = "x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n' + (f"[jev]\n{jev}" if jev else ""),
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    return vault


def _item(item_id: str, text: str, topics=("ai-coding",)) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text,
        created_at=DT,
        captured_at=DT,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="Resumen.",
            primary_topic=topics[0],
            topics=list(topics),
        ),
    )


def _seed_vocab(tmp_path: Path) -> None:
    save_vocab(
        [
            Topic(slug="ai-coding", description="Construir software con IA."),
            Topic(slug="startups", description="Fundar empresas."),
        ],
        tmp_path / "data" / "vocab.yaml",
    )


def _seed(tmp_path: Path) -> None:
    save_store(
        {"1": _item("1", "Claude Code hooks"), "2": _item("2", "Seed round tips", ("startups",))},
        tmp_path / "data" / "items.json",
    )
    _seed_vocab(tmp_path)


def _topics_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "jev" / "topics.json"


# --------------------------------------------------------------------------- the happy path


def test_jev_topics_assesses_saves_and_skips_current_on_rerun(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(nouls={"ai-coding": 0.93}, primary="ai-coding")
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    # The tallies are stdout: `xbrain jev topics > run.log` must put them in the file.
    assert "2 items por evaluar (0 evaluaciones guardadas)" in result.stdout
    assert "2 evaluadas · 0 fallidas" in result.stdout
    assert "jev-1.13.0" in result.stdout
    saved = load_assessments(_topics_path(tmp_path))
    assert saved["1"].membership == {"ai-coding": 0.93, "startups": 0.05}
    # The judge that ANSWERED, not the one the config asked for.
    assert saved["1"].provider == "fake"
    assert len(fake.calls) == 2

    monkeypatch.setattr(
        cli, "_jev_client", _refusing_client("no client expected when nothing is pending")
    )
    again = runner.invoke(app, ["jev", "topics"])
    assert again.exit_code == 0, again.output
    assert "0 items por evaluar · 2 vigentes (2 evaluaciones guardadas)" in again.stdout


def test_jev_topics_never_touches_items_json(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    before = (tmp_path / "data" / "items.json").read_bytes()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient())
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0
    assert (tmp_path / "data" / "items.json").read_bytes() == before
    # Not `glob("snapshots/*")`: that passes against an empty directory too.
    assert not (tmp_path / "data" / "snapshots").exists()


# --------------------------------------------------------------------------- selection flags


def test_jev_topics_asks_only_about_the_named_ids(tmp_path: Path, monkeypatch):
    """`--id` is the flag that stands between "one post" and a bill for the whole corpus."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics", "--id", "1"])

    assert result.exit_code == 0, result.output
    assert "1 item por evaluar (0 evaluaciones guardadas)" in result.stdout
    assert len(fake.calls) == 1
    assert list(load_assessments(_topics_path(tmp_path))) == ["1"]


def test_jev_topics_collapses_a_repeated_id_instead_of_billing_it_twice(tmp_path, monkeypatch):
    """Two paid calls for one post would produce two records under one `item_id`."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics", "--id", "1", "--id", "1", "--id", "2"])

    assert result.exit_code == 0, result.output
    assert "2 items por evaluar" in result.stdout
    assert len(fake.calls) == 2


def test_jev_topics_refuses_an_unknown_id_instead_of_selecting_nothing(tmp_path, monkeypatch):
    """A silently-empty selection would read as a successful no-op — the operator would
    think the item was already current."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", _refusing_client("an unknown id must cost nothing"))
    result = runner.invoke(app, ["jev", "topics", "--id", "9"])
    assert result.exit_code == 1
    assert "Error: ids desconocidos: 9" in result.stderr


def test_jev_topics_force_reasks_a_current_assessment_and_announces_the_rebill(
    tmp_path: Path, monkeypatch
):
    """Under `--force` the currency check is skipped, so `vigentes` is structurally 0 — it
    means "we did not look", not "nothing was current". Without `forzados` a full re-bill of
    a perfectly current corpus prints the same line as one whose contracts had expired."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0
    assert len(fake.calls) == 2

    forced = runner.invoke(app, ["jev", "topics", "--force"])

    assert forced.exit_code == 0, forced.output
    assert "2 items por evaluar · 2 forzados (2 evaluaciones guardadas)" in forced.stdout
    assert len(fake.calls) == 4


def test_jev_topics_reports_the_backlog_the_limit_left_behind(tmp_path: Path, monkeypatch):
    """The two counted skips cost nothing; the one `--limit` causes means "there is more
    backlog still to pay for", and it was the only one that was silent."""
    _setup_repo(tmp_path, monkeypatch)
    _seed_vocab(tmp_path)
    save_store(
        {str(n): _item(str(n), f"Post numero {n}") for n in (1, 2, 3)},
        tmp_path / "data" / "items.json",
    )
    fake = FakeJevClient()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics", "--limit", "1"])

    assert result.exit_code == 0, result.output
    assert "1 item por evaluar · 2 fuera del límite (0 evaluaciones guardadas)" in result.stdout
    assert len(fake.calls) == 1


def test_jev_topics_rejects_a_limit_below_one(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", _refusing_client("--limit 0 must cost nothing"))
    result = runner.invoke(app, ["jev", "topics", "--limit", "0"])
    assert result.exit_code == 1
    assert "Error: --limit debe ser >= 1" in result.stderr


def test_jev_topics_on_an_empty_corpus_says_the_corpus_is_empty(tmp_path: Path, monkeypatch):
    """All three counts at 0 is the one reading that means "there is nothing here" rather
    than "everything is current" — which is what the counts were added to distinguish."""
    _setup_repo(tmp_path, monkeypatch)
    _seed_vocab(tmp_path)
    save_store({}, tmp_path / "data" / "items.json")
    monkeypatch.setattr(cli, "_jev_client", _refusing_client("an empty corpus must cost nothing"))

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "0 items por evaluar (0 evaluaciones guardadas)" in result.stdout
    assert not _topics_path(tmp_path).exists()


def test_jev_topics_dry_run_counts_and_reports_the_key_without_calling(tmp_path, monkeypatch):
    """`--dry-run` returns before the client is built, so it validates neither the key nor
    the SDK import. Reporting the key is what stops a green dry-run from being followed by a
    real run that dies on the first thing it checks."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", _refusing_client("--dry-run must not call Jev"))
    result = runner.invoke(app, ["jev", "topics", "--dry-run", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "1 item por evaluar · 1 fuera del límite (0 evaluaciones guardadas)" in result.stdout
    assert "clave TYPESAFE_API_KEY: NO configurada" in result.stdout
    assert not (tmp_path / "data" / "jev").exists()


# --------------------------------------------------------------------------- building a client


def test_jev_topics_without_key_is_a_clean_operator_error(tmp_path: Path, monkeypatch):
    """A clean operator error is one that never reached the vendor's HTTP stack."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    built: list[object] = []
    monkeypatch.setattr(
        "xbrain.jev.typesafe.TypeSafeJevClient", lambda **kwargs: built.append(kwargs)
    )

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 1
    assert "Error: TYPESAFE_API_KEY no encontrada" in result.stderr
    assert "Traceback" not in result.stderr
    assert built == []


def test_jev_topics_builds_the_client_from_the_dotenv_key_and_the_configured_model(
    tmp_path: Path, monkeypatch
):
    """The one test that walks `cfg.repo_root` → `<repo>/.env` → key → adapter for real.

    `tests/conftest.py` redirects `dotenv_path` for EVERY test (correctly — it is what keeps
    the suite off the paid API), so this test restores the real lookup against its own tmp
    repo. Without it, `_jev_client` could read the wrong root and pass the wrong model and
    the suite would stay green: every other test replaces the factory wholesale.
    """
    _setup_repo(tmp_path, monkeypatch, jev='model = "jev-1.13.0"\n')
    _seed(tmp_path)
    monkeypatch.setattr("xbrain.jev.env.dotenv_path", lambda repo_root: repo_root / ".env")
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=ts-de-prueba\n",  # pragma: allowlist secret
        encoding="utf-8",
    )
    built: list[dict] = []

    class _RecordingAdapter(FakeJevClient):
        def __init__(self, *, api_key: str, model: str) -> None:
            super().__init__()
            built.append({"api_key": api_key, "model": model})

    monkeypatch.setattr("xbrain.jev.typesafe.TypeSafeJevClient", _RecordingAdapter)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert built == [
        {"api_key": "ts-de-prueba", "model": "jev-1.13.0"}  # pragma: allowlist secret
    ]


def test_jev_topics_names_the_remedy_when_the_sdk_is_not_installed(tmp_path, monkeypatch):
    """`ImportError` is not in `_OPERATOR_ERRORS`, so a half-finished `uv sync` reached the
    operator as a traceback — after their key was accepted and the selection printed."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-de-prueba")
    # `None` in `sys.modules` is how Python models "this import must fail".
    monkeypatch.setitem(sys.modules, "xbrain.jev.typesafe", None)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 1
    assert "el SDK de TypeSafe no está disponible" in result.stderr
    assert "uv sync" in result.stderr
    assert "Traceback" not in result.stderr


def test_jev_topics_passes_the_configured_concurrency_to_the_runner(tmp_path, monkeypatch):
    """`[jev].concurrency` is the only knob on how fast money is spent; hard-coding 8 in the
    call would keep every other test green."""
    _setup_repo(tmp_path, monkeypatch, jev="concurrency = 3\n")
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient())
    seen: list[int] = []
    real = cli.run_assessments

    def _spy(*args, **kwargs):
        seen.append(kwargs["concurrency"])
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_assessments", _spy)
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0
    assert seen == [3]


# --------------------------------------------------------------------------- cost reporting


def test_jev_topics_prices_the_run_from_the_table_at_the_listed_rate(tmp_path: Path, monkeypatch):
    """The cost line is the one number an operator maps to a bill, so it is pinned exactly.

    `INPUT_USD_PER_MTOK["typesafe"]` is 0.042 $/Mtok, so 2 × 6,000,000 input tokens is
    12,000,000 × 0.042 / 1e6 = 0.504 $. Swapping `/1e6` for `/1e3` prints 504.000, summing
    `output_tokens` prints 20 tokens and 0.000 — every plausible slip moves this string.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(provider="typesafe", input_tokens=6_000_000, output_tokens=10)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "12000000 tokens de entrada (~0.5040 $)" in result.stdout
    # Nothing unaccounted and nothing unpriced: neither marker appears.
    assert "sin recuento" not in result.stdout
    assert "sin tarifa" not in result.stdout


def test_jev_topics_prices_each_record_by_the_provider_that_answered_it(tmp_path, monkeypatch):
    """Per RECORD, not one rate for the run — and the unpriced judge is NAMED.

    Half the batch is answered by `typesafe` (priced) and half by `otro-juez` (absent from
    the table), so only 6,000,000 of the 12,000,000 tokens are billable: 0.252 $. A run
    priced at one flat rate would print 0.504 $ here and stay green in every
    single-provider test, which is exactly why this one mixes them.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(
        cli, "_jev_client", lambda cfg: _PerItemProviderClient(input_tokens=6_000_000)
    )

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert (
        "12000000 tokens de entrada (~0.2520 $ · proveedor sin tarifa: otro-juez)" in result.stdout
    )
    # Two models in one run is what `sorted({...})` exists for — `[jev].model` defaults to a
    # moving alias, so a long run legitimately straddles a version bump.
    assert "modelos jev-1.13.0, jev-1.14.0" in result.stdout


def test_jev_topics_says_how_much_of_the_run_it_could_not_measure(tmp_path, monkeypatch):
    """`input_tokens is None` is a real provider behaviour. Folding it into 0 reports a run
    that was paid for in full as free, which is the failure that costs trust."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(provider="typesafe", input_tokens=None)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "0 tokens de entrada (+2 sin recuento) (~0.0000 $)" in result.stdout


def test_jev_topics_costs_nothing_when_no_provider_in_the_run_is_priced(tmp_path, monkeypatch):
    """An unpriced provider costs 0.0 — it must not raise, and must not borrow a rate.

    `~0.0000 $` alone is indistinguishable from a genuinely free run, so the provider is
    named beside it.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(provider="fake", input_tokens=6_000_000)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "12000000 tokens de entrada (~0.0000 $ · proveedor sin tarifa: fake)" in result.stdout


# --------------------------------------------------------------------------- failures


def test_jev_topics_reports_failed_items_and_keeps_the_rest(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(fail_when=lambda state: "Seed" in state["post"])
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    result = runner.invoke(app, ["jev", "topics"])
    assert result.exit_code == 0, result.output
    assert "1 evaluada · 1 fallida" in result.stdout
    # The per-item failures are stderr: they must reach the terminal even when the tallies
    # are redirected into a log file.
    assert "FALLO 2: fake failure" in result.stderr
    assert list(load_assessments(_topics_path(tmp_path))) == ["1"]


@pytest.mark.parametrize(
    ("total", "failures", "tail"),
    # 12 items → 11 failures → 1 over the limit, the case that exposes a hard-coded plural.
    [(12, 11, "… y 1 fallo más"), (13, 12, "… y 2 fallos más")],
)
def test_jev_topics_truncates_a_long_failure_list(
    tmp_path: Path, monkeypatch, total: int, failures: int, tail: str
):
    """Only the first ten failures are printed; the rest are COUNTED, never dropped.

    The case is a PARTIAL failure — a provider rate-limiting or timing out across most of a
    large batch while some answers still land. (A run where NOTHING succeeds never gets
    here: `run_assessments` raises instead of returning an empty `RunResult`, and a
    malformed vocabulary raises before the pool even opens.) One line per failure would bury
    the summary above them, and dropping the overflow silently would understate the damage.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed_vocab(tmp_path)
    store = {"01": _item("01", "El unico que pasa")}
    store.update({f"{n:02d}": _item(f"{n:02d}", f"Fallo numero {n}") for n in range(2, total + 1)})
    save_store(store, tmp_path / "data" / "items.json")
    fake = FakeJevClient(fail_when=lambda state: "unico" not in state["post"])
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert f"1 evaluada · {failures} fallidas" in result.stdout
    assert result.stderr.count("  FALLO ") == 10
    assert tail in result.stderr
    assert list(load_assessments(_topics_path(tmp_path))) == ["01"]


def test_jev_topics_refuses_to_run_on_a_corrupt_sidecar(tmp_path: Path, monkeypatch):
    """Loading a corrupt side-car as `{}` would re-ask, and re-pay for, the whole corpus and
    then overwrite whatever was still readable with only the new records."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    path = _topics_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        cli, "_jev_client", _refusing_client("a corrupt side-car must cost nothing")
    )

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 1
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr
    # The path is the one thing the operator needs: three files load back to back.
    assert str(path) in result.stderr
    # Nothing was spent and nothing was rewritten: the file to repair is intact.
    assert path.read_text(encoding="utf-8") == "{not json"


# --------------------------------------------------------------------------- teardown & saving


def test_jev_topics_closes_the_client_it_built(tmp_path: Path, monkeypatch):
    """The run releases the connection pool it opened, on the success path."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0
    assert fake.closed is True


def test_jev_topics_closes_the_client_even_when_the_run_dies(tmp_path: Path, monkeypatch):
    """...and on the failure path, which is the one a `with`-less run forgets."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(fail_when=lambda state: True)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    result = runner.invoke(app, ["jev", "topics"])
    assert result.exit_code == 1
    assert "ninguna de las 2 evaluaciones terminó" in result.stderr
    assert fake.closed is True


def test_a_failing_close_never_costs_the_run_its_records(tmp_path: Path, monkeypatch, caplog):
    """Teardown is not the run's verdict.

    With `client.close()` ahead of the save, an `OSError` from the vendor's pool escaped
    before the side-car was ever written: a fully successful, fully billed run exited 1 with
    `data/jev/topics.json` non-existent, and the next run re-billed the whole corpus.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: _ClosingFailsClient())

    with caplog.at_level(logging.WARNING, logger="xbrain.cli"):
        result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert list(load_assessments(_topics_path(tmp_path))) == ["1", "2"]
    assert "2 evaluadas · 0 fallidas" in result.stdout
    # Noise, but never silent noise.
    assert "cerrar el cliente Jev falló" in caplog.text
    assert "Connection reset by peer" in caplog.text


def test_a_failing_close_does_not_downgrade_the_interrupt_exit_code(tmp_path, monkeypatch):
    """A `finally` that raises REPLACES the exception in flight — Python's default. That
    turned the operator's Ctrl-C signal (130) into a socket errno and exit 1."""
    _setup_repo(tmp_path, monkeypatch, jev="concurrency = 1\n")
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: _ClosingFailsClient(interrupt_after=1))

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 130, result.output
    assert list(load_assessments(_topics_path(tmp_path))) == ["1"]
    assert "Interrumpido:" in result.stderr


def test_jev_topics_names_the_bill_when_the_sidecar_cannot_be_written(tmp_path, monkeypatch):
    """`Error: [Errno 13] Permission denied` is true and useless: it is about a directory,
    and nothing in it says "you were just billed for 2 assessments and none was written"."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient())

    def _boom(assessments, path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(cli, "save_assessments", _boom)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 1
    assert "2 evaluaciones pagadas sin guardar" in result.stderr
    assert str(_topics_path(tmp_path)) in result.stderr
    # The summary is echoed BEFORE the save, so a save failure never suppresses the counters.
    assert "2 evaluadas · 0 fallidas" in result.stdout


# --------------------------------------------------------------------------- the interrupt


def test_jev_topics_checkpoints_what_it_paid_for_when_interrupted(tmp_path: Path, monkeypatch):
    """Ctrl-C keeps the records that were already answered.

    They are PAID FOR. `run_assessments` cancels the queued calls and re-raises, discarding
    its own collection, so if the command did not checkpoint, an operator who interrupted a
    3,000-item run would be billed again for every post that had already come back.
    """
    _setup_repo(tmp_path, monkeypatch, jev="concurrency = 1\n")
    _seed(tmp_path)
    # An earlier run's record for an item the corpus no longer holds. It makes the two
    # numbers in the message DIFFER (1 rescued, 2 on disk), so a message reporting the file
    # total for both — the reading an operator hears as "503 rescued" — goes red.
    save_assessments({"99": _stored_assessment("99")}, _topics_path(tmp_path))
    fake = FakeJevClient(interrupt_after=1)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    # 130 is what an UNCAUGHT KeyboardInterrupt already exits with (typer converts it —
    # `typer/core.py` raises `Exit(130)`; bare click would map it to `Abort` and exit 1).
    # Catching the interrupt in order to checkpoint must not change the code the shell sees,
    # and `_handle_cli_errors` would rewrite it to 1 if `typer.Exit` were not re-raised.
    assert result.exit_code == 130, result.output
    assert list(load_assessments(_topics_path(tmp_path))) == ["1", "99"]
    assert (
        f"Interrumpido: 1 evaluación nueva guardada (2 en total) en {_topics_path(tmp_path)}"
        in result.stderr
    )
    # What the interruption cost is the question the operator actually has here.
    assert "100 tokens de entrada" in result.stderr
    assert fake.closed is True


def test_the_interrupt_line_says_re_evaluated_under_force(tmp_path: Path, monkeypatch):
    """Under `--force` the rescued record is a REPLACEMENT, not a new assessment. The two
    numbers exist precisely so the operator does not misread them."""
    _setup_repo(tmp_path, monkeypatch, jev="concurrency = 1\n")
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient())
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0

    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient(interrupt_after=1))
    result = runner.invoke(app, ["jev", "topics", "--force"])

    assert result.exit_code == 130, result.output
    assert "Interrumpido: 1 evaluación re-evaluada guardada (2 en total)" in result.stderr


def test_the_sidecar_is_flushed_during_a_long_run_not_only_at_the_end(tmp_path, monkeypatch):
    """Only `KeyboardInterrupt` is caught, so a SIGTERM, a closed terminal or an OOM kill
    takes every record the dict is holding. The periodic flush bounds that loss.

    Asserted on the SAVE CALLS rather than by reading the file from a fake mid-run: the
    worker thread runs ahead of the main thread's `as_completed` loop, so "the 26th call
    sees 25 records on disk" is a race. `_checkpoint` runs in the main thread once per
    record, so the sequence of save sizes is deterministic.
    """
    _setup_repo(tmp_path, monkeypatch, jev="concurrency = 1\n")
    _seed_vocab(tmp_path)
    save_store(
        {f"{n:02d}": _item(f"{n:02d}", f"Post numero {n}") for n in range(1, 31)},
        tmp_path / "data" / "items.json",
    )
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient())
    sizes: list[int] = []
    real_save = cli.save_assessments

    def _recording_save(assessments, path):
        sizes.append(len(assessments))
        real_save(assessments, path)

    monkeypatch.setattr(cli, "save_assessments", _recording_save)

    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0

    # One durable flush at 25, then the final write — not a single write at the end.
    assert sizes == [25, 30]
    assert len(load_assessments(_topics_path(tmp_path))) == 30


def test_an_interrupt_that_rescued_nothing_writes_nothing(tmp_path: Path, monkeypatch):
    """A Ctrl-C before the first answer lands has nothing to save, and must say so.

    Reporting `0 evaluaciones nuevas guardadas · 0 tokens de entrada (~0.0000 $)` describes a
    save that did not happen and a bill that was never incurred. Worse, the save DID happen:
    it wrote `{}` over the side-car, so an operator who hit Ctrl-C a second too early
    destroyed every assessment they had ever paid for.
    """
    _setup_repo(tmp_path, monkeypatch, jev="concurrency = 1\n")
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient(interrupt_after=0))

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 130, result.output
    assert "Interrumpido: nada nuevo que guardar" in result.stderr
    # No cost line: nothing was answered, so there is no bill to report.
    assert "tokens de entrada" not in result.stderr
    assert not _topics_path(tmp_path).exists()


def test_an_interrupt_that_rescued_nothing_leaves_an_existing_sidecar_alone(
    tmp_path: Path, monkeypatch
):
    """The file every earlier run paid for is not touched by a run that answered nothing."""
    _setup_repo(tmp_path, monkeypatch, jev="concurrency = 1\n")
    _seed(tmp_path)
    save_assessments({"99": _stored_assessment("99")}, _topics_path(tmp_path))
    before = _topics_path(tmp_path).read_bytes()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient(interrupt_after=0))

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 130, result.output
    assert "Interrumpido: nada nuevo que guardar" in result.stderr
    assert _topics_path(tmp_path).read_bytes() == before


# ------------------------------------------------------------------- `xbrain jev report`


def _report_paths(tmp_path: Path) -> tuple[Path, Path]:
    jev_dir = tmp_path / "data" / "jev"
    return jev_dir / "topics-report.json", jev_dir / "topics-report.md"


def _assess_corpus(monkeypatch) -> None:
    """Fill the side-car with a judge that backs `ai-coding` (0.93) and little else (0.05).

    The seeded corpus assigns `ai-coding` to item 1 and `startups` to item 2, so this one
    fake produces one agreement and one disagreement in BOTH directions — item 2's
    `startups` is doubtful and its `ai-coding` is a missing candidate.

    That holds at any threshold AT OR BELOW 0.93. Above it (the `0.95` config test) the judge
    backs nothing at all and item 1's `ai-coding` turns doubtful too, which is exactly what
    that test reads.
    """
    monkeypatch.setattr(
        cli,
        "_jev_client",
        lambda cfg: FakeJevClient(nouls={"ai-coding": 0.93}, primary="ai-coding"),
    )
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0


def test_jev_report_writes_json_and_markdown(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    _assess_corpus(monkeypatch)

    result = runner.invoke(app, ["jev", "report", "--threshold", "0.5"])

    assert result.exit_code == 0, result.output
    assert "Umbral 0.5" in result.stdout
    assert "items comparados 2/2" in result.stdout
    # `sin juzgar` sits next to `dudosas`: without it, `enrich respaldado 50 %` beside
    # `dudosas 0` is a riddle rather than a summary.
    assert "dudosas 1 · sin juzgar 0 · candidatas 1" in result.stdout
    # Nothing was dropped, and the line says so rather than leaving it to be assumed.
    assert "0 caducadas" in result.stdout
    # The recap of what the side-car cost names the judge nobody prices, exactly as the run
    # itself does: `~0.0000 $` alone cannot say whether the work was free or merely unpriced.
    assert "proveedor sin tarifa: fake" in result.stdout
    json_path, md_path = _report_paths(tmp_path)
    # The paths are how an operator finds the files.
    assert f"→ {md_path}" in result.stdout and f"→ {json_path}" in result.stdout
    assert md_path.read_text(encoding="utf-8").startswith("# Jev · topics")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    # Item 2 is assigned `startups`, which this judge scores 0.05 — doubtful at 0.5 — while
    # backing `ai-coding` at 0.93, which enrich did not assign to it.
    assert payload["summary"]["doubtful_pairs"] == 1
    assert payload["summary"]["missing_pairs"] == 1
    assert payload["summary"]["items_compared"] == 2


def test_jev_report_without_assessments_refuses_and_names_the_command_that_fixes_it(
    tmp_path: Path, monkeypatch
):
    """A report over nothing is not a report of zero.

    It is a plausible-looking file of zeros written over the last good one, atomically —
    and `data/` is gitignored, the side-car is not snapshotted, and it costs money to
    regenerate. There is no copy to fall back to.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)

    result = runner.invoke(app, ["jev", "report"])

    assert result.exit_code == 1
    assert "no hay evaluaciones guardadas" in result.stderr
    assert "xbrain jev topics" in result.stderr
    assert not _report_paths(tmp_path)[0].exists()


def test_jev_report_defaults_to_the_configured_threshold(tmp_path: Path, monkeypatch):
    """No `--threshold` reads `[jev].threshold`, so the report and the config agree."""
    _setup_repo(tmp_path, monkeypatch, jev="threshold = 0.95\n")
    _seed(tmp_path)
    _assess_corpus(monkeypatch)

    result = runner.invoke(app, ["jev", "report"])

    assert result.exit_code == 0, result.output
    assert "Umbral 0.95" in result.stdout


def test_jev_report_refuses_a_threshold_above_one(tmp_path: Path, monkeypatch):
    """A probability cannot be 1.5, and above 1.0 nothing is ever backed: every assignment
    becomes doubtful and the report is a plausible-looking file of pure noise.

    The opposite half of the guard has its own test — the two failures are opposites, so one
    case cannot stand for both.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)

    result = runner.invoke(app, ["jev", "report", "--threshold", "1.5"])

    assert result.exit_code == 1
    assert "Error: --threshold debe estar en [0.0, 1.0]" in result.stderr
    assert not _report_paths(tmp_path)[0].exists()


def test_jev_report_excludes_assessments_a_vocabulary_change_retired(tmp_path: Path, monkeypatch):
    """A stale assessment is left OUT, never compared as if it were current.

    The contract binds a stored record to the questions Jev was ACTUALLY asked. Adding a
    topic rewrites the question set, so every stored record now describes an ask that no
    longer exists — and comparing against a question Jev was never shown is not a weaker
    signal, it is a wrong one. The report must drop to zero rather than keep quoting numbers
    off retired records.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    _assess_corpus(monkeypatch)
    before = runner.invoke(app, ["jev", "report"])
    assert "items comparados 2/2" in before.stdout
    json_path, md_path = _report_paths(tmp_path)
    good_json, good_md = json_path.read_bytes(), md_path.read_bytes()

    save_vocab(
        [
            Topic(slug="ai-coding", description="Construir software con IA."),
            Topic(slug="startups", description="Fundar empresas."),
            Topic(slug="devtools", description="Herramientas para programar."),
        ],
        tmp_path / "data" / "vocab.yaml",
    )
    result = runner.invoke(app, ["jev", "report"])

    assert result.exit_code == 1
    # The two retired records are NAMED. `0 items evaluados` was byte-identical to "nobody
    # has ever run `jev topics`", and the two readings differ by the price of the corpus.
    assert "0 evaluaciones vigentes de 2 guardadas (2 caducadas, 0 huérfanas)" in result.stderr
    assert "xbrain jev topics" in result.stderr
    # And the last good report is still there, untouched.
    assert json_path.read_bytes() == good_json and md_path.read_bytes() == good_md


def _good_report(tmp_path: Path, monkeypatch) -> tuple[bytes, bytes]:
    """Write one real report, and hand back the bytes of both files.

    Every refusal test below asserts these EXACT bytes survive: "it printed an error" is not
    the property that matters, "it did not eat the last good report" is.
    """
    _assess_corpus(monkeypatch)
    assert runner.invoke(app, ["jev", "report"]).exit_code == 0
    json_path, md_path = _report_paths(tmp_path)
    return json_path.read_bytes(), md_path.read_bytes()


def _assert_report_untouched(tmp_path: Path, before: tuple[bytes, bytes]) -> None:
    json_path, md_path = _report_paths(tmp_path)
    assert (json_path.read_bytes(), md_path.read_bytes()) == before


def test_jev_report_refuses_an_empty_vocabulary_and_keeps_the_last_report(
    tmp_path: Path, monkeypatch
):
    """The sharpest of the four: `build_topic_questions` refuses an empty vocabulary loudly,
    but only from INSIDE the comparison loop.

    With an empty side-car the loop never runs, so the behaviour used to flip on the contents
    of an unrelated file — and the silent branch was the one where the operator had no other
    signal.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    before = _good_report(tmp_path, monkeypatch)
    (tmp_path / "data" / "vocab.yaml").unlink()

    result = runner.invoke(app, ["jev", "report"])

    assert result.exit_code == 1
    assert "el vocabulario está vacío o falta" in result.stderr
    assert "xbrain vocab" in result.stderr
    _assert_report_untouched(tmp_path, before)


def test_jev_report_refuses_an_empty_store_and_keeps_the_last_report(tmp_path: Path, monkeypatch):
    """A side-car full of paid records and no items to compare them against is an operator
    error, not a corpus with nothing to say."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    before = _good_report(tmp_path, monkeypatch)
    (tmp_path / "data" / "items.json").unlink()

    result = runner.invoke(app, ["jev", "report"])

    assert result.exit_code == 1
    assert "no hay items que comparar" in result.stderr
    assert "xbrain extract" in result.stderr
    _assert_report_untouched(tmp_path, before)


def test_jev_report_refuses_when_the_side_car_disappears_and_keeps_the_last_report(
    tmp_path: Path, monkeypatch
):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    before = _good_report(tmp_path, monkeypatch)
    _topics_path(tmp_path).unlink()

    result = runner.invoke(app, ["jev", "report"])

    assert result.exit_code == 1
    assert "no hay evaluaciones guardadas" in result.stderr
    _assert_report_untouched(tmp_path, before)


def test_jev_report_accepts_the_closed_interval_and_never_calls_jev(tmp_path: Path, monkeypatch):
    """`t = 0` backs everything and `t = 1` backs only certainty — both legal, both meaningful.

    The same test holds the command to its own docstring ("No llama a Jev ni gasta nada") by
    making the call impossible rather than counting calls afterwards.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    _assess_corpus(monkeypatch)
    monkeypatch.setattr(cli, "_jev_client", _refusing_client("report must never call Jev"))

    for value, shown in (("0", "Umbral 0.0"), ("1", "Umbral 1.0")):
        result = runner.invoke(app, ["jev", "report", "--threshold", value])
        assert result.exit_code == 0, result.output
        assert shown in result.stdout


def test_jev_report_refuses_a_negative_threshold_too(tmp_path: Path, monkeypatch):
    """Below 0.0 the failure is the OPPOSITE of above 1.0 and just as silent: `noul >= t`
    holds for every pair, so everything is backed and every topic becomes a candidate — a
    report that reads as near-perfect agreement."""
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    before = _good_report(tmp_path, monkeypatch)

    result = runner.invoke(app, ["jev", "report", "--threshold", "-0.5"])

    assert result.exit_code == 1
    assert "Error: --threshold debe estar en [0.0, 1.0]" in result.stderr
    _assert_report_untouched(tmp_path, before)


def test_jev_report_agrees_with_one_stale_record(tmp_path: Path, monkeypatch):
    """Spanish agrees at 1 only, in the segment that reports what a vocabulary edit cost."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _item("1", "Claude Code hooks")}, tmp_path / "data" / "items.json")
    _seed_vocab(tmp_path)
    _assess_corpus(monkeypatch)
    stored = load_assessments(_topics_path(tmp_path))
    save_assessments(
        {
            "1": stored["1"],
            "2": stored["1"].model_copy(update={"item_id": "2", "contract": "f" * 64}),
        },
        _topics_path(tmp_path),
    )

    result = runner.invoke(app, ["jev", "report"])

    assert result.exit_code == 0, result.output
    # One orphan (id 2 is not in the store) — and the line agrees with each number.
    assert "1 huérfana" in result.stdout
    assert "0 caducadas" in result.stdout


def test_the_three_places_that_quote_the_bill_print_the_same_fragment(tmp_path: Path, monkeypatch):
    """`jev topics`, `jev report` and `topics-report.md` recap ONE side-car.

    They used to render it `~0.000 $` / `~0.0001 $` and `proveedor sin tarifa:` / `sin
    tarifa:` — a recap that prints a different figure from the bill it recaps.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(
        cli,
        "_jev_client",
        lambda cfg: FakeJevClient(nouls={"ai-coding": 0.93}, primary="ai-coding"),
    )
    topics = runner.invoke(app, ["jev", "topics"])
    report = runner.invoke(app, ["jev", "report"])

    assert topics.exit_code == 0 and report.exit_code == 0, report.output
    stored = tuple(load_assessments(_topics_path(tmp_path)).values())
    fragment = cli._jev_cost_line(stored)

    assert fragment == "200 tokens de entrada (~0.0000 $ · proveedor sin tarifa: fake)"
    assert fragment in topics.stdout
    assert fragment in report.stdout
    assert fragment in _report_paths(tmp_path)[1].read_text(encoding="utf-8")

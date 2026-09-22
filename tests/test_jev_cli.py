# tests/test_jev_cli.py
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
    """A `FakeJevClient` that attributes each answer to a different provider.

    `INPUT_USD_PER_MTOK` prices `typesafe` and nothing else. A run that priced the WHOLE
    batch at one rate — say the first record's — instead of pricing each record by the
    provider that answered it would bill the unpriced half too, and no single-provider
    test can see that: with one provider both formulas agree.
    """

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        answered = super().ask(state, questions)
        provider = "typesafe" if "Claude" in state["post"] else "otro-juez"
        return replace(answered, provider=provider)


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


def test_jev_topics_assesses_saves_and_skips_current_on_rerun(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(nouls={"ai-coding": 0.93}, primary="ai-coding")
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "2 items por evaluar · 0 vigentes · 0 sin evidencia" in result.output
    assert "2 evaluadas · 0 fallidas" in result.output
    assert "jev-1.13.0" in result.output
    saved = load_assessments(tmp_path / "data" / "jev" / "topics.json")
    assert saved["1"].membership == {"ai-coding": 0.93, "startups": 0.05}
    # The judge that ANSWERED, not the one the config asked for: provenance travels on the
    # result, so a record can never be attributed to the wrong provider.
    assert saved["1"].provider == "fake"
    assert len(fake.calls) == 2

    monkeypatch.setattr(
        cli, "_jev_client", _refusing_client("no client expected when nothing is pending")
    )
    again = runner.invoke(app, ["jev", "topics"])
    assert again.exit_code == 0, again.output
    assert "0 items por evaluar · 2 vigentes · 0 sin evidencia" in again.output


def test_jev_topics_dry_run_counts_without_calling(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", _refusing_client("--dry-run must not call Jev"))
    result = runner.invoke(app, ["jev", "topics", "--dry-run", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "1 items por evaluar · 0 vigentes · 0 sin evidencia" in result.output
    assert not (tmp_path / "data" / "jev").exists()


def test_jev_topics_without_key_is_a_clean_operator_error(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    result = runner.invoke(app, ["jev", "topics"])
    assert result.exit_code == 1
    assert "TYPESAFE_API_KEY" in result.output


def test_jev_topics_reports_failed_items_and_keeps_the_rest(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(fail_when=lambda state: "Seed" in state["post"])
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    result = runner.invoke(app, ["jev", "topics"])
    assert result.exit_code == 0, result.output
    assert "1 evaluadas · 1 fallidas" in result.output
    assert "FALLO 2: fake failure" in result.output
    assert list(load_assessments(tmp_path / "data" / "jev" / "topics.json")) == ["1"]


def test_jev_topics_closes_the_client_it_built(tmp_path: Path, monkeypatch):
    """The run releases the connection pool it opened, on the success path.

    `_jev_client` builds a client per invocation; without this the pool leaks once per run
    and nothing goes red, because a leaked socket does not fail a test that only reads the
    side-car.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0
    assert fake.closed is True


def test_jev_topics_closes_the_client_even_when_the_run_dies(tmp_path: Path, monkeypatch):
    """...and on the failure path, which is the one a `with`-less run forgets.

    Every item failing makes `run_assessments` raise, so the close must be in a `finally`
    and not on the line after the call.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(fail_when=lambda state: True)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    result = runner.invoke(app, ["jev", "topics"])
    assert result.exit_code == 1
    assert "ninguna de las 2 evaluaciones terminó" in result.output
    assert fake.closed is True


def test_jev_topics_never_touches_items_json(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    before = (tmp_path / "data" / "items.json").read_bytes()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient())
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0
    assert (tmp_path / "data" / "items.json").read_bytes() == before
    assert not list((tmp_path / "data").glob("snapshots/*"))


@pytest.mark.parametrize(
    ("total", "failures", "tail"),
    # 12 items → 11 failures → 1 over the limit, the case that exposes a hard-coded plural.
    [(12, 11, "… y 1 fallo más"), (13, 12, "… y 2 fallos más")],
)
def test_jev_topics_truncates_a_long_failure_list(
    tmp_path: Path, monkeypatch, total: int, failures: int, tail: str
):
    """Only the first ten failures are printed; the rest are COUNTED, never dropped.

    A dead key or a bad vocabulary fails nearly every item. One line each would bury the
    summary under thousands of identical lines, and dropping the overflow silently would
    understate the damage — so the tail is reported as a number, and agrees with itself.
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
    assert f"1 evaluadas · {failures} fallidas" in result.output
    # Ten shown, the rest accounted for — never 10 and a shrug.
    assert result.output.count("  FALLO ") == 10
    assert tail in result.output
    # The one record that succeeded is still saved: a noisy failure list never costs work
    # that was already paid for.
    assert list(load_assessments(tmp_path / "data" / "jev" / "topics.json")) == ["01"]


def test_jev_topics_checkpoints_what_it_paid_for_when_interrupted(tmp_path: Path, monkeypatch):
    """Ctrl-C keeps the records that were already answered.

    They are PAID FOR. `run_assessments` cancels the queued calls and re-raises, discarding
    its own collection, so if the command did not checkpoint, an operator who interrupted a
    3,000-item run would be billed again for every post that had already come back.
    """
    _setup_repo(tmp_path, monkeypatch, jev="concurrency = 1\n")
    _seed(tmp_path)
    # An earlier run's record for an item the corpus no longer holds. It makes the two
    # numbers in the message DIFFER (1 rescued, 2 on disk), so a message that reported the
    # file total for both — the reading an operator hears as "503 rescued" — goes red.
    save_assessments({"99": _stored_assessment("99")}, tmp_path / "data" / "jev" / "topics.json")
    fake = FakeJevClient(interrupt_after=1)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    # 130 is what an UNCAUGHT KeyboardInterrupt already exits with (click converts it).
    # Catching the interrupt in order to checkpoint must not change the code the shell
    # sees — and `_handle_cli_errors` would rewrite it to 1 if `typer.Exit` were not
    # re-raised, so this pins a regression the side-car assertions below cannot see.
    assert result.exit_code == 130, result.output
    topics_path = tmp_path / "data" / "jev" / "topics.json"
    # The record answered before the interrupt is kept, and the earlier run's is not lost.
    assert list(load_assessments(topics_path)) == ["1", "99"]
    assert (
        f"Interrumpido: 1 evaluaciones nuevas guardadas (2 en total) en {topics_path}"
        in result.output
    )
    # The pool is still released on the way out.
    assert fake.closed is True


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
    assert "12000000 tokens de entrada (~0.504 $)" in result.output


def test_jev_topics_prices_each_record_by_the_provider_that_answered_it(
    tmp_path: Path, monkeypatch
):
    """Per RECORD, not one rate for the run — and an unlisted provider contributes 0.0.

    Half the batch is answered by `typesafe` (priced) and half by `otro-juez` (absent from
    the table), so only 6,000,000 of the 12,000,000 tokens are billable: 0.252 $. A run
    priced at one flat rate would print 0.504 $ here and stay green in every
    single-provider test, which is exactly why this one mixes them.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = _PerItemProviderClient(input_tokens=6_000_000)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "12000000 tokens de entrada (~0.252 $)" in result.output


def test_jev_topics_costs_nothing_when_no_provider_in_the_run_is_priced(
    tmp_path: Path, monkeypatch
):
    """An unpriced provider costs 0.0 — it must not raise, and must not borrow a rate.

    `.get(provider, 0.0)` is what makes this a number instead of a `KeyError` in the middle
    of reporting a run that has already been paid for.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(provider="fake", input_tokens=6_000_000)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "12000000 tokens de entrada (~0.000 $)" in result.output

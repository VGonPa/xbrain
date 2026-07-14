# tests/test_entity_grounding.py
"""The deterministic entity-grounding check.

The judge ensemble shares a model and a rubric, so its three votes are one sample
drawn three times: its errors correlate, and a unanimous false negative is invisible
to an audit that only inspects the FAIL/divergent set. This module is the instrument
that CANNOT share that blind spot — no LLM, no judgment. It buys RECALL; the judge
keeps precision. So these tests are written to pin over-flagging as acceptable and
under-flagging as a bug.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.entity_grounding import (
    evidence_text,
    extract_entities,
    extract_uncertain,
    is_grounded,
    scan_store,
    ungrounded_entities,
)
from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Enrichment,
    Item,
    VideoFrame,
)


def _item(
    item_id: str = "1",
    *,
    handle: str = "someone",
    name: str = "Some One",
    tweet: str = "",
    transcript: str = "",
    frames: tuple[str, ...] = (),
    digest: str = "",
) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle=handle, name=name),
        text=tweet,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/v",
                    text=transcript,
                    has_speech=bool(transcript),
                    frames=[
                        VideoFrame(timestamp=float(i), local_path=f"{i}.png", description=d)
                        for i, d in enumerate(frames)
                    ],
                    digest=digest,
                )
            ],
        ),
    )


# ---------------------------------------------------------------- matching semantics


def test_grounding_is_token_boundary_aware_not_substring():
    """ "ARK" must not be grounded by "market". Naive `in` matching would pass it and
    silently clear a real invention — the failure this whole module exists to catch."""
    assert not is_grounded("ARK", "the market moved today")
    assert is_grounded("ARK", "ARK Invest published a note")


def test_grounding_ignores_case_and_accents():
    assert is_grounded("Vertex AI", "we deployed on vertex ai last week")
    assert is_grounded("Martín", "habla martin sobre el modelo")


def test_multiword_entity_must_match_as_a_unit():
    """ "Times" alone in the transcript does NOT ground "Financial Times" — otherwise a
    publication the source never named is cleared by an unrelated common word."""
    assert not is_grounded("Financial Times", "three times he repeated it")
    assert is_grounded("Financial Times", "as the Financial Times reported")


def test_handle_grounds_the_names_concatenated_inside_it():
    """@AnthropicAI grounds "Anthropic": the handle IS the author metadata surface, and
    a concatenated handle is a name with the spaces removed. Splitting it on case
    boundaries recovers the words — otherwise a self-posted clip could never attribute
    itself, which is exactly the needless vagueness the digest rubric forbids."""
    assert is_grounded("Anthropic", "@AnthropicAI")
    assert is_grounded("Startup Archive", "@StartupArchive_")
    # ...but it must not become a licence for any substring: "Ant" is not a word here.
    assert not is_grounded("Ant", "@AnthropicAI")


# ---------------------------------------------------------------- extraction


def test_extracts_multiword_names_handles_acronyms_and_course_codes():
    text = (
        "Entrevista de Dwarkesh Patel con Jensen Huang sobre CS336 y ARK Invest. Ver @lexfridman."
    )
    found = {e.lower() for e in extract_entities(text)}
    assert "dwarkesh patel" in found
    assert "jensen huang" in found
    assert "cs336" in found
    assert "ark invest" in found
    assert "@lexfridman" in found


def test_does_not_extract_our_own_template_headers():
    """ "Qué es" / "Puntos clave" / "Key points" are OUR digest scaffolding, not entities.
    Flagging them would drown every real finding in template noise."""
    text = "**Qué es:** Un clip.\n**Puntos clave:**\n- Algo.\n**Por qué importa:** Nada."
    assert extract_entities(text) == []


def test_does_not_extract_sentence_initial_common_words():
    """Spanish/English sentence-initial capitals are the dominant junk source. They are
    NOT entities and would otherwise flood the report."""
    text = "Clip de entrevista. Vídeo sin audio. The speaker explains. Cuando termina, algo."
    assert extract_entities(text) == []


def test_extraction_is_recall_tuned_for_an_unknown_capitalised_word():
    """An unrecognised capitalised mid-sentence word IS a candidate. We would rather
    flag a false positive (cheap: a human dismisses it) than miss an invention."""
    assert "Groq" in extract_entities("el modelo corre sobre Groq y es rápido")


def test_a_capital_inside_parentheses_is_not_a_line_opener():
    """`(Anthropic)` is punctuation-delimited but NOT sentence-initial — its capital is
    informative and it must stay a confident candidate. Conflating "punctuation break"
    with "sentence start" would demote the single sharpest finding in the corpus."""
    found = extract_entities("Clip de entrevista a Dario Amodei (Anthropic) sobre software")
    assert "Anthropic" in found
    assert "Dario Amodei" in found


def test_quotes_are_delimiters_not_letters():
    """A quoted title arrives as `'ChatGPT'`. Left glued, the token never matches the
    source's ChatGPT and the checker invents its own false positive. Internal apostrophes
    are part of the name and must survive."""
    assert "ChatGPT" in extract_entities("una inmersión en LLMs como 'ChatGPT' para todos")
    assert is_grounded("ChatGPT", "un vídeo sobre ChatGPT")
    assert "O'Brien" in extract_entities("el ensayo de O'Brien sobre agentes")


def test_line_initial_common_word_is_demoted_but_never_dropped():
    """Every Spanish bullet opens with a capitalised verb ("Muestra…", "Aparece…"). Their
    capital proves nothing, so they must not reach the headline — but they are NOT
    discarded either: silently dropping candidates is a hidden recall cap, and a hidden
    recall cap is exactly the defect this module exists to remove."""
    text = "- Muestra una terminal.\n- Aparece un juego retro."
    assert extract_entities(text) == []
    assert set(extract_uncertain(text)) == {"Muestra", "Aparece"}


def test_line_initial_name_survives_when_it_looks_like_a_name():
    """Position demotes only a LONE, ordinary-looking capital. An intrinsic name — internal
    caps, an acronym, a version — or a multi-word run keeps its confidence at line start,
    so "OpenAI lanza…" and "Claude Code muestra…" are never demoted."""
    assert "OpenAI" in extract_entities("- OpenAI lanza un modelo.")
    assert "Claude Code" in extract_entities("- Claude Code ejecuta el plan.")
    assert "CS336" in extract_entities("- CS336 explica el tema.")


# ---------------------------------------------------------------- the contract


def test_ungrounded_splits_a_grounded_name_from_an_ungrounded_affiliation():
    """The sharpest case in the corpus: the caption names the speaker, so his name is
    grounded — but his employer appears on NO surface and must still be flagged."""
    item = _item(
        handle="kimmonismus",
        name="Chubby",
        tweet="Dario Amodei: This disruption is happening faster than ever before.",
        transcript="the models are getting better at software engineering",
        digest="Clip de entrevista a Dario Amodei (Anthropic) sobre ingeniería de software.",
    )
    ungrounded = {e.lower() for e in ungrounded_entities(item, "digest")}
    assert "anthropic" in ungrounded, "the unsourced affiliation must be flagged"
    assert "dario amodei" not in ungrounded, "the caption names him — he is grounded"


def test_a_sentence_opening_verb_before_a_name_is_resolved_by_the_evidence():
    """ "Muestra Claude Code…" (verb + name) and "Claude Code ejecuta…" (name) are
    structurally identical — only a verb lexicon could separate them, and we refuse to
    ship one. So the EVIDENCE arbitrates: when the tail alone is grounded, the leading
    capital was a verb, nothing was invented, and nothing is flagged."""
    item = _item(
        transcript="el agente Claude Code ejecuta el plan",
        digest="- Muestra Claude Code ejecutando el plan.",
    )
    assert ungrounded_entities(item, "digest") == []


def test_a_sentence_opening_verb_before_an_UNGROUNDED_name_still_flags():
    """Same shape, but nothing supports the name: the item must still be flagged. The
    evidence-arbitration must never become an escape hatch."""
    item = _item(transcript="el agente ejecuta el plan", digest="- Muestra Claude Code.")
    assert ungrounded_entities(item, "digest") != []


def test_author_metadata_grounds_a_self_posted_clip():
    """@AnthropicAI posting its own video: naming Anthropic is supported, not invented."""
    item = _item(
        handle="AnthropicAI",
        name="Anthropic",
        tweet="New engineering blog.",
        frames=("A blank screen.",),
        digest="Vídeo que acompaña un post de ingeniería de Anthropic.",
    )
    assert ungrounded_entities(item, "digest") == []


def test_world_knowledge_inference_is_flagged():
    """@claudeai's display name is "Claude". A digest that says "Anthropic" inferred the
    parent company from world knowledge — no surface says it. THREE judges passed this
    unanimously; the deterministic check must not."""
    item = _item(
        handle="claudeai",
        name="Claude",
        tweet="Ads are coming to AI. But not to Claude. Keep thinking.",
        transcript="an AI assistant refuses to show an advert",
        digest="Anuncio en vídeo de Anthropic que satiriza la publicidad dentro de la IA.",
    )
    assert [e.lower() for e in ungrounded_entities(item, "digest")] == ["anthropic"]


def test_summary_evidence_includes_the_linked_article_but_digest_evidence_does_not():
    """Evidence is TARGET-DEPENDENT, because the two generators are handed different things.

    `export_video_digest_worksheet` never ships the linked article — the digest must
    summarise the VIDEO — so an article name in a digest is ungrounded. But the summary
    generator IS given the article body (its rubric says to summarise the article's
    substance), so a name taken from the article is grounded, not invented. Checking a
    summary against the digest's four surfaces would flag the generator for using evidence
    it was correctly given: a false positive manufactured by the checker.
    """
    item = _item(tweet="great read", transcript="", digest="")
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="api",
        summary="El artículo de OpenAI relata cinco meses de trabajo.",
    )
    item.content.sources.append(
        ContentSourceSuccess(
            kind="external_article",
            url="https://openai.com/x",
            text="OpenAI describes how a team built a product over five months.",
        )
    )
    assert "OpenAI" in evidence_text(item, "summary")
    assert "OpenAI" not in evidence_text(item, "digest")
    assert ungrounded_entities(item, "summary") == []


def test_evidence_text_concatenates_exactly_the_four_surfaces():
    """The evidence set must be the SAME four surfaces the digest rubric declares —
    transcript, frames, author metadata, tweet text. If this drifts, the check starts
    disagreeing with the rubric it enforces."""
    item = _item(
        handle="h",
        name="Display Name",
        tweet="tweet words",
        transcript="transcript words",
        frames=("frame words",),
        digest="ignored",
    )
    evidence = evidence_text(item, "digest")
    for surface in ("transcript words", "frame words", "h", "Display Name", "tweet words"):
        assert surface in evidence
    assert "ignored" not in evidence, "the OUTPUT must never be its own evidence"


# ---------------------------------------------------------------- sweep


def test_scan_store_reports_only_items_with_ungrounded_entities():
    store = {
        "1": _item("1", handle="AnthropicAI", name="Anthropic", digest="Post de Anthropic."),
        "2": _item("2", handle="claudeai", name="Claude", digest="Anuncio de Anthropic."),
        "3": _item("3", digest=""),  # no digest → not scanned
    }
    records = scan_store(store, "digest")
    assert [r.item_id for r in records] == ["2"]
    assert [e.lower() for e in records[0].ungrounded] == ["anthropic"]


def test_scan_store_is_target_agnostic():
    """The module takes a target so `summary` can follow the digest without a rewrite."""
    store = {"1": _item("1", handle="a", name="A", transcript="hello", digest="Corre sobre Groq.")}
    assert scan_store(store, "digest")[0].ungrounded == ["Groq"]
    assert scan_store(store, "summary") == []  # no summary generated → nothing to scan


def test_scan_store_rejects_an_unknown_target():
    with pytest.raises(ValueError, match="target"):
        scan_store({}, "not-a-target")


# --- The ASR/variant class: the audit found 0% digest precision from exactly this --------
#
# The primary evidence surface is a speech-recognition transcript, and ASR mangles proper
# nouns. The generator's JOB is to recover the real name. An exact matcher then flags every
# name the system got right. These pin each variant class the audit caught in the wild.


@pytest.mark.parametrize(
    ("entity", "evidence"),
    [
        ("OpenAI", "pricing of open ai models"),  # ASR spacing
        ("DeepMind", "deep mind, Disney Research and NVIDIA"),
        ("ChatGPT", "this is a chat gpt moment"),
        ("Claude Code SDK", "we released the cloud code sdk"),  # ASR corruption
        ("Optimizely", "the founder of optimize lee"),
        ("QK-clip", "a new technique called uh qk clip"),  # hyphenation
        ("HIIT", "high-intensity interval training"),  # acronym ↔ expansion
        ("Financial Times", "trusted FT journalism"),  # expansion ↔ acronym
        ("Claude Cowork", "the Claude's Cowork approach"),  # possessive
        ("Yann LeCun", "RT @ylecun: a great paper"),  # lowercase handle
        ("Miguel González-Fierro", "RT @miguelgfierro: hola"),
    ],
)
def test_a_name_the_generator_correctly_recovered_is_grounded(entity, evidence):
    """The generator un-mangled what ASR mangled. Flagging that is measuring the
    transcriber, not the generator."""
    assert is_grounded(entity, evidence), f"{entity!r} is on the evidence — flagging it is our bug"


def test_variant_matching_does_not_dissolve_the_boundary_guarantee():
    """Being variant-aware must not become being permissive: the whole instrument rests on
    NOT grounding a name the evidence never carries."""
    assert not is_grounded("ARK", "the market moved today")
    assert not is_grounded("Anthropic", "ads are coming to AI, but not to Claude")
    assert not is_grounded("Sam Altman", "the CEO discussed hiring plans")
    assert not is_grounded("Financial Times", "three times he repeated it")


def test_short_names_never_fuzzy_match():
    """ "Andes" is 86% similar to the word "and"; "Hayes" to "hay". Below the length gate a
    fuzzy hit is coincidence, not ASR — and the old `_singular` truncation made it worse by
    stripping `es` and then re-adding it in the regex."""
    assert not is_grounded("Andes", "trained and evaluated")
    assert not is_grounded("Hayes", "there is hay in the barn")
    assert not is_grounded("Reyes", "el rey de España")
    # ...while the real plural case still works.
    assert is_grounded("PDFs", "converted the pdf")
    assert is_grounded("Claudes", "two Claude instances")


def test_scan_store_records_an_output_whose_only_candidates_are_uncertain():
    """The docstring promised the uncertain bucket is never silently dropped; `scan_store`
    dropped it for any output with no CONFIDENT flag — 71 digests and 760 summaries. The
    module exists because a hidden recall cap is what the ensemble had; it had grown one."""
    item = _item(digest="- Arranca con una pantalla en blanco.")
    records = scan_store({"1": item}, "digest")
    assert records, "an uncertain-only output must still produce a record"
    assert records[0].ungrounded == []
    assert records[0].uncertain == ["Arranca"]


def test_missing_verdicts_file_raises_instead_of_reporting_zero():
    """`{}` printed "0 with a UNANIMOUS PASS" — which reads as *the judges caught
    everything* when the file was never found. A silent zero is the worst failure mode for
    a report whose subject is silent zeros."""
    from xbrain.entity_grounding import load_ensemble_verdicts

    with pytest.raises(FileNotFoundError):
        load_ensemble_verdicts(Path("/nonexistent/verify-report.json"), "digest")


def test_topics_target_is_rejected_as_structurally_unscannable():
    """Topic slugs are lowercase kebab-case, so the extractor yields zero candidates for
    every one of 2,168 outputs — a confident clean bill of health from a check incapable of
    finding anything."""
    with pytest.raises(ValueError, match="topics"):
        scan_store({}, "topics")


# --- The real lever: translation + generic terms -------------------------------------
#
# At the tuned threshold EVERY surviving false positive is one of two classes, and NO
# threshold touches either. The output is Spanish; the evidence is English. The generator
# is translating correctly, and we were flagging it for doing so.


@pytest.mark.parametrize(
    ("entity", "evidence"),
    [
        ("PIB", "GDP growth slowed last quarter"),
        ("IMC", "his BMI dropped four points"),
        ("ADN", "the DNA of the organisation"),
        ("Marte", "a mission to Mars"),
        ("Luna", "landing on the Moon"),
        ("Ucrania", "the war in Ukraine"),
        ("España", "he moved to Spain"),
        ("Europa", "across Europe"),
        ("EEUU", "the United States market"),
        ("Navidad", "released on Christmas Day"),
    ],
)
def test_a_correctly_translated_name_is_grounded(entity, evidence):
    """A Spanish summary of an English source SHOULD say `Marte`, not `Mars`. Flagging the
    translation is flagging the generator for doing its job."""
    assert is_grounded(entity, evidence), (
        f"{entity!r} is the correct translation — not an invention"
    )


def test_the_lexicon_does_not_ground_an_unrelated_name():
    """Translation tolerance must not become a licence: a name with no translation link is
    still ungrounded."""
    assert not is_grounded("Marte", "a mission to Jupiter")
    assert not is_grounded("Anthropic", "the DNA of the organisation")


@pytest.mark.parametrize(
    "term", ["USD", "PDF", "WOD", "CVs", "Q2", "SOTA", "kHz", "HIIT", "IMC", "PIB"]
)
def test_generic_terms_and_units_are_not_named_entities(term):
    """`USD`, `PDF`, `Q2` are not names at all. Flagging them is pure noise, and noise is
    what makes a report stop being read — which is its own kind of false negative."""
    assert extract_entities(f"el informe usa {term} para el análisis") == []


def test_the_gate_counts_only_the_confident_bucket():
    """Ingest is ~84 items/week. At the tuned threshold the CONFIDENT flag rate is ~12%
    (~10/week: ~3 real, ~7 false alarms) — a payable tax for a token-free check. The
    `uncertain` bucket is ~46 items/week: in a gate it would bury the 3 real findings under
    50 line-openers. It stays in the REPORT and out of the GATE."""
    from xbrain.entity_grounding import gate_failures

    confident_only = _item(digest="Clip de entrevista a Anthropic sobre software.")
    uncertain_only = _item("2", digest="- Arranca con una pantalla en blanco.")
    store = {"1": confident_only, "2": uncertain_only}

    records = scan_store(store, "digest")
    assert {r.item_id for r in records} == {"1", "2"}, "both are still REPORTED"
    assert [r.item_id for r in gate_failures(records)] == ["1"], "only the confident one GATES"

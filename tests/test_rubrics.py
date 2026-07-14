# tests/test_rubrics.py
from pathlib import Path

from xbrain.models import Topic
from xbrain.rubrics import load_guardrails, load_rubric, load_vocab, save_vocab


def test_load_rubric_returns_file_text():
    assert "summary" in load_rubric("summary").lower()


def test_load_rubric_substitutes_language_placeholder():
    """When language is provided, {language} is replaced verbatim."""
    text = load_rubric("summary", language="English")
    assert "{language}" not in text
    assert "**Language:** English" in text


def test_load_rubric_supports_spanish_language():
    text = load_rubric("topic-page", language="Spanish")
    assert "{language}" not in text
    # The placeholder appears twice in topic-page (overview + notes)
    assert text.count("in Spanish") == 2


def test_load_rubric_preserves_placeholder_when_language_none():
    """No-language calls (tests, inspection) keep the literal `{language}`."""
    text = load_rubric("summary")
    assert "{language}" in text


def test_load_rubric_topics_has_no_placeholder():
    """rubric-topics emits only slugs; no language placeholder; passing one is a no-op."""
    a = load_rubric("topics")
    b = load_rubric("topics", language="English")
    assert a == b
    assert "{language}" not in a


def test_load_rubric_defensive_check_catches_unsubstituted_placeholder(tmp_path, monkeypatch):
    """A typo like {Language} (capital L) survives str.replace and would
    silently ship the literal placeholder to the LLM. The defensive regex
    catches it and raises a loud ValueError naming the typo.
    """
    import pytest

    from xbrain import rubrics as rubrics_mod

    typo_dir = tmp_path / "rubrics"
    typo_dir.mkdir()
    (typo_dir / "rubric-typo.md").write_text(
        "**Language:** {Language}, regardless of the post.\n",  # capital L typo
        encoding="utf-8",
    )
    monkeypatch.setattr(rubrics_mod, "_RUBRICS_DIR", typo_dir)

    with pytest.raises(ValueError, match=r"\{Language\}"):
        load_rubric("typo", language="English")


def test_load_guardrails_returns_enrichment_constraints():
    g = load_guardrails()
    assert g["enrichment"]["topics_max"] == 4
    assert g["enrichment"]["summary_required"] is True


def test_save_then_load_vocab_roundtrips(tmp_path: Path):
    path = tmp_path / "vocab.yaml"
    topics = [
        Topic(slug="ai-coding", description="LLMs writing software."),
        Topic(slug="misc", description="Posts that do not fit a topic."),
    ]
    save_vocab(topics, path)
    loaded = load_vocab(path)
    assert [t.slug for t in loaded] == ["ai-coding", "misc"]
    assert loaded[0].description == "LLMs writing software."


def test_load_vocab_missing_file_returns_empty(tmp_path: Path):
    assert load_vocab(tmp_path / "nope.yaml") == []


def test_topic_page_rubric_loads():
    from xbrain.rubrics import load_rubric

    text = load_rubric("topic-page")
    assert "overview" in text
    assert "notes" in text


def test_describe_image_rubric_loads_and_substitutes_language():
    """The describe-image rubric ships a `{language}` placeholder; the
    loader must substitute it, and the defensive check must not trip on
    correctly-spelt placeholders.
    """
    text = load_rubric("describe-image", language="English")
    assert "{language}" not in text
    assert "English" in text
    # Sanity: the contract keys must appear in the prompt so the LLM
    # produces the right JSON shape.
    assert "is_decorative" in text
    assert "description" in text
    assert "index" in text


def test_describe_image_rubric_preserves_placeholder_when_language_none():
    """No-language calls (tests, manual inspection) keep the literal placeholder."""
    text = load_rubric("describe-image")
    assert "{language}" in text


def test_video_digest_rubric_loads_and_substitutes_language():
    """The video-digest rubric ships a `{language}` placeholder the loader must
    substitute; its structural section keys must reach the LLM."""
    text = load_rubric("video-digest", language="Spanish")
    assert "{language}" not in text
    assert "Spanish" in text
    assert "Key points" in text
    assert "What it is" in text


def test_video_digest_rubric_preserves_placeholder_when_language_none():
    text = load_rubric("video-digest")
    assert "{language}" in text


def test_verify_rubric_loads_and_substitutes_language():
    """The verify rubric ships a `{language}` placeholder + names its axes/verdicts."""
    text = load_rubric("verify", language="Spanish")
    assert "{language}" not in text
    assert "Spanish" in text
    assert "Faithfulness" in text
    assert "FAIL" in text


def test_verify_audit_rubric_loads_and_substitutes_language():
    """The audit rubric ships a `{language}` placeholder + names CONFIRM/REVOKE."""
    text = load_rubric("verify-audit", language="Spanish")
    assert "{language}" not in text
    assert "Spanish" in text
    assert "CONFIRM" in text
    assert "REVOKE" in text


def test_verify_rubric_author_block_licenses_only_who_posted():
    """F2: the `[Author]` block says who POSTED the item — nothing about who wrote or
    spoke its CONTENT. The two come apart on a repost/quote/clip of someone else: a
    digest naming the poster as the speaker of a clip is a MISATTRIBUTION the judge
    caught before this rule existed. The rubric must grant only the true half.
    """
    text = load_rubric("verify", language="English").lower()
    # The permissive wording that licensed attributing anyone's words to the poster.
    assert "attributing the post to its own author is supported" not in text
    # What the block DOES establish: who posted it.
    assert "posted" in text
    # What it does NOT establish: authorship/voice of the content, on a repost/quote/clip.
    assert "repost" in text or "quote" in text or "clip" in text
    assert "speaker" in text
    assert "does not establish" in text or "not establish" in text


def test_summary_rubric_only_summarises_shared_content_when_present():
    """F3: no fetcher downloads a quoted post, so ordering the generator to
    'summarise the substantive content being shared' orders it to invent. The rule
    must be conditional on that content actually being in the source."""
    text = load_rubric("summary", language="English").lower()
    assert "summarise the substantive content being shared." not in text
    assert "when it is present" in text or "when its content is present" in text
    assert "post's own text" in text


# --- Summary faithfulness: the evidence contract (PR-E) ----------------------
#
# The summary rubric carried only "never invent facts, numbers or claims" — the same
# abstraction that failed on digests, because the model does not classify recognising a
# famous name as inventing. It produced 307 ungrounded names across 2,168 summaries.
# These guards pin the mechanical rule that replaces it.


def test_summary_rubric_declares_its_evidence_surfaces():
    """The summary's evidence set is WIDER than the digest's: the enrich worksheet ships
    the fetched article, the poster's own thread and the image descriptions. The rule must
    ADMIT them — forbidding evidence the generator was correctly given would flag it for
    doing its job (the exact bug found in the entity checker itself)."""
    text = load_rubric("summary").lower()
    for surface in ("display name", "tweet text", "thread", "article", "image descriptions"):
        assert surface in text, f"summary rubric no longer admits {surface!r} as evidence"


def test_summary_rubric_forbids_naming_entities_no_surface_names():
    """Same mechanism as the digest rubric: the entity enumeration IS the mechanism,
    since the generic rule empirically failed across 2,168 summaries."""
    text = load_rubric("summary").lower()
    for entity in ("interviewer", "employer", "publication", "university", "author"):
        assert entity in text, f"summary rubric no longer forbids naming an unnamed {entity}"
    assert "world knowledge" in text
    assert "neutral descriptor" in text


def test_summary_rubric_refuses_the_url_and_domain_as_a_source_of_names():
    """A link to nytimes.com does not license naming "The New York Times". The generator
    sees every link's URL+domain; the judge only sees them when the fetch FAILED. Reading a
    publication's name off a domain is the "Financial Times" failure with extra steps."""
    text = load_rubric("summary").lower()
    assert "domain" in text
    assert "topic signal" in text


def test_summary_rubric_keeps_quote_verbatim_and_do_not_sharpen():
    text = load_rubric("summary").lower()
    assert "verbatim" in text
    assert "sharpen" in text


def test_summary_rubric_keeps_the_86_attribution_and_unfetched_guardrails():
    """#86's constraints must survive intact and stay coherent with the new rule: the
    author block licenses WHO POSTED only, and unfetched content is never reconstructed."""
    text = load_rubric("summary").lower()
    assert "posted" in text
    assert "not fetched" in text or "never fetched" in text
    assert "reconstruct" in text


def test_verify_rubric_treats_a_url_as_topic_signal_never_as_a_name():
    """D1 — the deepest defect on the board. `rubric-verify` used to carve out the URL
    ("unsupported ... beyond the URL/domain itself"), which made the domain EVIDENCE:
    the judge was structurally unable to flag `Axios` reconstructed from
    `axios.com/2025/05/28/ai-jobs...`, or `Nature` from `nature.com`. The generator's
    rubric says the opposite — a domain is topic signal, never a name. Both must now say
    the same thing, or the judge licenses exactly what the generator is forbidden."""
    text = load_rubric("verify", language="English")
    assert "beyond the URL/domain itself" not in text  # the carve-out that excused it
    lowered = text.lower()
    assert "topic signal" in lowered
    # naming the publication/company a link belongs to is UNSUPPORTED
    assert "publication" in lowered
    assert "never a name" in lowered or "not a name" in lowered


def test_verify_rubric_declares_the_evidence_surfaces_per_target():
    """The judge must be told WHICH surfaces support a claim, and they differ by target:
    a digest is judged against the video, a summary also against the article/thread/
    images its generator was handed."""
    text = load_rubric("verify", language="English")
    assert "digest" in text and "summary" in text
    assert "video transcript" in text.lower()
    assert "fetched article body" in text.lower()

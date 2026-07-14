"""Deterministic entity-grounding check — the instrument that cannot share the judge's
blind spot.

WHY THIS EXISTS. The verification ensemble is three judges sharing one model and one
rubric: that is one sample drawn three times, not three independent samples. Its errors
correlate by construction, so unanimity measures agreement, not truth. The judge≠party
audit then inspects only the CONSEQUENTIAL set (FAIL + divergent), which makes a
*unanimous false negative* invisible by design — the ensemble's most likely error is the
one the audit is guaranteed never to look at. Two digests in the corpus name "Anthropic"
on no evidence and were passed unanimously by all three judges. This module catches them
with no model and no judgment at all.

THE DIVISION OF LABOUR. Recall comes from HERE (a mechanical check cannot inherit an
LLM's blind spot); precision stays with the judge (it adjudicates what this raises). So
every ambiguous call in this module is resolved **towards flagging**: a false positive
costs one human dismissal, a false negative is what has been shipping silently.

WHAT IT CHECKS. Every named entity in a generated output must appear on one of the
evidence surfaces its rubric declares. An entity present in the output and absent from all
of them is an *ungrounded candidate*.

WHAT IT IS BLIND TO — READ THIS BEFORE QUOTING ANY NUMBER FROM IT.
This instrument checks that PROPER NOUNS APPEAR SOMEWHERE ON THE EVIDENCE. It never checks
that anything asserted ABOUT them is true, and it never looks at a single NUMBER.

* Claims about entities are invisible. "Sam Altman dijo que despedirá a la mitad" against
  evidence where he discusses *hiring* extracts `Sam Altman`, finds it grounded, and
  passes CLEAN. Every false attribution, invented mechanism and fabricated causal link is
  of this shape, and an independent audit found them in ~8% of the outputs this module
  called clean.
* Numbers are never examined. An invented "92% en MMLU", a false date, a fabricated
  funding round: invisible, always.
* Lowercase and two-letter names are not extracted at all.

A CLEAN VERDICT MEANS "NO UNKNOWN PROPER NOUNS". IT DOES NOT MEAN "NOT HALLUCINATED".
No statement of the form "N% of the corpus is hallucination-free" is supported by this
tool, and the most damaging hallucination for a knowledge base — a confident false claim
about a real, correctly-named entity — is precisely the one it cannot see.

THE EVIDENCE IS ASR OUTPUT, AND ASR MANGLES PROPER NOUNS. The transcript says "open ai",
"cloud code sdk", "mustafa sullivan"; the generator correctly recovers `OpenAI`,
`Claude Code SDK`, `Mustafa Suleyman`. An exact-string matcher flags exactly the names the
system got RIGHT — measured at ~0% digest precision before this was fixed. Matching is
therefore variant-aware (squashed spacing, acronym↔expansion, handle abbreviation, bounded
fuzzy) — see `is_grounded`. That is not leniency; it is the difference between measuring
the generator and measuring the transcriber.

NO NLP DEPENDENCY. A statistical NER model would add a heavyweight dependency (and its
own probabilistic blind spot) to a check whose entire value is being non-probabilistic.
The heuristics below are plain `re` + `unicodedata` + `difflib`, fully pinned by tests.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path

from xbrain.evidence import evidence_text
from xbrain.models import Item
from xbrain.verification import ALL_TARGETS, VerifyTarget, _output_for

# Words that are never a named entity. Sentence-initial capitalisation in Spanish and
# English is the dominant junk source, and our own digest template ("Qué es", "Puntos
# clave", "Key points"…) is scaffolding, not content. Without this list the report is
# drowned in noise and stops being read — which is its own kind of false negative.
_STOPWORDS = frozenset(
    """
    a al algo ademas ahora antes anuncio aqui asi aunque cada caso casos charla clave
    clip como con cuando de del demo desde discurso dos durante ejemplo ejemplos el ella
    ellos en entre entrevista esa escena ese esta estas este esto estos explicacion
    extracto final fragmento grabacion gran hasta hay hilo idea ideas imagen inicio la
    las lo los luego mas mientras misma mismo muy nada no nota nueva nuevo o otra otro
    panel pantalla para parte pero podcast por porque post posts presentacion primero
    principal punto puntos que resumen se segun ser sesion si sin sobre solo son su sus
    tambien tema temas tiene todo tras tres tutorial un una unas unos ver vez video y ya
    about after also an and are as at be before but by clip during end first for from he
    her here his how i in interview is it its key keynote main my new now of on one or
    our part point points second she start summary talk than that the their then these
    they third this those three to two video was we were what when while who why with
    without you your
    """.split()
)

# Generic technical nouns. They may EXTEND an entity ("Vertex AI", "Claude Code") but may
# never START one — a lone "IA"/"AI"/"API" is a field, not a name. Flagging every generic
# "IA" across a Spanish corpus would be pure noise; dropping it from a multi-word name
# would lose the name. This distinction is what buys us both.
_WEAK_TOKENS = frozenset("ai ia api apis llm llms ml ui ux sdk cli gpu gpus cpu os ceo cto".split())

_COMBINING = re.compile(r"[̀-ͯ]")
# Word boundary inside a compound: `AnthropicAI` → `Anthropic AI`. A handle is a name with
# the spaces removed, so a case transition IS a word boundary. Lowercase concatenations
# (`@lexfridman`) are deliberately NOT split — they are ambiguous, and the display name
# (shipped beside the handle since #87) already carries the spaced form.
_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_COMPOUND = re.compile(r"@?\w*[a-z0-9]_?[A-Z]\w*")
# Markdown that carries no entity signal. `_` is NOT stripped: it lives inside handles.
_MARKDOWN = re.compile(r"[*#`>]+")
# A bullet marker is not a word: without stripping it, the first WORD of "- Muestra una
# terminal" looks mid-sentence and its uninformative capital is trusted.
_BULLET = re.compile(r"^[ \t]*(?:[-•·]|\d+[.)])[ \t]+", re.MULTILINE)
# Sentence/clause boundary: only HERE is a capital uninformative.
_SENTENCE = re.compile(r"[.!?:;]+\s|\n+")
# Internal punctuation ends an entity but NOT the sentence: "Dario Amodei (Anthropic)" is
# two entities, and Anthropic's capital still means something — it is not a line opener.
_BREAK = re.compile(r"[^\w@'’\-\s]+")


def fold(text: str) -> str:
    """Case-fold, strip accents, collapse whitespace — the comparison form.

    Both sides of every comparison pass through here, so "Martín" grounds "martin"
    and a digest's capitalisation never decides a match.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return " ".join(_COMBINING.sub("", decomposed).casefold().split())


def _expand_compounds(text: str) -> str:
    """Append the space-separated form of every CamelCase/underscore compound.

    `@AnthropicAI` also reads as `Anthropic AI`, so a clip posted by the company's own
    account grounds its own name — the self-attribution the digest rubric requires.
    """
    extra = [
        _CASE_BOUNDARY.sub(" ", m.group().lstrip("@").replace("_", " "))
        for m in _COMPOUND.finditer(text)
    ]
    return " ".join([text, *extra])


# How close a squashed evidence window must be to the entity before we call it the same
# name. THE PRIMARY EVIDENCE SURFACE IS ASR OUTPUT, AND ASR MANGLES PROPER NOUNS: the
# transcript says "cloud code sdk", "optimize lee", "mustafa sullivan"; the generator
# correctly recovers "Claude Code SDK", "Optimizely", "Mustafa Suleyman". An exact matcher
# flags exactly the names the system got RIGHT — measured at ~0% digest precision.
#
# 0.80, set from a hand-labelled adjudication, not by feel. Recall on real inventions is
# FLAT at 100% across the whole sweep, even at 0.70 — so the threshold was never the safety
# property. `_FUZZY_MIN_LEN` is (short names like `Claude`/`España` cannot fuzzy-match at
# all, which is exactly where coincidence would bite). What 0.82 → 0.80 buys, entity by
# entity: `Mustafa Suleyman`~"mustafasullivan" (exactly 0.800), `Claude Code`~"claudecode"
# (1.000 — an exact match the window logic misses when the tokens are non-adjacent in the
# evidence), `Projection`~"proyeccion" (an EN↔ES cognate). The first COINCIDENTAL ground
# appears at ~0.73 (`BOND Capital`~"bondchatgpt"). Safe band 0.78–0.82. Never below 0.78.
_FUZZY_THRESHOLD = 0.80
_POSSESSIVE = re.compile(r"'s\b|’s\b")
_WORD = re.compile(r"[a-z0-9]+")


# EN ⇄ ES. The outputs are Spanish; the evidence is usually English. The generator writes
# `Marte` for "Mars" and `PIB` for "GDP" — CORRECTLY — and an exact matcher flags it for
# doing its job. Translation was one of the two false-positive classes that no threshold can
# touch; this is the lever that actually moves precision. Deliberately small: only the terms
# the corpus actually produced, because a sprawling lexicon becomes a way to ground things
# that were never said.
_LEXICON = {
    "pib": "gdp",
    "imc": "bmi",
    "adn": "dna",
    "marte": "mars",
    "luna": "moon",
    "tierra": "earth",
    "ucrania": "ukraine",
    "espana": "spain",
    "europa": "europe",
    "eeuu": "unitedstates",
    "eeuu.": "unitedstates",
    "canarias": "canaryislands",
    "navidad": "christmas",
    "congreso": "congress",
    "universidad": "university",
    "roma": "rome",
    "occidente": "west",
    "normadecapa": "layernorm",
    "proyeccion": "projection",
}
_LEXICON |= {v: k for k, v in _LEXICON.items()}  # both directions

# Not names at all: units, currencies, file formats, quarters, generic acronyms. Flagging
# them is pure noise, and a report drowned in noise stops being read — which is its own kind
# of false negative. The other class no threshold touches.
_GENERIC_TERMS = frozenset(
    """
    usd eur gbp pdf pdfs csv json html xml url urls wod hiit imc pib adn sota kpi roi mrr arr
    cv cvs q1 q2 q3 q4 khz mhz ghz kb mb gb tb ms fps rpm bpm iva irpf pyme ipo b2b b2c saas
    faq rss sdk api cli gpu cpu ram ssd usb pdf debe
    ias ceos ctos llms apis uis uxs
    """.split()
)

# A token that is nothing but roman numerals is a century or a sequence number ("siglo XIX",
# "Luis XIV"), never a name on its own.
_ROMAN = re.compile(r"^[ivxlcdm]+$")


def _norm_tokens(text: str) -> list[str]:
    """Fold, drop possessives, split into alphanumeric word tokens."""
    return _WORD.findall(_POSSESSIVE.sub("", fold(text)))


def _translations(squashed: str) -> set[str]:
    """The name as the OTHER language would render it, if we know the pair."""
    hit = _LEXICON.get(squashed)
    return {hit} if hit else set()


def _depluralise(token: str) -> str:
    """Strip ONE trailing `s` — never `es`.

    The old rule stripped `es` too, and the regex then re-added `(?:e?s)?`, so "Andes"
    matched the word "and", "Hayes" matched "hay", "Jobs" matched "job". Stripping a single
    `s` keeps the real case (`PDFs`→`pdf`, `Claudes`→`claude`) and kills the false one
    (`Andes`→`ande`, which matches nothing).
    """
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _variants(tokens: list[str]) -> set[str]:
    """Every squashed form of a name that a source might plausibly carry.

    A name survives ASR, hyphenation and handle-mangling in a small, enumerable set of
    shapes — and matching them is not "being lenient", it is refusing to flag a name the
    generator recovered correctly:

    * squashed — `open ai`/`chat gpt`/`qk clip` are one word once the spaces go;
    * acronym — `HIIT` is `high-intensity interval training`;
    * handle abbreviations — `@ylecun` IS `Yann LeCun`, `@miguelgfierro` IS
      `Miguel González-Fierro`. Each token is kept whole or cut to its initial, which
      generates exactly the handle conventions people actually use.
    """
    plain = [_depluralise(t) for t in tokens]
    forms = {"".join(tokens), "".join(plain), "".join(t[0] for t in plain if t)}
    forms |= _translations("".join(plain)) | _translations("".join(tokens))
    if len(plain) <= 4:  # 2^4 shapes — bounded, and real handles are short
        for mask in range(1 << len(plain)):
            forms.add("".join(t if mask >> i & 1 else t[0] for i, t in enumerate(plain)))
    return {f for f in forms if len(f) > 1}


def _windows(tokens: list[str], size: int) -> set[str]:
    """Squashed token windows of `size`, in BOTH the raw and de-pluralised forms.

    De-pluralising was applied destructively at first, which quietly mangled every word that
    merely ends in `s`: "christmas" became "christma", "states" became "state", and the
    matching name never landed. A plural is an ALTERNATIVE spelling of the window, not a
    replacement for it.
    """
    windows = {"".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}
    stemmed = [_depluralise(t) for t in tokens]
    windows |= {"".join(stemmed[i : i + size]) for i in range(len(stemmed) - size + 1)}
    return windows


# Below this length a fuzzy match is coincidence, not ASR corruption: "andes" is 86%
# similar to the word "and", "hayes" to "hay". Short names must match exactly or not at
# all — this gate, not the threshold, is what keeps the boundary guarantee.
_FUZZY_MIN_LEN = 8


def _fuzzy_hit(needle: str, candidates: set[str]) -> bool:
    """True when some candidate is within the ASR-corruption threshold of `needle`."""
    if len(needle) < _FUZZY_MIN_LEN:
        return False
    return any(
        SequenceMatcher(None, needle, c).ratio() >= _FUZZY_THRESHOLD
        for c in candidates
        if abs(len(c) - len(needle)) <= max(3, len(needle) // 4)
    )


def is_grounded(entity: str, evidence: str) -> bool:
    """True when `entity` appears on the evidence — in ANY shape a source plausibly uses.

    Boundary-aware, never substring: `ARK` is still not grounded by "market", because every
    comparison is against whole-token windows. Multi-word names still match as a unit:
    "Times" alone does not ground `Financial Times`. What is new is that the comparison is
    made on the SQUASHED form of a window, plus its acronym and handle abbreviations, plus
    a bounded fuzzy match — because the evidence is ASR output and the generator's job is
    to un-mangle it. See `_variants`.
    """
    ent = _norm_tokens(entity)
    if not ent:
        return False
    ev = _norm_tokens(_expand_compounds(evidence))
    if not ev:
        return False
    forms = _variants(ent)
    # Exact/variant: any squashed window of 1..len+1 tokens equal to any form of the name.
    for size in range(1, min(len(ent) + 2, len(ev) + 1)):
        if forms & _windows(ev, size):
            return True
    # Acronym→expansion: `HIIT` vs "high intensity interval training".
    squashed = "".join(_depluralise(t) for t in ent)
    for size in range(2, min(6, len(ev) + 1)):
        acronyms = {"".join(w[0] for w in ev[i : i + size]) for i in range(len(ev) - size + 1)}
        if squashed in acronyms:
            return True
    # ASR corruption: `Mustafa Suleyman` vs "mustafa sullivan".
    raw = "".join(ent)
    return _fuzzy_hit(raw, _windows(ev, len(ent)) | _windows(ev, max(1, len(ent) - 1)))


def _is_version_code(core: str) -> bool:
    """`GPT-4`, `CS336`, `StyleTTS2` — letters AND digits mean a code, not a word."""
    return any(c.isdigit() for c in core) and any(c.isalpha() for c in core)


def _is_acronym(core: str) -> bool:
    """`ARK`, `SORTTAB`. Two letters is too weak — `EE`/`UU`/`IA` are abbreviations."""
    return core.isupper() and sum(c.isalpha() for c in core) >= 3


def _is_camel_case(core: str) -> bool:
    """`OpenAI`, `macOS`. Camel needs a lowercase: `IA` and `EE` are not camel."""
    return any(c.isupper() for c in core[1:]) and any(c.islower() for c in core)


def _is_intrinsically_named(token: str) -> bool:
    """Name-like on its own evidence, independent of POSITION.

    Sentence- and bullet-initial capitalisation carries no information (every Spanish
    bullet opens "Muestra…", "Aparece…", "Tesis:…"), so a token that starts a line is a
    name only if something OTHER than its first capital says so. A single letter is never
    a name, and a leading digit disqualifies — `1M`/`000M` are quantities.
    """
    core = token.lstrip("@")
    if token.startswith("@"):
        return True
    if len(core) < 2 or core[0].isdigit():
        return False
    return _is_version_code(core) or _is_acronym(core) or _is_camel_case(core)


def _is_not_a_name(folded: str) -> bool:
    """A word that is never a name whatever its capitalisation: a stopword, a unit or
    currency (`USD`, `kHz`), a file format (`PDF`), or a roman numeral (`siglo XIX`)."""
    if not folded or folded in _STOPWORDS or folded in _GENERIC_TERMS:
        return True
    return bool(_ROMAN.match(folded))


def _is_abbreviation(token: str, folded: str) -> bool:
    """May EXTEND a name, never start one: `AI`/`IA`/`API`, and the `EE`/`UU` that "EE.UU."
    shatters into."""
    two_letter_caps = token.isupper() and sum(c.isalpha() for c in token) == 2
    return folded in _WEAK_TOKENS or two_letter_caps


def _classify(token: str) -> str:
    """`strong` (may start an entity) · `weak` (may only extend one) · `none`.

    The not-a-name check runs FIRST: `USD` and `PDF` are acronyms, so asking
    `_is_intrinsically_named` before the generic list would claim them as names, and no
    threshold could ever undo that.
    """
    folded = fold(token)
    if _is_not_a_name(folded):
        return "none"
    if _is_intrinsically_named(token):
        return "strong"
    if _is_abbreviation(token, folded):
        return "weak"
    return "strong" if token[0].isupper() and len(token) > 1 else "none"


@dataclass(frozen=True)
class _Candidate:
    """A candidate entity plus the two things position tells us about it."""

    text: str
    confident: bool
    # "Muestra Claude Code…" and "Claude Code ejecuta…" are structurally IDENTICAL: a
    # sentence-opening capital followed by a name. Only a verb lexicon could tell them
    # apart — so we do not guess. We mark the head as droppable and let the EVIDENCE
    # decide: if the tail alone is grounded, the head was a verb and nothing is flagged.
    head_droppable: bool


def _grade(run: list[str], start: int) -> _Candidate:
    """Grade a run by its WORD position: only word 0 sits where a capital proves nothing."""
    opener = start == 0 and not _is_intrinsically_named(run[0])
    confident = not opener or len(run) > 1
    return _Candidate(" ".join(run), confident, opener and len(run) > 1)


def _entities_in_sentence(sentence: str) -> list[_Candidate]:
    """Group entity tokens in one sentence, tracking each run's WORD index.

    Internal punctuation (`|`) ends a run without resetting the word count: only word 0
    of the sentence sits in the position where a capital proves nothing.
    """
    entities: list[_Candidate] = []
    run: list[str] = []
    start = 0
    for index, raw in enumerate(_BREAK.sub(" | ", sentence).split()):
        # A quote is a delimiter, not a letter. Without stripping it, the digest's
        # 'ChatGPT' arrives as `ChatGPT'` and never matches the source's ChatGPT — a
        # false positive the checker invented. Internal apostrophes stay (O'Brien).
        token = raw.strip("'’\"")
        kind = "none" if not token or token == "|" else _classify(token)
        if kind == "strong" or (kind == "weak" and run):
            start = index if not run else start
            run.append(token)
            continue
        if run:
            entities.append(_grade(run, start))
            run = []
    if run:
        entities.append(_grade(run, start))
    return entities


def _candidates(text: str) -> list[_Candidate]:
    """Every candidate named entity in `text`, each tagged confident/uncertain."""
    plain = _BULLET.sub("", _MARKDOWN.sub(" ", text))
    found: list[_Candidate] = []
    for sentence in _SENTENCE.split(plain):
        found += _entities_in_sentence(sentence)
    return list(dict.fromkeys(found))


def extract_entities(text: str) -> list[str]:
    """The CONFIDENT named entities in `text`, in order, de-duplicated.

    Adjacent capitalised tokens group into one entity, but never across punctuation — so
    "Dario Amodei (Anthropic)" yields TWO candidates and the grounded name can be told
    apart from the ungrounded affiliation.
    """
    return [c.text for c in _candidates(text) if c.confident]


def extract_uncertain(text: str) -> list[str]:
    """Candidates whose only claim to being a name is a line-initial capital.

    NOT dropped — reported in their own bucket. Silently discarding them would be a
    hidden recall cap, and this module exists because a hidden recall cap is what the
    judge ensemble already had. Kept out of the headline because ~90% of them are Spanish
    bullet-openers, and a report nobody can read is its own kind of false negative.
    """
    return [c.text for c in _candidates(text) if not c.confident]


def _candidate_grounded(candidate: _Candidate, evidence: str) -> bool:
    """Grounded when the candidate matches — or, for a sentence opener, when its TAIL does.

    A grounded tail means the leading capital was a verb ("Muestra Claude Code" where the
    source says "Claude Code"), not part of the name. The evidence settles what no heuristic
    could, so nothing is invented and nothing is flagged.
    """
    if is_grounded(candidate.text, evidence):
        return True
    tail = candidate.text.split(" ", 1)
    return candidate.head_droppable and is_grounded(tail[1], evidence)


def ungrounded_entities(item: Item, target: str) -> list[str]:
    """Confident entities named in the item's `target` output that no surface supports."""
    return _ungrounded(item, target, confident=True)


def uncertain_entities(item: Item, target: str) -> list[str]:
    """The line-initial-capital candidates that no surface supports (the review bucket)."""
    return _ungrounded(item, target, confident=False)


# A URL — bare or schemed — inside an evidence surface. It is CONTENT the surface really
# carries (1,281 of the 2,168 items have one inside their own tweet text, and `[Tweet]` is
# the post's words verbatim), but it grounds NOTHING: a domain is topic signal, never a
# name. Both rubrics say so; only this module does the substring matching, so only this
# module can enforce it.
_URL = re.compile(
    r"""(?xi)
    \b(?:https?://|www\.)\S+          # a schemed or www URL
    | \b[\w-]+(?:\.[\w-]+)*\.         # or a bare host: labels, then a TLD…
      (?:com|org|net|io|ai|co|dev|me|app|xyz|edu|gov|news|blog|so|to|ly|be|tv)
      \b(?:/\S*)?                     # …plus any path
    """
)


def strip_urls(text: str) -> str:
    """Remove every URL from an evidence blob, leaving the words around it intact.

    THE HOLE THIS CLOSES. The checker asks "does this name appear anywhere in the
    evidence?" by substring. A URL rides into the evidence with the tweet's own words —
    and `axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic` contains the
    literal strings "axios" and "anthropic". So the naive search GROUNDS "Axios" and
    "Anthropic" in the slug of a link that was never fetched, and reports the summary
    that invented them as clean.

    That is not a hypothetical: it is the exact defect this whole workstream started
    from. `xbrain.evidence` states the invariant (no surface is derived from `item.links`)
    and is careful to say what it does NOT mean — a URL inside the tweet's own text does
    ride in. Enforcing it belongs HERE, at the only place a name is matched against a
    blob.

    A space replaces the URL rather than nothing, so the words on either side stay
    separate tokens ("Anthropic" beside a link is still grounded; it is the SLUG that
    stops grounding).
    """
    return _URL.sub(" ", text)


def _ungrounded(item: Item, target: str, *, confident: bool) -> list[str]:
    """Candidates of the requested confidence that no evidence surface supports."""
    checked = _as_target(target)
    output = _output_for(item, checked)
    if not output:
        return []
    evidence = strip_urls(evidence_text(item, checked))
    return [
        c.text
        for c in _candidates(output)
        if c.confident is confident and not _candidate_grounded(c, evidence)
    ]


@dataclass(frozen=True)
class EntityRecord:
    """One output carrying at least one ungrounded entity.

    `uncertain` is carried alongside — never dropped — so the report states what it set
    aside instead of quietly capping its own recall.
    """

    item_id: str
    target: str
    handle: str
    ungrounded: list[str]
    uncertain: list[str]


def _as_target(target: str) -> VerifyTarget:
    """Validate a target, loudly. A target that cannot be scanned must never return zero
    findings and read as a clean bill of health."""
    if target not in ALL_TARGETS:
        raise ValueError(f"unknown target {target!r} — expected one of {', '.join(ALL_TARGETS)}")
    if target == "topics":
        # Topics are lowercase kebab slugs (`ai-in-science`), so the extractor yields ZERO
        # candidates for all 2,168 outputs — a confident clean bill of health from a check
        # structurally incapable of finding anything. Grounding topics means checking them
        # against the induced vocab, which is a different instrument.
        raise ValueError(
            "target 'topics' cannot be entity-scanned: topic slugs are lowercase kebab-case, "
            "so no named entity is ever extracted and the result is a vacuous PASS. "
            "Check topics against the induced vocab instead."
        )
    return target  # type: ignore[return-value]


def scan_store(store: dict[str, Item], target: str = "digest") -> list[EntityRecord]:
    """Sweep every item's `target` output for ungrounded entities.

    Costs no tokens, so the whole corpus is swept — there is no sampling error to reason
    about, and no reason to prefer a sample.
    """
    _as_target(target)
    records = []
    for item in store.values():
        ungrounded = ungrounded_entities(item, target)
        uncertain = uncertain_entities(item, target)
        # A record whenever EITHER bucket has something. `if ungrounded:` dropped every
        # output whose only ungrounded candidates were uncertain — 71 digests and 760
        # summaries, silently. The docstring promised these were never discarded; the code
        # discarded them. This module exists because a hidden recall cap is what the judge
        # ensemble had, and it had quietly grown one of its own.
        if ungrounded or uncertain:
            records.append(EntityRecord(item.id, target, item.author.handle, ungrounded, uncertain))
    return records


def outputs_present(store: dict[str, Item], target: str) -> int:
    """How many items actually HAVE a `target` output — the true denominator.

    Reporting against the whole store would silently shrink the flagged rate by counting
    the 1,973 items that were never digested at all.
    """
    return sum(1 for item in store.values() if _output_for(item, _as_target(target)))


def gate_failures(records: list[EntityRecord]) -> list[EntityRecord]:
    """The records a FORWARD GATE should fail on: the CONFIDENT bucket only.

    Sizing, measured: ingest is ~84 items/week; the confident flag rate is ~12%, so ~10
    flags/week — roughly 3 real and 7 false alarms, and ~2 false alarms once the lexicon and
    the generic-term list are in. That is a payable tax for a deterministic, token-free
    check.

    The `uncertain` bucket (~46 items/week) stays OUT. It is reported — never silently
    dropped — but in a gate it would bury the 3 real findings under 50 Spanish line-openers,
    and a gate nobody reads is a gate that does not exist.

    AND IT IS NOT A HALLUCINATION GATE. Invented names are ~26% of real defects; the largest
    class — the generator describing content it was never shown (unfetched links, unshipped
    quoted tweets) — is ~46% and is invisible to this instrument BY CONSTRUCTION. Called what
    it is (a cheap deterministic check for names that appear on no evidence surface) it earns
    its place. Called a hallucination gate, it manufactures exactly the false confidence this
    whole effort exists to destroy.
    """
    return [record for record in records if record.ungrounded]


def load_ensemble_verdicts(path: Path, target: str) -> dict[str, dict]:
    """Index a `verify-report.json`'s records for `target` by item id.

    Used to answer the question the ensemble cannot answer about itself: of the outputs
    this check flags, how many did the judges pass UNANIMOUSLY? That count is the
    ensemble's false-negative floor.
    """
    if not path.exists():
        # Returning {} printed "0 with a UNANIMOUS PASS", which reads as *the judges caught
        # everything* when the file was simply never found. A silent zero is the worst
        # possible failure for a report whose whole subject is silent zeros.
        raise FileNotFoundError(f"verdicts report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {r["item_id"]: r for r in payload.get("records", []) if r.get("target") == target}


def _is_unanimous_pass(record: dict) -> bool:
    """A verdict the ensemble agreed on with no dissent — the invisible-failure class."""
    return record.get("verdict") == "PASS" and not record.get("divergent")


def summarise_scan(records: list[EntityRecord], verdicts: dict[str, dict]) -> dict:
    """The headline numbers, including the one the ensemble cannot see about itself."""
    records = [r for r in records if r.ungrounded]  # headline counts CONFIDENT flags only
    judged = [r for r in records if r.item_id in verdicts]
    missed = [r for r in judged if _is_unanimous_pass(verdicts[r.item_id])]
    return {
        "flagged": len(records),
        "entities": sum(len(r.ungrounded) for r in records),
        "judged_by_ensemble": len(judged),
        "unanimous_pass_but_ungrounded": len(missed),
        # NOT "false negatives": at the precision this check has been measured at, most of
        # these are its own false alarms. The name stays neutral so no reader can quote it
        # as a recall claim about the judges — that is exactly the error already made once.
        "unanimous_pass_ids": [r.item_id for r in missed],
        "false_negative_ids": [r.item_id for r in missed],
    }


def render_entity_report(
    records: list[EntityRecord], summary: dict, scanned: int, target: str
) -> tuple[str, str]:
    """Render `(json_report, markdown_report)`."""
    payload = {
        "target": target,
        "scanned": scanned,
        "summary": summary,
        "records": [vars(r) for r in records],
    }
    lines = [
        f"# Entity grounding — `{target}`",
        "",
        f"**{summary['flagged']}/{scanned}** outputs carry at least one entity this check "
        f"could not find on the evidence ({summary['entities']} entities). "
        f"{summary['unanimous_pass_but_ungrounded']} of them were passed unanimously by the "
        "judges.",
        "",
        "> **These are CANDIDATES, not confirmed hallucinations, and this table supports no "
        "claim about the judges' recall.** An earlier version of this check reported ~0% "
        "precision on digests: the evidence is ASR output, it mangles proper nouns, and the "
        "check was flagging the names the generator had correctly recovered. Matching is now "
        "variant-aware, but precision has NOT been independently re-measured — treat every "
        "row as a lead to verify, never as a finding.",
        "",
        "> **What this check is blind to:** it verifies that proper nouns appear somewhere on "
        "the evidence. It never checks whether anything asserted *about* them is true, and it "
        "never looks at a single number. A clean verdict means *no unknown proper nouns* — it "
        "does NOT mean *not hallucinated*.",
        "",
        "| item | author | ungrounded candidates | judges |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r.item_id}` | @{r.handle} | {', '.join(r.ungrounded)} | "
        f"{'unanimous PASS' if r.item_id in summary['false_negative_ids'] else '—'} |"
        for r in records
        if r.ungrounded
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False), "\n".join(lines) + "\n"

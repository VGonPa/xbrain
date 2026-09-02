"""Human output, from the SAME response model `--json` serialises (spec §7.6).

ONE MODEL, TWO RENDERINGS — never two shapes. Spec §7.6: *el CLI sin `--json` puede renderizar
una tabla o bloques legibles, pero consume el mismo modelo de respuesta.* So every function
here takes a `SearchResponse`, an `EvidenceBundle` or a report and produces text; none of them
reaches back into the store, the index or the services. A renderer that re-derived anything
would be a third definition of what a result is, after the service and the JSON.

WHAT THE HUMAN VIEW MUST ALWAYS SHOW (spec §7.6, verbatim): item, author, date and URL;
surface and origin; excerpt; the channels that produced the match; staleness and truncation
warnings; and how to obtain the source with `xbrain get`.

The last one is CLAUDE.md rule 7 in the cheapest possible form. Before building an instrument
to detect a defect, ask whether SHOWING the evidence makes it self-evident — and a line that
names the exact `xbrain get` command is what turns "this summary says X" into "here is how to
read what it summarised", in one copy-paste.

THIS IS ALSO WHERE `no_underlying_source` BECOMES A WORD. The frozen `SearchResult` has no
`warnings` field (Plan 01 froze it at `schema_version: "1"` and Plan 02 §0 forbids amending
the contract here), so the JSON says it structurally — `verify_with: []` is reachable in
exactly one case — and the human view says it in Spanish. The gap for a JSON consumer is
recorded in the execution report as a finding for Plan 01, not closed by a quiet field.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence

from xbrain.knowledge.contracts import (
    FALLBACK_STRATEGY,
    NOT_IMPLEMENTED_SUFFIX,
    EvidenceBundle,
    SearchResponse,
    SearchResult,
)
from xbrain.knowledge.index_build import BuildReport, StatusReport, UpdateReport
from xbrain.knowledge.search_service import no_underlying_source
from xbrain.models import Author

# What each degradation MEANS and what to do about it. A flag a reader has to look up is a
# flag they will ignore; spec §9.3 asks the response to NAME the command that fixes it.
DEGRADED_TEXT: dict[str, str] = {
    "index_behind_store": (
        "⚠ El índice va por detrás del store: `items.json`, `vocab.yaml` o `topics.json` "
        "cambió después de construirlo. La evidencia puede estar obsoleta — actualiza con "
        "`xbrain index update`."
    ),
    "no_embeddings": (
        "· Estrategia léxica (sin embeddings): recupera nombres propios, cifras y frases "
        "exactas, no similitud conceptual."
    ),
}


def render_search(response: SearchResponse) -> str:
    """The human rendering of a search (spec §7.6)."""
    lines: list[str] = [f'"{response.query}" · estrategia {response.strategy}']
    lines += _index_lines(response)
    if not response.results:
        lines.append("Sin resultados. Prueba con menos filtros o términos más distintivos.")
        return "\n".join(lines)
    for result in response.results:
        lines.append("")
        lines += _result_lines(result)
    return "\n".join(lines)


def _index_lines(response: SearchResponse) -> list[str]:
    """What the index says about itself — the warnings, before the results.

    Before, not after: a reader who stops at the first result must already have seen that the
    evidence may be stale. A warning under the fold is a warning nobody read.
    """
    lines = [_degraded_line(flag) for flag in response.index.degraded]
    if response.index.corrupt_chunks_excluded:
        lines.append(
            f"⚠ {response.index.corrupt_chunks_excluded} chunk(s) excluido(s): su fingerprint "
            "no cuadra con su texto. Reconstruye con `xbrain index build --force`."
        )
    if response.truncated:
        lines.append("⚠ Resultado truncado.")
    return lines


def _degraded_line(flag: str) -> str:
    """One degradation as a sentence. The strategy family is computed, not tabulated.

    `<requested>_not_implemented` is one flag per declared-but-unimplemented `Strategy`, so a
    literal table here would be a second copy of the enum that goes stale the day Plan 03
    implements one of them (rule 5). Falling through to the bare `⚠ vector_not_implemented`
    would also have left the human view saying less than the JSON, which is the surface a
    reader actually reads.
    """
    if flag.endswith(NOT_IMPLEMENTED_SUFFIX):
        requested = flag[: -len(NOT_IMPLEMENTED_SUFFIX)]
        return (
            f"⚠ La estrategia `{requested}` no tiene backend todavía: ha respondido "
            f"`{FALLBACK_STRATEGY}`. Estos resultados NO son de `{requested}`."
        )
    return DEGRADED_TEXT.get(flag, f"⚠ {flag}")


def _result_lines(result: SearchResult) -> list[str]:
    """One item: metadata, the labelled summary, every match, and how to verify it."""
    lines = [
        f"{result.rank}. {result.item_id}  {_author_label(result.author)}"
        f" · {result.created_at.date().isoformat()}",
        f"   {result.url}",
    ]
    if result.topics:
        lines.append(f"   topics: {', '.join(result.topics)}")
    if result.summary is not None:
        verdict = (
            f" [{result.summary.verification_status}]" if result.summary.verification_status else ""
        )
        lines.append(
            f"   resumen ({result.summary.origin}{verdict}): {_one_line(result.summary.text)}"
        )
    for match in result.matches:
        lines.append(
            f"   · [{match.surface_type}] origin={match.origin} trust={match.trust_class}"
            f" · via {'+'.join(match.matched_by)}"
        )
        if label := _own_author(match.attribution, result.author):
            # The surface's own author, when it is not the item's (A-1): a quoted post is
            # somebody else's words, and the cheapest guard against reading them as the
            # poster's is to say whose they are next to the excerpt (CLAUDE.md rule 7).
            lines.append(f"     {label}")
        lines.append(f"     {_one_line(match.excerpt)}")
    if not result.matches:
        # A profile-only candidate. Saying so matters: the profile is a composed string
        # nobody wrote, so there is no excerpt to show and none may be invented (spec §5.1.A).
        lines.append("   · coincide por el perfil del item; no hay fragmento citable")
    lines += _verify_lines(result)
    return lines


def _verify_lines(result: SearchResult) -> list[str]:
    """How to reach the source — or the statement that there is none (spec §3.5).

    `no_underlying_source` is the only case where `verify_with` is empty, so the branch is
    exhaustive: either there is a command to run, or there is a warning to read.
    """
    if no_underlying_source(result):
        return [
            "   ⚠ no_underlying_source: la coincidencia es texto derivado y este item no "
            "conserva ninguna fuente primaria que la sustente."
        ]
    if not result.verify_with:
        return []
    surfaces = " ".join(f"--surface {name}" for name in result.verify_with)
    return [f"   → verifica con: xbrain get {result.item_id} {surfaces}"]


def render_get(
    bundle: EvidenceBundle,
    *,
    surfaces: Sequence[str] = (),
    query: str | None = None,
) -> str:
    """The human rendering of an evidence bundle (spec §7.3, §7.6).

    THIS VIEW IS FOR A HUMAN, NOT FOR AN AGENT. The surface for agents is `--json`, where
    `origin` and `trust_class` travel as siblings of every text and nothing can be confused
    with the frame (spec §10.4). Here the frame is text too, so the body is FENCED (G-7):
    every line of a surface or chunk is prefixed with `│ `, and a title is collapsed to one
    line, so a quoted post that carries `[user_note] origin=user trust=user_text` on a line
    of its own — a forged header, byte-identical to the renderer's — stays visibly inside
    the body instead of standing where a header stands. The text is still shown whole; it is
    evidence. It just cannot impersonate the label above it.

    `surfaces` and `query` are the REQUEST the bundle answers, and they are here for one
    line: the continuation (H2). The cursor is an offset into a sequence — the chunks of the
    selected surfaces in emitter order, or their ranking for a query — and the frozen bundle
    does not carry what defined that sequence. A continuation printed without them resumed
    inside the default selection and returned an empty page, or was refused by the cursor
    decoder: a truncation whose only published continuation cannot be run is the silent cut
    spec §9.3 forbids. The budget is not echoed: it bounds a page, it does not define the
    sequence, and the pages stay disjoint and complete under any budget.
    """
    item = bundle.item
    lines = [
        f"{item.item_id}  {_author_label(item.author)}"
        f" · {item.created_at.date().isoformat()} · {item.source}",
        f"  {item.url}",
        f"  topics: {', '.join(item.topics) or '—'}",
        f"  superficies disponibles: {', '.join(item.available_surfaces) or '—'}",
    ]
    for target, verdict in sorted(bundle.verification.items()):
        lines.append(f"  verificación {target}: {verdict.verdict}")
    lines += _failure_lines(bundle)
    for surface in bundle.surfaces:
        lines += [
            "",
            f"[{surface.surface_type}] origin={surface.origin} trust={surface.trust_class}"
            + (f" · {_one_line(surface.title)}" if surface.title else "")
            + _author_suffix(surface.attribution, item.author),
            *_fenced(surface.text),
        ]
    for chunk in bundle.chunks:
        # The chunk's OWN author on its header (H3): a quoted post paginated or prioritised
        # by `--query` arrives here as a chunk, and this header is the only line between it
        # and the bundle header that names the poster.
        lines += [
            "",
            f"[{chunk.surface_type} {chunk.char_start}:{chunk.char_end}] origin={chunk.origin}"
            + _author_suffix(chunk.attribution, item.author),
            *_fenced(chunk.text),
        ]
    if bundle.truncated:
        lines += [
            "",
            "⚠ Truncado. Continúa con: " + _continuation(item.item_id, bundle, surfaces, query),
        ]
    return "\n".join(lines)


def _continuation(
    item_id: str, bundle: EvidenceBundle, surfaces: Sequence[str], query: str | None
) -> str:
    """The `xbrain get` that resumes THIS sequence at `bundle.cursor`, runnable as printed.

    Every `--surface` repeated in the order given, `--query` shell-quoted (a query is free
    text and may carry spaces or quotes), then the cursor. Nothing else: what is not in
    this line does not change which chunk comes next.
    """
    parts = [f"xbrain get {item_id}"]
    parts += [f"--surface {name}" for name in surfaces]
    if query:
        parts.append(f"--query {shlex.quote(query)}")
    parts.append(f"--cursor {bundle.cursor}")
    return " ".join(parts)


def _own_author(attribution: Author | None, item_author: Author) -> str:
    """`autor: @handle (Name)` when the text has an author of its own, else ``.

    ONE rule for a search match, a whole surface and a chunk (A-1, H3): the text's author is
    named whenever it is not the item's. Spec §3.7.3 — a quoted tweet keeps its own author —
    is enforceable in the human view only if every place that shows a body applies the same
    test; the chunk branch of `render_get` did not, and the surface branch had a different
    one (`surface_type == "quoted_post"`), so a quoted post that arrived as a CHUNK — paged,
    or prioritised by `--query` — sat under the poster's name with nothing saying otherwise.
    """
    if attribution is None or attribution == item_author:
        return ""
    return f"autor: {_author_label(attribution)}"


def _author_label(author: Author) -> str:
    """`@handle (Name)` as ONE printable line — the only way an author reaches a human (D-3).

    A name and a handle come from X verbatim and are UNTRUSTED like a body is: a newline in
    `author.name` printed, at column 0, a line byte-identical to a renderer header and
    another byte-identical to a fence line — the forge G-7 exists to stop, through the field
    next to the body that nobody fenced — and an `ESC[2K` in a handle reached the TTY through
    the result line (the round-06 gate, D-3). Population today 0 of 2,404, and the
    population is what X accepts, not what the corpus holds. Three placements — the bundle
    header, the result line, the `autor:` label of a match, a surface or a chunk — and one
    function, pinned by a test that swaps it for a sentinel and expects the sentinel in all
    three (rule 5).
    """
    return f"@{_one_line(author.handle)} ({_one_line(author.name)})"


def _author_suffix(attribution: Author | None, item_author: Author) -> str:
    """The `_own_author` label as a header suffix, or `` — the `get` placement of the rule."""
    label = _own_author(attribution, item_author)
    return f" · {label}" if label else ""


def _failure_lines(bundle: EvidenceBundle) -> list[str]:
    """Failed fetches and bodiless links — STATE, never a silence (spec §4, m7).

    Two loops because they are two facts: one records an attempt that failed, the other
    records that there is no body and why. Merging them would make a link nobody tried
    indistinguishable from one that returned a 404.
    """
    lines = []
    for failure in bundle.failures:
        lines.append(f"  ⚠ fetch falló: {failure.kind} {failure.url} ({failure.failure_reason})")
    for link in bundle.unfetched_links:
        detail = f" — {link.detail}" if link.detail else ""
        lines.append(f"  ⚠ sin cuerpo: {link.url} ({link.reason}){detail}")
    return lines


def render_status(report: StatusReport) -> str:
    """`index status` for a human (acceptance 2)."""
    if report.manifest is None:
        return "\n".join(["Índice: INCOMPLETO o inexistente.", f"  {report.advice}"])
    manifest = report.manifest
    lines = [
        f"Índice construido {manifest.built_at.isoformat(timespec='seconds')}",
        f"  esquema {manifest.schema_version}"
        f" · emisor {manifest.surface_version}"
        f" · chunker {manifest.chunker_version} {manifest.chunker_params}",
        f"  tokenizer {manifest.tokenize!r} · conectiva {manifest.connective}"
        f" · embeddings {manifest.embeddings or '—'}",
        "  contenidos: "
        + " · ".join(f"{name} {count}" for name, count in sorted(report.counts.items())),
        "  omitidos: "
        + " · ".join(f"{name} {count}" for name, count in sorted(manifest.skipped.items())),
    ]
    lines.append(
        f"  respecto al store: +{report.items_added} nuevos"
        f" · {report.items_changed} cambiados"
        f" · -{report.items_removed} borrados"
        f" · {report.topics_changed} topics con miembros desfasados"
        f" · señal barata {'DESFASADA' if report.behind else 'al día'}"
    )
    if report.incomplete:
        # A manifest exists and the code cannot use it, or the base does not hold what it
        # declares (C-3): said BEFORE the advice, so the header above is not read as health.
        lines.append("  ⚠ Índice INCOMPLETO o inutilizable: ninguna consulta lo usará.")
    if report.advice:
        lines.append(f"  → {report.advice}")
    return "\n".join(lines)


def render_build(report: BuildReport) -> str:
    """`index build` for a human. COUNTS, never text (spec §10.8, §12.7)."""
    prefix = "[dry-run] " if report.dry_run else ""
    lines = [
        f"{prefix}{report.items_written} items · {report.topics_written} topics"
        f" · {report.surfaces_written} superficies · {report.chunks_written} chunks"
        f" · {report.profiles_written} perfiles",
        "  omitidos: " + " · ".join(f"{k} {v}" for k, v in sorted(report.skipped.items())),
        f"  {report.duration_seconds:.1f}s",
    ]
    if report.failed:
        lines.append(f"  ⚠ {len(report.failed)} item(s) fallaron")
    return "\n".join(lines)


def render_update(report: UpdateReport) -> str:
    """`index update` for a human."""
    prefix = "[dry-run] " if report.dry_run else ""
    return "\n".join(
        [
            f"{prefix}+{report.items_added} nuevos · {report.items_changed} cambiados"
            f" · -{report.items_removed} borrados"
            + (" · topics y perfiles recalculados" if report.topics_rebuilt else "")
            + (
                f" · {report.topics_refreshed} topics con miembros recalculados"
                if report.topics_refreshed
                else ""
            ),
            f"  chunks +{report.chunks_inserted} / -{report.chunks_deleted}"
            f" · perfiles +{report.profiles_inserted} / -{report.profiles_deleted}",
            f"  {report.duration_seconds:.1f}s",
        ]
    )


# Every C0 control except TAB and LF, plus DEL and the C1 range. What a terminal would
# INTERPRET rather than print: `ESC` opens every ANSI sequence, BEL rings, U+009B is CSI on a
# terminal that honours 8-bit controls. TAB is kept because a body may be indented; LF is the
# line structure `_fenced` turns into fence lines and `_one_line` collapses. The other line
# separators (`\r`, `\v`, `\f`, NEL, U+2028…) never reach this pattern as text: `splitlines`
# and `split` consume them first.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _printable(text: str) -> str:
    """The text with every control a terminal would act on removed (M-3).

    This is output SAFETY, not cosmetics. The fence of G-7 exists so a reader can tell the
    renderer's frame from an untrusted body — and an `ESC[2K` (erase line) or `ESC[1A`
    (cursor up) stored in a tweet reached the terminal intact under a pseudo-TTY through
    `get` and through the excerpt of `search`, so the body could erase the very fence and
    header that frame it (BEL passed even through a pipe). The characters are dropped, not
    escaped: a body is evidence and is shown whole, but what it may not do is drive the
    terminal.
    """
    return _CONTROL_CHARACTERS.sub("", text)


def _fenced(text: str) -> list[str]:
    """A body as lines that cannot stand where a header stands (G-7) — nor erase it (M-3).

    `│ ` in front of every line — including an empty one, so a paragraph break inside the
    body is still visibly inside it. Nothing is truncated or reflowed: the fence and the
    removal of terminal controls are the whole transformation.
    """
    return [f"│ {_printable(line)}" for line in text.splitlines()] or ["│ "]


def _one_line(text: str, width: int = 160) -> str:
    """Collapse a body to one readable line, with no control a terminal would act on (M-3).

    A hard cap, because an excerpt is already bounded but a summary is not, and a terminal
    rendering that wraps a 700-character paragraph across nine lines buries every other
    result on the screen.
    """
    flat = " ".join(_printable(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def render_lines(values: Sequence[str]) -> str:
    """Join pre-built lines. Exists so the CLI never has to know the separator."""
    return "\n".join(values)

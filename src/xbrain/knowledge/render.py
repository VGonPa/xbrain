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

# What each degradation MEANS and what to do about it. A flag a reader has to look up is a
# flag they will ignore; spec §9.3 asks the response to NAME the command that fixes it.
DEGRADED_TEXT: dict[str, str] = {
    "index_behind_store": (
        "⚠ El índice va por detrás del store: `data/items.json` cambió después de "
        "construirlo. La evidencia puede estar obsoleta — actualiza con `xbrain index update`."
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
        f"{result.rank}. {result.item_id}  @{result.author.handle} ({result.author.name})"
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
        if match.attribution is not None and match.attribution != result.author:
            # The surface's own author, when it is not the item's (A-1): a quoted post is
            # somebody else's words, and the cheapest guard against reading them as the
            # poster's is to say whose they are next to the excerpt (CLAUDE.md rule 7).
            lines.append(f"     autor: @{match.attribution.handle} ({match.attribution.name})")
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


def render_get(bundle: EvidenceBundle) -> str:
    """The human rendering of an evidence bundle (spec §7.3, §7.6).

    THIS VIEW IS FOR A HUMAN, NOT FOR AN AGENT. The surface for agents is `--json`, where
    `origin` and `trust_class` travel as siblings of every text and nothing can be confused
    with the frame (spec §10.4). Here the frame is text too, so the body is FENCED (G-7):
    every line of a surface or chunk is prefixed with `│ `, and a title is collapsed to one
    line, so a quoted post that carries `[user_note] origin=user trust=user_text` on a line
    of its own — a forged header, byte-identical to the renderer's — stays visibly inside
    the body instead of standing where a header stands. The text is still shown whole; it is
    evidence. It just cannot impersonate the label above it.
    """
    item = bundle.item
    lines = [
        f"{item.item_id}  @{item.author.handle} ({item.author.name})"
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
            + (
                f" · @{surface.attribution.handle}"
                if surface.attribution and surface.surface_type == "quoted_post"
                else ""
            ),
            *_fenced(surface.text),
        ]
    for chunk in bundle.chunks:
        lines += [
            "",
            f"[{chunk.surface_type} {chunk.char_start}:{chunk.char_end}] origin={chunk.origin}",
            *_fenced(chunk.text),
        ]
    if bundle.truncated:
        lines += [
            "",
            f"⚠ Truncado. Continúa con: xbrain get {item.item_id} --cursor {bundle.cursor}",
        ]
    return "\n".join(lines)


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


def _fenced(text: str) -> list[str]:
    """A body as lines that cannot stand where a header stands (G-7).

    `│ ` in front of every line — including an empty one, so a paragraph break inside the
    body is still visibly inside it. Nothing is truncated or reflowed: the fence is the whole
    transformation, and it is reversible by eye.
    """
    return [f"│ {line}" for line in text.splitlines()] or ["│ "]


def _one_line(text: str, width: int = 160) -> str:
    """Collapse a body to one readable line.

    A hard cap, because an excerpt is already bounded but a summary is not, and a terminal
    rendering that wraps a 700-character paragraph across nine lines buries every other
    result on the screen.
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def render_lines(values: Sequence[str]) -> str:
    """Join pre-built lines. Exists so the CLI never has to know the separator."""
    return "\n".join(values)

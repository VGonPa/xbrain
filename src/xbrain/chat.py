"""Dashboard chat over the generated XBrain markdown vault."""

from __future__ import annotations

import math
import re
import textwrap
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xbrain.llm_client import LlmProvider, build_llm_client
from xbrain.llm_json import json_from_response
from xbrain.notes_io import GEN_END, GEN_START

MAX_QUESTION_CHARS = 1200
MAX_SOURCES = 6
MAX_RETRIEVAL_CANDIDATES = 24
CHUNK_CHAR_LIMIT = 1800
SYNTHESIS_SOURCE_CHAR_LIMIT = 900
EVIDENCE_NOTE_CHAR_LIMIT = 700
SYNTHESIS_MAX_TOKENS = 1200
ANSWER_MAX_TOKENS = 900

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]{2,}")
_STOPWORDS = frozenset(
    {
        "aqui",
        "about",
        "algo",
        "also",
        "and",
        "as",
        "are",
        "by",
        "como",
        "con",
        "cual",
        "cuales",
        "de",
        "del",
        "des",
        "desde",
        "el",
        "donde",
        "els",
        "en",
        "es",
        "esta",
        "estan",
        "este",
        "esto",
        "for",
        "hay",
        "in",
        "is",
        "it",
        "la",
        "las",
        "lo",
        "los",
        "of",
        "on",
        "or",
        "para",
        "per",
        "por",
        "que",
        "the",
        "to",
        "un",
        "una",
        "what",
        "with",
    }
)

_SEMANTIC_CONCEPTS = {
    "ai": {
        "ai",
        "ia",
        "artificial intelligence",
        "inteligencia artificial",
        "llm",
        "llms",
        "model",
        "models",
        "modelo",
        "modelos",
        "genai",
    },
    "agents": {
        "agent",
        "agents",
        "agente",
        "agentes",
        "agentic",
        "agentica",
        "agentico",
        "workflow",
        "workflows",
        "orchestration",
        "orquestacion",
        "multi-agent",
        "multiagente",
    },
    "business": {
        "business",
        "negocio",
        "negocios",
        "startup",
        "startups",
        "company",
        "empresa",
        "revenue",
        "ingresos",
        "sales",
        "ventas",
        "marketing",
        "ecommerce",
        "commerce",
    },
    "career": {
        "career",
        "carrera",
        "empleo",
        "job",
        "jobs",
        "skills",
        "habilidades",
        "engineer",
        "ingeniero",
        "ingenieria",
        "hireable",
        "promotion",
        "promocion",
    },
    "learning": {
        "learn",
        "learning",
        "aprender",
        "aprendizaje",
        "study",
        "estudio",
        "curso",
        "courses",
        "guide",
        "guia",
        "roadmap",
        "tutorial",
    },
    "productivity": {
        "productivity",
        "productividad",
        "workflow",
        "habits",
        "habitos",
        "system",
        "sistema",
        "automation",
        "automatizacion",
        "leverage",
        "apalancamiento",
    },
    "knowledge": {
        "knowledge",
        "conocimiento",
        "second brain",
        "segundo cerebro",
        "obsidian",
        "notes",
        "notas",
        "pkm",
        "library",
        "biblioteca",
    },
    "communication": {
        "communication",
        "comunicacion",
        "writing",
        "escritura",
        "persuasion",
        "storytelling",
        "copywriting",
        "content",
        "contenido",
    },
}


@dataclass(frozen=True)
class MarkdownChunk:
    """One searchable chunk from a generated markdown file."""

    title: str
    path: Path
    text: str
    index: int


@dataclass(frozen=True)
class ChatSource:
    """A source snippet sent to the LLM and returned to the dashboard."""

    id: str
    title: str
    path: Path
    excerpt: str
    score: float

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "path": str(self.path),
            "excerpt": self.excerpt,
            "score": round(self.score, 3),
        }


@dataclass(frozen=True)
class ChatAnswer:
    """Structured response returned by `/api/chat`."""

    answer: str
    sources: list[ChatSource]
    provider: LlmProvider
    model: str
    scanned_files: int
    retrieved_sources: int
    candidate_sources: int
    synthesis_used: bool
    retrieval_mode: str

    def to_payload(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "sources": [source.to_payload() for source in self.sources],
            "provider": self.provider,
            "model": self.model,
            "scanned_files": self.scanned_files,
            "retrieved_sources": self.retrieved_sources,
            "candidate_sources": self.candidate_sources,
            "synthesis_used": self.synthesis_used,
            "retrieval_mode": self.retrieval_mode,
        }


@dataclass(frozen=True)
class EvidenceNote:
    """One compact note distilled from a retrieved source candidate."""

    source_id: str
    note: str


def _generated_region(markdown: str) -> str:
    """Return only XBrain's generated block when markers are present."""
    start = markdown.find(GEN_START)
    if start == -1:
        return markdown
    start += len(GEN_START)
    end = markdown.find(GEN_END, start)
    if end == -1:
        return markdown[start:]
    return markdown[start:end]


def _plain_text(markdown: str) -> str:
    """Normalize markdown into searchable prose while keeping useful labels."""
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"!\[\[([^\]]+)\]\]", r" \1 ", text)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^---\s*$.*?^---\s*$", " ", text, flags=re.S | re.M)
    text = re.sub(r"[#>*_`~|-]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _stem_token(token: str) -> str:
    for suffix in (
        "ciones",
        "ments",
        "mente",
        "acion",
        "idad",
        "ing",
        "ed",
        "es",
        "s",
    ):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _title_for(path: Path, markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or path.stem
    return path.stem.replace("-", " ")


def _markdown_paths(output_dir: Path) -> list[Path]:
    """Return generated item/topic pages, excluding index/log noise."""
    paths: list[Path] = []
    for dirname in ("items", "topics"):
        directory = output_dir / dirname
        if directory.exists():
            paths.extend(sorted(directory.glob("*.md")))
    return paths


def _split_chunks(text: str, *, limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > limit:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for piece in textwrap.wrap(
                paragraph,
                width=limit,
                break_long_words=False,
                replace_whitespace=False,
            ):
                chunks.append(piece)
            continue
        next_len = current_len + len(paragraph) + (2 if current else 0)
        if current and next_len > limit:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = next_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def load_markdown_chunks(output_dir: Path) -> tuple[list[MarkdownChunk], int]:
    """Load searchable chunks from the generated markdown vault."""
    chunks: list[MarkdownChunk] = []
    paths = _markdown_paths(output_dir)
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        generated = _generated_region(raw)
        plain = _plain_text(generated)
        if not plain:
            continue
        title = _title_for(path, generated)
        for index, chunk in enumerate(_split_chunks(plain)):
            chunks.append(MarkdownChunk(title=title, path=path.resolve(), text=chunk, index=index))
    return chunks, len(paths)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0) for match in _WORD_RE.finditer(_normalize_text(text)))
        if token not in _STOPWORDS
    ]


def _lexical_score_chunk(query: Counter[str], chunk: MarkdownChunk) -> float:
    body = Counter(_tokens(chunk.text))
    title = Counter(_tokens(chunk.title))
    if not body and not title:
        return 0.0
    score = 0.0
    for token, weight in query.items():
        score += min(body.get(token, 0), 3) * weight
        score += min(title.get(token, 0), 2) * weight * 1.8
    return score


def _semantic_features(text: str) -> Counter[str]:
    """Build a small local concept vector for hybrid retrieval.

    This is deliberately dependency-free: it boosts bilingual/domain aliases,
    light stems and phrases from the generated corpus. It complements exact
    keyword search without adding a heavyweight embedding stack to the VPS.
    """
    normalized = _normalize_text(text)
    tokens = _tokens(normalized)
    features: Counter[str] = Counter()
    for token in tokens:
        features[f"tok:{token}"] += 1
        stem = _stem_token(token)
        if stem != token:
            features[f"stem:{stem}"] += 1
    token_set = set(tokens)
    for concept, aliases in _SEMANTIC_CONCEPTS.items():
        hits = 0
        for alias in aliases:
            normalized_alias = _normalize_text(alias)
            if " " in normalized_alias or "-" in normalized_alias:
                if normalized_alias in normalized:
                    hits += 1
            elif normalized_alias in token_set:
                hits += 1
        if hits:
            features[f"concept:{concept}"] += min(hits, 4) * 3
    return features


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    if dot <= 0:
        return 0.0
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _hybrid_score_chunk(
    query: Counter[str],
    query_vector: Counter[str],
    chunk: MarkdownChunk,
) -> float:
    lexical = _lexical_score_chunk(query, chunk)
    title_vector = _semantic_features(chunk.title)
    body_vector = _semantic_features(chunk.text)
    semantic = _cosine(query_vector, body_vector) + (_cosine(query_vector, title_vector) * 1.6)
    return lexical + semantic * 8.0


def retrieve_sources(
    output_dir: Path,
    question: str,
    *,
    limit: int = MAX_SOURCES,
    allowed_paths: set[Path] | None = None,
) -> tuple[list[ChatSource], int]:
    """Retrieve source snippets, optionally scoped to generated note paths."""
    query = Counter(_tokens(question))
    chunks, scanned_files = load_markdown_chunks(output_dir)
    if allowed_paths is not None:
        allowed = {path.resolve() for path in allowed_paths}
        chunks = [chunk for chunk in chunks if chunk.path in allowed]
    if not query or not chunks:
        return [], scanned_files
    query_vector = _semantic_features(question)

    scored = [
        (score, chunk)
        for chunk in chunks
        if (score := _hybrid_score_chunk(query, query_vector, chunk)) > 0
    ]
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].path), pair[1].index))

    sources: list[ChatSource] = []
    seen_paths: set[Path] = set()
    for score, chunk in scored:
        if chunk.path in seen_paths:
            continue
        seen_paths.add(chunk.path)
        sources.append(
            ChatSource(
                id=f"S{len(sources) + 1}",
                title=chunk.title,
                path=chunk.path,
                excerpt=chunk.text[:CHUNK_CHAR_LIMIT].strip(),
                score=score,
            )
        )
        if len(sources) >= limit:
            break
    return sources, scanned_files


def _synthesis_system_prompt() -> str:
    return (
        "Eres el paso de selección de evidencias de Ask XBrain. Tu tarea es comprimir "
        "fragmentos recuperados antes de la respuesta final.\n"
        "Reglas obligatorias:\n"
        "- Usa solo los fragmentos proporcionados.\n"
        "- Conserva detalles concretos que respondan a la pregunta.\n"
        "- Descarta fragmentos irrelevantes aunque compartan palabras clave.\n"
        "- Cada nota debe estar asociada a un único ID de fuente existente.\n"
        "- Devuelve un único objeto JSON: "
        '{"evidence":[{"source":"S1","note":"..."}],"missing":"..."}'
    )


def _synthesis_user_prompt(question: str, sources: list[ChatSource]) -> str:
    blocks = [
        f"Pregunta del usuario:\n{question}",
        "",
        (
            "Candidatos recuperados. Selecciona los que realmente aportan evidencia "
            "y resume cada uno en una nota compacta:"
        ),
    ]
    for source in sources:
        blocks += [
            "",
            f"[{source.id}] {source.title}",
            f"Path: {source.path}",
            "Contenido:",
            source.excerpt[:SYNTHESIS_SOURCE_CHAR_LIMIT].strip(),
        ]
    return "\n".join(blocks)


def _evidence_from_response(data: dict[str, Any], sources: list[ChatSource]) -> list[EvidenceNote]:
    raw = data.get("evidence", [])
    if not isinstance(raw, list):
        return []
    valid_ids = {source.id for source in sources}
    evidence: list[EvidenceNote] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        source_id = entry.get("source")
        note = entry.get("note")
        if not isinstance(source_id, str) or source_id not in valid_ids or source_id in seen:
            continue
        if not isinstance(note, str) or not note.strip():
            continue
        seen.add(source_id)
        evidence.append(
            EvidenceNote(source_id=source_id, note=note.strip()[:EVIDENCE_NOTE_CHAR_LIMIT])
        )
    return evidence


def _synthesize_evidence(
    client: Any,
    *,
    model: str,
    question: str,
    sources: list[ChatSource],
) -> list[EvidenceNote]:
    response = client.messages.create(
        model=model,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        system=_synthesis_system_prompt(),
        messages=[{"role": "user", "content": _synthesis_user_prompt(question, sources)}],
    )
    data = json_from_response(response, context="dashboard chat evidence synthesis")
    return _evidence_from_response(data, sources)


def _system_prompt() -> str:
    return (
        "Eres el chat de biblioteca de XBrain. Responde solo con la informacion incluida "
        "en las evidencias o fragmentos de markdown proporcionados.\n"
        "Reglas obligatorias:\n"
        "- No uses conocimiento externo ni supongas hechos que no aparezcan en el contexto.\n"
        "- Si el contexto no basta, dilo de forma clara y concreta.\n"
        "- Responde en el mismo idioma de la pregunta.\n"
        "- Cita los IDs de fuente relevantes dentro de la respuesta, como [S1] o [S2].\n"
        "- Prioriza respuestas accionables: conclusión breve, puntos clave y límites.\n"
        "- Devuelve un unico objeto JSON: "
        '{"answer":"...","sources":["S1","S2"]}'
    )


def _user_prompt(
    question: str,
    sources: list[ChatSource],
    evidence: list[EvidenceNote] | None = None,
) -> str:
    blocks = [f"Pregunta del usuario:\n{question}", "", "Fragmentos disponibles:"]
    if evidence:
        source_by_id = {source.id: source for source in sources}
        for note in evidence:
            source = source_by_id.get(note.source_id)
            if source is None:
                continue
            blocks += [
                "",
                f"[{source.id}] {source.title}",
                f"Path: {source.path}",
                "Evidencia sintetizada:",
                note.note,
            ]
    else:
        for source in sources:
            blocks += [
                "",
                f"[{source.id}] {source.title}",
                f"Path: {source.path}",
                "Contenido:",
                source.excerpt,
            ]
    return "\n".join(blocks)


def _selected_sources(data: dict[str, Any], sources: list[ChatSource]) -> list[ChatSource]:
    raw_ids = data.get("sources", [])
    if not isinstance(raw_ids, list):
        return []
    valid_ids = {str(source_id) for source_id in raw_ids}
    return [source for source in sources if source.id in valid_ids]


def answer_question(
    output_dir: Path,
    question: str,
    *,
    provider: LlmProvider,
    model: str,
    base_url: str | None = None,
    client: Any = None,
    max_sources: int = MAX_SOURCES,
    allowed_paths: set[Path] | None = None,
) -> ChatAnswer:
    """Answer one question using only generated markdown context."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ValueError(f"question is too long; max {MAX_QUESTION_CHARS} characters")

    candidates, scanned_files = retrieve_sources(
        output_dir,
        cleaned,
        limit=max(max_sources, MAX_RETRIEVAL_CANDIDATES),
        allowed_paths=allowed_paths,
    )
    if not candidates:
        return ChatAnswer(
            answer=(
                "No he encontrado informacion suficiente en los markdown generados de XBrain "
                "para responder a esa pregunta."
            ),
            sources=[],
            provider=provider,
            model=model,
            scanned_files=scanned_files,
            retrieved_sources=0,
            candidate_sources=0,
            synthesis_used=False,
            retrieval_mode="hybrid",
        )

    active_client = client or build_llm_client(provider, base_url=base_url)
    final_sources = candidates[:max_sources]
    evidence: list[EvidenceNote] | None = None
    synthesis_used = False
    if len(candidates) > max_sources:
        try:
            synthesized = _synthesize_evidence(
                active_client,
                model=model,
                question=cleaned,
                sources=candidates,
            )
        except Exception:  # noqa: BLE001 - evidence synthesis is an optional quality layer
            synthesized = []
        if synthesized:
            evidence = synthesized[:max_sources]
            source_by_id = {source.id: source for source in candidates}
            final_sources = [
                source_by_id[note.source_id] for note in evidence if note.source_id in source_by_id
            ]
            synthesis_used = bool(final_sources)
        if not final_sources:
            final_sources = candidates[:max_sources]
            evidence = None
            synthesis_used = False

    response = active_client.messages.create(
        model=model,
        max_tokens=ANSWER_MAX_TOKENS,
        system=_system_prompt(),
        messages=[{"role": "user", "content": _user_prompt(cleaned, final_sources, evidence)}],
    )
    data = json_from_response(response, context="dashboard chat")
    answer = str(data.get("answer", "")).strip()
    if not answer:
        raise ValueError("dashboard chat response has no answer")
    selected = _selected_sources(data, final_sources)
    if not selected and "no he encontrado" not in answer.lower():
        selected = final_sources[: min(3, len(final_sources))]
    return ChatAnswer(
        answer=answer,
        sources=selected,
        provider=provider,
        model=model,
        scanned_files=scanned_files,
        retrieved_sources=len(candidates),
        candidate_sources=len(candidates),
        synthesis_used=synthesis_used,
        retrieval_mode="hybrid",
    )

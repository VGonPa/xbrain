from xbrain.chat import answer_question, retrieve_sources
from xbrain.llm_client import TextBlock, TextResponse
from xbrain.notes_io import GEN_END, GEN_START


def _write_item(output_dir, name, body, tail=""):
    items = output_dir / "items"
    items.mkdir(parents=True, exist_ok=True)
    path = items / name
    path.write_text(f"{GEN_START}\n{body}\n{GEN_END}{tail}", encoding="utf-8")
    return path


def test_retrieve_sources_reads_generated_markdown_only(tmp_path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    _write_item(
        output_dir,
        "2026-01-01-retencion-1.md",
        "# Retencion\n\nLa estrategia de retencion usa onboarding y comunidad.",
        tail="\n\n## Mis notas\n\nSECRETO personal fuera del bloque generado.",
    )

    sources, scanned = retrieve_sources(output_dir, "retencion onboarding")

    assert scanned == 1
    assert len(sources) == 1
    assert sources[0].title == "Retencion"
    assert "onboarding" in sources[0].excerpt
    assert "SECRETO" not in sources[0].excerpt


def test_retrieve_sources_uses_hybrid_concept_matching(tmp_path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    _write_item(
        output_dir,
        "2026-01-01-business-ai-1.md",
        "# Business AI systems\n\nAgents can automate sales workflows and increase revenue.",
    )

    sources, scanned = retrieve_sources(output_dir, "negocio con IA")

    assert scanned == 1
    assert len(sources) == 1
    assert sources[0].title == "Business AI systems"


def test_retrieve_sources_can_be_scoped_to_filtered_library_notes(tmp_path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    allowed = _write_item(
        output_dir,
        "2026-01-01-agents-1.md",
        "# Selected agents\n\nAgents automate selected workflows.",
    )
    _write_item(
        output_dir,
        "2026-01-02-agents-2.md",
        "# Hidden agents\n\nAgents automate unrelated workflows.",
    )

    sources, scanned = retrieve_sources(
        output_dir,
        "agents workflows",
        allowed_paths={allowed},
    )

    assert scanned == 2
    assert [source.title for source in sources] == ["Selected agents"]


def test_answer_question_uses_configured_client_and_sources(tmp_path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    _write_item(
        output_dir,
        "2026-01-01-rag-1.md",
        "# RAG local\n\nEl indice local recupera fragmentos markdown antes de llamar al modelo.",
    )

    class FakeMessages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return TextResponse([TextBlock('{"answer":"Usa un indice local.","sources":["S1"]}')])

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    client = FakeClient()
    answer = answer_question(
        output_dir,
        "Como funciona el indice local?",
        provider="nanogpt",
        model="zai-org/glm-5.2",
        base_url="https://nano-gpt.example/api/v1",
        client=client,
    )

    assert answer.answer == "Usa un indice local."
    assert [source.id for source in answer.sources] == ["S1"]
    assert answer.model == "zai-org/glm-5.2"
    call = client.messages.calls[0]
    assert call["model"] == "zai-org/glm-5.2"
    assert "conocimiento externo" in call["system"]
    assert "RAG local" in call["messages"][0]["content"]


def test_answer_question_synthesizes_many_candidates_before_answering(tmp_path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    for index in range(8):
        _write_item(
            output_dir,
            f"2026-01-{index + 1:02d}-agents-{index}.md",
            f"# Agents {index}\n\nLos agentes de IA automatizan workflows numero {index}.",
        )

    class FakeMessages:
        def __init__(self):
            self.calls = []
            self.payloads = [
                TextResponse(
                    [
                        TextBlock(
                            '{"evidence":['
                            '{"source":"S2","note":"S2 aporta una tactica concreta."},'
                            '{"source":"S5","note":"S5 aporta un limite operativo."}'
                            '],"missing":""}'
                        )
                    ]
                ),
                TextResponse(
                    [
                        TextBlock(
                            '{"answer":"Usa tacticas concretas [S2] y revisa limites [S5].",'
                            '"sources":["S2","S5"]}'
                        )
                    ]
                ),
            ]

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return self.payloads.pop(0)

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    client = FakeClient()
    answer = answer_question(
        output_dir,
        "Que ideas utiles hay sobre agentes de IA?",
        provider="nanogpt",
        model="zai-org/glm-5.2",
        client=client,
    )

    assert answer.synthesis_used is True
    assert answer.candidate_sources == 8
    assert [source.id for source in answer.sources] == ["S2", "S5"]
    assert len(client.messages.calls) == 2
    assert "selección de evidencias" in client.messages.calls[0]["system"]
    assert "Candidatos recuperados" in client.messages.calls[0]["messages"][0]["content"]
    assert "Evidencia sintetizada" in client.messages.calls[1]["messages"][0]["content"]


def test_answer_question_without_sources_does_not_call_llm(tmp_path):
    output_dir = tmp_path / "vault" / "x-knowledge"
    _write_item(output_dir, "2026-01-01-rag-1.md", "# RAG\n\nContenido sobre indices.")

    class FailingMessages:
        def create(self, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("LLM should not be called without retrieved sources")

    class FakeClient:
        messages = FailingMessages()

    answer = answer_question(
        output_dir,
        "zzznomatch",
        provider="nanogpt",
        model="zai-org/glm-5.2",
        client=FakeClient(),
    )

    assert "No he encontrado informacion suficiente" in answer.answer
    assert answer.sources == []

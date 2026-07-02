# tests/test_worksheet_consumer_wiring.py
"""Guard the claude-code *consumers* of the topic worksheet (#34, #56).

The producer tests (`test_topic_synth.py`) prove `export_topic_worksheet`
EMITS `image_descriptions` and `video_transcripts`. They cannot catch a
consumer that silently drops those fields — and a dropped field means the
claude-code topics track never sees the evidence, which is exactly the bug
this branch fixes. These text-level assertions fail if a future edit removes
the new fields from either consumer artifact (the resynth workflow's per-topic
extraction command / prompt, or the skill's field list).
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = _REPO_ROOT / ".claude/workflows/resynth-topic-overviews.js"
_SKILL = _REPO_ROOT / ".claude/skills/enriching-x-knowledge/SKILL.md"


def test_resynth_workflow_extracts_new_evidence_fields():
    """The resynth workflow's per-topic extraction MUST print the new evidence.

    The agent is told to work ONLY with what the command prints ("No leas otros
    ficheros"), so a field absent from the extraction is invisible to it.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "image_descriptions" in text
    assert "video_transcripts" in text


def test_resynth_workflow_prompts_for_new_evidence_blocks():
    """The agent prompt must name the new evidence so the LLM weighs it."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "Images across" in text
    assert "Video transcripts across" in text


def test_skill_lists_new_topic_evidence_fields():
    """The skill's topic-object field list MUST mention the new fields, else a
    session driving the flow by hand never reads them."""
    text = _SKILL.read_text(encoding="utf-8")
    assert "image_descriptions" in text
    assert "video_transcripts" in text

"""The `vocab` stage — induce the topic taxonomy from the corpus.

Map-reduce: each chunk of posts proposes candidate topics (map); one
consolidation call merges all candidates into exactly `target_count` topics
(reduce). The Anthropic client is injected so tests run offline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from xbrain.llm_json import json_from_response
from xbrain.models import Item, Topic
from xbrain.rubrics import load_rubric

_MAP_MAX_TOKENS = 1000
_REDUCE_MAX_TOKENS = 2000


def _chunks(items: list[Item], size: int) -> list[list[Item]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _call(client, model: str, max_tokens: int, system: str, user: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return json_from_response(response, context="vocab")


def induce_vocab(
    store: dict[str, Item],
    target_count: int,
    model: str,
    output_language: str,
    client=None,
    chunk_size: int = 80,
) -> list[Topic]:
    """Induce `target_count` topics from the items in `store`."""
    if client is None:
        from anthropic import Anthropic  # lazy: tests inject a fake

        client = Anthropic()

    system = load_rubric("vocab", language=output_language)
    items = list(store.values())

    # --- Map: each chunk proposes candidate topics ---
    candidates: list[dict] = []
    for chunk in _chunks(items, chunk_size):
        posts = "\n".join(f"- {it.text}" for it in chunk)
        user = (
            "MAP STEP. Propose candidate topics for these posts. Respond with "
            'JSON: {"candidates": [{"slug": "...", "description": "..."}]}\n\n' + posts
        )
        result = _call(client, model, _MAP_MAX_TOKENS, system, user)
        cands = result.get("candidates")
        if not isinstance(cands, list):
            raise ValueError(
                "vocab map step: response has no 'candidates' list — "
                "the map call failed or was truncated"
            )
        candidates.extend(cands)

    # --- Reduce: consolidate into exactly target_count topics ---
    cand_block = "\n".join(f"- {c.get('slug')}: {c.get('description')}" for c in candidates)
    user = (
        f"REDUCE STEP. Consolidate these candidate topics into exactly "
        f"{target_count} final topics. Respond with JSON: "
        '{"topics": [{"slug": "...", "description": "..."}]}\n\n' + cand_block
    )
    final = _call(client, model, _REDUCE_MAX_TOKENS, system, user)
    topics: list[Topic] = []
    for entry in final.get("topics", []):
        try:
            topics.append(Topic(**entry))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid topic in vocab reduce output: {entry!r} ({exc})") from exc
    return topics


def export_vocab_worksheet(
    store: dict[str, Item], target_count: int, path: Path, output_language: str
) -> None:
    """Export the corpus + rubric so an executor can induce the taxonomy.

    The no-API-cost track for the `vocab` stage: a Claude Code session (or a
    person) reads this worksheet, induces the taxonomy and fills `topics`.
    XBrain only moves JSON — it never handles a Claude OAuth token.
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_count": target_count,
        "instructions": (
            "Induce a controlled topic taxonomy from `corpus`, following the "
            "embedded `rubric`. Use a map-reduce method: chunk the corpus, "
            "propose candidate topics per chunk, then consolidate to exactly "
            f"{target_count} topics. Write the final taxonomy into `topics` as "
            "objects {slug, description} — `slug` is kebab-case ([a-z0-9-]). "
            "Then run: xbrain vocab --apply <this file>."
        ),
        "rubric": load_rubric("vocab", language=output_language),
        "corpus": [item.text for item in store.values()],
        "topics": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_vocab_worksheet(path: Path) -> list[dict]:
    """Read the filled `topics` list from a vocab worksheet."""
    if not path.exists():
        raise FileNotFoundError(f"Worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("worksheet must be a JSON object")
    topics = data.get("topics", [])
    if not isinstance(topics, list):
        raise ValueError("worksheet `topics` must be a list")
    return topics


def apply_vocab_worksheet(
    topics_data: list[dict],
) -> tuple[list[Topic], list[tuple[str, list[str]]]]:
    """Validate worksheet topic entries into `Topic`s. Returns `(valid, invalid)`.

    Rejects malformed entries, invalid slugs (the `Topic` pattern) and duplicate
    slugs — the taxonomy's join key must be unique and well-formed.
    """
    valid: list[Topic] = []
    invalid: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for entry in topics_data:
        if not isinstance(entry, dict):
            invalid.append(("", ["topic entry is not a JSON object"]))
            continue
        slug = str(entry.get("slug", ""))
        extra = set(entry) - {"slug", "description"}
        if extra:
            invalid.append((slug, [f"unexpected keys: {sorted(extra)}"]))
            continue
        try:
            topic = Topic(**entry)
        except Exception as exc:  # noqa: BLE001 - collect, do not abort the batch
            invalid.append((slug, [f"invalid topic: {exc}"]))
            continue
        if topic.slug in seen:
            invalid.append((topic.slug, ["duplicate slug"]))
            continue
        seen.add(topic.slug)
        valid.append(topic)
    return valid, invalid

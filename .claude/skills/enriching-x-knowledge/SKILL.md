---
name: enriching-x-knowledge
description: Use when the user wants to enrich their XBrain knowledge base (item summaries + topics) or synthesize its topic-page overviews with their Claude subscription instead of the paid API. Drives the `xbrain enrich` and `xbrain topics` worksheet flows end to end.
---

# Enriching the XBrain knowledge base

Enrich pending XBrain items (summary + topics) at no API cost — this Claude Code
session is the executor.

## When to use

The user asks to enrich, classify or summarise their XBrain items / X knowledge
base and wants the `claude-code` track (their Claude subscription) rather than the
paid `api` executor.

## Procedure

1. **Export the worksheet.** Run `xbrain enrich --executor claude-code`. It writes
   `data/enrich-worksheet.json` with the pending items, the topic vocabulary and
   the rubrics. If it reports 0 pending items, stop. If it reports there is no
   vocabulary, run `xbrain vocab` first, then retry.

2. **Read `data/enrich-worksheet.json`.** It has `rubrics` (summary + topics),
   `vocab` (the allowed topic slugs) and `items`.

3. **Produce one judgment per item**, following the embedded rubrics exactly:
   - `summary`: 1-3 sentences, Spanish, faithful to the item.
   - `primary_topic`: exactly one slug from `vocab`.
   - `topics`: 1-4 slugs from `vocab`, including `primary_topic`.
   - Emit only `{item_id, summary, primary_topic, topics}` — never a filename,
     never a slug outside `vocab`. Use `misc` only for genuine no-topic noise,
     never because an article was not fetched (classify from the post text and
     the link domain).
   - If there are many items, process them in chunks (~40) or dispatch one
     subagent per chunk for accuracy. Append every judgment to the worksheet's
     `judgments` array.

4. **Apply.** Run `xbrain enrich --apply data/enrich-worksheet.json`. It validates
   and attaches the valid judgments. If it reports rejected items, fix those
   judgments in the worksheet and run `--apply` again.

## Topic-page overviews

After items are enriched, the `topics` stage synthesizes a prose overview for
each topic page — also via a worksheet, at no API cost.

1. **Export the worksheet.** Run `xbrain topics --executor claude-code`. It
   writes `data/topic-worksheet.json` with the topics needing (re)synthesis,
   their post summaries and the topic-page rubric, and writes the topic pages
   with their current post lists. If it reports 0 topics pending, stop.

2. **Read `data/topic-worksheet.json`.** It has `rubric` and `topics` (each with
   a `slug`, a `description` and the `summaries` of its posts).

3. **Produce one judgment per topic**, following the embedded rubric exactly:
   - `overview`: 1-3 paragraphs of plain Spanish prose, faithful to the
     summaries.
   - `notes`: 0-15 plain-prose strings, one important idea each.
   - Emit only `{slug, overview, notes}`. **Never write a wikilink (`[[...]]`),
     a filename or any identifier** — you have summaries, not posts, and the
     code adds the links. The validator rejects any judgment that contains `[[`.
   - Append every judgment to the worksheet's `judgments` array.

4. **Apply.** Run `xbrain topics --apply data/topic-worksheet.json`. It
   validates the judgments, stores the overviews and rewrites the topic pages.
   If it reports rejected topics, fix those judgments and run `--apply` again.

## Notes

- No API key and no per-token cost — this session does the LLM work.
- `data/enrich-worksheet.json` is gitignored and disposable once applied.

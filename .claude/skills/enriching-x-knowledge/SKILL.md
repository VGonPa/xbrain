---
name: enriching-x-knowledge
description: Use when the user wants to induce their XBrain topic vocabulary, enrich their knowledge base (item summaries + topics), synthesize its topic-page overviews, digest bookmarked videos into long-form readable notes, or verify enrichment outputs (LLM-as-judge) with their Claude subscription instead of the paid API. Drives the `xbrain vocab`, `xbrain enrich`, `xbrain topics`, `xbrain video-digest` and `xbrain verify` worksheet flows end to end.
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

## Vocabulary induction

The `vocab` stage induces the controlled topic taxonomy from the corpus — also
via a worksheet, at no API cost.

1. **Export the worksheet.** Run `xbrain vocab --executor claude-code`. It writes
   `data/vocab-worksheet.json` with the full corpus (`corpus`), the induction
   `rubric` and the `target_count`.

2. **Read `data/vocab-worksheet.json`.** Induce the taxonomy with a map-reduce
   method: chunk `corpus`, propose candidate topics per chunk (dispatch one
   subagent per chunk for a large corpus), then consolidate the candidates into
   exactly `target_count` topics. Follow the embedded `rubric` — always include
   a `misc` topic; topics must be distinct and comparably grained.

3. **Fill `topics`.** Write the final taxonomy into the worksheet's `topics`
   array as `{slug, description}` objects — `slug` is kebab-case (`[a-z0-9-]`),
   `description` is one sentence. Never reuse a slug.

4. **Apply.** Run `xbrain vocab --apply data/vocab-worksheet.json` (add
   `--regenerate` to re-mark every item for re-enrichment against the new
   taxonomy). It validates each entry and writes `data/vocab.yaml`. If it
   reports rejected topics, fix them and run `--apply` again.

## Topic-page overviews

After items are enriched, the `topics` stage synthesizes a prose overview for
each topic page — also via a worksheet, at no API cost.

1. **Export the worksheet.** Run `xbrain topics --executor claude-code`. It
   writes `data/topic-worksheet.json` with the topics needing (re)synthesis,
   their post summaries and the topic-page rubric, and writes the topic pages
   with their current post lists. If it reports 0 topics pending, stop.

2. **Read `data/topic-worksheet.json`.** It has `rubric` and `topics` (each with
   a `slug`, a `description`, the `summaries` of its posts, and — when the posts
   carry them — `image_descriptions` (prose for the topic's content-bearing
   photos) and `video_transcripts` (bounded transcript excerpts of its videos)).

3. **Produce one judgment per topic**, following the embedded rubric exactly:
   - `overview`: 1-3 paragraphs of plain Spanish prose, faithful to the
     summaries. When a topic carries `image_descriptions` or `video_transcripts`,
     weigh them as visual/transcript evidence alongside the summaries — they carry
     topic signal a one-line summary may only hint at.
   - `notes`: 0-15 plain-prose strings, one important idea each.
   - Emit only `{slug, overview, notes}`. **Never write a wikilink (`[[...]]`),
     a filename or any identifier** — you have summaries, not posts, and the
     code adds the links. The validator rejects any judgment that contains `[[`.
   - Append every judgment to the worksheet's `judgments` array.

4. **Apply.** Run `xbrain topics --apply data/topic-worksheet.json`. It
   validates the judgments, stores the overviews and rewrites the topic pages.
   If it reports rejected topics, fix those judgments and run `--apply` again.

## Video digests

`video-digest` turns a bookmarked video (already transcribed by `digest-video`)
into a long-form readable `digest` — also via a worksheet, at no API cost.

1. **Export the worksheet.** Run `xbrain video-digest --executor claude-code`. It
   writes `data/video-digest-worksheet.json` with the videos pending a digest
   (their transcript + frame descriptions) and the `rubric-video-digest.md` rubric.
   If it reports no videos pending, stop. If none has a transcript, run
   `xbrain digest-video` first.

2. **Read `data/video-digest-worksheet.json`** and, per entry, write one `digest`
   following the embedded rubric: the *What it is · Key points · Why it matters*
   shape, grounded in the transcript (what the video says) and the frame
   descriptions (what it shows), faithful with no hype. For a mute video, build it
   from the frames alone. Append `{item_id, digest}` to the `judgments` array.

3. **Apply.** Run `xbrain video-digest --apply data/video-digest-worksheet.json`.
   It snapshots `data/`, writes each `source.digest`, and `generate` then leads the
   note with the digest. Fix any rejected entries and re-run `--apply`.

## Verifying enrichment outputs

`verify` is an LLM-as-judge pass that scores existing enrichment outputs (a
`summary`, a video `digest`, or a `topics` assignment) for faithfulness +
adherence — report-only, it never mutates the store.

1. **Export the worksheet.** Run `xbrain verify --target all --executor claude-code`
   (or `--target summary|digest|topics`). It writes `data/verify-worksheet.json`
   with, per `(item, target)`, the source, the generated output, its generation
   rubric and the `rubric-verify.md` verify rubric. If it reports nothing to
   verify, run `enrich` (and, for `digest`, `video-digest`) first.

2. **Judge each entry**, defaulting SKEPTICAL, per the verify rubric: **faithfulness**
   (every claim/number/name grounded in the source — one unsupported claim is a
   FAIL) and **adherence** (the output obeys its own generation rubric). Append one
   `{item_id, target, verdict, faithfulness, adherence, flags}` per entry. For a
   real ensemble, copy the worksheet **once per judge** and fill each independently.

3. **Apply.** Run `xbrain verify --apply ws1.json --apply ws2.json …` — one file per
   judge. It aggregates them (faithfulness is unforgiving: one judge's FAIL sinks the
   group) and writes `data/verify-report.{json,md}`. The `.md` leads with the
   FAIL/REVIEW verdicts + flagged claims for a human to act on.

## Notes

- No API key and no per-token cost — this session does the LLM work.
- `data/enrich-worksheet.json` is gitignored and disposable once applied.
- `video-digest` and `verify` share this same `claude-code` worksheet handoff and
  read the same `[enrich].executor` default; neither has an `api` track.

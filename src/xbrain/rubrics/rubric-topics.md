# Rubric — Topic assignment

Assign topics to one X post from the controlled vocabulary provided.

- You receive the vocabulary as a list of `slug` + `description`. Use **only**
  those slugs. Never invent a slug.
- Choose exactly **one `primary_topic`** — the post's "home", the single topic
  it most belongs to.
- Optionally add **0-3 secondary topics** — other genuinely relevant topics.
- `topics` is the full set: `primary_topic` first, then the secondaries. Total
  length 1-4.
- Emit `topic_confidence` as `high`, `medium`, or `low`:
  - `high`: the vocabulary fits the post naturally.
  - `medium`: the assignment is reasonable but imperfect.
  - `low`: the vocabulary lacks a good home for the post.
- Emit `suggested_new_topics` as 0-5 kebab-case slugs when `topic_confidence`
  is `medium` or `low` because the controlled vocabulary is missing a useful
  topic. These are diagnostics only. Do **not** put suggested slugs in
  `primary_topic` or `topics` unless they already appear in the vocabulary.
- Output **only** the judgment object — slugs from the vocabulary. Never output
  filenames, note titles, wikilinks or any identifier the vocabulary did not
  give you, except inside `suggested_new_topics`.

## Classifying when there is no fetched article

Many posts link to an article that could not be downloaded (especially X's own
articles). **A missing article is NOT a reason to fall back to `misc`.**

- Classify from the **post's own text** and from the **link's URL and domain**.
  The domain alone is strong signal: `arxiv.org` → research, `github.com` →
  code, a known newsletter → its subject.
- Use `misc` **only** when the post has genuinely no identifiable subject (a
  pure greeting, a single word, an image with no text). Never use `misc` merely
  because the article body is absent.
- If `misc` is the best available vocabulary slug but the post clearly has a
  subject, set `topic_confidence` to `low` and add one or more
  `suggested_new_topics`.

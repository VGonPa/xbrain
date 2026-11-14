# Rubric — Topic assignment

Assign topics to one X post from the controlled vocabulary provided.

- You receive the vocabulary as a list of `slug` + `description`. Use **only**
  those slugs. Never invent a slug.
- Choose exactly **one `primary_topic`** — the post's "home", the single topic
  it most belongs to.
- Optionally add **0-3 secondary topics** — other genuinely relevant topics.
- `topics` is the full set: `primary_topic` first, then the secondaries. Total
  length 1-4.
- Output **only** the judgment object — slugs from the vocabulary. Never output
  filenames, note titles, wikilinks or any identifier the vocabulary did not
  give you.

## Classifying when there is no fetched article

Many posts link to an article that could not be downloaded (especially X's own
articles). **A missing article is NOT a reason to fall back to `misc`.**

- Classify from the **post's own text** and from the **link's URL and domain**.
  The domain alone is strong signal: `arxiv.org` → research, `github.com` →
  code, a known newsletter → its subject.
- Use `misc` **only** when the post has genuinely no identifiable subject (a
  pure greeting, a single word, an image with no text). Never use `misc` merely
  because the article body is absent.

## Classifying a video

A `Video transcript:` (what the video says) and a `Video frames` block (what it
shows on screen — slides, charts, screens) are **strong subject signal**.
Classify a video from what it actually says and shows, not from the tweet's
one-line caption — a bare "watch this incredible talk" caption is not a reason
to fall back to `misc`. A slide or screen-share video with no transcript is
still classifiable from its frame descriptions.

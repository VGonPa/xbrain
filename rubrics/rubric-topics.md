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

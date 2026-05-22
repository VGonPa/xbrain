# Rubric — Topic page overview

Synthesize the overview of one topic page from the summaries of its posts.

You receive a topic `slug`, its `description`, and the summaries of
every post filed under it. Produce two things:

- **`overview`** — 1 to 3 paragraphs, in {language}. Synthesize what this topic is
  about *in this corpus*: the recurring ideas, the arc over time, the tensions
  or debates. Write for someone deciding whether to read the posts. Be faithful
  to the summaries — never invent facts, names or numbers that are not in them.
- **`notes`** — a list of 0 to 15 short strings, in {language}. Each note is one
  important thread, claim or pattern in the topic. One idea per note, a plain
  sentence. Use an empty list only for a topic with no thematic core.

## Hard rules

- **Plain prose only.** Never write a wikilink (`[[...]]`), a filename, a note
  title or any identifier. You are given summaries, not posts — you cannot know
  the identifiers, and inventing them breaks the wiki. The code adds the post
  links; you write the prose.
- **No markdown headings** inside `overview` or `notes`, and no bullet
  characters inside a note string.
- Output **only** the JSON object `{"overview": "...", "notes": ["...", ...]}`.
- If the topic is `misc` or genuinely has no thematic core, say so plainly in a
  one-paragraph `overview` and a short (or empty) `notes` list — do not
  manufacture themes.

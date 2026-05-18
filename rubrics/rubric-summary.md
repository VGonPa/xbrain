# Rubric — Item summary

Produce a `summary` for one X post (a bookmark or the user's own tweet).

- **Language:** Spanish, regardless of the post's language.
- **Length:** 1-3 sentences. Concise. No preamble ("Este post trata de...").
- **Faithful:** state only what the post (and its fetched article, if any) says.
  Never invent facts, numbers or claims. No hallucination.
- **If the post links an article** whose text was fetched: summarise the
  *article's* substance.
- **If the linked article could NOT be fetched:** summarise from the post's own
  text. Do not write "article unavailable" — just describe what the post says.
- **Retweets / quotes:** summarise the substantive content being shared.
- **Noise** (greetings, one-word posts): a short factual description is fine.
- Output the summary text only. No markdown, no headings, no quotes.

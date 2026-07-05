# Rubric — Item summary

Produce a `summary` for one X post (a bookmark or the user's own tweet).

- **Language:** {language}, regardless of the post's language.
- **Length:** 1-3 sentences. Concise. No preamble ("Este post trata de...").
- **Faithful:** state only what the post — and its fetched article, video
  transcript or on-screen frames, if any — actually says. Never invent facts,
  numbers or claims. No hallucination.
- **If the post links an article** whose text was fetched: summarise the
  *article's* substance.
- **If the post has a video** (a `Video transcript:` and/or `Video frames`
  block): summarise what the *video* is about — grounded in the transcript (what
  it says) and the frame descriptions (what it shows on screen). For a slide or
  screen-share video with no transcript, summarise from the frame descriptions.
  Still 1-3 sentences: capture the whole talk's subject, not just its opening.
- **If the linked article could NOT be fetched:** summarise from the post's own
  text. Do not write "article unavailable" — just describe what the post says.
- **Retweets / quotes:** summarise the substantive content being shared.
- **Noise** (greetings, one-word posts): a short factual description is fine.
- Output the summary text only. No markdown, no headings, no quotes.

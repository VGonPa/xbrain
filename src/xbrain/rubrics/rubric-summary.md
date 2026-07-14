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
- **If the linked article could NOT be fetched** (the item says so explicitly —
  an `unfetched_links_note` or a "content NOT fetched" line): summarise from the
  post's own text. Do not write "article unavailable" — just describe what the
  post says. NEVER describe, reconstruct or guess the linked content from its
  URL, its domain or your own knowledge of it.
- **Retweets / quotes:** summarise the shared content **when it is present** in the
  item (a fetched article, a transcript, a quoted body). When it is not — the item
  says `content NOT fetched`, or carries only a `quoted_content_note` — summarise
  the post's own text instead. Never reconstruct the shared content you cannot see.
- **Attribution:** the post's author is who POSTED it. On a repost, a quote or a clip
  of someone else, the words are a third party's — do not name the poster as the
  speaker or author of that content, and do not name a speaker the item never names.
- **Noise** (greetings, one-word posts): a short factual description is fine.
- Output the summary text only. No markdown, no headings, no quotes.

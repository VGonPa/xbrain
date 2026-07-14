# Rubric — Item summary

Produce a `summary` for one X post (a bookmark or the user's own tweet).

- **Language:** {language}, regardless of the post's language.
- **Length:** 1-3 sentences. Concise. No preamble ("Este post trata de...").
- **Faithful — the evidence is exactly these surfaces:**
  1. the **tweet text** (the post's own words),
  2. the **author metadata** of the account that posted it (its `@handle` and its
     display name),
  3. the poster's own **thread**,
  4. the **fetched article** body and its title,
  5. the **video** title, transcript and frame descriptions,
  6. the **image descriptions**,
  7. the **quoted post** — its body AND the `@handle (Name)` of the account that wrote
     it, shown together under a `Quoted post — @handle (Name)` label. When that label
     is present, the quoted body is evidence and so is that account's name.

  Nothing else is evidence. Not your world knowledge, not recognising the topic, the
  voice or the byline, and **not a link's URL or domain** — a domain is topic signal,
  never a name and never content. State only what these surfaces say or show. Never
  invent facts, numbers or claims. No hallucination. Three rules make this mechanical:
  - **Never name what no surface names.** Do not name the speaker, interviewer, host,
    author, company, employer, product, publication, podcast, university, course code,
    paper or model unless that exact name appears in one of the surfaces above.
    Recognising who someone probably is — from the topic, the writing, the setting or
    your own world knowledge — is NOT evidence. When no surface names them, use a
    neutral descriptor ("the speaker", "the author", "a cloud provider"). A link to
    `nytimes.com` does not license naming the publication.
  - **Quote verbatim.** Reproduce a quoted span exactly as the source renders it,
    apparent ASR errors included. Never repair, normalise or complete a quote into the
    phrase you think was meant.
  - **Do not sharpen.** Never resolve a vague term into a specific one the evidence
    never uses ("beans" stays "beans"; it does not become "coffee"), and never add a
    duration, date, figure, version or affiliation the evidence does not state.
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
  item (a fetched article, a transcript, a quoted body under its `Quoted post` label).
  A quote-tweet's own text is often a bare reaction ("Read this", "This is huge") — the
  substance is in the quoted post, so summarise THAT, and say what it actually says.
  When it is not present — the item says `content NOT fetched`, or carries only a
  `quoted_content_note` — summarise the post's own text instead. Never reconstruct the
  shared content you cannot see: if the reaction does not say what it is reacting to,
  neither do you.
- **Attribution:** the post's author is who POSTED it. On a repost, a quote or a clip
  of someone else, the words are a third party's — do not name the poster as the
  speaker or author of that content, and do not name a speaker the item never names.
  On a quote-tweet the quoted account is named for you in the `Quoted post — @handle
  (Name)` label: THAT is the author of the quoted words, and the poster is not.
- **Noise** (greetings, one-word posts): a short factual description is fine.
- Output the summary text only. No markdown, no headings, no quotes.

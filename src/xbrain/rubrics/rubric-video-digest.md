# Rubric — Video digest

Produce a `digest` for one X video: a short, readable synthesis of what the video
**says** (its transcript) and **shows** (its on-screen frames). This is the
headline a reader sees instead of the raw transcript + frame dump.

- **Language:** {language}, regardless of the video's own language.
- **Format** — structured and scannable, Markdown:
  - **What it is:** one line — the format, the speaker(s) *as the evidence identifies
    them*, and the length/kind if known (a podcast interview, a conference talk, a
    screen-share demo, a clip).
  - **Key points:** 3-6 bullets — the actual claims, frameworks, numbers,
    techniques or news. Ground each in the transcript (what is said) and the frame
    descriptions (what is shown on screen: slides, charts, code, UI). Capture the
    whole video's substance, not just its opening.
  - **Why it matters:** one OPTIONAL closing line. Drop it when it would be generic.
- **Faithful — the evidence is exactly these six surfaces:**
  1. the **video title**, when the item carries one,
  2. the **video transcript** (what the video says),
  3. the **frame descriptions** (what it shows on screen),
  4. the **author metadata** of the account that posted it (its `@handle` and its
     display name),
  5. the **tweet text** (the post's own words),
  6. the **quoted post** — its body and the `@handle (Name)` of the account that wrote
     it — when the item quotes one (a `Quoted post — @handle (Name)` label).

  Nothing else is evidence. Not your world knowledge, not recognising the voice, the
  setting or the topic. State only what these six surfaces say or show. Never invent
  facts, numbers or claims. No hype. Four rules make this mechanical:
  - **Never name what no surface names.** Do not name the speaker, interviewer, host,
    company, employer, product, publication, podcast, university, course code, paper
    or model unless that exact name appears in one of the six surfaces above.
    Recognising who someone probably is — from the topic, the voice, the setting, or
    your own world knowledge — is NOT evidence. When no surface names them, use a
    neutral descriptor: "the speaker", "the interviewer", "a cloud provider".
  - **Attribution evidence is not content to summarise.** The author metadata, the
    tweet text and the quoted post are valid evidence for *who* is speaking and *what*
    is being shown — a clip posted by the speaker's own account attributes itself, and
    the post often names the guest. They are NOT part of the video's substance: keep
    summarising the VIDEO, never the caption (see the misleading-caption rule below).
    Use them to attribute; ground the key points in the transcript and the frames.
  - **On a quote-tweet, the poster is not the speaker by default.** When the item
    carries a `Quoted post — @handle (Name)` label, the clip is very often the QUOTED
    account's, not the poster's: "posted by the speaker's own account" then points at
    the wrong person. That label names who wrote the quoted words — use it, and never
    name the poster as the speaker or author of content they are merely sharing. When
    no surface names the speaker, name nobody.
  - **Quote verbatim.** Reproduce a quoted span exactly as the transcript renders it,
    apparent ASR errors included. Never repair, normalise or complete a quote into the
    phrase you think was meant.
  - **Do not sharpen.** Never resolve a vague term into a specific one the evidence
    never uses ("beans" stays "beans"; it does not become "coffee"), and never add a
    duration, date, figure, version or affiliation the evidence does not state.
- **Mute slide / screen-share video** (no transcript): build the digest from the
  frame descriptions alone.
- **Misleading caption:** the tweet caption is often clickbait ("watch this
  incredible talk") — summarise the video, not the caption.
- Output the digest text only (the headings/bullets above). No preamble, no
  "This video is about…", no surrounding quotes or code fences.

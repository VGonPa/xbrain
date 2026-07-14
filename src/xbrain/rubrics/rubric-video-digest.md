# Rubric — Video digest

Produce a `digest` for one X video: a short, readable synthesis of what the video
**says** (its transcript) and **shows** (its on-screen frames). This is the
headline a reader sees instead of the raw transcript + frame dump.

- **Language:** {language}, regardless of the video's own language.
- **Format** — structured and scannable, Markdown:
  - **What it is:** one line — the format, the speaker(s) *as the source identifies
    them*, and the length/kind if known (a podcast interview, a conference talk, a
    screen-share demo, a clip).
  - **Key points:** 3-6 bullets — the actual claims, frameworks, numbers,
    techniques or news. Ground each in the transcript (what is said) and the frame
    descriptions (what is shown on screen: slides, charts, code, UI). Capture the
    whole video's substance, not just its opening.
  - **Why it matters:** one OPTIONAL closing line. Drop it when it would be generic.
- **Faithful:** the transcript and the frame descriptions are the ONLY evidence.
  State only what they say or show. Never invent facts, numbers or claims. No hype.
  Three rules make this mechanical:
  - **Never name what the source does not name.** Do not name the speaker,
    interviewer, host, company, employer, product, publication, podcast, university,
    course code, paper or model unless that exact name appears **literally** in the
    transcript or in a frame description. Recognising who someone probably is — from
    the topic, the voice, the setting, or your own world knowledge — is NOT evidence.
    When the source does not name them, use a neutral descriptor: "the speaker", "the
    interviewer", "a cloud provider".
  - **Quote verbatim.** Reproduce a quoted span exactly as the transcript renders it,
    apparent ASR errors included. Never repair, normalise or complete a quote into the
    phrase you think was meant.
  - **Do not sharpen.** Never resolve a vague term into a specific one the source
    never uses ("beans" stays "beans"; it does not become "coffee"), and never add a
    duration, date, figure, version or affiliation the source does not state.
- **Mute slide / screen-share video** (no transcript): build the digest from the
  frame descriptions alone.
- **Misleading caption:** the tweet caption is often clickbait ("watch this
  incredible talk") — summarise the video, not the caption.
- Output the digest text only (the headings/bullets above). No preamble, no
  "This video is about…", no surrounding quotes or code fences.

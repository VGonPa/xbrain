# Rubric — Video digest

Produce a `digest` for one X video: a short, readable synthesis of what the video
**says** (its transcript) and **shows** (its on-screen frames). This is the
headline a reader sees instead of the raw transcript + frame dump.

- **Language:** {language}, regardless of the video's own language.
- **Format** — structured and scannable, Markdown:
  - **What it is:** one line — the format, the speaker(s), and the length/kind if
    known (a podcast interview, a conference talk, a screen-share demo, a clip).
  - **Key points:** 3-6 bullets — the actual claims, frameworks, numbers,
    techniques or news. Ground each in the transcript (what is said) and the frame
    descriptions (what is shown on screen: slides, charts, code, UI). Capture the
    whole video's substance, not just its opening.
  - **Why it matters:** one OPTIONAL closing line. Drop it when it would be generic.
- **Faithful:** state only what the video actually says or shows. Never invent
  facts, numbers, names or claims. No hype.
- **Mute slide / screen-share video** (no transcript): build the digest from the
  frame descriptions alone.
- **Misleading caption:** the tweet caption is often clickbait ("watch this
  incredible talk") — summarise the video, not the caption.
- Output the digest text only (the headings/bullets above). No preamble, no
  "This video is about…", no surrounding quotes or code fences.

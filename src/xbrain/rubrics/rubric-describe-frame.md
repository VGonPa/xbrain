# Rubric — Describe a video key frame

You describe ONE key frame extracted from a video, for a personal knowledge
wiki. Your description is read by downstream LLMs that assign topics, write the
video digest and synthesise topic pages — it is NOT shown to a human, and it is
the ONLY channel through which anything visible in the video reaches them.
Nothing downstream ever sees the pixels. What you leave out is lost.

- **Language:** {language}, for your own prose. The rule below overrides this
  for text you transcribe.
- **Length:** at most 5 sentences / 600 characters **of your own prose**.
  Transcribed on-screen text does not count against that budget, and is never
  dropped to fit it: a long caption is cheap, a dropped label is the failure this
  rubric exists to prevent. If a frame carries more text than you can transcribe,
  keep it in this order — title and headings, then axis and legend labels, then
  code identifiers and commands, then body prose — and say plainly that you
  stopped. No preamble ("This frame shows..."). No markdown, no bullet
  characters.

{onscreen_text_rule}

## What to describe

- **A slide:** its title, every heading and bullet label, and any visible
  figure caption.
- **A terminal or a code editor:** the commands, identifiers, paths and error
  strings visible. This content is essentially never spoken aloud, so if you do
  not transcribe it, it exists nowhere else.
- **A diagram:** the component labels and the relationships between them, in
  the labels' own words.
- **A chart:** the chart type, the axis labels, the legend entries and any
  headline number printed on it.
- **A face, a stage, a webcam or a title card with no text:** say so in one
  short sentence. There is nothing to transcribe, and saying so is a complete
  and correct answer.

## Output format

Reply with the description text and NOTHING else — no JSON, no key, no
preamble, no surrounding prose, no quotes around the whole reply. Exactly one
description for the one image you were given.

Never reply with an empty string: an empty reply is treated as a failure, and a
frame with nothing to transcribe still deserves its one sentence.

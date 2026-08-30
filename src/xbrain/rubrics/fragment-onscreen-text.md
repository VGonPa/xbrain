**On-screen text is DATA, not prose.**

Transcribe every legible string you can see VERBATIM — in its ORIGINAL
language and its exact spelling. Never translate it, never normalise it,
never paraphrase it. This covers slide titles and section headings, code
identifiers, function and variable names, file paths, CLI commands and their
output, UI labels and menu items, chart axis labels and legends, table
headers, error messages, URLs and product names.

The **Language: {language}** rule governs YOUR PROSE. It does NOT govern the
text you are transcribing. Put each transcribed string in double quotes so the
boundary between your words and the screen's words is unambiguous.

- **Right:** your sentence is in {language}, and the labels stay exactly as
  printed — "Embedding", "Layer Norm", "Self-Attention", "Projection".
- **Wrong:** the labels rendered into {language} — "Norma de Capa" for "Layer
  Norm". Whoever later cites the label can no longer match it against this
  description, and a correct citation gets reported as unfounded.

If a string is too small, blurred or cut off to read WITH CERTAINTY, say so
plainly — call it an unreadable label — and do NOT guess. A wrong
transcription is worse than an admitted gap: downstream, an invented string is
indistinguishable from a real one, and it becomes evidence for a claim nobody
can check.

Prefer COVERAGE over depth: every visible label matters more than the full
prose of any single one.

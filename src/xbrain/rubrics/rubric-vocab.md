# Rubric — Vocabulary induction

Induce a topic taxonomy from a corpus of X posts.

- **Map step:** given a chunk of posts, propose candidate topics. Each candidate
  is a short kebab-case `slug` plus a one-sentence `description`. Propose topics
  about *subject matter*, not format.
- **Reduce step:** given all candidates, consolidate to exactly the requested
  `target_count` topics. Merge near-duplicates; split topics too broad; drop
  topics with negligible support.
- Every final topic has a unique kebab-case `slug` and a one-sentence
  `description`. **`description` language: {language}.** Slugs stay
  kebab-case ASCII regardless of language — they are the stable identifier.
- Always include a `misc` topic (description: "Posts that do not fit a specific
  topic." — translate this default description into {language}).
- Topics must be distinct, comparably grained, and together cover the corpus.

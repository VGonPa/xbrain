/*
 * resynth-topic-overviews — re-synthesize stale xbrain topic-page overviews,
 * one agent per topic (each reads only its own topic's post summaries).
 *
 * WHY: `xbrain topics` refreshes the mechanical post-lists on every run, but it
 * does NOT rewrite the prose `overview` / `notes`. As the corpus grows, those
 * overviews go stale and the pages show "⚠ Overview desactualizado: +N posts
 * desde la última síntesis". `topics --resynth` rewrites them from the current
 * post set. Fanning the work out — one agent per topic — lets each agent spend
 * its full context on a single topic (some carry 200-326 post summaries) and
 * runs them in parallel, instead of one agent diluting across all of them.
 *
 * The agents synthesize; they do NOT write the worksheet (N writers on one JSON
 * would corrupt it). Each RETURNS {slug, overview, notes}; a single assemble +
 * validate step (step 4 below) writes the worksheet once.
 *
 * RUNBOOK (from the xbrain repo root, with the venv):
 *
 *   # 1. Export the resynth worksheet (stale topics + their summaries + rubric):
 *   .venv/bin/xbrain topics --resynth --executor claude-code
 *
 *   # 2. Dump the topic slugs to feed as this workflow's `args`:
 *   .venv/bin/python -c "import json;print(json.dumps([t['slug'] for t in json.load(open('data/topic-worksheet.json'))['topics']]))"
 *
 *   # 3. Run this workflow (Claude drives it):
 *   #      Workflow({ name: 'resynth-topic-overviews', args: <the slug array from step 2> })
 *   #    It returns [{slug, overview, notes}, ...] — one entry per topic.
 *
 *   # 4. Save that returned array to data/overviews-result.json, then assemble +
 *   #    validate into the worksheet (coverage, no wikilinks, cap notes to 15):
 *   .venv/bin/python - <<'PY'
 *   import json
 *   res = json.load(open('data/overviews-result.json'))
 *   ws = json.load(open('data/topic-worksheet.json'))
 *   assert {t['slug'] for t in ws['topics']} == {r['slug'] for r in res}, 'cobertura incompleta'
 *   for r in res:
 *       blob = r['overview'] + ' '.join(r['notes'])
 *       assert r['overview'].strip() and '[[' not in blob, r['slug']  # no vacío, sin wikilinks
 *   ws['judgments'] = [{'slug': r['slug'], 'overview': r['overview'], 'notes': r['notes'][:15]} for r in res]
 *   json.dump(ws, open('data/topic-worksheet.json', 'w'), ensure_ascii=False, indent=2)
 *   print('worksheet:', len(ws['judgments']), 'judgments')
 *   PY
 *
 *   # 5. Apply + regenerate the vault, then confirm zero stale warnings:
 *   .venv/bin/xbrain topics --apply data/topic-worksheet.json
 *   .venv/bin/xbrain generate
 *   grep -rl "Overview desactualizado" <vault>/learnings/x-knowledge/topics | wc -l   # -> 0
 *
 * Model: agents inherit the session model. Pin `model: 'opus'` (below) so a
 * resynth always runs on Opus regardless of the session's model.
 */

export const meta = {
  name: 'resynth-topic-overviews',
  description: 'Re-synthesize stale xbrain topic overviews, one Opus agent per topic',
  phases: [{ title: 'Synthesize', detail: 'one agent per topic reads its own posts', model: 'opus' }],
}

// The standard export location written by `xbrain topics --resynth --executor claude-code`.
const WORKSHEET = '/Users/vgonpa/devel/xbrain/data/topic-worksheet.json'

const OVERVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['overview', 'notes'],
  properties: {
    overview: {
      type: 'string',
      description: '1 a 3 párrafos en español, prosa plana, sin wikilinks ni headings.',
    },
    notes: {
      type: 'array',
      items: { type: 'string' },
      description: '0 a 15 frases cortas en español, una idea por nota, sin bullets ni wikilinks.',
    },
  },
}

// `args` may arrive as a real array or as a JSON string — parse defensively.
const slugs = Array.isArray(args) ? args : JSON.parse(args)
log(`Re-sintetizando ${slugs.length} overviews de topic, un agente Opus por topic`)
phase('Synthesize')

const results = await parallel(
  slugs.map((slug) => () =>
    agent(
      `Eres un sintetizador experto de una base de conocimiento personal (xbrain de Víctor). Tu única tarea: escribir el OVERVIEW del topic "${slug}", leyendo SOLO los posts de ESE topic.

PASO 1 — Extrae tu topic y la rúbrica ejecutando exactamente:
  cd /Users/vgonpa/devel/xbrain && .venv/bin/python -c "import json; d=json.load(open('${WORKSHEET}')); t=[x for x in d['topics'] if x['slug']=='${slug}'][0]; print('=== RUBRICA ==='); print(d['rubric']); print('=== DESCRIPTION ==='); print(t['description']); print('=== SUMMARIES ('+str(len(t['summaries']))+') ==='); [print('- '+s) for s in t['summaries']]"

PASO 2 — Lee la RÚBRICA entera y respétala al pie de la letra. Lee la DESCRIPTION y TODOS los SUMMARIES: cada summary es el resumen de un post del topic — ESOS son "los posts" que debes leer y sintetizar. No leas otros ficheros; trabaja solo con lo que imprime ese comando.

PASO 3 — Sintetiza para el topic "${slug}":
  - overview: 1 a 3 párrafos en español. Qué es este topic EN ESTE CORPUS: las ideas recurrentes, el arco temporal, las tensiones o debates. Escrito para alguien que decide si leer los posts. FIEL a los summaries — nunca inventes nombres, números ni hechos que no estén en ellos. Prosa plana: sin wikilinks [[...]], sin nombres de fichero, sin identificadores, sin headings markdown. Si el topic es "misc" o de verdad no tiene núcleo temático, dilo con naturalidad en un párrafo y deja notes corto o vacío — no fabriques temas.
  - notes: lista de 0 a 15 frases cortas en español, una idea/hilo/patrón importante por nota, frase plana sin bullets ni wikilinks.

Devuelve SOLO el objeto {overview, notes}.`,
      { label: `synth:${slug}`, phase: 'Synthesize', schema: OVERVIEW_SCHEMA, model: 'opus' }
    ).then((r) => (r ? { slug, overview: r.overview, notes: r.notes } : null))
  )
)

const filled = results.filter(Boolean)
log(`Completados ${filled.length}/${slugs.length} overviews`)
return filled

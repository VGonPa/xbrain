# Barrido del umbral del grafo — Plan 04.5

**Fecha de la medición:** 2026-09-15 · **Estado:** medición local firmada (Plan 04 §1.3 y §11.7), no un
check de CI · **Instrumento:** `xbrain eval --strategy hybrid_graph --sweep-graph <rejilla>`

> Todas las cifras de este documento son una fotografía del corpus, del golden set y del código que se
> nombran en la §1, en ese momento. Se re-derivan con los comandos de la §6; no se citan después de que el
> corpus se mueva (CLAUDE.md, regla 2).

## 0. Resultado

**Umbral aplicado:** `min_shared_items = 5` · `min_weight = 0.05`

- **Ninguna de las 16 combinaciones aporta.** Todas empeoran `recall@10` frente a `hybrid`
  (Δ entre **−0,1208** y **−0,1763** sobre una base de **0,6301**), y todas pierden más de 3 pp de
  precisión en al menos un estrato (`cruzado_idioma` en las 16, `semantico` además en 8). **Ninguno** de los
  62 a 99 candidatos que el grafo metió en los top-10 era relevante.
- **Se aplica `min_shared_items = 5`, `min_weight = 0.05`** porque el índice siempre construye un grafo y
  algún umbral tiene que estar en vigor: es la primera por la regla de la §2 — el menor daño a `recall@10`
  (empatado con las otras tres celdas de `min_weight = 0.05`), el menor ruido de esas cuatro (94) y, entre
  las tres que empatan también ahí, el grafo más disperso (160 aristas frente a 164). Sustituye al par en
  vigor, `2 / 0.0`, que el código declaraba «sin barrer».
- **`hybrid_graph` NO se promueve** (Plan 04 §3, criterio §11.7): `GRAPH_ENABLED_BY_DEFAULT` sigue en
  `False` y `search` sigue en `lexical` por defecto. Nada en este PR cambia qué estrategia sirve `search`.
- **Lo que sí mide el umbral es la forma del grafo.** Con `2 / 0.0` quedan 408 aristas y un grado medio de
  **9,07** frente a un tope de 10 vecinos por topic: casi todos los topics llenan el tope, que es el grafo
  que «deja de discriminar» del Plan 04 §1.1. Con `5 / 0.05` quedan 160 y un grado medio de 3,56.

## 1. Qué se midió, sobre qué, con qué instrumento

| | |
|---|---|
| Corpus | `data/items.json` sha256 `2773310f…` — 2.495 items · 45 topics · `vocab.yaml` sha256 `e73fbede…` · `topics.json` sha256 `d2f46a72…`. Los tres sha256 son idénticos antes y después de la corrida |
| Fingerprints del índice | `store_fingerprint` `93f994d4…` · `vocab_fingerprint` `55da1032…` · `topics_fingerprint` `9af12df5…` (los que selló el manifest de la corrida) |
| Golden set | `eval/golden-set.yaml` v3, sha256 `ed6dd760…`: 23 casos, **18 medidos**; 5 declarados no medibles (§3) |
| Código | xbrain `719954c` (el instrumento; este documento y el umbral aplicado van en el commit siguiente) |
| Rejilla | `min_shared_items ∈ {2, 3, 5, 8}` × `min_weight ∈ {0.0, 0.02, 0.05, 0.10}` — la del Plan 04 §1.3, entera |
| Profundidad | `k = 10` items por caso; un resultado directo expulsado de esa profundidad cuenta como puesto 11 |
| Recuperación | `search_service.search` — la única puerta en la que existe `hybrid_graph` — sobre un índice propio en `data/eval-index/graph-sweep/`, construido con el escritor de `xbrain index build` y reescrito celda a celda con `index update`; cada celda comprueba contra su manifest que midió los umbrales que dice |
| Máquina | Apple M2 · 16 GB · Python 3.13.7 · SQLite 3.50.4 · **179 s** de reloj para las 16 celdas |

**Lo que el instrumento NO mide, dicho antes de los números.**

1. **La base es léxica.** La corrida no tenía embedder configurado y el índice del barrido no tiene plano
   vectorial, así que `search --strategy hybrid` respondió `lexical` declarando
   `embeddings_not_configured, no_embeddings`, y `hybrid_graph` reordenó ese mismo ranking declarando lo
   mismo. El Δ es el del grafo sobre el ranking léxico. Sobre el ranking fusionado de un `hybrid` con
   vectores **no se ha medido**, y `hybrid` tampoco está promovido (bake-off del Plan 03.7).
2. **La unidad es el item que sirve `search`**, no el owner que puntúa `xbrain eval`: estas cifras no son
   comparables con el `recall@10` 0,7395 del arnés.
3. **El estrato `expansión` sigue sin casos.** Poblarlo (criterio §11.9) no entra en este PR. Sin él ningún
   caso mide lo único que el grafo podría aportar — un relevante alcanzable sólo por vecindad —, así que
   los estratos que deciden aquí miden sobre todo el **daño** a la recuperación directa. Se declara, no se
   fabrica un caso.
4. **La procedencia `real`.** Los 18 casos medidos son `construido`.
5. **El porcentaje de paths con sustento resoluble no es una medición aquí.** `graph_expand` rechaza entera
   una expansión con un id que el store no resuelve (`_require_resolvable`), así que un path servido sin
   sustento es un error, no un porcentaje: su 100 % no podría salir de otra manera (regla 2).
6. **Sólo se barren los dos umbrales.** `GRAPH_WEIGHT` (1,0), `GRAPH_SEEDS` (1) y
   `max_neighbors_per_node` (10) quedan en vigor y sin barrer — y la §4 muestra que son ellos, no el umbral,
   los que deciden el daño.

## 2. La regla, fijada antes de medir

Por orden (`evaluation.rank_graph_rows`, atada por `tests/test_knowledge_graph_sweep.py` sobre filas
construidas, de modo que ningún resultado puede moverla):

1. Una celda en la que el grafo no corrió en algún caso mide otra estrategia: va la última.
2. Una celda que pierde **más de 3 pp** de `precision@10` en algún estrato queda **descartada** (la
   definición de «degradar materialmente» del Plan 04 §3). Las descartadas van detrás de todas las demás.
3. Dentro de cada grupo decide el **Δ recall@10** frente a `hybrid` (útil), luego **menos ruido** (entrantes
   no relevantes), luego **menos degradación** de los resultados directos, luego el **grafo más disperso**
   y, entre grafos idénticos, los **umbrales menos restrictivos**.
4. Si ninguna celda sobrevive al paso 2, se aplica igualmente la primera — el índice siempre construye un
   grafo — y el veredicto dice con palabras que ninguna aporta y que `hybrid_graph` no se promueve.

**La cronología, porque la regla 2 lo exige.** El paso 2 y el orden del paso 3 hasta «grafo más disperso»
quedaron registrados en una pregunta al coordinador (`msg_5f599d175fcd`) **antes** de medir nada. Después,
y antes de escribir el código de la regla, una sonda de 4 celdas sobre el mismo corpus dio las mismas
cifras que la tabla para esas celdas. Tras la sonda se añadieron tres cosas que **no cambian el orden**:
el paso 1 (en las 16 celdas el grafo corrió), el paso 4 y el desempate entre grafos idénticos. El ganador
sale sin ellas: `5 / 0.05` gana a `2 / 0.05` y `3 / 0.05` por el grafo más disperso (160 frente a 164), un
criterio del orden registrado; el desempate nuevo sólo ordena `2 / 0.05` frente a `3 / 0.05`, que
persistieron el mismo grafo.

## 3. La tabla — las 16 combinaciones, en el orden de la regla

Base: `hybrid` respondido como `lexical` · `recall@10` **0,6301** sobre 18 casos medidos.

| min_shared_items | min_weight | aristas | grado medio | recall@10 | Δ recall@10 | entrantes | útiles | ruido | precisión entrantes | degradación | descartada por | en vigor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|:---:|
| 5 | 0.05 | 160 | 3.56 | 0.5093 | -0.1208 | 94 | 0 | 94 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 3.75 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 2 | 0.05 | 164 | 3.64 | 0.5093 | -0.1208 | 94 | 0 | 94 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 3.75 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 3 | 0.05 | 164 | 3.64 | 0.5093 | -0.1208 | 94 | 0 | 94 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 3.75 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 8 | 0.05 | 146 | 3.24 | 0.5093 | -0.1208 | 96 | 0 | 96 | 0.0000 | 4.1282 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 3.75 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 2 | 0.0 | 408 | 9.07 | 0.4609 | -0.1692 | 62 | 0 | 62 | 0.0000 | 3.0000 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) | sí |
| 3 | 0.0 | 372 | 8.27 | 0.4609 | -0.1692 | 63 | 0 | 63 | 0.0000 | 3.0449 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 3 | 0.02 | 326 | 7.24 | 0.4609 | -0.1692 | 66 | 0 | 66 | 0.0000 | 3.1987 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 2 | 0.02 | 339 | 7.53 | 0.4609 | -0.1692 | 66 | 0 | 66 | 0.0000 | 3.1987 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 5 | 0.0 | 316 | 7.02 | 0.4609 | -0.1692 | 67 | 0 | 67 | 0.0000 | 3.1859 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 5 | 0.02 | 301 | 6.69 | 0.4609 | -0.1692 | 70 | 0 | 70 | 0.0000 | 3.3013 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 8 | 0.0 | 249 | 5.53 | 0.4609 | -0.1692 | 74 | 0 | 74 | 0.0000 | 3.4038 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 8 | 0.02 | 247 | 5.49 | 0.4609 | -0.1692 | 76 | 0 | 76 | 0.0000 | 3.5064 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 2 | 0.1 | 52 | 1.16 | 0.4537 | -0.1763 | 99 | 0 | 99 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 5.00 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 3 | 0.1 | 52 | 1.16 | 0.4537 | -0.1763 | 99 | 0 | 99 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 5.00 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 5 | 0.1 | 52 | 1.16 | 0.4537 | -0.1763 | 99 | 0 | 99 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 5.00 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 8 | 0.1 | 52 | 1.16 | 0.4537 | -0.1763 | 99 | 0 | 99 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 5.00 pp en `semantico` (> 3 pp, Plan 04 §3) |  |

Columnas: **aristas** = filas `CO_OCCURS_WITH` persistidas (las dos direcciones); **grado medio** = esas
aristas por topic con asignaciones; **entrantes** = items del top-10 de `hybrid_graph` que no estaban en el
top-10 de `hybrid`, sumados sobre los 18 casos; **útiles** = los relevantes de esos; **degradación** = media
de puestos perdidos por los resultados que ya estaban en el top-10 de `hybrid`.

Pérdida de precisión en los estratos que la columna «descartada por» no nombra: `exacto` **0,00 pp** en las 16;
`enterrado` 2,50 pp y `multimodal` 1,67 pp en las 16 — por debajo del umbral, así que no descartan.

**No medidos, con su razón (spec §8.6.8):** S7, S8 y S9 — su verdad son topics y `search` sirve items, así
que su recall sería 0/0, no 0,0; F1 y F2 — declaran filtros (`created_from`, `created_to`, `source` /
`content_kinds`) que `hybrid_graph` no aplica, y puntuarlos sería fabricar un cero.

**Veredicto del instrumento, literal:** NINGUNA COMBINACIÓN APORTA: ninguna mejora recall@10 frente a
`hybrid` sin perder más de 3 pp de precisión en algún estrato. Se aplica min_shared_items=5,
min_weight=0.05, la primera por la regla (Δ recall@10 -0.1208, ruido 94, degradación 4.0705 puestos),
porque el índice siempre construye un grafo; `hybrid_graph` NO se promueve (Plan 04 §3).

## 4. Por qué ninguna aporta, y qué forma tiene el daño

**El daño no lo decide el umbral: lo decide el peso del término del grafo.** `rank_with_graph` suma a cada
item alcanzado un término RRF `GRAPH_WEIGHT / (RRF_K + puesto_en_el_grafo)` = `1 / (60 + p)` sobre el
`1 / (60 + puesto)` que ya tiene. El primer item alcanzado recibe `1/61` — exactamente la puntuación entera
del primer resultado léxico —, y en general el p-ésimo alcanzado queda por encima de todo resultado directo
a partir del puesto p, esté donde esté en el léxico dentro del horizonte de candidatos: basta con que el
canal léxico lo haya puntuado. Con una sola semilla
(`GRAPH_SEEDS = 1`) y dos saltos, lo que se alcanza son los otros items de los topics de la cabeza: vecinos
de tema, no respuestas. De ahí `útiles = 0` en las 16 celdas.

**El umbral sólo decide CUÁNTOS vecinos entran, y en sentido contrario al intuitivo.** `graph_expand` sirve
las aristas de un topic con un presupuesto de `max_neighbors_per_node = 10`, y las de coocurrencia (peso de
Jaccard > 0) van antes que las asignaciones a items (peso 0). Con un umbral permisivo el presupuesto de cada
topic se gasta en topics vecinos y se alcanzan **menos** items (62 entrantes con 408 aristas); con uno
estricto sobra presupuesto para items y entran **más** (99 con 52 aristas). Por eso la curva no es monótona:
`min_weight = 0.05` daña menos el recall que `0.0` y que `0.10`.

**Lo que haría falta para que el grafo aportase no es un umbral.** Barrer `GRAPH_WEIGHT` y `GRAPH_SEEDS`, y
poblar el estrato `expansión` para que exista un caso en el que el grafo *pueda* acertar, son trabajo
posterior y fuera de este PR. Hasta entonces el grafo sirve a `graph_expand` — explorar y explicar —, no a
ordenar resultados.

## 5. Qué cambia en el código

- `graph_build.DEFAULT_GRAPH_MIN_SHARED_ITEMS` pasa de 2 a **5** y `DEFAULT_GRAPH_MIN_WEIGHT` de 0.0 a
  **0.05**. `config.py` e `index_build.IndexOptions` los importan, así que el default de `[index]` y el
  bloque `graph` que sella cada build se mueven con ellos; `config.toml.example` documenta el valor.
- `tests/test_knowledge_graph_sweep.py` ata la línea **Umbral aplicado** de este documento al default del
  módulo, al de `load_config`, al de `IndexOptions`, al manifest de un build real y a `config.toml.example`,
  y exige que la tabla publique las 16 celdas de la rejilla del Plan 04 §1.3. Mover uno sin re-medir es rojo.
- **Un índice ya construido se pone al día con `xbrain index update`**, que reescribe el plano del grafo
  cuando los umbrales difieren de los que selló su manifest; el plano léxico no se toca.
- `GRAPH_ENABLED_BY_DEFAULT` y la estrategia por defecto de `search` **no cambian**.

## 6. Cómo re-derivarlo

Nada escribe en el store: el barrido lee los tres ficheros una vez y escribe su índice y su informe dentro
de la raíz que se le da. Los enlaces simbólicos mantienen el store donde está.

```bash
SRC=/ruta/al/clon/de/xbrain     # cualquier clon con el historial de VGonPa/xbrain
STORE=/ruta/al/store/data       # contiene items.json, vocab.yaml y topics.json medidos
WORK=$(mktemp -d)
git clone --no-checkout "$SRC" "$WORK/xbrain"
git -C "$WORK/xbrain" checkout --detach 719954c
(cd "$WORK/xbrain" && uv sync --locked)   # con el índice privado de pip de esta máquina: --index-url https://pypi.org/simple

ROOT=$WORK/root
mkdir -p "$ROOT/data"
for f in items.json vocab.yaml topics.json; do ln -s "$STORE/$f" "$ROOT/data/$f"; done
shasum -a 256 "$ROOT"/data/items.json "$ROOT"/data/vocab.yaml "$ROOT"/data/topics.json   # compara con la §1
printf '[paths]\nvault = "vault"\noutput_subdir = ""\ndata_dir = "data"\n\n[x]\nhandle = "u"\n' > "$ROOT/config.toml"

cd "$WORK/xbrain" && XBRAIN_REPO_ROOT="$ROOT" uv run xbrain eval --strategy hybrid_graph \
  --sweep-graph "min_shared_items=2,3,5,8 min_weight=0.0,0.02,0.05,0.10" \
  --golden-set "$WORK/xbrain/eval/golden-set.yaml"
```

Escribe `$ROOT/data/eval-graph-sweep.json` y `.md`, y el índice del barrido en
`$ROOT/data/eval-index/graph-sweep/`. **Sin embedder configurado** en ese `config.toml`, que es como se midió
(§1.1). Re-ejecutado desde un commit que ya contenga el umbral aplicado, la tabla es la misma y sólo cambia
la columna «en vigor», que pasa a marcar `5 / 0.05`.

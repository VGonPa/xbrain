# Barrido del umbral del grafo — Plan 04.5

**Fecha de la medición:** 2026-09-15, re-firmada el mismo día con el estrato `expansión` poblado y otra vez tras la
ronda 2 del PR #198 (U3 completo a 24 relevantes; el estrato leído de `graph_edges`) · **Estado:**
medición local firmada (Plan 04 §1.3, §11.7 y §11.9), no un check de CI · **Instrumento:**
`xbrain eval --strategy hybrid_graph --sweep-graph <rejilla>`

> Todas las cifras de este documento son una fotografía del corpus, del golden set y del código que se
> nombran en la §1, en ese momento. Se re-derivan con los comandos de la §6; no se citan después de que el
> corpus se mueva (CLAUDE.md, regla 2).

## 0. Resultado

**Umbral aplicado:** `min_shared_items = 5` · `min_weight = 0.05`

- **Ninguna de las 16 combinaciones aporta.** Todas empeoran `recall@10` frente a `hybrid`
  (Δ entre **−0,1204** y **−0,1759** sobre una base de **0,6296**), y todas pierden más de 3 pp de
  precisión en al menos un estrato (`cruzado_idioma` en las 16; `expansion` y `semantico` además en 8).
- **El estrato `expansión` (§11.9) está poblado y medido, y dice que no.** **33** pares (caso, relevante)
  quedan fuera del top 10 de `hybrid` y un camino del grafo persistido los une a la semilla, repartidos en
  **4** casos (D1b, D1c, P1, U3). En las 16 celdas el grafo metió **0 de esos 33** en el top 10, y los 62 a
  99 candidatos que sí metió eran, todos, no relevantes. Es la condición de salida de la fila 04.5 —«la
  expansión aporta, o se dice con números que no»— con el denominador que la primera firma no tenía.
- **La firma anterior decía «0 de esos 31», y el defecto era el denominador, no el cero.** U3 enumeraba 22
  de los 24 items que su propio criterio literal incluye; los dos que faltaban (capturados el 2026-09-11)
  también son alcanzables y tampoco entraron. Completarlo movió la población de 31 a 33, la base de
  0,6301 a 0,6296 y el recall y el Δ de cada celda en 0,0004 o menos; no movió el orden de las 16 filas, ni
  el ganador, ni una sola caída de precisión (§2, «Re-firmado tras la ronda 2»).
- **Se aplica `min_shared_items = 5`, `min_weight = 0.05`** porque el índice siempre construye un grafo y
  algún umbral tiene que estar en vigor: es la primera por la regla de la §2 — el menor daño a `recall@10`
  (empatado con las otras tres celdas de `min_weight = 0.05`), el menor ruido de esas cuatro (94) y, entre
  las tres que empatan también ahí, el grafo más disperso (160 aristas frente a 164). Sustituyó al par
  `2 / 0.0`, que el código declaraba «sin barrer».
- **`hybrid_graph` NO se promueve** (Plan 04 §3, criterio §11.7): `GRAPH_ENABLED_BY_DEFAULT` sigue en
  `False` y `search` sigue en `lexical` por defecto. Nada en este PR cambia qué estrategia sirve `search`.
- **Lo que sí mide el umbral es la forma del grafo.** Con `2 / 0.0` quedan 408 aristas y un grado medio de
  **9,07** frente a un tope de 10 vecinos por topic: casi todos los topics llenan el tope, que es el grafo
  que «deja de discriminar» del Plan 04 §1.1. Con `5 / 0.05` quedan 160 y un grado medio de 3,56.

## 1. Qué se midió, sobre qué, con qué instrumento

| | |
|---|---|
| Corpus | `store-2495` (`data/items.json` · `vocab.yaml` · `topics.json`; sus sha256, en la [tabla de versiones medidas](knowledge-index.md#measured-versions)) — 2.495 items · 45 topics. Los tres sha256 son idénticos antes y después de la corrida |
| Fingerprints del índice | `index-2495` en la misma tabla: `store_fingerprint` · `vocab_fingerprint` · `topics_fingerprint` (los que selló el manifest de la corrida) |
| Golden set | `eval/golden-set.yaml` v3, `golden@d1423c8` en la misma tabla: 23 casos, **18 medidos** (4 de ellos en el estrato `expansion`); 5 declarados no medibles (§3). U3 enumera 24 relevantes; en la firma anterior (`golden@a88c753`) enumeraba 22 |
| Código | xbrain `d1423c8`: el instrumento de `719954c`, la clasificación del estrato `expansion` de `92fd2d5` —que desde `ec162d3` lee la tabla `graph_edges` y no la salida de `graph_expand`— y sus etiquetas en el golden set. El umbral aplicado entró en `3beeea5`; este documento se re-firma en el commit siguiente |
| Rejilla | `min_shared_items ∈ {2, 3, 5, 8}` × `min_weight ∈ {0.0, 0.02, 0.05, 0.10}` — la del Plan 04 §1.3, entera |
| Profundidad | `k = 10` items por caso; un resultado directo expulsado de esa profundidad cuenta como puesto 11 |
| Recuperación | `search_service.search` — la única puerta en la que existe `hybrid_graph` — sobre un índice propio en `data/eval-index/graph-sweep/`, construido con el escritor de `xbrain index build` y reescrito celda a celda con `index update`; cada celda comprueba contra su manifest que midió los umbrales que dice |
| Máquina | Apple M2 · 16 GB · Python 3.13.7 · SQLite 3.50.4 · **317 s** de reloj para las 16 celdas más la clasificación del estrato `expansion` (ésta sola, 140 s leyendo `graph_expand` y 144 s leyendo `graph_edges`, en dos corridas aparte sobre el mismo store con `golden@a88c753`) |

**Lo que el instrumento NO mide, dicho antes de los números.**

1. **La base es léxica.** La corrida no tenía embedder configurado y el índice del barrido no tiene plano
   vectorial, así que `search --strategy hybrid` respondió `lexical` declarando
   `embeddings_not_configured, no_embeddings`, y `hybrid_graph` reordenó ese mismo ranking declarando lo
   mismo. El Δ es el del grafo sobre el ranking léxico. Sobre el ranking fusionado de un `hybrid` con
   vectores **no se ha medido**, y `hybrid` tampoco está promovido (bake-off del Plan 03.7).
2. **La unidad es el item que sirve `search`**, no el owner que puntúa `xbrain eval`: estas cifras no son
   comparables con el `recall@10` 0,7395 del arnés.
3. **La población del estrato `expansión` es relativa a esa base léxica.** «Directo» es «en el top 10 de
   `hybrid` respondido `lexical`»: con un plano vectorial parte de los 33 pares puede volverse directa, y
   entonces la etiqueta del golden set se revisa. El barrido re-deriva la población en cada corrida y nombra
   la deriva entre etiqueta y medición en los dos sentidos (en esta corrida, ninguna). La población sale de
   los relevantes ya verificados del golden set: no se escribió ninguna pregunta nueva (§2).
4. **La procedencia `real`.** Los 18 casos medidos son `construido`.
5. **El porcentaje de paths con sustento resoluble no es una medición aquí.** `graph_expand` rechaza entera
   una expansión con un id que el store no resuelve (`_require_resolvable`), así que un path servido sin
   sustento es un error, no un porcentaje: su 100 % no podría salir de otra manera (regla 2).
6. **Sólo se barren los dos umbrales.** `GRAPH_WEIGHT` (1,0), `GRAPH_SEEDS` (1) y
   `max_neighbors_per_node` (10) quedan en vigor y sin barrer — y la §4 muestra que son ellos, no el umbral,
   los que deciden el daño.
7. **La verdad de terreno de los casos definidos por un literal se re-censó sobre este store** (ronda 2 del
   PR #198): D1a, D1b, D1c, P1, P2, S3, V1, X1, X2, X3, F1 y F2 coinciden con su enumeración; U3 no coincidía
   y se completó (22 → 24). **V2 se sigue midiendo con una fuga del anexo A.3** («Hashimoto» en 3
   `x_article`, ya declarada en `docs/embeddings-bakeoff.md` §2): es un defecto de otra clase —una pregunta
   que no discrimina, no una población incompleta— y este PR no lo toca.

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

**El estrato `expansión`, fijado igual.** Cada par (caso medido, relevante) se clasifica contra el ranking
que `hybrid_graph` re-ordena y la semilla desde la que expande (`evaluation.classify_expansion`):
`direct` si está en el top 10; `graph_reachable` si no lo está, un camino de a lo sumo `GRAPH_MAX_HOPS`
aristas de la tabla `graph_edges` lo une a la semilla —**sin presupuesto de vecinos**— y un canal lo puntuó
dentro de `GRAPH_CANDIDATE_HORIZON`; `graph_unscored` si ese camino existe pero nadie lo puntuó (el grafo
nunca admite lo no puntuado, Plan 04 §3.4); `unreachable` en otro caso. Un caso lleva la etiqueta si tiene al menos un par
`graph_reachable`. Esa regla se escribió **antes** de leer ninguna salida de la sonda que la aplicó. El
presupuesto queda fuera a propósito: es lo que decide si el grafo llega a tiempo, que es lo que el barrido
mide, y aplicarlo llamaría «sin cobertura» a un fallo del grafo. La población son los relevantes del golden
set, cuya verdad ya estaba verificada bajo el anexo A.3: una pregunta escrita después de ver qué alcanza el
grafo sería verdad de terreno hecha a la medida del mecanismo (spec §8.3).

**Re-firmado con el estrato poblado.** Las etiquetas no mueven ningún recall, ruido ni degradación —son los
mismos 18 casos—, así que el orden y el ganador son los de la primera firma. Lo que cambia es que `expansion`
entra en la columna «descartada por» y que la columna «útiles» gana su denominador.

**Lo que la pertenencia lee, y lo que no.** Tres cosas: la verdad de terreno, el ranking `hybrid` —la base
que `hybrid_graph` re-ordena, sin la cual «pero no directamente» no significa nada— y la tabla `graph_edges`,
recorrida en anchura. **Nunca la salida de `graph_expand` ni la de `hybrid_graph`.** La primera versión
preguntaba a `graph_expand` qué alcanzaba, y la ronda 2 del PR #198 lo reprodujo: con la misma pregunta y los
mismos relevantes, omitir un nodo de la respuesta del servicio pasaba ese par de `graph_reachable` a
`unreachable`. La población contra la que se lee «útiles» se movía con el servicio cuya ayuda esa columna
mide. Ahora un test falsea esa respuesta de tres maneras y exige que no se mueva un solo par; y sobre este
store la clasificación nueva y la antigua dieron los 58 pares de `golden@a88c753` idénticos byte a
byte: la definición no cambió, cambió de dónde se lee.

**Re-firmado tras la ronda 2.** U3 enumeraba 22 relevantes y su criterio literal («harness engineering» en
alguna superficie) da 24 sobre este store: `2094209008580325405` y `2097020722669289481`, capturados el
2026-09-11, después de enumerar. El caso no declara ventana, así que se completó en vez de excluirlos;
`docs/embeddings-bakeoff.md` §2 ya los contaba. Los dos son `graph_reachable` (puestos 438 y 441 de `hybrid`)
y ninguno entró en el top 10 de ninguna celda. Movió la población (31 → 33), la base (0,6301 → 0,6296) y el
recall y el Δ de cada celda, en 0,0004 o menos. No movió el orden de las 16 filas, el ganador, `útiles = 0`
ni ninguna caída de precisión. Había tres maneras de que saliera otra cosa y no ocurrió ninguna: que una
celda levantara uno de los dos (útiles ≠ 0), que alguno fuera directo (la población se habría quedado en
31) o que el nuevo denominador de U3 invirtiera dos celdas cercanas.

## 3. La tabla — las 16 combinaciones, en el orden de la regla

Base: `hybrid` respondido como `lexical` · `recall@10` **0,6296** sobre 18 casos medidos.

| min_shared_items | min_weight | aristas | grado medio | recall@10 | Δ recall@10 | entrantes | útiles | ruido | precisión entrantes | degradación | descartada por | en vigor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|:---:|
| 5 | 0.05 | 160 | 3.56 | 0.5093 | -0.1204 | 94 | 0 | 94 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 7.50 pp en `expansion` (> 3 pp, Plan 04 §3); la precisión cae 3.75 pp en `semantico` (> 3 pp, Plan 04 §3) | sí |
| 2 | 0.05 | 164 | 3.64 | 0.5093 | -0.1204 | 94 | 0 | 94 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 7.50 pp en `expansion` (> 3 pp, Plan 04 §3); la precisión cae 3.75 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 3 | 0.05 | 164 | 3.64 | 0.5093 | -0.1204 | 94 | 0 | 94 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 7.50 pp en `expansion` (> 3 pp, Plan 04 §3); la precisión cae 3.75 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 8 | 0.05 | 146 | 3.24 | 0.5093 | -0.1204 | 96 | 0 | 96 | 0.0000 | 4.1282 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 7.50 pp en `expansion` (> 3 pp, Plan 04 §3); la precisión cae 3.75 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 2 | 0.0 | 408 | 9.07 | 0.4606 | -0.1690 | 62 | 0 | 62 | 0.0000 | 3.0000 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 3 | 0.0 | 372 | 8.27 | 0.4606 | -0.1690 | 63 | 0 | 63 | 0.0000 | 3.0449 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 3 | 0.02 | 326 | 7.24 | 0.4606 | -0.1690 | 66 | 0 | 66 | 0.0000 | 3.1987 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 2 | 0.02 | 339 | 7.53 | 0.4606 | -0.1690 | 66 | 0 | 66 | 0.0000 | 3.1987 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 5 | 0.0 | 316 | 7.02 | 0.4606 | -0.1690 | 67 | 0 | 67 | 0.0000 | 3.1859 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 5 | 0.02 | 301 | 6.69 | 0.4606 | -0.1690 | 70 | 0 | 70 | 0.0000 | 3.3013 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 8 | 0.0 | 249 | 5.53 | 0.4606 | -0.1690 | 74 | 0 | 74 | 0.0000 | 3.4038 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 8 | 0.02 | 247 | 5.49 | 0.4606 | -0.1690 | 76 | 0 | 76 | 0.0000 | 3.5064 | la precisión cae 4.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3) |  |
| 2 | 0.1 | 52 | 1.16 | 0.4537 | -0.1759 | 99 | 0 | 99 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 7.50 pp en `expansion` (> 3 pp, Plan 04 §3); la precisión cae 5.00 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 3 | 0.1 | 52 | 1.16 | 0.4537 | -0.1759 | 99 | 0 | 99 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 7.50 pp en `expansion` (> 3 pp, Plan 04 §3); la precisión cae 5.00 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 5 | 0.1 | 52 | 1.16 | 0.4537 | -0.1759 | 99 | 0 | 99 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 7.50 pp en `expansion` (> 3 pp, Plan 04 §3); la precisión cae 5.00 pp en `semantico` (> 3 pp, Plan 04 §3) |  |
| 8 | 0.1 | 52 | 1.16 | 0.4537 | -0.1759 | 99 | 0 | 99 | 0.0000 | 4.0705 | la precisión cae 6.00 pp en `cruzado_idioma` (> 3 pp, Plan 04 §3); la precisión cae 7.50 pp en `expansion` (> 3 pp, Plan 04 §3); la precisión cae 5.00 pp en `semantico` (> 3 pp, Plan 04 §3) |  |

Columnas: **aristas** = filas `CO_OCCURS_WITH` persistidas (las dos direcciones); **grado medio** = esas
aristas por topic con asignaciones; **entrantes** = items del top-10 de `hybrid_graph` que no estaban en el
top-10 de `hybrid`, sumados sobre los 18 casos; **útiles** = los relevantes de esos, que por construcción son
pares del estrato `expansion`; **degradación** = media de puestos perdidos por los resultados que ya estaban
en el top-10 de `hybrid`.

Pérdida de precisión en los estratos que la columna «descartada por» no nombra: `exacto` **0,00 pp** en las 16;
`enterrado` 2,50 pp y `multimodal` 1,67 pp en las 16; `expansion` y `semantico` 2,50 pp en las 8 celdas de
`min_weight` 0.0 y 0.02 — por debajo del umbral, así que no descartan.

**Estrato `expansión`, literal del instrumento:** Estrato `expansion` (Plan 04 §11.9): 33 pares relevantes alcanzables sólo por el grafo en 4 casos (D1b, D1c, P1, U3) — 18 directos, 0 alcanzables sin puntuar, 9 inalcanzables. La columna «útiles» cuenta cuántos de esos 33 entraron en el top 10.
En las 16 celdas, **útiles = 0**: ninguno de los 33.

**No medidos, con su razón (spec §8.6.8):** S7, S8 y S9 — su verdad son topics y `search` sirve items, así
que su recall sería 0/0, no 0,0; F1 y F2 — declaran filtros (`created_from`, `created_to`, `source` /
`content_kinds`) que `hybrid_graph` no aplica, y puntuarlos sería fabricar un cero.

**Veredicto del instrumento, literal:** NINGUNA COMBINACIÓN APORTA: ninguna mejora recall@10 frente a
`hybrid` sin perder más de 3 pp de precisión en algún estrato. Se aplica min_shared_items=5,
min_weight=0.05, la primera por la regla (Δ recall@10 -0.1204, ruido 94, degradación 4.0705 puestos),
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

**Y no es por falta de población.** Los 33 pares que sólo el grafo podía levantar existen, y ninguna celda
levanta uno. Lo que dice el código —no medido celda a celda—: dentro de ese presupuesto de 10, las
asignaciones de un topic se sirven ordenadas por peso, primario antes que secundario e id del item
(`graph_service._ranked`), así que qué vecinos entran no depende en nada de la pregunta. Barrer
`GRAPH_WEIGHT`, `GRAPH_SEEDS`, `max_neighbors_per_node` o ese orden es trabajo posterior y fuera de este PR;
el estrato ya existe para medirlo. Hasta entonces el grafo sirve a `graph_expand` — explorar y explicar —,
no a ordenar resultados.

## 5. Qué cambia en el código

- `graph_build.DEFAULT_GRAPH_MIN_SHARED_ITEMS` pasa de 2 a **5** y `DEFAULT_GRAPH_MIN_WEIGHT` de 0.0 a
  **0.05**. `config.py` e `index_build.IndexOptions` los importan, así que el default de `[index]` y el
  bloque `graph` que sella cada build se mueven con ellos; `config.toml.example` documenta el valor.
- `tests/test_knowledge_graph_sweep.py` **deriva el ganador de los números de la tabla**, no de la línea que
  lo nombra: reconstruye cada fila (exigiendo que re-renderice byte a byte en la publicada), la re-ordena con
  `rank_graph_rows` — el orden publicado tiene que ser el que la regla da a esos números — y toma
  `GraphSweepReport.winner`. La línea **Umbral aplicado**, el veredicto literal de la §3, el default del
  módulo, el de `load_config`, el manifest de un build real y `config.toml.example` tienen que nombrar esa
  celda, y la tabla tiene que publicar las 16 de la rejilla del Plan 04 §1.3. Mover la línea, los defaults y
  el ejemplo **juntos** a otra celda es rojo, porque los números medidos no se movieron con ellos.
- `evaluation.classify_expansion` y el bloque `expansion` del informe del barrido (población, casos, pares
  por clase y la etiqueta contrastada con la medición); `eval/golden-set.yaml` etiqueta `expansion` en D1b,
  D1c, P1 y U3, con la medición en sus notas.
- **Ronda 2 del PR #198.** `classify_expansion` lee la alcanzabilidad de la tabla `graph_edges` y ya no de
  `graph_expand`, y conserva la negativa ante un índice por detrás del store; `eval/golden-set.yaml` completa
  U3 a 24 relevantes. Tres tests nuevos en `tests/test_knowledge_graph_sweep.py`: falsear la salida del
  servicio no mueve ningún par, la fuente es la tabla y un índice obsoleto se rechaza.
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
git -C "$WORK/xbrain" checkout --detach d1423c8
(cd "$WORK/xbrain" && uv sync --locked)   # con el índice privado de pip de esta máquina: --index-url https://pypi.org/simple
shasum -a 256 "$WORK/xbrain/eval/golden-set.yaml"                                   # compara con golden@d1423c8 (tabla de versiones)

ROOT=$WORK/root
mkdir -p "$ROOT/data"
for f in items.json vocab.yaml topics.json; do ln -s "$STORE/$f" "$ROOT/data/$f"; done
shasum -a 256 "$ROOT"/data/items.json "$ROOT"/data/vocab.yaml "$ROOT"/data/topics.json   # compara con store-2495 (tabla de versiones)
printf '[paths]\nvault = "vault"\noutput_subdir = ""\ndata_dir = "data"\n\n[x]\nhandle = "u"\n' > "$ROOT/config.toml"

cd "$WORK/xbrain" && XBRAIN_REPO_ROOT="$ROOT" uv run xbrain eval --strategy hybrid_graph \
  --sweep-graph "min_shared_items=2,3,5,8 min_weight=0.0,0.02,0.05,0.10" \
  --golden-set "$WORK/xbrain/eval/golden-set.yaml"
```

Escribe `$ROOT/data/eval-graph-sweep.json` y `.md` —con el bloque `expansion`: la población, sus pares
clase a clase y la deriva de la etiqueta— y el índice del barrido en `$ROOT/data/eval-index/graph-sweep/`.
**Sin embedder configurado** en ese `config.toml`, que es como se midió (§1.1). Desde `a88c753` el umbral en
vigor ya es `5 / 0.05`, así que la columna «en vigor» marca esa fila, como en la tabla.

# Bake-off de embeddings — Plan 03.7

**Fecha de la medición:** 2026-09-13 · **Estado:** medición local firmada (Plan 03 §13.8–§13.10), no un
check de CI · **Instrumento:** `xbrain eval --strategy vector|hybrid --embeddings-model <m>`

> Todas las cifras de este documento son una fotografía del corpus, del golden set y de la máquina que se
> nombran en la §1, en ese momento. Se re-derivan con los comandos de la §9; no se citan después de que el
> corpus se mueva (CLAUDE.md, regla 2).

## 0. Resultado

**No hay ganador, y el bake-off está INCOMPLETO: se midió 1 candidato de los ≥ 3 que exige el criterio
§13.8.** Se publica así, con los números, en vez de rellenar el hueco.

- **Medido: `paraphrase-multilingual-MiniLM-L12-v2`, el suelo barato, en `vector` y en `hybrid`.
  Pierde en las dos.** `vector` recupera los 4 casos cruzados que deciden (recall@10 0,75 → 1,00) pero
  borra el estrato exacto (0,75 → 0,00) y empeora el semántico; `hybrid` conserva los exactos y **no
  mejora ni el semántico (recall@10 0,7024 → 0,6548) ni el cruzado (0,75 → 0,75, MRR@20 0,625 → 0,5833)**.
- **No medidos: `multilingual-e5-small` (corrida interrumpida a los 859 s), `multilingual-e5-base` (no
  arrancó), `bge-m3` y `jina-embeddings-v3` (no se corrieron).** La máquina estaba en presión de memoria
  antes de la primera corrida y el swap crece en el mismo contenedor APFS que el disco: la corrida de
  e5-small añadió 1 GB de swap, dejó el disco en **187 MiB libres** y se paró de forma controlada (§6.3).
- **`hybrid` NO se promueve a default** (spec §8.6.4, §13.15): el único candidato medido no mejora los dos
  estratos que deciden. Esto **no** es «ningún modelo bate al léxico»: es «el suelo no lo bate, y los que
  podrían no se han medido todavía».
- **`fusion.py` no cambia.** El barrido de `RRF_K` y pesos no se ejecutó (§8), y aunque se hubiera
  ejecutado sobre MiniLM no podría justificar mover las constantes de un `hybrid` que no se promueve.
- **Hallazgo de coste que no depende del modelo:** con el embedder de referencia cada consulta paga una
  carga del modelo en un subproceso — **p50 5,5–7,2 s sólo en embeber** frente a **31,6 ms** del léxico
  completo (§6.2). `hybrid` por defecto exige antes un embedder persistente, sea cual sea el ganador.

**Corrección tras la revisión del PR #186.** La primera versión firmada leía el MRR sobre la ventana del
recuperador y no sobre los 20 owners que publica: un rango que el informe no mostraba decidía un desempate.
Todos los MRR de este documento son ahora **MRR@20**: los léxicos, re-medidos con el arnés corregido; los de
`vector`/`hybrid`, recalculados sin modelo a partir de los rankings conservados (§9.5). **Un solo veredicto
cambia:** `hybrid` en `exacto` pasa de «empeora» a **«empata»** (§5.2). La decisión no cambia. También se
publican ahora las poblaciones de cada media de superficies (§3, §5.1) y los comandos efectivos de la
corrida (§9).

---

## 1. Qué se midió, sobre qué, con qué instrumento

| | |
|---|---|
| Corpus | `data/items.json` sha256 `4fed54a0…` — 2.474 items · 45 topics · 10.570 superficies · 22.933 chunks (chunker `v3`, `800/0`) · `vocab.yaml` sha256 `e73fbede…` · `topics.json` sha256 `7a40f4f1…` (sha256 completos en la §9.2) |
| Golden set | `eval/golden-set.yaml` v3 tal como está versionado (sha256 `ed6dd760…`): 23 casos puntuables + 8 escenarios archivados; **los 23 resuelven** contra ese store |
| Profundidad | 20 owners por caso, `k ∈ {1, 5, 10, 20}` — la misma para las tres estrategias. **Todas las métricas por owner se leen de esos 20 owners publicados (`retrieved`), MRR incluido: es MRR@20.** La ventana que el recuperador entrega es más honda y distinta por estrategia (léxico: 4 chunks por owner, ampliable; `vector`/`hybrid`: hasta 1.000 chunks por canal) y no se puntúa más allá de esos 20 |
| Índice vectorial | uno por modelo en `data/eval-index/<modelo>/`, escrito por `index_build.build(..., vectors=…)`: el mismo escritor y el mismo plano que `xbrain index build --embeddings` |
| `vector` / `hybrid` | la ventana fusionada del propio `search_service` (`FUSED_CHUNK_WINDOW` = 1.000 chunks por canal) y RRF con las constantes de `fusion.py` en vigor (`RRF_K` = 60, pesos 1 / 1); se puntúa por OWNER sobre el ranking de chunks, igual que la línea base léxica |
| Embedder | `scripts/xbrain-embed` (sentence-transformers 6.0.1 · torch 2.14.0 · MPS) vía `[embeddings].command`, pesos `safetensors` en caché local, `HF_HUB_OFFLINE=1`, `batch_size` 1.024 |
| Máquina | Apple M2 · 16 GB · **el swap ya estaba en uso antes de la primera corrida**: 13,2 GB usados de 14,3 GB, por otras sesiones abiertas |

**Lo que el instrumento NO mide, dicho antes de los números.**

1. **El plano de perfiles.** La línea base léxica tampoco lo puntúa, y `hybrid` se compara con ella en la
   misma unidad; el relleno por perfil de `hybrid` (Plan 03 §4.3) queda fuera de esta medición.
2. **Los filtros bajo `vector` y `hybrid`.** El plano vectorial no tiene columnas de filtro y un filtro
   aplicado después de puntuar no es un filtro, así que **F1 y F2 quedan NO MEDIDOS** en esas dos
   estrategias — jamás 0,0 — y el estrato `filtros` sólo tiene número léxico.
3. **La procedencia `real`.** Los 23 casos son `construido`: la columna `real` sale `sin cobertura` en las
   tres estrategias. Lo que decide aquí es regresión construida, no utilidad real (spec §8.2).
4. **La latencia de un servidor de embeddings persistente.** `xbrain-embed` carga el modelo en cada
   invocación y `search` lo invoca una vez por consulta: la latencia publicada es la de ese contrato.

## 2. Precondición: qué casos deciden, id a id (Plan 03 §3.3)

Antes de correr nada se listaron los casos de los estratos que deciden y se comprobó, sobre el store de la
§1, que sus ids resuelven (los 23 lo hacen) y — donde la verdad de terreno se define por un literal — que la
población enumerada sigue siendo la de hoy.

| Caso | Estratos | Enumerado | Comprobación de hoy | Decide |
|---|---|---:|---|---|
| **D1a** | semántico · cruzado · enterrado | 1 | «Leadership, Lab, and Crowd» sólo en el `external_article` de `1958350546831630810` | **sí** |
| **D1b** | semántico · cruzado | 1 | «80s computer terminal / GUI hasn't been invented»: 1 item, el enumerado | **sí** |
| **D1c** | semántico · enterrado | 12 | «context rot»: **12 items, exactamente los 12 enumerados** | **sí** |
| **P1** | semántico · enterrado | 6 | «Simon Willison» o `@simonw`: 6 items, los 6 enumerados | **sí** |
| **P2** | semántico | 1 | el handle vive en la atribución, no en el texto: `JosephJacks_` es autor de un solo `quoted_post`, el enumerado | **sí** |
| S1 · S2 · S4 · D1d | semántico / cruzado | 1 c/u | verificados leyendo en v1/v2 y resuelven hoy; su hecho es una combinación que un literal no re-deriva | sí, declarado |
| V1 · X1 · X2 · X3 | exacto | 1–3 | cada literal re-derivado: 1 · 3 · 2 · 1 items, en las superficies nombradas | sí (guardarraíl) |
| **U3** | semántico · cruzado | 22 | «harness engineering»: **24 items hoy** (`2094209008580325405`, `2097020722669289481` llegaron después) | **no** |
| **S8** | topic · semántico | topic | «Tim Urban» ya **no** está en la nota de `agency-and-mindset` (re-sintetizada el 2026-09-11 14:12 UTC); hoy vive en el item `1901711876318204315` | **no** |
| **V2** | enterrado · exacto | 1 | fuga (anexo A.3): «Hashimoto … harness» en **3** `x_article` (`2047145274200768969`, `2050631735529095575`, `2051019159488663824`) | **no** |

Fuera de los estratos que deciden, **S7 y S9** (estrato `topic`) tienen el mismo defecto que S8: sus hechos ya
no están en la nota del topic. Se publican aquí y no se tocan: el golden set no es de este PR.

**El conjunto que decide:** `cruzado_idioma` — **S1, S2, D1a, D1b** (4) · `semantico` — **S4, D1a, D1b,
D1c, D1d, P1, P2** (7) · `exacto`, el guardarraíl — **V1, X1, X2, X3** (4). Las exclusiones viajan como
argumento de `compare_reports(..., exclude=…)`, con su razón; la §5.2 publica también la comparación **sin**
ellas.

## 3. Línea base léxica

Misma corrida, mismo corpus, misma profundidad. Todos los casos puntuables:

| estrato | casos | recall@1 | recall@10 | precision@10 | MRR@20 | nDCG@10 | superficies@10 | casos que miden superficies |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cruzado_idioma | 5 | 0.4091 | 0.6182 | 0.1 | 0.7 | 0.5835 | 0.25 | 4 — sin U3 |
| enterrado | 8 | 0.5208 | 0.7396 | 0.25 | 0.6687 | 0.6464 | 0.5 | 6 — sin D1c, P1 |
| exacto | 5 | 0.3667 | 0.8 | 0.62 | 0.62 | 0.6578 | 0.8 | 5 |
| filtros | 2 | 0.4167 | 1.0 | 1.0 | 1.0 | 1.0 | sin cobertura | 0 — F1, F2 no nombran superficies |
| multimodal | 6 | 0.75 | 0.8333 | 0.2333 | 0.8333 | 0.8333 | 0.6667 | 6 |
| resumen | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| semantico | 9 | 0.3569 | 0.5564 | 0.1333 | 0.6468 | 0.5333 | 0.1667 | 6 — sin D1c, P1, U3 |
| topic | 3 | 0.3333 | 0.3333 | 0.0333 | 0.3571 | 0.3333 | 0.3333 | 3 |
| expansion | — | sin cobertura | sin cobertura | sin cobertura | — | — | — | — |
| **construido** | **23** | 0.5165 | **0.7395** | 0.3 | **0.7357** | 0.6995 | 0.5 | 18 — sin D1c, P1, U3, F1, F2 |

**Poblaciones.** `casos` es el denominador de recall, precision, MRR@20 y nDCG en todas las filas: los 23
casos enumeran owners y ninguno volvió vacío, así que su `measured` coincide con `casos` en el JSON.
`superficies@10` **no** se divide por `casos`: sólo promedia los casos que nombran superficies (última
columna). D1c, P1 y U3 no nombran ninguna, y F1 y F2 tampoco. Así, el 0,1667 semántico es la media de **6**
casos, no de 9, y el 0,5 construido la de **18**, no de 23.

**Unidades de k.** `recall@k`, `precision@k` y `nDCG@k` cuentan OWNERS: los k primeros owners distintos del
ranking. `superficies@k` cuenta CHUNKS: los k primeros chunks de la ventana, que pueden pertenecer a menos de
k owners. `recall@10` y `superficies@10` no son comparables entre sí; cada una sólo se compara con su propio
valor en otra corrida.

El `recall@10` de 0,7395 es el par publicado por el Plan 02 para `800/0`, y el MRR@20 sale **0,7357**. La
primera versión de este informe publicaba 0,7366 y lo atribuía a la profundidad (20 owners aquí, 10 allí).
La causa era otra: V1 aportaba 1/51 desde un rango que no está entre los 20 owners publicados. El 0,7357 del
Plan 02 tampoco era un MRR@10: se leyó sobre la ventana de profundidad 10 e incluía a S8 en el rango 14.
Leído de los 10 owners publicados, sale 0,7326 (re-medido el 2026-09-13 con el mismo corpus). **Reproducible:**
antes de la corrección, dos corridas léxicas seguidas dieron los mismos `retrieved` y las mismas métricas en
los 23 casos. Después de ella, el comando léxico de la §9.4 reproduce esta tabla celda a celda.

## 4. Candidatos

| Candidato | Dim | Estado | Por qué |
|---|---:|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | **medido** (`vector` y `hybrid`) | el suelo barato |
| `intfloat/multilingual-e5-small` | 384 | **interrumpido** a los 859 s de `vector`, construyendo el plano | presión de memoria y disco (§6.3) |
| `intfloat/multilingual-e5-base` | 768 | **no arrancó** | la corrida se detuvo antes de llegar a él |
| `BAAI/bge-m3` | 1024 | **no se corrió** | ~2,3 GB de pesos con 0,9–1,3 GiB libres |
| `jinaai/jina-embeddings-v3` | 1024 | **no se corrió** | el mismo disco; y sus adaptadores de tarea (`retrieval.query` / `retrieval.passage`) no se pueden elegir con el wrapper de referencia, que sólo aplica prefijos: medirlo sin ellos sería medir otro modelo |

Prefijos: ninguno para MiniLM; `query: ` / `passage: ` para la familia E5, que los exige.

## 5. Resultados

### 5.1 Por estrato, todos los casos — recall@10 · MRR@20 · nDCG@10 · superficies@10

Cada celda promedia recall@10, MRR@20 y nDCG@10 sobre **todos** los casos de su columna, y `superficies@10`
(unidad: chunk, §3) sólo sobre los que nombran superficies. Las poblaciones son las mismas en las tres
estrategias salvo donde se indica:

| columna | casos | miden superficies@10 | sin superficies enumeradas |
|---|---:|---:|---|
| cruzado_idioma | 5 | 4 | U3 |
| semantico | 9 | 6 | D1c, P1, U3 |
| exacto | 5 | 5 | — |
| enterrado | 8 | 6 | D1c, P1 |
| multimodal | 6 | 6 | — |
| topic | 3 | 3 | — |
| resumen | 1 | 1 | — |
| filtros | 2 en léxico; no medido en `vector`/`hybrid` | 0 | F1, F2 |
| construido, léxico | 23 | 18 | D1c, P1, U3, F1, F2 |
| construido, `vector`/`hybrid` | 21 | 18 | D1c, P1, U3 (F1 y F2 no medidos) |

| estrategia | cruzado_idioma | semantico | exacto | enterrado | multimodal | topic | resumen | filtros | construido |
|---|---|---|---|---|---|---|---|---|---|
| lexical | 0.6182 · 0.7000 · 0.5835 · 0.2500 | 0.5564 · 0.6468 · 0.5333 · 0.1667 | 0.8000 · 0.6200 · 0.6578 · 0.8000 | 0.7396 · 0.6687 · 0.6464 · 0.5000 | 0.8333 · 0.8333 · 0.8333 · 0.6667 | 0.3333 · 0.3571 · 0.3333 · 0.3333 | 0.0000 · 0.0000 · 0.0000 · 0.0000 | 1.0000 · 1.0000 · 1.0000 · sin cobertura | 0.7395 · 0.7357 · 0.6995 · 0.5000 |
| MiniLM `vector` | 0.8091 · 0.2508 · 0.3799 · 0.5000 | 0.5051 · 0.3602 · 0.3728 · 0.5000 | 0.2000 · 0.2000 · 0.2000 · 0.2000 | 0.6875 · 0.4241 · 0.4736 · 0.6667 | 0.6667 · 0.5833 · 0.6052 · 0.5000 | 0.3333 · 0.1111 · 0.1667 · 0.3333 | 0.0000 · 0.0000 · 0.0000 · 0.0000 | **no medido** (F1, F2) | 0.5498 · 0.3687 · 0.4041 · 0.4444 |
| MiniLM `hybrid` | 0.6091 · 0.6667 · 0.5440 · 0.2500 | 0.5143 · 0.6852 · 0.5378 · 0.1667 | 0.8000 · 0.8000 · 0.7733 · 0.8000 | 0.6979 · 0.6875 · 0.6400 · 0.5000 | 0.8333 · 0.8333 · 0.8200 · 0.8333 | 0.3333 · 0.3333 · 0.3333 · 0.3333 | 0.0000 · 0.0000 · 0.0000 · 0.0000 | **no medido** (F1, F2) | 0.6966 · 0.7381 · 0.6765 · 0.5556 |

`construido` en `vector`/`hybrid` promedia 21 casos (F1 y F2 no medidos) y en léxico 23: **no se comparan
entre sí**; la comparación que decide es la pareada de la §5.2.

### 5.2 Puertas §8.6.3 y §8.6.4 contra léxico, pareadas por caso (`compare_reports`)

Las dos estrategias de cada fila tienen la misma profundidad (`limit` = 20 owners), y `compare_reports` se
niega a comparar dos informes con profundidades distintas: el MRR@20 que desempata se lee con la misma regla
en los dos lados.

**Con las exclusiones de la §2 — el conjunto que decide:**

| estrategia | estrato | casos pareados | léxico recall@10 · MRR@20 | MiniLM recall@10 · MRR@20 | veredicto |
|---|---|---|---|---|---|
| `vector` | exacto | V1, X1, X2, X3 | 0.7500 · 0.7500 | 0.0000 · 0.0000 | **empeora** |
| `vector` | semantico | D1a, D1b, D1c, D1d, P1, P2, S4 | 0.7024 · 0.6786 | 0.6429 · 0.4473 | **empeora** |
| `vector` | cruzado_idioma | D1a, D1b, S1, S2 | 0.7500 · 0.6250 | 1.0000 · 0.2857 | mejora |
| `hybrid` | exacto | V1, X1, X2, X3 | 0.7500 · 0.7500 | 0.7500 · 0.7500 | empata |
| `hybrid` | semantico | D1a, D1b, D1c, D1d, P1, P2, S4 | 0.7024 · 0.6786 | 0.6548 · 0.7381 | **empeora** |
| `hybrid` | cruzado_idioma | D1a, D1b, S1, S2 | 0.7500 · 0.6250 | 0.7500 · 0.5833 | **empeora** |

`hybrid` **empata** en `exacto`: X1, X2 y X3 siguen en 1,0 / 1,0, y V1 — que ninguna estrategia encuentra —
no está entre los 20 owners de ninguna de las dos (MRR@20 0 en ambas). La primera versión publicaba aquí
«empeora» (MRR 0,7549 → 0,7545), pero ese movimiento era V1 puntuado desde rangos que el informe no
publicaba: 51 en léxico y 56 en `hybrid`, dos ventanas distintas comparadas como si fueran una. Corregido el
horizonte, cambia el veredicto de `exacto` y no cambia la decisión: `hybrid` sigue sin mejorar el semántico y
el cruzado.

**Sin exclusiones — sensibilidad:**

| estrategia | estrato | casos | léxico recall@10 · MRR@20 | MiniLM recall@10 · MRR@20 | veredicto |
|---|---|---|---|---|---|
| `vector` | exacto | V1, V2, X1, X2, X3 | 0.8000 · 0.6200 | 0.2000 · 0.2000 | empeora |
| `vector` | semantico | + S8, U3 | 0.5564 · 0.6468 | 0.5051 · 0.3602 | empeora |
| `vector` | cruzado_idioma | + U3 | 0.6182 · 0.7000 | 0.8091 · 0.2508 | mejora |
| `hybrid` | exacto | V1, V2, X1, X2, X3 | 0.8000 · 0.6200 | 0.8000 · 0.8000 | mejora |
| `hybrid` | semantico | + S8, U3 | 0.5564 · 0.6468 | 0.5143 · 0.6852 | empeora |
| `hybrid` | cruzado_idioma | + U3 | 0.6182 · 0.7000 | 0.6091 · 0.6667 | empeora |

Las exclusiones **no cambian la decisión**: con o sin ellas, `hybrid` no mejora ni el semántico ni el
cruzado. Lo único que mueven es el `exacto` de `hybrid`, de «empata» a «mejora». Lo mueve V2 (MRR@20 0,1 →
1,0), justamente el caso cuya verdad de terreno tiene una fuga.

### 5.3 Por caso decisorio — recall@10 / MRR@20

| caso | léxico | MiniLM `vector` | MiniLM `hybrid` |
|---|---|---|---|
| D1a | 0.000 / 0.000 | 1.000 / 0.143 | 0.000 / 0.000 |
| D1b | 1.000 / 0.500 | 1.000 / 0.333 | 1.000 / 1.000 |
| D1c | 0.083 / 0.250 | 0.333 / 0.333 | 0.083 / 0.167 |
| D1d | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| P1 | 0.833 / 1.000 | 0.167 / 0.250 | 0.500 / 1.000 |
| P2 | 1.000 / 1.000 | 0.000 / 0.071 | 1.000 / 1.000 |
| S1 | 1.000 / 1.000 | 1.000 / 0.333 | 1.000 / 0.333 |
| S2 | 1.000 / 1.000 | 1.000 / 0.333 | 1.000 / 1.000 |
| S4 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| V1 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| X1 | 1.000 / 1.000 | 0.000 / 0.000 | 1.000 / 1.000 |
| X2 | 1.000 / 1.000 | 0.000 / 0.000 | 1.000 / 1.000 |
| X3 | 1.000 / 1.000 | 0.000 / 0.000 | 1.000 / 1.000 |
| S8 (excluido) | 0.000 / 0.071 | 0.000 / 0.000 | 0.000 / 0.000 |
| U3 (excluido) | 0.091 / 1.000 | 0.045 / 0.111 | 0.045 / 1.000 |
| V2 (excluido) | 1.000 / 0.100 | 1.000 / 1.000 | 1.000 / 1.000 |

Un MRR@20 de 0 significa «no está entre los 20 owners publicados». No significa «no está en la ventana».

Lo que se lee: el canal vectorial de MiniLM **encuentra lo que el léxico no puede** (D1a, D1c 0,083 →
0,333) y **pierde lo que el léxico da gratis** (handles X1, siglas X2, cifras X3, la atribución P2). La fusión
RRF con pesos 1 / 1 recupera lo segundo y, sobre este corpus, **ahoga lo primero**: D1a y D1c vuelven a
su valor léxico. Es la clase de resultado que el barrido de pesos existe para mover — y la razón por la que
no se puede decidir con un solo modelo.

## 6. Coste: indexación, disco, latencia y memoria

### 6.1 Indexación y disco

| | MiniLM `vector` | MiniLM `hybrid` |
|---|---|---|
| índice | **construido en 439,6 s** (base léxica + 22.281 textos distintos de 22.933 chunks, en lotes de 1.024) | reutilizado (no se re-embebió) |
| plano vectorial en disco | **38.089.193 bytes** (`vectors.f32` + `vectors.meta.json`), 22.281 filas | el mismo |
| corrida completa (`/usr/bin/time`) | 619,6 s | 158,9 s |
| RSS máximo | 1,15 GB | 1,15 GB |

### 6.2 Latencia por consulta (spec §8.4)

| | p50 total | p95 total | embeber p50 · p95 | recuperar p50 · p95 |
|---|---:|---:|---|---|
| léxico | **31,6 ms** | 94,0 ms | — | — |
| MiniLM `vector` | 7.683,8 ms | 10.588,3 ms | 7.225,0 · 9.952,8 ms | 422,4 · 550,2 ms |
| MiniLM `hybrid` | 5.974,5 ms | 11.219,2 ms | 5.511,1 · 10.121,6 ms | 476,0 · 1.135,0 ms |

**Embeber la consulta es ~94 % de la latencia**, y no por el modelo sino por el contrato: `xbrain-embed`
atiende una petición por invocación y carga el modelo cada vez. La recuperación sola cuesta ~0,4–0,5 s
(ventana de 1.000 chunks por canal, verificada), un orden de magnitud por encima del léxico. Las dos
cifras se midieron con la máquina en presión de memoria (§6.3), así que son un techo, no un suelo.

### 6.3 ¿Corre en esta máquina sin swap? (spec §5.5) — **No, y así se paró el bake-off**

- **Antes de la primera corrida** el swap ya tenía 13,2 GB usados de 14,3 GB (otras sesiones); a lo largo
  de las corridas de MiniLM osciló entre 13,2 y 13,8 GB usados.
- **e5-small `vector`** arrancó a las 11:22:58. Durante la corrida el swap total pasó de **14.336 MB a
  15.360 MB**: dos ficheros de swap nuevos (`swapfile14` a las 11:24, `swapfile15` a las 11:35, 512 MB
  cada uno) **en el mismo contenedor APFS que el disco de datos** (`/System/Volumes/VM` y
  `/System/Volumes/Data` informaban los mismos MiB libres). El disco libre cayó de 1.430 MiB a 586 MiB.
- **Estaba progresando, pero a ~3,6 % de CPU**: en 40 s de muestreo el embedder sumó 1,4 s de CPU, su RSS
  bajaba mientras trabajaba (280 → 164 MB, páginas expulsadas) y alternaba estados `SN`/`UN`, con 55–75 MB
  de páginas libres en el sistema. Con MiniLM, un lote de 1.024 costaba ~19 s.
- **Se paró de forma controlada a los 859 s**, antes del tope de 31 min fijado para esa corrida, porque el
  siguiente fichero de swap (≥ 512 MB) habría llevado el contenedor por debajo de ~100 MiB de una vez; en
  esta misma sesión un disco lleno ya dejó sin poder ejecutar nada. Tras la parada el vigilante de disco
  llegó a leer **187 MiB libres**. Se conservaron el `run.log` con la causa, los tres informes completos y
  los ficheros de salida; e5-small `hybrid` y e5-base no arrancaron.

**Lectura honesta:** esto es una medición de *esta máquina con esta carga*, no de e5-small. Lo que sí dice
del producto es que, en un portátil de 16 GB compartido, el contrato «un subproceso y una carga del modelo
por lote o por consulta» convierte la presión de memoria en presión de disco.

## 7. D1a en las dos estrategias (criterio §13.10, no criterio de merge)

Pregunta en español que parafrasea «Leadership, Lab, and Crowd» de un artículo inglés, sin solape léxico.
Rango del item `1958350546831630810` entre los 20 owners materializados:

| estrategia | rango | recall@10 | MRR@20 |
|---|---:|---:|---:|
| léxico | **> 20** | 0 | 0 |
| MiniLM `vector` | **7** | 1 | 0,143 |
| MiniLM `hybrid` | **> 20** | 0 | 0 |

El canal vectorial lo encuentra y la fusión 1 / 1 lo vuelve a sacar del top 20. (La primera versión daba a
`hybrid` un MRR de 0,040, que salía de un rango 25 que no estaba entre los 20 owners publicados.) No es
criterio de merge (m10); es la ilustración de una línea de la §5.3.

## 8. Barrido de la fusión (`RRF_K`, pesos) — **no ejecutado, y `fusion.py` no cambia**

El instrumento existe y está probado (`xbrain eval --strategy hybrid --embeddings-model <m> --sweep-fusion
"rrf_k=… w_lexical=… w_vector=…"`: un plano y una embebida por consulta para toda la rejilla; la
combinación en vigor siempre se mide y gana los empates). **No se ejecutó**, por dos razones que se dicen
por separado:

1. **Medición:** al pararse e5-small ya no quedaba ningún candidato con pesos en disco. Los de MiniLM se
   borraron a las 11:31, con sus dos corridas terminadas, para dar margen a las de e5; los de e5 se borraron
   tras la parada, con el disco en estado crítico. Volver a descargarlos repetía la presión que paró la
   corrida.
2. **Decisión:** aunque se hubiera barrido MiniLM, mover las constantes de `fusion.py` exige un `hybrid`
   que pase las puertas, y el suyo no las pasa (§5.2). Unas constantes ajustadas sobre un candidato que no
   se promueve cambiarían el producto sin justificación medida. **`RRF_K` = 60 y los pesos 1 / 1 se
   quedan.** La §5.3 deja escrito qué debería mirar el barrido: si subir `w_vector` rescata D1a y D1c sin
   perder X1–X3 y P2.

## 9. Cómo re-derivarlo

Esta sección da los comandos **efectivos** de la corrida publicada: MiniLM en `lexical`, `vector` y
`hybrid`, con la raíz, el config, el snapshot y el golden set explícitos. Los ficheros originales se
conservan fuera del repositorio, en `zz-support-files/docs/reports/2026-09-13-bakeoff-03.7-evidence/`
(ignorado por Git) del checkout donde se midió: los tres informes `eval-*.json`/`.md` con sus
`stdout`/`stderr`, `run.log`, el `run-bakeoff.sh` que lanzó `vector`/`hybrid` y el `config.toml` de cada
candidato, con `SHA256SUMS`. El snapshot del corpus no se copió; se identifica por los sha256 de la §9.2.

### 9.1 Entorno

| | versión | de dónde sale |
|---|---|---|
| arnés (`xbrain`) | el worktree de `VGonPa/plan03-7-bakeoff`, antes del commit `547a860` | las corridas son de las 11:08–11:22 y el commit, de las 11:42 (2026-09-13, +02:00). Los informes no registran commit, así que no se puede probar que el árbol fuera byte a byte `547a860`. La tabla léxica sí se re-midió con el arnés corregido de este PR |
| CPython de `xbrain` | 3.13.7 (`.venv` del checkout, vía `uv run`) | `.venv/bin/python --version` |
| `uv` | 0.8.24 | `uv --version` |
| entorno del embedder | CPython 3.12 · `sentence-transformers` 6.0.1 · `torch` 2.14.0 · MPS disponible | impreso al instalar (`torch.__version__`, `sentence_transformers.__version__`, `torch.backends.mps.is_available()`). El entorno se borró después; el resto de dependencias transitivas no se registró |

```bash
W=/ruta/al/checkout/xbrain           # el árbol de este documento
B=/ruta/fuera/del/repo/bakeoff       # snapshot, configs, índices de evaluación e informes
S=/ruta/fuera/del/repo/embed         # entorno y pesos del embedder

uv venv --python 3.12 $S/embedenv
VIRTUAL_ENV=$S/embedenv uv pip install --index-url https://pypi.org/simple \
    'sentence-transformers==6.0.1' 'torch==2.14.0'
$S/embedenv/bin/python -c "import torch, sentence_transformers as s; \
print(torch.__version__, s.__version__, torch.backends.mps.is_available())"   # 2.14.0 6.0.1 True
```

**Pesos.** Sólo `safetensors`, sin los duplicados `onnx/` ni `*.bin` de los repositorios: sin ese filtro,
e5-base ocupa tres veces lo que usa. Son los mismos patrones con los que se descargaron los tres candidatos:

```bash
HF_HOME=$S/hf $S/embedenv/bin/python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],
    allow_patterns=['*.json', '*.txt', '*.model', 'model.safetensors', '1_Pooling/*', '2_Normalize/*', 'sentencepiece*'],
    ignore_patterns=['onnx/*', 'openvino/*'])" sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Presupuesto medido:** ~0,9 GB el entorno, 0,47 GB MiniLM o e5-small, 1,1 GB e5-base, ~0,1 GB por índice
de evaluación, **más el crecimiento del swap si la máquina está en presión de memoria** (§6.3).

### 9.2 Snapshot de sólo lectura

La corrida no leyó el `data/` de trabajo: leyó una copia de los tres ficheros de entrada, para que los
índices de evaluación se escribieran fuera de él. Las cifras sólo se esperan iguales si los sha256 coinciden.

```bash
mkdir -p $B/data $B/reports
cp -p $W/data/items.json $W/data/vocab.yaml $W/data/topics.json $B/data/
shasum -a 256 $B/data/items.json $B/data/vocab.yaml $B/data/topics.json $W/eval/golden-set.yaml
```

| fichero | sha256 |
|---|---|
| `items.json` | `4fed54a0bee5e747fffa7efcace8502733defdc445d0e9a88194a9c293312cde` |
| `vocab.yaml` | `e73fbedecdcaf6a8cf9609fb72487481a1917b2611bf96c4529097dcf2cca595` |
| `topics.json` | `7a40f4f12d285d44cb4205c0c85ce5c79448872c0fc3c7f92894b5d7ca28363e` |
| `eval/golden-set.yaml` | `ed6dd7600f946fba0d367eaa7bd020092820dc1684da79c74b820a91fe3fa319` |

### 9.3 La raíz y un `config.toml` por candidato

`XBRAIN_REPO_ROOT` le dice a `xbrain` en qué directorio buscar `config.toml` **y contra cuál resolver las
rutas relativas**. Tiene dos consecuencias que la primera versión de esta receta omitía:

- `[paths].data_dir` tiene que apuntar al snapshot, así que se escribe absoluto.
- El valor por defecto `--golden-set eval/golden-set.yaml` pasa a buscarse **dentro de la raíz**, donde no
  existe, y el comando falla con `Error: golden set no encontrado: $B/cfg-…/eval/golden-set.yaml`. Por eso
  `--golden-set` se pasa siempre como ruta absoluta.

Los prefijos son del MODELO y viajan en config, no en código. El config de MiniLM:

```bash
mkdir -p $B/cfg-minilm-l12
cat > $B/cfg-minilm-l12/config.toml <<EOF
[paths]
vault = "vault"
output_subdir = "x-knowledge"
data_dir = "$B/data"

[x]
handle = "<handle>"

[embeddings]
command = "env HF_HOME=$S/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false $S/embedenv/bin/python $W/scripts/xbrain-embed"
batch_size = 1024
timeout_seconds = 1800
query_prefix = ""
passage_prefix = ""
EOF
```

### 9.4 Las corridas de MiniLM

Con los valores por defecto (`--limit 10`, `k ∈ {1, 5, 10, 20}`) la profundidad sale 20 owners, porque es el
máximo de `limit` y del mayor k. `vector` construye `$B/data/eval-index/<modelo>/` y `hybrid` lo reutiliza
mientras el manifest declare el mismo modelo y el store no se haya movido (el informe dice `construido` o
`reutilizado`). Si el manifest declara OTRO modelo, el comando falla sin tocar nada.

```bash
cd $W
export XBRAIN_REPO_ROOT=$B/cfg-minilm-l12
G=$W/eval/golden-set.yaml
M=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

uv run xbrain eval --strategy lexical --golden-set $G --report $B/reports/eval-lexical.json \
    > $B/reports/lexical.stdout
/usr/bin/time -l uv run xbrain eval --strategy vector --embeddings-model $M --golden-set $G \
    --report $B/reports/eval-minilm-l12-vector.json \
    > $B/reports/minilm-l12-vector.stdout 2> $B/reports/minilm-l12-vector.stderr
/usr/bin/time -l uv run xbrain eval --strategy hybrid --embeddings-model $M --golden-set $G \
    --report $B/reports/eval-minilm-l12-hybrid.json \
    > $B/reports/minilm-l12-hybrid.stdout 2> $B/reports/minilm-l12-hybrid.stderr
```

Qué está verificado de estos comandos y qué no:

- **El léxico** se ejecutó el 2026-09-13, con el arnés corregido, sobre un snapshot con los sha256 de la
  §9.2 y una raíz con `[paths].data_dir` absoluto, y **reproduce la §3 celda a celda**. La corrida original
  usó otra raíz (`XBRAIN_REPO_ROOT=$B`, con `data_dir = "data"` relativo a ella) y el mismo `--golden-set`
  absoluto. Las dos resuelven el mismo snapshot: la re-corrida con el arnés anterior dio el mismo
  `retrieved` y las mismas métricas en los 23 casos que el informe original.
- **`vector` y `hybrid`** son los comandos de `run-bakeoff.sh`, que los lanzó con
  `env XBRAIN_REPO_ROOT=$B/cfg-minilm-l12` en línea, que es lo mismo que el `export` de arriba. **No se han
  vuelto a ejecutar:** los pesos se borraron (§8). Sus cifras de §5 salen de los informes originales,
  recalculando el MRR como indica la §9.5.

**Candidatos pendientes (e5-small, e5-base).** La misma receta con otro `M` y otro config. Cambia sólo
`[embeddings]`: `query_prefix = "query: "` y `passage_prefix = "passage: "`, que la familia E5 exige. Hay que
correrlos en una máquina sin presión de memoria (§6.3):

```bash
M=intfloat/multilingual-e5-small      # o intfloat/multilingual-e5-base
export XBRAIN_REPO_ROOT=$B/cfg-e5-small
uv run xbrain eval --strategy vector --embeddings-model $M --golden-set $G --report $B/reports/eval-e5-small-vector.json
uv run xbrain eval --strategy hybrid --embeddings-model $M --golden-set $G --report $B/reports/eval-e5-small-hybrid.json
uv run xbrain eval --strategy hybrid --embeddings-model $M --golden-set $G --sweep-fusion "rrf_k=20,60,120 w_vector=0.5,1,2"
```

### 9.5 MRR@20 sobre los informes conservados

Los informes originales de `vector`/`hybrid` conservan en `retrieved` los 20 owners de cada caso, pero su
`mrr` se leyó sobre la ventana completa. Para cada caso, el MRR@20 es `1/rango` del primer owner relevante
dentro de `retrieved`, o 0 si no aparece. Los owners relevantes son `relevant_items` y `relevant_topics` del
golden set, y las medias por estrato salen de `evaluation._aggregate`, igual que en una corrida.

El recálculo está en `recompute_mrr20.py`, en la carpeta de evidencias. Antes de reescribir nada comprueba
dos cosas: que las 780 celdas de recall, precision y nDCG se reconstruyen desde `retrieved`, y que todos los
agregados distintos de `mrr` salen idénticos a los del informe original. Aplicado al informe léxico, da
exactamente lo mismo que la re-corrida del arnés corregido (§9.4).

### 9.6 La comparación

Se usa el instrumento que se publica, con las exclusiones como argumento. Si los dos informes tienen
distinto `limit`, `compare_reports` lanza un error en vez de desempatar con MRR leídos a dos profundidades:

```bash
uv run python -c "
import json
from xbrain.knowledge.evaluation import compare_reports
lexical = json.load(open('$B/reports/eval-lexical.json'))
candidate = json.load(open('$B/reports/eval-minilm-l12-hybrid.json'))
exclude = {'U3': 'población crecida', 'S8': 'hecho fuera de la nota del topic', 'V2': 'fuga A.3'}
print(json.dumps(compare_reports(lexical, candidate, k=10, exclude=exclude), ensure_ascii=False, indent=2))"
```

## 10. Puertas del spec §8.6 y criterios del Plan 03 §13

| Puerta / criterio | Estado | Con qué número |
|---|---|---|
| §8.6.1 — el evaluador valida schema, ids, filtros y enums | **cumple** | `load_cases` valida el fichero versionado en CI; `resolve_cases` resolvió los 23 casos contra el store de la §1 |
| §8.6.2 — reproducible ante empates | **cumple para léxico; no re-medido para `vector`/`hybrid`** | dos corridas léxicas idénticas en los 23 casos, y la re-corrida con el arnés corregido reproduce la §3; para vectores sólo lo cubren los desempates por `chunk_id` probados en 03.3/03.5 — no quedó modelo para repetir la corrida real |
| §8.6.3 — `hybrid` no degrada `exacto` | **cumple (empata)** | recall@10 0,75 → 0,75 · MRR@20 0,75 → 0,75 sobre V1, X1, X2, X3 (§5.2). La primera versión decía «no cumple por la regla» con MRR 0,7549 → 0,7545: era V1 leído fuera de los 20 owners publicados |
| §8.6.4 — `hybrid` mejora `semantico` y `cruzado_idioma` | **no cumple** | semántico recall@10 0,7024 → 0,6548; cruzado 0,75 → 0,75 con MRR@20 0,625 → 0,5833 |
| §8.6.5 — un match en derivados conduce a una fuente | **no medido aquí** | el arnés puntúa owners y no hidrata `verify_with`; la propiedad la prueban las suites de servicio de 03.5/03.6, no esta corrida |
| §8.6.8 — fallos y skips publicados, ningún cero fabricado | **cumple** | F1/F2 no medidos bajo `vector`/`hybrid`; `expansion` y `real` sin cobertura; `thread`/`user_note` declaradas sin datos; e5 interrumpido y publicado |
| §13.8 — bake-off con ≥ 3 candidatos, ganador y perdedores | **NO CUMPLE: 1 de 3** | MiniLM medido y perdedor en las dos estrategias; e5-small interrumpido a 859 s; e5-base, bge-m3 y jina-v3 sin medir |
| §13.9 — puertas 1–5 y 8 con números | **cumple como declaración** | esta tabla |
| §13.10 — D1a en ambas estrategias | **cumple** | léxico > 20 · `vector` 7 · `hybrid` > 20 (§7) |

**Decisión:** `lexical` sigue siendo el default y **`hybrid` no se promueve**. Ninguno de estos criterios
bloquea el merge (lo bloquea `check.sh`, §13.13), pero el §13.8 queda **abierto**: el bake-off hay que
completarlo con e5-small, e5-base y, si el disco lo permite, bge-m3, en una máquina sin presión de memoria.
El instrumento está en el código y la §9 dice cómo.

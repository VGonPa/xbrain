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
  mejora ni el semántico (recall@10 0,7024 → 0,6548) ni el cruzado (0,75 → 0,75, MRR 0,625 → 0,5933)**.
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

---

## 1. Qué se midió, sobre qué, con qué instrumento

| | |
|---|---|
| Corpus | `data/items.json` sha256 `4fed54a0…` — 2.474 items · 45 topics · 10.570 superficies · 22.933 chunks (chunker `v3`, `800/0`) · `vocab.yaml` sha256 `e73fbede…` · `topics.json` sha256 `7a40f4f1…` |
| Golden set | `eval/golden-set.yaml` v3 tal como está versionado: 23 casos puntuables + 8 escenarios archivados; **los 23 resuelven** contra ese store |
| Profundidad | 20 owners por caso, `k ∈ {1, 5, 10, 20}` — la misma para las tres estrategias |
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

| estrato | casos | recall@1 | recall@10 | precision@10 | MRR | nDCG@10 | superficies@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cruzado_idioma | 5 | 0.4091 | 0.6182 | 0.1 | 0.7 | 0.5835 | 0.25 |
| enterrado | 8 | 0.5208 | 0.7396 | 0.25 | 0.6687 | 0.6464 | 0.5 |
| exacto | 5 | 0.3667 | 0.8 | 0.62 | 0.6239 | 0.6578 | 0.8 |
| filtros | 2 | 0.4167 | 1.0 | 1.0 | 1.0 | 1.0 | sin cobertura |
| multimodal | 6 | 0.75 | 0.8333 | 0.2333 | 0.8366 | 0.8333 | 0.6667 |
| resumen | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| semantico | 9 | 0.3569 | 0.5564 | 0.1333 | 0.6468 | 0.5333 | 0.1667 |
| topic | 3 | 0.3333 | 0.3333 | 0.0333 | 0.3571 | 0.3333 | 0.3333 |
| expansion | — | sin cobertura | sin cobertura | sin cobertura | — | — | — |
| **construido** | **23** | 0.5165 | **0.7395** | 0.3 | **0.7366** | 0.6995 | 0.5 |

El `recall@10` de 0,7395 es el par publicado por el Plan 02 para `800/0`; el MRR sale 0,7366 y no 0,7357
porque aquí la profundidad es 20 owners y allí 10. **Reproducible:** dos corridas léxicas seguidas dieron
los mismos `retrieved` y las mismas métricas en los 23 casos, idénticos a los de esta tabla.

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

### 5.1 Por estrato, todos los casos — recall@10 · MRR · nDCG@10 · superficies@10

| estrategia | cruzado_idioma | semantico | exacto | enterrado | multimodal | topic | resumen | filtros | construido |
|---|---|---|---|---|---|---|---|---|---|
| lexical | 0.6182 · 0.7000 · 0.5835 · 0.2500 | 0.5564 · 0.6468 · 0.5333 · 0.1667 | 0.8000 · 0.6239 · 0.6578 · 0.8000 | 0.7396 · 0.6687 · 0.6464 · 0.5000 | 0.8333 · 0.8366 · 0.8333 · 0.6667 | 0.3333 · 0.3571 · 0.3333 · 0.3333 | 0.0000 · 0.0000 · 0.0000 · 0.0000 | 1.0000 · 1.0000 · 1.0000 · sin cobertura | 0.7395 · 0.7366 · 0.6995 · 0.5000 |
| MiniLM `vector` | 0.8091 · 0.2508 · 0.3799 · 0.5000 | 0.5051 · 0.3606 · 0.3728 · 0.5000 | 0.2000 · 0.2125 · 0.2000 · 0.2000 | 0.6875 · 0.4241 · 0.4736 · 0.6667 | 0.6667 · 0.5915 · 0.6052 · 0.5000 | 0.3333 · 0.1164 · 0.1667 · 0.3333 | 0.0000 · 0.0120 · 0.0000 · 0.0000 | **no medido** (F1, F2) | 0.5498 · 0.3724 · 0.4041 · 0.4444 |
| MiniLM `hybrid` | 0.6091 · 0.6747 · 0.5440 · 0.2500 | 0.5143 · 0.6936 · 0.5378 · 0.1667 | 0.8000 · 0.8036 · 0.7733 · 0.8000 | 0.6979 · 0.6925 · 0.6400 · 0.5000 | 0.8333 · 0.8363 · 0.8200 · 0.8333 | 0.3333 · 0.3483 · 0.3333 · 0.3333 | 0.0000 · 0.0091 · 0.0000 · 0.0000 | **no medido** (F1, F2) | 0.6966 · 0.7430 · 0.6765 · 0.5556 |

`construido` en `vector`/`hybrid` promedia 21 casos (F1 y F2 no medidos) y en léxico 23: **no se comparan
entre sí**; la comparación que decide es la pareada de la §5.2.

### 5.2 Puertas §8.6.3 y §8.6.4 contra léxico, pareadas por caso (`compare_reports`)

**Con las exclusiones de la §2 — el conjunto que decide:**

| estrategia | estrato | casos pareados | léxico recall@10 · MRR | MiniLM recall@10 · MRR | veredicto |
|---|---|---|---|---|---|
| `vector` | exacto | V1, X1, X2, X3 | 0.7500 · 0.7549 | 0.0000 · 0.0156 | **empeora** |
| `vector` | semantico | D1a, D1b, D1c, D1d, P1, P2, S4 | 0.7024 · 0.6786 | 0.6429 · 0.4473 | **empeora** |
| `vector` | cruzado_idioma | D1a, D1b, S1, S2 | 0.7500 · 0.6250 | 1.0000 · 0.2857 | mejora |
| `hybrid` | exacto | V1, X1, X2, X3 | 0.7500 · 0.7549 | 0.7500 · 0.7545 | **empeora** |
| `hybrid` | semantico | D1a, D1b, D1c, D1d, P1, P2, S4 | 0.7024 · 0.6786 | 0.6548 · 0.7438 | **empeora** |
| `hybrid` | cruzado_idioma | D1a, D1b, S1, S2 | 0.7500 · 0.6250 | 0.7500 · 0.5933 | **empeora** |

El «empeora» de `hybrid` en `exacto` es **un solo movimiento y fuera del top 20**: X1, X2 y X3 siguen en
1,0 / 1,0, y V1 — que ninguna estrategia encuentra — pasa de MRR 0,020 a 0,018. Por la regla que el
instrumento aplica (recall, luego MRR) es un empeoramiento; en la práctica, `hybrid` no perdió ningún
exacto que el léxico encontrase. Se dice las dos cosas.

**Sin exclusiones — sensibilidad:**

| estrategia | estrato | casos | léxico recall@10 · MRR | MiniLM recall@10 · MRR | veredicto |
|---|---|---|---|---|---|
| `vector` | exacto | V1, V2, X1, X2, X3 | 0.8000 · 0.6239 | 0.2000 · 0.2125 | empeora |
| `vector` | semantico | + S8, U3 | 0.5564 · 0.6468 | 0.5051 · 0.3606 | empeora |
| `vector` | cruzado_idioma | + U3 | 0.6182 · 0.7000 | 0.8091 · 0.2508 | mejora |
| `hybrid` | exacto | V1, V2, X1, X2, X3 | 0.8000 · 0.6239 | 0.8000 · 0.8036 | mejora |
| `hybrid` | semantico | + S8, U3 | 0.5564 · 0.6468 | 0.5143 · 0.6936 | empeora |
| `hybrid` | cruzado_idioma | + U3 | 0.6182 · 0.7000 | 0.6091 · 0.6747 | empeora |

Las exclusiones **no cambian la decisión**: con o sin ellas, `hybrid` no mejora ni el semántico ni el
cruzado. Lo único que mueven es el `exacto` de `hybrid`, y lo mueve V2 (MRR 0,1 → 1,0), justamente el caso
cuya verdad de terreno tiene una fuga.

### 5.3 Por caso decisorio — recall@10 / MRR

| caso | léxico | MiniLM `vector` | MiniLM `hybrid` |
|---|---|---|---|
| D1a | 0.000 / 0.000 | 1.000 / 0.143 | 0.000 / 0.040 |
| D1b | 1.000 / 0.500 | 1.000 / 0.333 | 1.000 / 1.000 |
| D1c | 0.083 / 0.250 | 0.333 / 0.333 | 0.083 / 0.167 |
| D1d | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| P1 | 0.833 / 1.000 | 0.167 / 0.250 | 0.500 / 1.000 |
| P2 | 1.000 / 1.000 | 0.000 / 0.071 | 1.000 / 1.000 |
| S1 | 1.000 / 1.000 | 1.000 / 0.333 | 1.000 / 0.333 |
| S2 | 1.000 / 1.000 | 1.000 / 0.333 | 1.000 / 1.000 |
| S4 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| V1 | 0.000 / 0.020 | 0.000 / 0.045 | 0.000 / 0.018 |
| X1 | 1.000 / 1.000 | 0.000 / 0.013 | 1.000 / 1.000 |
| X2 | 1.000 / 1.000 | 0.000 / 0.004 | 1.000 / 1.000 |
| X3 | 1.000 / 1.000 | 0.000 / 0.000 | 1.000 / 1.000 |
| S8 (excluido) | 0.000 / 0.071 | 0.000 / 0.004 | 0.000 / 0.036 |
| U3 (excluido) | 0.091 / 1.000 | 0.045 / 0.111 | 0.045 / 1.000 |
| V2 (excluido) | 1.000 / 0.100 | 1.000 / 1.000 | 1.000 / 1.000 |

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

| estrategia | rango | recall@10 | MRR |
|---|---:|---:|---:|
| léxico | **> 20** | 0 | 0 |
| MiniLM `vector` | **7** | 1 | 0,143 |
| MiniLM `hybrid` | **> 20** | 0 | 0,040 |

El canal vectorial lo encuentra y la fusión 1 / 1 lo vuelve a sacar del top 20. No es criterio de merge
(m10); es la ilustración de una línea de la §5.3.

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

Todo corre en una **raíz aislada**. `XBRAIN_REPO_ROOT` apunta a un directorio que NO es un checkout y que
contiene un `config.toml` y una copia de sólo lectura de los tres ficheros de entrada; así los índices de
evaluación nunca se escriben en el `data/` de trabajo. Dos trampas, medidas sobre la versión anterior de esta
sección en `90dc74e` (las dos salían con exit 1):

- **`--golden-set` y `--report` relativos se resuelven contra `XBRAIN_REPO_ROOT`, no contra el checkout.**
  Sin `--golden-set` absoluto: `Error: golden set no encontrado: <raíz>/eval/golden-set.yaml`.
- **`load_config` exige `[paths]` y un `[x].handle` no vacío**, aunque `eval` no los use. Con un
  `config.toml` que sólo tenga `[embeddings]`: `Error: 'paths'`.

### 9.1 Versión y entorno de la corrida publicada

| | |
|---|---|
| xbrain | corridas de 11:08 a 11:22 sobre el árbol sin commitear de `VGonPa/plan03-7-bakeoff` (base `e38959f`), commiteado como **`547a860`** a las 11:42. `lexical` re-corrido en `547a860` y en `90dc74e`: mismos `retrieved` y métricas en los 23 casos. `vector`/`hybrid` no se pueden re-correr: los pesos se borraron (§8) |
| Python y uv de xbrain | CPython 3.13.7 · uv 0.8.24 · el extra `embeddings` del `uv.lock` (`numpy` 2.5.3); los comandos corren con `uv run` desde el checkout |
| Golden set | `eval/golden-set.yaml` sha256 `ed6dd7600f946fba0d367eaa7bd020092820dc1684da79c74b820a91fe3fa319`, idéntico en `e38959f`, `547a860` y `90dc74e` (último cambio: `427fea9`) |
| Entrada copiada | `items.json` `4fed54a0bee5e747fffa7efcace8502733defdc445d0e9a88194a9c293312cde` · `vocab.yaml` `e73fbedecdcaf6a8cf9609fb72487481a1917b2611bf96c4529097dcf2cca595` · `topics.json` `7a40f4f12d285d44cb4205c0c85ce5c79448872c0fc3c7f92894b5d7ca28363e`. La copia de la corrida conserva esos sha256, pero hoy sus permisos no son de sólo lectura; el `chmod` de la §9.4 sí lo es |
| Embedder | `scripts/xbrain-embed` en un entorno aparte (§9.3): sentence-transformers 6.0.1 · torch 2.14.0 · MPS · `HF_HUB_OFFLINE=1`. **No quedaron registradas ni la revisión de los pesos de Hugging Face ni la versión exacta de Python de ese entorno.** Los pesos ya no existen: re-correr descarga la revisión vigente, que puede no ser la medida |
| Profundidad | `limit: 20` en los tres informes. La corrida publicada no pasó `--limit`: los 20 salieron de `max(--limit 10 por defecto, mayor k por defecto 20)`. Los comandos de la §9.5 los fijan con `--limit 20`, para no depender de esos defaults |

### 9.2 Artefactos

Fuera de Git, igual que `data/`. El original está en el scratchpad de la sesión del bake-off, bajo
`/private/tmp`, que no sobrevive a un reinicio; hay una copia con las mismas sumas en
`zz-support-files/docs/reports/2026-09-13-plan03-7-bakeoff/` del checkout principal (ignorado por Git).

| fichero | sha256 | qué es |
|---|---|---|
| `eval-lexical.json` · `.md` | `6161f8eba4f28aeae2d7a8a07f9ffa766649b3d99f9b23b663b1f2e23863ff01` · `3c85462561958e3137842e5630f419e85eaa8c1531392ba95a21fa2d880ac3f6` | línea base léxica (§3) |
| `eval-minilm-l12-vector.json` · `.md` | `53349515a3708088c2de4e6c9410c08ecc9f7e3cb5b7e2e2b8baa11829fe6d3b` · `a821528196c4416e3a02aef9cc00d8bb2c2e7a0a0fd787e666bad26f94a48796` | MiniLM `vector`, índice construido (§6.1) |
| `eval-minilm-l12-hybrid.json` · `.md` | `e9257c4f6e3f97a26b4a3d8c971115e3b2485d78cea8e339a8ccb225c956e4af` · `276a115b7fe03c5ebccf16fc33a00ae0c5b190632f1f6bcdb8cd8f7f447300cc` | MiniLM `hybrid`, índice reutilizado |
| `run.log` | `22114ee72698a937eb42e3c7926f48b5b6695694301c979dbf42931168cb1b5a` | swap y disco por corrida, y la parada de e5-small (§6.3) |
| `run-bakeoff.sh` | `d0ef4a3b2ba542dcc66d8a6924903d00b419250f1617fa0947348392d4c34239` | el script de `vector`/`hybrid`; no incluye la corrida léxica, de las 11:08, anterior a él |

**Una re-corrida no reproduce esos sha256, ni debe.** `corpus.source` y `embeddings.command_version` llevan
rutas absolutas, y `latency` e `indexing.seconds` son tiempos. Lo que tiene que coincidir es
`cases[].retrieved`, las métricas por caso, `by_stratum` y `by_provenance`. Desde `90dc74e` el informe añade
además `mrr@k` y `metric_units`, que los artefactos de arriba no traen.

### 9.3 El embedder, fuera de xbrain

Un entorno propio con `sentence-transformers` (xbrain no lleva ninguna librería de modelos) y los pesos en
`safetensors`, sin los duplicados `onnx/` ni `*.bin` de los repositorios — sin eso, e5-base ocupa tres veces
lo que usa. **Presupuesto medido:** ~0,9 GB el entorno, 0,47 GB MiniLM o e5-small, 1,1 GB e5-base, ~0,1 GB
por índice de evaluación, **más el crecimiento del swap si la máquina está en presión de memoria** (§6.3).

```bash
uv venv --python 3.12 embedenv
VIRTUAL_ENV=embedenv uv pip install --index-url https://pypi.org/simple sentence-transformers
HF_HOME=hf embedenv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('intfloat/multilingual-e5-base',
    allow_patterns=['*.json', '*.txt', '*.model', 'model.safetensors', '1_Pooling/*', '2_Normalize/*', 'sentencepiece*'],
    ignore_patterns=['onnx/*', 'openvino/*'])"
```

### 9.4 La raíz aislada y un `config.toml` por candidato

Los prefijos son del MODELO y viajan en config, no en código. `[paths]` y `[x]` están porque `load_config`
los exige: `eval` no lee ni `vault` ni `handle`, y la corrida léxica tampoco lee `[embeddings]`. El
directorio `data/` de la raíz queda escribible, porque `vector` escribe ahí `eval-index/<modelo>/`; los
tres ficheros, no.

```bash
XBRAIN_SRC=/ruta/xbrain      # checkout de xbrain: 547a860 para las cifras publicadas (§9.1)
STORE=/ruta/xbrain/data      # el data/ de trabajo: sólo se lee, una vez
BAKEOFF=/ruta/bakeoff        # la raíz aislada, fuera de cualquier checkout
EMBEDENV=/ruta/embedenv      # §9.3
HF=/ruta/hf                  # §9.3

mkdir -p "$BAKEOFF/data" "$BAKEOFF/reports"
cp "$STORE/items.json" "$STORE/vocab.yaml" "$STORE/topics.json" "$BAKEOFF/data/"
chmod a-w "$BAKEOFF/data/items.json" "$BAKEOFF/data/vocab.yaml" "$BAKEOFF/data/topics.json"
(cd "$BAKEOFF/data" && shasum -a 256 -c - <<'EOF'
4fed54a0bee5e747fffa7efcace8502733defdc445d0e9a88194a9c293312cde  items.json
e73fbedecdcaf6a8cf9609fb72487481a1917b2611bf96c4529097dcf2cca595  vocab.yaml
7a40f4f12d285d44cb4205c0c85ce5c79448872c0fc3c7f92894b5d7ca28363e  topics.json
EOF
)

candidate_config() {  # $1 = directorio · $2 = query_prefix · $3 = passage_prefix
  mkdir -p "$1"
  cat > "$1/config.toml" <<EOF
[paths]
vault = "vault"
output_subdir = "x-knowledge"
data_dir = "$BAKEOFF/data"

[x]
handle = "bakeoff"

[embeddings]
command = "env HF_HOME=$HF HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false $EMBEDENV/bin/python $XBRAIN_SRC/scripts/xbrain-embed"
batch_size = 1024
timeout_seconds = 1800
query_prefix = "$2"
passage_prefix = "$3"
EOF
}
candidate_config "$BAKEOFF/cfg-lexical" "" ""
candidate_config "$BAKEOFF/cfg-minilm-l12" "" ""
candidate_config "$BAKEOFF/cfg-e5-base" "query: " "passage: "
```

Si `shasum -c` dice `FAILED`, el corpus se ha movido: lo que se mida ya no re-deriva la §3–§7, mide otro
corpus (CLAUDE.md, regla 2).

### 9.5 Las corridas

Desde el checkout, con el golden set en ruta **absoluta** y la profundidad fijada. Los tres primeros comandos
son los de los artefactos de la §9.2. `vector` construye `$BAKEOFF/data/eval-index/<modelo>/` y `hybrid` lo
reutiliza mientras el manifest declare el mismo modelo y el store no se haya movido (el informe dice
`construido` o `reutilizado`); si el manifest declara OTRO modelo, el comando falla sin tocar nada. El barrido
sin `--limit 20` corre a profundidad 10 (`max(--limit, k)`) y su MRR no se compara con el de la §5.

`vector` y `hybrid` necesitan `numpy`, que viaja en el extra `embeddings` y no en el entorno que crea un
`uv run` a secas: sin él fallan con `el plano vectorial necesita numpy`. `uv sync` es exacto y quita lo que no
se le pide, así que en un checkout de desarrollo añade `--extra dev`, como hace CI.

```bash
cd "$XBRAIN_SRC"
uv sync --locked --extra embeddings
GOLDEN="$XBRAIN_SRC/eval/golden-set.yaml"
MINILM=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
E5=intfloat/multilingual-e5-base

XBRAIN_REPO_ROOT="$BAKEOFF/cfg-lexical" uv run xbrain eval --strategy lexical \
  --golden-set "$GOLDEN" --limit 20 --report "$BAKEOFF/reports/eval-lexical.json"
XBRAIN_REPO_ROOT="$BAKEOFF/cfg-minilm-l12" uv run xbrain eval --strategy vector --embeddings-model "$MINILM" \
  --golden-set "$GOLDEN" --limit 20 --report "$BAKEOFF/reports/eval-minilm-l12-vector.json"
XBRAIN_REPO_ROOT="$BAKEOFF/cfg-minilm-l12" uv run xbrain eval --strategy hybrid --embeddings-model "$MINILM" \
  --golden-set "$GOLDEN" --limit 20 --report "$BAKEOFF/reports/eval-minilm-l12-hybrid.json"

# Para completar el §13.8, lo mismo con cada candidato que falta:
XBRAIN_REPO_ROOT="$BAKEOFF/cfg-e5-base" uv run xbrain eval --strategy vector --embeddings-model "$E5" \
  --golden-set "$GOLDEN" --limit 20 --report "$BAKEOFF/reports/eval-e5-base-vector.json"
XBRAIN_REPO_ROOT="$BAKEOFF/cfg-e5-base" uv run xbrain eval --strategy hybrid --embeddings-model "$E5" \
  --golden-set "$GOLDEN" --limit 20 --report "$BAKEOFF/reports/eval-e5-base-hybrid.json"
XBRAIN_REPO_ROOT="$BAKEOFF/cfg-e5-base" uv run xbrain eval --strategy hybrid --embeddings-model "$E5" \
  --golden-set "$GOLDEN" --limit 20 --sweep-fusion "rrf_k=20,60,120 w_vector=0.5,1,2" \
  --report "$BAKEOFF/reports/sweep-e5-base.json"
```

### 9.6 La comparación

Con el instrumento publicado y las exclusiones como argumento. **La versión de `XBRAIN_SRC` decide qué
informes acepta:**

- en **`547a860`**, `compare_reports` ordena por `recall@10` y luego por el `mrr` sin corte, y sobre los
  artefactos de la §9.2 reproduce las tablas de la §5.2;
- desde **`90dc74e`** (F1) ordena por `mrr@10`, que esos artefactos no traen: sobre ellos marca cada estrato
  `sin cobertura`, con todos los casos sin parear por «mrr@10 ausente o no medido». Con esa versión se
  comparan informes producidos por ella misma con los comandos de la §9.5.

```bash
cd "$XBRAIN_SRC" && uv run python - "$BAKEOFF/reports/eval-lexical.json" "$BAKEOFF/reports/eval-minilm-l12-hybrid.json" <<'EOF'
import json, sys
from xbrain.knowledge.evaluation import compare_reports
lexical, candidate = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:3])
exclude = {"U3": "población crecida", "S8": "hecho fuera de la nota del topic", "V2": "fuga A.3"}
print(json.dumps(compare_reports(lexical, candidate, k=10, exclude=exclude), ensure_ascii=False, indent=2))
EOF
```

## 10. Puertas del spec §8.6 y criterios del Plan 03 §13

| Puerta / criterio | Estado | Con qué número |
|---|---|---|
| §8.6.1 — el evaluador valida schema, ids, filtros y enums | **cumple** | `load_cases` valida el fichero versionado en CI; `resolve_cases` resolvió los 23 casos contra el store de la §1 |
| §8.6.2 — reproducible ante empates | **cumple para léxico; no re-medido para `vector`/`hybrid`** | dos corridas léxicas idénticas en los 23 casos; para vectores sólo lo cubren los desempates por `chunk_id` probados en 03.3/03.5 — no quedó modelo para repetir la corrida real |
| §8.6.3 — `hybrid` no degrada `exacto` | **no cumple por la regla, sin pérdida de aciertos** | recall@10 0,75 → 0,75; MRR 0,7549 → 0,7545 por V1 fuera del top 20 (§5.2) |
| §8.6.4 — `hybrid` mejora `semantico` y `cruzado_idioma` | **no cumple** | semántico recall@10 0,7024 → 0,6548; cruzado 0,75 → 0,75 con MRR 0,625 → 0,5933 |
| §8.6.5 — un match en derivados conduce a una fuente | **no medido aquí** | el arnés puntúa owners y no hidrata `verify_with`; la propiedad la prueban las suites de servicio de 03.5/03.6, no esta corrida |
| §8.6.8 — fallos y skips publicados, ningún cero fabricado | **cumple** | F1/F2 no medidos bajo `vector`/`hybrid`; `expansion` y `real` sin cobertura; `thread`/`user_note` declaradas sin datos; e5 interrumpido y publicado |
| §13.8 — bake-off con ≥ 3 candidatos, ganador y perdedores | **NO CUMPLE: 1 de 3** | MiniLM medido y perdedor en las dos estrategias; e5-small interrumpido a 859 s; e5-base, bge-m3 y jina-v3 sin medir |
| §13.9 — puertas 1–5 y 8 con números | **cumple como declaración** | esta tabla |
| §13.10 — D1a en ambas estrategias | **cumple** | léxico > 20 · `vector` 7 · `hybrid` > 20 (§7) |

**Decisión:** `lexical` sigue siendo el default y **`hybrid` no se promueve**. Ninguno de estos criterios
bloquea el merge (lo bloquea `check.sh`, §13.13), pero el §13.8 queda **abierto**: el bake-off hay que
completarlo con e5-small, e5-base y, si el disco lo permite, bge-m3, en una máquina sin presión de memoria.
El instrumento está en el código y la §9 dice cómo.

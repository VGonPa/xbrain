#!/bin/bash
# Ejecuta la batería dirigida del §15 del Plan 02 e imprime su recuento.
#
# Ésta es la mitad de MEDICIÓN del listón del §5.3: la selección vive versionada en
# `eval/plan02-acceptance.yaml` y la mitad fail-closed (que ningún node id desaparezca) la
# guarda `tests/test_plan02_acceptance.py`, que corre dentro de la suite. Este guion existe
# porque una cifra va con el comando que la re-deriva (§6.6 del documento de entrega).
#
#   bash scripts/plan02-acceptance.sh              # ejecuta y cuenta
#   bash scripts/plan02-acceptance.sh --collect    # sólo recuenta, sin ejecutar
#
# Lee la CONCLUSIÓN que imprime pytest, no el código de salida de una tubería (regla 9): el
# `$?` de un `| tail` es el de `tail`. Por eso pytest no lleva tubería detrás.
#
# `mapfile` NO se usa: bash 3.2 (el de macOS) no lo tiene, y un guion de re-derivación que
# sólo corre en la máquina de quien lo escribió no re-deriva nada.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

SELECTION="eval/plan02-acceptance.yaml"
SUMMARY=$(uv run python -c "
import pathlib, yaml
doc = yaml.safe_load(pathlib.Path('$SELECTION').read_text(encoding='utf-8'))
nodes = [e['node'] for c in doc['criteria'] for e in c['tests']]
print(len(doc['criteria']), len(nodes), doc['listed_nodes'])
print('\n'.join(nodes))
")
HEAD=$(printf '%s\n' "$SUMMARY" | head -1)
NODES=$(printf '%s\n' "$SUMMARY" | tail -n +2)

echo "Selección: $SELECTION"
echo "Criterios · node ids · listón declarado: $HEAD"
echo

# shellcheck disable=SC2086  # the node ids are pytest arguments, one per word, and hold no spaces
if [ "${1:-}" = "--collect" ]; then
    uv run pytest --collect-only -q $NODES
else
    uv run pytest -q $NODES
fi

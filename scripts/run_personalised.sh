#!/usr/bin/env bash
# Personalisierte Regler: einer je Held-out-Patient und Beobachtungsraum.
#
# Bewertet werden sie auf genau denselben Patienten und denselben
# Episoden-Seeds wie der generalistische Regler. Nur so ist der Unterschied
# zwischen "ein Regler fuer die Kohorte" und "ein Regler je Patient"
# zurechenbar.
#
# Budget je Regler bewusst kleiner als beim Generalisten (ein Patient ist eine
# einfachere Aufgabe): 80 Lehrer-Episoden fuer den Warmstart, 250000 Schritte
# Feinabstimmung.
set -euo pipefail
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src

PATIENTS="adolescent#008 adolescent#009 adolescent#010 adult#008 adult#009 adult#010 child#007 child#008 child#009 child#010"

for P in $PATIENTS; do
  TAG=$(echo "$P" | tr -d '#')
  for OBS in privileged cgm; do
    echo "=== personalisiert $P / $OBS  $(date +%H:%M:%S) ==="
    BC="runs/pretrained/bc_${OBS}_${TAG}"
    if [ ! -f "${BC}.zip" ]; then
      "$PY" src/pretrain.py --obs "$OBS" --patient "$P" \
          --episodes 80 --out "$BC" 2>&1 | { grep -E "^\[bc\]" || true; }
    fi
    "$PY" src/train.py --obs "$OBS" --patient "$P" \
        --steps 250000 --n-envs 24 --n-steps 256 --seed 0 \
        --eval-freq 50000 --checkpoint-freq 50000 --eval-episodes 10 \
        --reset-log-std -2.5 --resume "${BC}.zip" 2>&1 \
      | { grep -E "^\[t1d\] Lauf|Traceback|Error" || true; }
  done
done
echo "=== PERSONALISIERT FERTIG $(date +%H:%M:%S) ==="

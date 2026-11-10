#!/usr/bin/env bash
# Generalist: Warmstart und Feinabstimmung auf den 20 Trainingspatienten,
# anschliessend Auswertung auf den 10 nie trainierten Held-out-Patienten.
set -euo pipefail
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src

for OBS in privileged cgm; do
  echo "=== BC generalist $OBS  $(date +%H:%M:%S) ==="
  "$PY" src/pretrain.py --obs $OBS --patient train --episodes 300 \
      --out runs/pretrained/bc_${OBS}_generalist 2>&1 \
    | { grep -E "^\[bc\]|Epoche" || true; }
done

for OBS in privileged cgm; do
  echo "=== Feinabstimmung generalist $OBS  $(date +%H:%M:%S) ==="
  "$PY" src/train.py --obs $OBS --patient train \
      --steps 600000 --n-envs 24 --n-steps 256 --seed 0 \
      --eval-freq 50000 --checkpoint-freq 50000 --eval-episodes 20 \
      --reset-log-std -2.5 --resume runs/pretrained/bc_${OBS}_generalist.zip 2>&1 \
    | { grep -E "^\[t1d\]|Traceback|Error" || true; }
done
echo "=== GENERALIST FERTIG $(date +%H:%M:%S) ==="

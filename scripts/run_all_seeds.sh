#!/usr/bin/env bash
# Vollstaendige Wiederholung des 2x2-Versuchs fuer zwei Zufallsstartwerte.
#
# Notwendig geworden durch einen Fehler in der Seed-Vergabe: bei 24 Arbeits-
# prozessen und einem festen Block von 10000 reichten die Trainings-Seeds bis
# 239999 und ueberlappten den Auswertungsbereich ab 100000. Die personalisierten
# Regler trainierten dadurch auf 19 der 20 Episoden, auf denen sie bewertet
# wurden. Die Blockgroesse fuellt jetzt genau [0, HELDOUT_SEED_BASE).
set -euo pipefail
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src

PATIENTS="adolescent#008 adolescent#009 adolescent#010 adult#008 adult#009 adult#010 child#007 child#008 child#009 child#010"

for SEED in 0 1; do
  for OBS in privileged cgm; do
    echo "=== generalist $OBS seed $SEED  $(date +%H:%M:%S) ==="
    "$PY" src/train.py --obs "$OBS" --patient train \
        --steps 600000 --n-envs 24 --n-steps 256 --seed "$SEED" \
        --eval-freq 100000 --checkpoint-freq 100000 --eval-episodes 20 \
        --reset-log-std -2.5 --resume "runs/pretrained/bc_${OBS}_generalist.zip" 2>&1 \
      | { grep -E "^\[t1d\] Lauf|Traceback|Error" || true; }
  done

  for P in $PATIENTS; do
    TAG=$(echo "$P" | tr -d '#')
    for OBS in privileged cgm; do
      BC="runs/pretrained/bc_${OBS}_${TAG}"
      [ -f "${BC}.zip" ] || "$PY" src/pretrain.py --obs "$OBS" --patient "$P" \
            --episodes 80 --out "$BC" 2>&1 | { grep -E "^\[bc\] Val" || true; }
      "$PY" src/train.py --obs "$OBS" --patient "$P" \
          --steps 250000 --n-envs 24 --n-steps 256 --seed "$SEED" \
          --eval-freq 100000 --checkpoint-freq 100000 --eval-episodes 10 \
          --reset-log-std -2.5 --resume "${BC}.zip" 2>&1 \
        | { grep -E "Traceback|Error" || true; }
    done
    echo "  $P seed $SEED fertig  $(date +%H:%M:%S)"
  done

  echo "=== AUSWERTUNG seed $SEED  $(date +%H:%M:%S) ==="
  "$PY" src/eval_cohort.py --episodes 20 --seed "$SEED" 2>&1 | tail -14
done
echo "=== ALLE SEEDS FERTIG $(date +%H:%M:%S) ==="

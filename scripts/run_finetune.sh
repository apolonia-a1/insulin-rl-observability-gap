#!/usr/bin/env bash
# Vorversuch auf einem Patienten (adult#001); vom Kohortenversuch
# (run_cohort.sh, run_all_seeds.sh) abgeloest.
# PPO-Feinabstimmung ausgehend vom klinischen Warmstart. Beide Laeufe teilen
# Hyperparameter, Seeds und Dynamik; nur der Beobachtungsraum unterscheidet sie.
set -euo pipefail
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src
COMMON="--patient adult#001 --steps 600000 --n-envs 24 --n-steps 256 --seed 0 \
        --eval-freq 50000 --checkpoint-freq 50000 --eval-episodes 20 --reset-log-std -2.5"
for OBS in privileged cgm; do
  echo "=============== finetune $OBS  $(date +%H:%M:%S) ==============="
  "$PY" src/train.py --obs $OBS $COMMON \
      --resume runs/pretrained/bc_${OBS}_adult001.zip 2>&1 \
    | { grep -E "^\[t1d\]|total_timesteps|ep_rew_mean|Traceback|Error" || true; }
done
echo "=============== FERTIG $(date +%H:%M:%S) ==============="

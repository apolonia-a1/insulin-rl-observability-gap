#!/usr/bin/env bash
# Ablation zur Fenstergroesse des CGM-Agenten.
#
# Frage: Der CGM-Agent sah bisher 12 Schritte = 36 Minuten Sensorverlauf. Bleibt
# die Informationsluecke bestehen, wenn er 60 Schritte = 3 Stunden sieht, also
# das Fuenffache? Das ist die naheliegendste Kritik am Hauptergebnis.
#
# Ein-Faktor-Versuch. Veraendert wird ausschliesslich `cgm_hist_len`, also die
# Laenge des Sensorfensters. Insulinhistorie (12) und Ankuendigungsfenster
# bleiben unveraendert, der privilegierte Beobachtungsraum ebenfalls. Deshalb
# muss der privilegierte Arm nicht neu trainiert werden: die bestehenden Laeufe
# aus runs/train/ sind weiterhin die gueltige Vergleichsbedingung.
#
# Alles landet unter runs/ablation_cgm_w60/, damit eval_cohort.py den
# Hauptversuch in runs/train/ nicht mit diesen Laeufen verwechselt.
set -euo pipefail
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src

W=60                       # 60 Schritte zu 3 min = 180 min
GROUP=ablation_cgm_w60
BC=runs/pretrained/bc_cgm_generalist_w${W}

echo "=== ABLATION CGM-Fenster ${W} Schritte (= $((W*3)) min)  Start $(date +%H:%M:%S) ==="

if [ ! -f "${BC}.zip" ]; then
  echo "=== BC Warmstart, Fenster ${W}  $(date +%H:%M:%S) ==="
  "$PY" src/pretrain.py --obs cgm --patient train --episodes 300 \
      --cgm-hist-len "$W" --out "$BC" 2>&1 | { grep -E "^\[bc\]|Epoche" || true; }
else
  echo "=== BC bereits vorhanden, uebersprungen ==="
fi

for SEED in 0 1; do
  echo "=== Feinabstimmung cgm w${W} seed ${SEED}  $(date +%H:%M:%S) ==="
  "$PY" src/train.py --obs cgm --patient train \
      --steps 600000 --n-envs 24 --n-steps 256 --seed "$SEED" \
      --eval-freq 100000 --checkpoint-freq 100000 --eval-episodes 20 \
      --reset-log-std -2.5 --cgm-hist-len "$W" --run-group "$GROUP" \
      --resume "${BC}.zip" 2>&1 | { grep -E "^\[t1d\]|Traceback|Error" || true; }
done

for SEED in 0 1; do
  echo "=== Auswertung w${W} seed ${SEED}  $(date +%H:%M:%S) ==="
  "$PY" src/eval_cohort.py --episodes 20 --seed "$SEED" \
      --group "$GROUP" --only-generalist --out "ablation_w${W}_seed${SEED}" 2>&1 | tail -12
done

echo "=== ABLATION FERTIG $(date +%H:%M:%S) ==="

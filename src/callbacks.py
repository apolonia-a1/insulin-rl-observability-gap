"""Per-episode CSV logging for training and evaluation.

Metrik-Infrastruktur: eine Zeile pro Episode, sofort auf Platte geschrieben.

Uebernommen aus meinem frueheren RL-Projekt (Mega-Man-X-Agent).
Der Grundsatz dort wie hier: alle Auswertungen und Abbildungen des Berichts
muessen sich spaeter aus den CSV-Dateien erzeugen lassen, ohne einen einzigen
Lauf zu wiederholen. Deshalb wird pro Episode geschrieben und geflusht, nicht
am Ende aggregiert.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Spaltenreihenfolge der Episodenbilanz; identisch in episodes.csv und eval.csv.
STAT_COLS = [
    "episode_seed", "patient", "obs_mode", "outcome", "steps", "reward_total",
    "r_risk", "r_hypo", "r_term", "tir", "tbr_total", "tbr_level1", "tbr_level2",
    "tar_total", "tar_level1", "tar_level2", "mean_bg", "std_bg", "cv",
    "min_bg", "max_bg", "lbgi", "hbgi", "risk_index", "total_insulin_U", "total_cho_g",
]


class EpisodeLogger(BaseCallback):
    """Schreibt jede Trainingsepisode nach episodes.csv und aggregiert nach TensorBoard."""

    def __init__(self, run_dir, verbose=0):
        super().__init__(verbose)
        self.run_dir = Path(run_dir)

    def _on_training_start(self):
        self.t0 = time.time()
        self.ep_count = 0
        self._f = open(self.run_dir / "episodes.csv", "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["episode", "global_step", "wall_time", "env_id"] + STAT_COLS)
        self.recent = deque(maxlen=100)

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            st = info.get("episode_stats")
            if st is None:
                continue
            self.ep_count += 1
            self._w.writerow(
                [self.ep_count, self.num_timesteps, round(time.time() - self.t0, 1), i]
                + [st[k] for k in STAT_COLS]
            )
            self._f.flush()
            self.recent.append(st)
        return True

    def _on_rollout_end(self):
        if not self.recent:
            return
        r = list(self.recent)
        for key in ("tir", "tbr_total", "tbr_level2", "tar_total", "mean_bg",
                    "cv", "risk_index", "total_insulin_U"):
            self.logger.record(f"t1d/{key}_100", float(np.mean([s[key] for s in r])))
        self.logger.record("t1d/terminated_rate_100",
                           float(np.mean([s["outcome"] == "terminated" for s in r])))

    def _on_training_end(self):
        hours = (time.time() - self.t0) / 3600.0
        summary = {
            "total_steps": int(self.num_timesteps),
            "total_episodes": self.ep_count,
            "wall_hours": round(hours, 3),
            "steps_per_sec": round(self.num_timesteps / max(1e-9, hours * 3600), 1),
        }
        if self.recent:
            summary["final_tir_100"] = float(np.mean([s["tir"] for s in self.recent]))
        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self._f.close()


class EvalEpisodeLogger(EvalCallback):
    """EvalCallback plus eine Zeile pro einzelner Evaluationsepisode.

    SB3 protokolliert nur Reward und Episodenlaenge. Fuer den Bericht werden die
    klinischen Metriken jeder einzelnen Held-out-Episode gebraucht, damit
    Ergebnistabellen Einzellaeufe zeigen koennen und nicht nur Mittelwerte.
    """

    def __init__(self, run_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_dir = Path(run_dir)
        self._eval_ep = 0

    def _on_training_start(self):
        super()._on_training_start()
        self._f = open(self.run_dir / "eval.csv", "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["global_step", "eval_episode"] + STAT_COLS)

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_.get("done"):
            st = locals_["info"].get("episode_stats")
            if st is not None:
                self._eval_ep += 1
                self._w.writerow([self.num_timesteps, self._eval_ep] + [st[k] for k in STAT_COLS])
                self._f.flush()

    def _on_training_end(self):
        super()._on_training_end()
        self._f.close()

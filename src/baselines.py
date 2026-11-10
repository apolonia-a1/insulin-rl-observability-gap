"""Clinical reference controllers (constant basal, basal-bolus) on the same episode protocol as the RL agent.

Klinische Referenzregler auf demselben Episodenprotokoll wie der RL-Agent.

Zwei Referenzen:

* `basal`  - konstante stationaere Basalrate, ohne jede Regelung. Die untere
  Schranke, gegen die jede Regelung sich absetzen muss.
* `bb`     - Basal-Bolus-Therapie (`simglucose.controller.basal_bolus_ctrller`),
  also die uebliche klinische Praxis mit Kohlenhydratfaktor und
  Korrekturfaktor aus `Quest.csv`. Dieser Regler bekommt die Mahlzeit
  angekuendigt, wie es ein Patient beim Bolusrechner auch tut.

Aufruf:
    python3 src/baselines.py --patient adult#001 --split heldout --episodes 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metrics import glycemic_metrics, meets_targets, summarize
from t1d_env import HELDOUT_SEED_BASE, T1DEnv, resolve_patients

PROJECT = Path(__file__).resolve().parent.parent


def run_constant_basal(env: T1DEnv, seed: int) -> dict:
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, term, trunc, info = env.step_insulin(env.basal_ss)
        done = term or trunc
    return info["episode_stats"]


class _Obs:
    """Minimaler Beobachtungsadapter; der BBController liest nur `.CGM`."""

    __slots__ = ("CGM",)

    def __init__(self, cgm: float):
        self.CGM = cgm


def run_basal_bolus(env: T1DEnv, seed: int) -> dict:
    """Basal-Bolus-Therapie, die uebliche klinische Praxis.

    Der Regler bekommt die aufgenommenen Kohlenhydrate des jeweils letzten
    Schritts und berechnet daraus den Bolus. Diese Verzoegerung um einen Schritt
    ist die Konvention der Simulationsschleife von simglucose selbst
    (`sim_engine`), die hier bewusst uebernommen wird, damit die Referenzzahlen
    mit der Literatur vergleichbar bleiben.
    """
    from simglucose.controller.basal_bolus_ctrller import BBController

    ctrl = BBController()
    env.reset(seed=seed)
    current = env.patient_name
    sample_time = env._core.sample_time
    cgm, meal = env._last_cgm, 0.0
    done, info = False, {}
    while not done:
        act = ctrl.policy(
            _Obs(cgm), 0.0, False,
            patient_name=current, meal=meal, sample_time=sample_time,
        )
        _, _, term, trunc, info = env.step_insulin(act.basal + act.bolus)
        cgm, meal = info["cgm"], info["meal"]
        done = term or trunc
    return info["episode_stats"]


def evaluate(controller: str, patient: str, seeds: list[int], days: int) -> tuple[list[dict], dict]:
    runner = {"basal": run_constant_basal, "bb": run_basal_bolus}[controller]
    rows = []
    for pat in resolve_patients(patient):
        env = T1DEnv(patient_name=pat, obs_mode="cgm", days=days)
        rows += [runner(env, s) for s in seeds]
        env.close()

    agg = {}
    for key in ("tir", "tbr_total", "tbr_level1", "tbr_level2", "tar_total",
                "tar_level1", "tar_level2", "mean_bg", "cv", "risk_index",
                "lbgi", "hbgi", "total_insulin_U", "reward_total"):
        vals = np.array([r[key] for r in rows], dtype=float)
        agg[key] = float(vals.mean())
        agg[key + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    agg["n_episodes"] = len(rows)
    agg["n_terminated"] = sum(r["outcome"] == "terminated" for r in rows)
    agg["episodes_tir_over_70"] = sum(r["tir"] > 70.0 for r in rows)
    agg["by_patient"] = {}
    for pat in sorted({r["patient"] for r in rows}):
        sub = [r for r in rows if r["patient"] == pat]
        agg["by_patient"][pat] = {
            "tir": float(np.mean([r["tir"] for r in sub])),
            "tbr_total": float(np.mean([r["tbr_total"] for r in sub])),
            "episodes_tir_over_70": sum(r["tir"] > 70.0 for r in sub),
            "n": len(sub)}
    agg["by_group"] = {}
    for g in ("adolescent", "adult", "child"):
        sub = [r for r in rows if r["patient"].startswith(g)]
        if sub:
            agg["by_group"][g] = {
                "tir": float(np.mean([r["tir"] for r in sub])),
                "tbr_total": float(np.mean([r["tbr_total"] for r in sub])),
                "episodes_tir_over_70": sum(r["tir"] > 70.0 for r in sub),
                "n": len(sub)}
    return rows, agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="adult#001",
                   help="Name, Komma-Liste oder Kohortenschluessel (train/heldout/all)")
    p.add_argument("--controller", default="all", choices=["all", "basal", "bb"])
    p.add_argument("--split", default="heldout", choices=["heldout", "train"])
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--out", default=None, help="Verzeichnis fuer die CSV-Ausgabe")
    args = p.parse_args()

    base = HELDOUT_SEED_BASE if args.split == "heldout" else 0
    seeds = list(range(base, base + args.episodes))
    controllers = ["basal", "bb"] if args.controller == "all" else [args.controller]

    out_dir = Path(args.out) if args.out else PROJECT / "runs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for c in controllers:
        rows, agg = evaluate(c, args.patient, seeds, args.days)
        tag = f"{c}_{args.patient.replace('#', '')}_{args.split}"
        with open(out_dir / f"{tag}_episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary[c] = agg
        print(f"\n=== {c}  {args.patient}  {args.split}  n={len(seeds)} ===")
        print(summarize(agg))
        print("  Zielerreichung:", meets_targets(agg))
        print(f"  Abbrueche: {agg['n_terminated']}/{agg['n_episodes']}  "
              f"| Insulin {agg['total_insulin_U']:.1f} U/Tag  "
              f"| ueber 70 %: {agg['episodes_tir_over_70']}/{agg['n_episodes']}")
        if agg["by_group"]:
            for g, a in agg["by_group"].items():
                print(f"    {g:11s}: TIR {a['tir']:5.1f} | TBR {a['tbr_total']:4.2f} | "
                      f"ueber 70 %: {a['episodes_tir_over_70']}/{a['n']}")

    with open(out_dir / f"summary_{args.patient.replace('#','')}_{args.split}.json", "w") as f:
        json.dump({"patient": args.patient, "split": args.split, "seeds": [seeds[0], seeds[-1]],
                   "days": args.days, "controllers": summary}, f, indent=2)
    print(f"\ngeschrieben nach {out_dir}")


if __name__ == "__main__":
    main()

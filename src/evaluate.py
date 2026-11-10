"""Held-out evaluation of one trained agent.

Held-out-Evaluation eines trainierten Agenten.

Die Held-out-Szenarien stammen aus einem Seed-Block, der vom Trainingsblock
disjunkt ist (`HELDOUT_SEED_BASE`). Jede Episode wird deterministisch
ausgewertet und einzeln protokolliert, damit die Ergebnistabellen des Berichts
Einzellaeufe zeigen koennen statt nur Mittelwerte.

Aufruf:
    python3 src/evaluate.py runs/adult001/<lauf>/checkpoints/best_model.zip --obs privileged
    python3 src/evaluate.py <modell> --obs cgm --episodes 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metrics import meets_targets, summarize
from t1d_env import HELDOUT_SEED_BASE, T1DEnv, resolve_patients

PROJECT = Path(__file__).resolve().parent.parent

AGG_KEYS = ("tir", "tbr_total", "tbr_level1", "tbr_level2", "tar_total",
            "tar_level1", "tar_level2", "mean_bg", "std_bg", "cv", "lbgi",
            "hbgi", "risk_index", "total_insulin_U", "reward_total", "steps")


def aggregate(rows: list[dict]) -> dict:
    agg = {}
    for key in AGG_KEYS:
        vals = np.array([r[key] for r in rows], dtype=float)
        agg[key] = float(vals.mean())
        agg[key + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    agg["n_episodes"] = len(rows)
    agg["n_terminated"] = sum(r["outcome"] == "terminated" for r in rows)
    # Anteil der Episoden, die das TIR-Konsensziel einzeln erreichen
    agg["episodes_tir_over_70"] = sum(r["tir"] > 70.0 for r in rows)
    return agg


def run_episodes(model, env: T1DEnv, seeds: list[int], deterministic=True) -> list[dict]:
    rows = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        rows.append(info["episode_stats"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("--obs", choices=["cgm", "privileged"], required=True,
                   help="muss zum Training des Modells passen")
    p.add_argument("--patient", default="adult#001",
                   help="Patientenname, Komma-Liste oder Kohortenschluessel "
                        "(train / heldout / all)")
    p.add_argument("--episodes", type=int, default=60,
                   help="Episoden JE PATIENT")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--stochastic", action="store_true",
                   help="stochastische statt deterministischer Politik")
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    from stable_baselines3 import PPO

    seeds = list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + args.episodes))
    patients = resolve_patients(args.patient)
    model = PPO.load(args.model_path, device="cpu")

    rows = []
    for pat in patients:
        env = T1DEnv(patient_name=pat, obs_mode=args.obs, days=args.days)
        rows += run_episodes(model, env, seeds, deterministic=not args.stochastic)
        env.close()
    agg = aggregate(rows)
    by_patient = {pat: aggregate([r for r in rows if r["patient"] == pat]) for pat in patients}
    by_group = {}
    for g in ("adolescent", "adult", "child"):
        sub = [r for r in rows if r["patient"].startswith(g)]
        if sub:
            by_group[g] = aggregate(sub)

    tag = args.tag or f"{args.obs}_{args.patient.replace('#','')}"
    out_dir = Path(args.out) if args.out else Path(args.model_path).resolve().parent.parent / "heldout"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"heldout_{tag}_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / f"heldout_{tag}_summary.json", "w") as f:
        json.dump({"model": str(args.model_path), "obs_mode": args.obs,
                   "patient_spec": args.patient, "patients": patients,
                   "episodes_per_patient": args.episodes,
                   "deterministic": not args.stochastic,
                   "seed_range": [seeds[0], seeds[-1]], "aggregate": agg,
                   "by_patient": by_patient, "by_group": by_group}, f, indent=2)

    print(f"\n=== Held-out  {args.obs}  {args.patient}  "
          f"{len(patients)} Patienten x {args.episodes} Episoden = {len(rows)} ===")
    print(summarize(agg))
    print("  Zielerreichung (Mittelwerte):", meets_targets(agg))
    print(f"  Episoden mit TIR > 70 %: {agg['episodes_tir_over_70']}/{agg['n_episodes']}")
    print(f"  Abbrueche: {agg['n_terminated']}/{agg['n_episodes']}")
    print(f"  Insulin {agg['total_insulin_U']:.1f} U/Tag  |  TIR SD {agg['tir_sd']:.1f}")
    if len(patients) > 1:
        print("  je Gruppe:")
        for g, a in by_group.items():
            print(f"    {g:11s}: TIR {a['tir']:5.1f} +/- {a['tir_sd']:4.1f} | "
                  f"TBR {a['tbr_total']:4.2f} | ueber 70 %: "
                  f"{a['episodes_tir_over_70']}/{a['n_episodes']}")
        print("  je Patient:")
        for pat, a in by_patient.items():
            print(f"    {pat:15s}: TIR {a['tir']:5.1f} | TBR {a['tbr_total']:4.2f} | "
                  f"ueber 70 %: {a['episodes_tir_over_70']}/{a['n_episodes']}")
    print(f"  geschrieben nach {out_dir}")


if __name__ == "__main__":
    main()

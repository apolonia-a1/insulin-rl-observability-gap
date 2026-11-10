"""Evaluation of the 2x2 experiment (observation x personalisation) on the held-out patients.

Auswertung des 2x2-Versuchs: Information mal Personalisierung.

Vergleicht auf denselben zehn Held-out-Patienten und denselben Episoden-Seeds:

* den generalistischen Regler (auf 20 anderen Patienten trainiert) und
* die personalisierten Regler (je einer, auf dem jeweiligen Patienten trainiert),

jeweils in beiden Beobachtungsraeumen. Schreibt eine Ergebnistabelle sowie die
Aufschluesselung nach Gruppe und Patient.

Aufruf:
    python3 src/eval_cohort.py --episodes 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from evaluate import aggregate, run_episodes
from metrics import meets_targets, summarize
from t1d_env import HELDOUT_PATIENTS, HELDOUT_SEED_BASE, T1DEnv

SEED = 0
GROUP = "train"          # Unterverzeichnis der generalistischen Laeufe

PROJECT = Path(__file__).resolve().parent.parent


def newest_run(pattern: str) -> str | None:
    hits = sorted(glob.glob(str(PROJECT / "runs" / pattern)))
    return hits[-1] if hits else None


def load_model(path: str):
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def run_cgm_hist_len(run: str) -> int | None:
    """Liest die CGM-Fensterlaenge aus der Provenienz des Laufs.

    Die Auswertungsumgebung muss denselben Beobachtungsraum aufspannen wie das
    Training, sonst passt das geladene Netz nicht zur Eingabe. Der Wert wird
    deshalb aus `config.json` des Laufs gelesen und nicht als Flagge uebergeben:
    eine Flagge koennte still von dem abweichen, womit das Modell trainiert
    wurde.
    """
    cfg = Path(run) / "config.json"
    if not cfg.exists():
        return None
    with open(cfg) as f:
        return json.load(f).get("obs_params", {}).get("cgm_hist_len")


def eval_generalist(obs_mode: str, seeds: list[int], days: int) -> list[dict] | None:
    """Ein Modell, auf allen Held-out-Patienten ausgewertet."""
    run = newest_run(f"{GROUP}/*_{obs_mode}_seed{SEED}")
    if run is None:
        print(f"  [warn] kein generalistischer Lauf fuer {obs_mode}")
        return None
    ckpt = Path(run) / "checkpoints" / "best_model.zip"
    if not ckpt.exists():
        print(f"  [warn] kein best_model in {run}")
        return None
    model = load_model(str(ckpt))
    chl = run_cgm_hist_len(run)
    rows = []
    for pat in HELDOUT_PATIENTS:
        env = T1DEnv(patient_name=pat, obs_mode=obs_mode, days=days, cgm_hist_len=chl)
        rows += run_episodes(model, env, seeds)
        env.close()
    return rows


def eval_personalised(obs_mode: str, seeds: list[int], days: int) -> list[dict] | None:
    """Je Patient das eigene Modell, ausgewertet auf genau diesem Patienten."""
    rows, missing = [], []
    for pat in HELDOUT_PATIENTS:
        tag = pat.replace("#", "")
        run = newest_run(f"{tag}/*_{obs_mode}_seed{SEED}")
        ckpt = Path(run) / "checkpoints" / "best_model.zip" if run else None
        if ckpt is None or not ckpt.exists():
            missing.append(pat)
            continue
        model = load_model(str(ckpt))
        env = T1DEnv(patient_name=pat, obs_mode=obs_mode, days=days,
                     cgm_hist_len=run_cgm_hist_len(run))
        rows += run_episodes(model, env, seeds)
        env.close()
    if missing:
        print(f"  [warn] {obs_mode}: keine personalisierten Modelle fuer {missing}")
    return rows or None


def breakdown(rows: list[dict]) -> dict:
    out = {"overall": aggregate(rows), "by_group": {}, "by_patient": {}}
    for g in ("adolescent", "adult", "child"):
        sub = [r for r in rows if r["patient"].startswith(g)]
        if sub:
            out["by_group"][g] = aggregate(sub)
    for pat in HELDOUT_PATIENTS:
        sub = [r for r in rows if r["patient"] == pat]
        if sub:
            out["by_patient"][pat] = aggregate(sub)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20, help="Episoden je Patient")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--seed", type=int, default=0,
                   help="welcher Trainingsdurchlauf ausgewertet wird")
    p.add_argument("--group", default="train",
                   help="Unterverzeichnis der generalistischen Laeufe unter runs/")
    p.add_argument("--only-generalist", action="store_true",
                   help="nur die generalistischen Zellen, fuer Nebenversuche ohne "
                        "personalisierte Gegenstuecke")
    p.add_argument("--out", default=None, help="Ausgabeverzeichnis unter runs/")
    args = p.parse_args()
    global SEED, GROUP
    SEED = args.seed
    GROUP = args.group

    seeds = list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + args.episodes))
    out_dir = PROJECT / "runs" / (args.out or f"cohort_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [("generalist", eval_generalist)]
    if not args.only_generalist:
        arms.append(("personalisiert", eval_personalised))

    results, all_rows = {}, []
    for pers, fn in arms:
        for obs in ("privileged", "cgm"):
            key = f"{pers}_{obs}"
            print(f"\n>>> {key}")
            rows = fn(obs, seeds, args.days)
            if not rows:
                continue
            for r in rows:
                r = dict(r); r["arm"] = key
                all_rows.append(r)
            results[key] = breakdown(rows)
            a = results[key]["overall"]
            print("   ", summarize(a))
            print(f"    Ziele: {meets_targets(a)}")
            print(f"    ueber 70 %: {a['episodes_tir_over_70']}/{a['n_episodes']} | "
                  f"Abbrueche {a['n_terminated']}")
            for g, ga in results[key]["by_group"].items():
                print(f"      {g:11s}: TIR {ga['tir']:5.1f} +/- {ga['tir_sd']:4.1f} | "
                      f"TBR {ga['tbr_total']:5.2f}")

    if all_rows:
        with open(out_dir / "cohort_episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    with open(out_dir / "cohort_summary.json", "w") as f:
        json.dump({"heldout_patients": HELDOUT_PATIENTS,
                   "episodes_per_patient": args.episodes,
                   "seed_range": [seeds[0], seeds[-1]],
                   "arms": results}, f, indent=2)

    # --- Die 2x2-Tabelle ---
    print("\n" + "=" * 66)
    print(f"{'':>16} {'privilegiert':>14} {'nur CGM':>14} {'Differenz':>12}")
    for pers in ("generalist", "personalisiert"):
        pr = results.get(f"{pers}_privileged", {}).get("overall", {}).get("tir")
        cg = results.get(f"{pers}_cgm", {}).get("overall", {}).get("tir")
        if pr is not None and cg is not None:
            print(f"{pers:>16} {pr:13.1f}% {cg:13.1f}% {pr-cg:11.1f}")
    print("=" * 66)
    for obs in ("privileged", "cgm"):
        g = results.get(f"generalist_{obs}", {}).get("overall", {}).get("tir")
        pp = results.get(f"personalisiert_{obs}", {}).get("overall", {}).get("tir")
        if g is not None and pp is not None:
            print(f"  Personalisierungsgewinn ({obs}): {pp-g:+.1f} Prozentpunkte")
    print(f"\ngeschrieben nach {out_dir}")


if __name__ == "__main__":
    main()

"""Morris screening of the UVA/Padova patient parameters (SALib).

Morris-Screening ueber die Patientenparameter des UVA/Padova-Modells.

Methodisch entspricht das dem Sensitivitaetsverfahren, das in der Vorlesung am
Noble-Modell des EKG gezeigt wird (SALib, Morris, elementare Effekte). Uebertragen
auf diese Arbeit lautet die Frage: **welche physiologischen Parameter bestimmen,
wie gut sich ein Patient ueberhaupt regeln laesst**, und unterscheidet sich das
zwischen der klinischen Referenztherapie und dem gelernten Regler.

Zielgroesse (QoI) ist die Time in Range ueber mehrere Mahlzeitenszenarien.

Aufruf:
    python3 src/sensitivity.py --controller bb --trajectories 20
    python3 src/sensitivity.py --controller model --model runs/.../best_model.zip --obs cgm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from metrics import glycemic_metrics
from t1d_env import HELDOUT_SEED_BASE, T1DEnv

PROJECT = Path(__file__).resolve().parent.parent

# Zehn physiologisch interpretierbare Parameter des Modells. Auswahl bewusst in
# der Groessenordnung des Vorlesungsbeispiels (dort ebenfalls zehn), damit die
# Darstellung vergleichbar bleibt.
PARAMS = [
    ("kabs", "Resorptionsrate aus dem Darm"),
    ("kmax", "maximale Magenentleerungsrate"),
    ("kp1", "Basisrate der endogenen Glukoseproduktion"),
    ("kp3", "Insulinwirkung auf die Glukoseproduktion"),
    ("Vmx",  "insulinabhaengige Glukoseaufnahme"),
    ("Km0",  "Michaelis-Konstante der Glukoseaufnahme"),
    ("ka1",  "Resorption des subkutanen Insulins"),
    ("kd",   "Dissoziation des subkutanen Insulins"),
    ("ksc",  "Diffusion Plasma zu subkutaner Glukose"),
    ("u2ss", "stationaerer Insulinbedarf"),
]

REL_RANGE = 0.20   # Variation um +/- 20 % um den Nominalwert


def build_problem(patient: str) -> tuple[dict, dict]:
    """SALib-Problemdefinition mit patientenspezifischen Nominalwerten."""
    from t1d_env import _patient_params
    p = _patient_params(patient)
    nominal = {name: float(getattr(p, name)) for name, _ in PARAMS}
    problem = {
        "num_vars": len(PARAMS),
        "names": [n for n, _ in PARAMS],
        "bounds": [[nominal[n] * (1 - REL_RANGE), nominal[n] * (1 + REL_RANGE)]
                   for n, _ in PARAMS],
    }
    return problem, nominal


def _run_episode(env: T1DEnv, seed: int, policy) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        if policy is None:                       # konstante Basalrate
            _, _, term, trunc, info = env.step_insulin(env.basal_ss)
        elif policy == "bb":
            raise RuntimeError("bb wird ueber run_bb behandelt")
        else:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info["episode_stats"]


def evaluate_qoi(overrides: dict, patient: str, controller: str, model,
                 obs_mode: str, seeds: list[int]) -> float:
    """Zielgroesse: mittlere Time in Range ueber die Szenarien."""
    env = T1DEnv(patient_name=patient, obs_mode=obs_mode, days=1,
                 param_overrides=overrides)
    tirs = []
    for s in seeds:
        if controller == "bb":
            from baselines import run_basal_bolus
            st = run_basal_bolus(env, s)
        elif controller == "basal":
            st = _run_episode(env, s, None)
        else:
            st = _run_episode(env, s, model)
        tirs.append(st["tir"])
    env.close()
    return float(np.mean(tirs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="adult#001")
    p.add_argument("--controller", choices=["bb", "basal", "model"], default="bb")
    p.add_argument("--model", default=None)
    p.add_argument("--obs", choices=["cgm", "privileged"], default="cgm")
    p.add_argument("--trajectories", type=int, default=20, help="Morris-Trajektorien N")
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--scenarios", type=int, default=3, help="Szenarien je Parametersatz")
    p.add_argument("--seed", type=int, default=0, help="Seed der Morris-Stichprobe")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model = None
    if args.controller == "model":
        if not args.model:
            raise SystemExit("--controller model erfordert --model")
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")

    problem, nominal = build_problem(args.patient)
    X = morris_sample.sample(problem, N=args.trajectories, num_levels=args.levels, seed=args.seed)
    seeds = list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + args.scenarios))

    print(f"[sens] {len(X)} Parametersaetze x {args.scenarios} Szenarien "
          f"= {len(X)*args.scenarios} Episoden, Regler {args.controller}")
    t0 = time.time()
    Y = np.empty(len(X))
    for i, row in enumerate(X):
        overrides = {name: float(v) for name, v in zip(problem["names"], row)}
        Y[i] = evaluate_qoi(overrides, args.patient, args.controller, model, args.obs, seeds)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(X)}  ({el:.0f}s, ETA {el/(i+1)*(len(X)-i-1):.0f}s)", flush=True)

    Si = morris_analyze.analyze(problem, X, Y, num_levels=args.levels,
                                conf_level=0.95, print_to_console=False)

    order = np.argsort(Si["mu_star"])[::-1]
    desc = dict(PARAMS)
    print(f"\n=== Morris-Screening, QoI = Time in Range, Regler {args.controller} ===")
    print(f"{'Parameter':<8} {'mu*':>8} {'sigma':>8}   Bedeutung")
    for i in order:
        n = problem["names"][i]
        print(f"{n:<8} {Si['mu_star'][i]:8.2f} {Si['sigma'][i]:8.2f}   {desc[n]}")
    print(f"\nTIR-Spanne ueber alle Parametersaetze: {Y.min():.1f} bis {Y.max():.1f} % "
          f"(nominal-nah: Median {np.median(Y):.1f} %)")

    out_dir = Path(args.out) if args.out else PROJECT / "runs" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.controller}_{args.patient.replace('#','')}"
    np.savez(out_dir / f"morris_{tag}.npz", X=X, Y=Y,
             names=np.array(problem["names"]), mu_star=Si["mu_star"], sigma=Si["sigma"])
    with open(out_dir / f"morris_{tag}.json", "w") as f:
        json.dump({
            "patient": args.patient, "controller": args.controller,
            "model": args.model, "obs_mode": args.obs,
            "trajectories": args.trajectories, "levels": args.levels, "seed": args.seed,
            "scenarios": args.scenarios, "rel_range": REL_RANGE,
            "nominal": nominal,
            "results": [
                {"name": problem["names"][i], "description": desc[problem["names"][i]],
                 "mu_star": float(Si["mu_star"][i]), "mu": float(Si["mu"][i]),
                 "sigma": float(Si["sigma"][i])}
                for i in order
            ],
            "qoi_min": float(Y.min()), "qoi_max": float(Y.max()),
            "qoi_median": float(np.median(Y)),
            "wall_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"geschrieben nach {out_dir}")


if __name__ == "__main__":
    main()

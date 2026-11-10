"""Behavioural-cloning warm start from the clinical basal-bolus controller.

Warmstart der Politik durch Imitation der klinischen Referenztherapie.

Hintergrund. Der Aktionsraum ist logarithmisch und umfasst den Faktor 160 um die
Basalrate. Damit entsteht ein Zielkonflikt, der sich rein ueber die
Explorationsbreite nicht loesen laesst: eng genug, um den Patienten nicht zu
gefaehrden, heisst zu eng, um einen Mahlzeitenbolus je zu ziehen; weit genug, um
Boli zu entdecken, heisst laufend lebensbedrohliche Dosen.

Loesung: die Politik wird zunaechst darauf trainiert, den Basal-Bolus-Regler zu
imitieren (Behavioral Cloning), und erst danach mit PPO weiter optimiert. Das ist
dieselbe Lehrer-Schueler-Struktur, die dem Kernthema der Arbeit zugrunde liegt:

* Der Lehrer (Basal-Bolus) sieht die angekuendigte Mahlzeit.
* Der `privileged`-Schueler sieht sie ebenfalls und kann den Lehrer im Prinzip
  vollstaendig nachbilden.
* Der `cgm`-Schueler sieht sie **nicht**. Wie gut er den Lehrer trotzdem
  nachbildet, ist genau die gesuchte Wahrnehmungsluecke.

Aufruf:
    python3 src/pretrain.py --obs privileged --episodes 200
    python3 src/pretrain.py --obs cgm --episodes 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from t1d_env import MAX_MULT, T1DEnv

PROJECT = Path(__file__).resolve().parent.parent


def collect(patient: str, obs_mode: str, n_episodes: int, days: int, seed0: int = 0,
            cgm_hist_len: int | None = None):
    """Zeichnet Beobachtungen und die zugehoerigen Lehrer-Aktionen auf."""
    from simglucose.controller.basal_bolus_ctrller import BBController
    from baselines import _Obs

    env = T1DEnv(patient_name=patient, obs_mode=obs_mode, days=days,
                 cgm_hist_len=cgm_hist_len)
    ctrl = BBController()
    X, Y = [], []
    stats = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        # Bei einer Kohorte zieht der Reset je Episode einen anderen Patienten.
        # Der Lehrer braucht dessen Namen, denn Kohlenhydrat- und Korrekturfaktor
        # stehen patientenweise in Quest.csv.
        current = env.patient_name
        sample_time = env._core.sample_time
        cgm, meal = env._last_cgm, 0.0
        done = False
        while not done:
            act = ctrl.policy(_Obs(cgm), 0.0, False, patient_name=current,
                              meal=meal, sample_time=sample_time)
            u = float(np.clip(act.basal + act.bolus, 0.0, env.max_basal))
            # dieselbe Umrechnung wie in T1DEnv.step_insulin
            a = float(np.clip(np.log(max(u, 1e-12) / env.basal_ss) / np.log(MAX_MULT), -1.0, 1.0))
            X.append(obs.copy())
            Y.append([a])
            obs, _, term, trunc, info = env.step_insulin(u)
            cgm, meal = info["cgm"], info["meal"]
            done = term or trunc
        stats.append(info["episode_stats"])
    env.close()
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="adult#001",
                   help="Name, Komma-Liste oder Kohortenschluessel (train/heldout/all)")
    p.add_argument("--obs", choices=["cgm", "privileged"], required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--cgm-hist-len", type=int, default=None,
                   help="Laenge des CGM-Fensters in Schritten zu 3 min "
                        "(Standard 12 = 36 min). Aendert nur den CGM-Beobachtungsraum, "
                        "nicht die Insulinhistorie und nicht den privilegierten Raum.")
    args = p.parse_args()

    from stable_baselines3 import PPO
    import t1d_env

    print(f"[bc] sammle {args.episodes} Lehrer-Episoden ({args.obs}) ...", flush=True)
    t0 = time.time()
    X, Y, stats = collect(args.patient, args.obs, args.episodes, args.days,
                          cgm_hist_len=args.cgm_hist_len)
    tir = float(np.mean([s["tir"] for s in stats]))
    print(f"[bc] {len(X)} Paare in {time.time()-t0:.0f}s | Lehrer-TIR {tir:.1f}% "
          f"| Aktion Mittel {Y.mean():+.3f} sd {Y.std():.3f} max {Y.max():.3f}")

    venv = t1d_env.make_vec_env(args.patient, args.obs, n_envs=1, days=args.days,
                                cgm_hist_len=args.cgm_hist_len)
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[args.hidden] * 2, vf=[args.hidden] * 2),
                                   log_std_init=args.log_std_init))
    policy = model.policy
    policy.train()

    Xt = torch.as_tensor(X)
    Yt = torch.as_tensor(Y)
    n_val = max(1, len(Xt) // 10)
    Xtr, Ytr, Xva, Yva = Xt[:-n_val], Yt[:-n_val], Xt[-n_val:], Yt[-n_val:]

    # Nur der Aktionspfad wird angepasst; der Kritiker bleibt untrainiert und
    # wird spaeter von PPO selbst gelernt.
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    def mean_action(xb):
        feats = policy.extract_features(xb)
        if isinstance(feats, tuple):
            feats = feats[0]
        return policy.action_net(policy.mlp_extractor.forward_actor(feats))

    print(f"[bc] trainiere {args.epochs} Epochen auf {len(Xtr)} Paaren ...", flush=True)
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = torch.nn.functional.mse_loss(mean_action(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            with torch.no_grad():
                va = float(torch.nn.functional.mse_loss(mean_action(Xva), Yva))
            print(f"  Epoche {ep+1:3d}: Train-MSE {tot/len(Xtr):.5f}  Val-MSE {va:.5f}", flush=True)

    with torch.no_grad():
        va = float(torch.nn.functional.mse_loss(mean_action(Xva), Yva))
        pred = mean_action(Xva).numpy().ravel()
    corr = float(np.corrcoef(pred, Yva.numpy().ravel())[0, 1])

    out = Path(args.out) if args.out else PROJECT / "runs" / "pretrained" / f"bc_{args.obs}_{args.patient.replace('#','')}"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    with open(str(out) + "_info.json", "w") as f:
        json.dump({"patient": args.patient, "obs_mode": args.obs,
                   "episodes": args.episodes, "pairs": int(len(X)),
                   "teacher_tir": tir, "epochs": args.epochs,
                   "val_mse": va, "val_corr": corr,
                   "action_mean": float(Y.mean()), "action_sd": float(Y.std()),
                   "log_std_init": args.log_std_init,
                   "cgm_hist_len": args.cgm_hist_len or t1d_env.HIST_LEN,
                   "obs_dim": int(X.shape[1])}, f, indent=2)
    print(f"[bc] Val-MSE {va:.5f} | Korrelation Vorhersage/Lehrer {corr:.3f}")
    print(f"[bc] gespeichert: {out}.zip")


if __name__ == "__main__":
    main()

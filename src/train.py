"""PPO training (Stable-Baselines3) with privileged or CGM-only observations.

PPO-Training der Insulinregelung, wahlweise privilegiert oder nur mit CGM.

Aufruf:
    python src/train.py --obs privileged --patient train --steps 600000 --n-envs 24 \
        --resume runs/pretrained/bc_privileged_generalist.zip --reset-log-std -2.5
    python src/train.py --obs cgm --patient train --steps 600000 --n-envs 24 \
        --resume runs/pretrained/bc_cgm_generalist.zip --reset-log-std -2.5

Erzeugt runs/<patient>/<zeitstempel>_<obs>_seed<N>/ mit config.json,
episodes.csv, eval.csv, tensorboard/, checkpoints/ und run_summary.json.
Die config.json haelt alle Hyperparameter und Paketversionen fest, damit jeder
Lauf reproduzierbar bleibt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import t1d_env
from callbacks import EpisodeLogger, EvalEpisodeLogger

PROJECT = Path(__file__).resolve().parent.parent

# PPO-Hyperparameter. Bewusst nahe an den SB3-Standardwerten fuer kontinuierliche
# Regelaufgaben gehalten.
HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="adult#001",
                   help="Name, Komma-Liste oder Kohortenschluessel (train/heldout/all)")
    p.add_argument("--obs", choices=["cgm", "privileged"], default="cgm")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=512,
                   help="Rollout-Schritte pro Umgebung; n_steps * n_envs bleibt die Rollout-Groesse")
    p.add_argument("--days", type=int, default=1, help="Simulierte Tage pro Episode")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Diskontfaktor. Eine Episode hat 480 Schritte; gamma=0.99 sieht rund "
                        "100 Schritte weit, was fuer die Mahlzeitendynamik knapp ist. 0.997 "
                        "verlaengert den Kredithorizont auf die Groessenordnung einer Mahlzeit.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--log-std-init", type=float, default=-1.0,
                   help="Anfaengliche Explorationsbreite der Politik. Der Wert entscheidet, "
                        "ob ein Mahlzeitenbolus ueberhaupt je gezogen wird: ein Bolus "
                        "entspricht Aktion 0,76 bis 0,95. Bei 0 (SB3-Standard) streut die "
                        "Startpolitik ueber den gesamten Bereich bis zum 160-fachen der "
                        "Basalrate und toetet den Patienten fast jede Episode. Bei -1,5 "
                        "(sigma 0,223) liegt ein Bolus bei 4,0 sigma, also 0,014 Treffern je "
                        "Episode, und wird nie entdeckt. Bei -1,0 (sigma 0,368) sind es "
                        "2,4 sigma und rund 3,5 Treffer je Episode, etwa so viele wie es "
                        "Mahlzeiten gibt.")
    p.add_argument("--eval-freq", type=int, default=25_000, help="Umgebungsschritte zwischen Evaluationen")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--n-scenarios", type=int, default=None,
                   help="Begrenzt die Zahl unterschiedlicher Trainingsszenarien "
                        "(fuer Overfitting-Experimente). Standard: unbegrenzt.")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cpu",
                   help="Die Politik ist ein kleines MLP (64x64). Auf der GPU dominiert der "
                        "Transferaufwand je Minibatch, daher ist die CPU hier schneller.")
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--run-group", default=None,
                   help="Unterverzeichnis unter runs/ statt des Patientennamens. "
                        "Fuer Nebenversuche, damit sie den Hauptversuch nicht verdraengen.")
    p.add_argument("--cgm-hist-len", type=int, default=None,
                   help="Laenge des CGM-Fensters in Schritten zu 3 min "
                        "(Standard 12 = 36 min). Aendert nur den CGM-Beobachtungsraum.")
    p.add_argument("--reset-log-std", type=float, default=None,
                   help="Setzt die Explorationsbreite beim Fortsetzen neu. Nach einem "
                        "Warmstart per Imitation muss der Agent das Bolussieren nicht mehr "
                        "ENTDECKEN, sondern nur noch verfeinern. Die breite Exploration, die "
                        "fuer die Entdeckung noetig waere, zerstoert hier den bereits "
                        "brauchbaren Mittelwert und toetet den Patienten in jeder Episode.")
    args = p.parse_args()

    HYPERPARAMS["n_steps"] = args.n_steps
    HYPERPARAMS["gamma"] = args.gamma
    HYPERPARAMS["n_epochs"] = args.n_epochs

    import numpy, scipy, torch
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.obs}_seed{args.seed}"
    # `--run-group` trennt Nebenversuche vom Hauptversuch. Ohne das landet eine
    # Ablation in `runs/train/` und `eval_cohort.newest_run` zieht sie als
    # juengsten Treffer heran, wodurch sie das Hauptergebnis still ersetzt.
    cohort = args.run_group or args.patient.replace("#", "").replace(",", "_")
    run_dir = PROJECT / "runs" / cohort / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patient": args.patient,
        "patients": t1d_env.resolve_patients(args.patient),
        "obs_mode": args.obs,
        "days_per_episode": args.days,
        "steps_per_episode": int(args.days * 24 * 60 / 3),
        "total_steps": args.steps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "n_scenarios": args.n_scenarios,
        "hyperparams": HYPERPARAMS,
        "net_hidden": args.hidden,
        "log_std_init": args.log_std_init,
        "reset_log_std": args.reset_log_std,
        "reward_version": t1d_env.REWARD_VERSION,
        "reward_weights": {"w_risk": t1d_env.W_RISK, "max_risk": t1d_env.MAX_RISK,
                           "w_hypo": t1d_env.W_HYPO, "w_severe": t1d_env.W_SEVERE,
                           "terminal": "verbleibende Schritte mal maximales Risiko"},
        "action_space": {"kind": "log-multiplier auf basal_ss", "max_mult": t1d_env.MAX_MULT},
        # Den tatsaechlich benutzten Wert festhalten, nicht die Modulkonstante:
        # sonst behauptet die Provenienz 12, waehrend der Lauf mit 60 trainiert hat.
        "obs_params": {"hist_len": t1d_env.HIST_LEN,
                       "cgm_hist_len": args.cgm_hist_len or t1d_env.HIST_LEN,
                       "announce_window": t1d_env.ANNOUNCE_WINDOW},
        "sensor": t1d_env.SENSOR, "pump": t1d_env.PUMP,
        "heldout_seed_base": t1d_env.HELDOUT_SEED_BASE,
        "train_seed_offset": (args.seed * 137) % max(1, t1d_env.HELDOUT_SEED_BASE // args.n_envs),
        "versions": {
            "python": sys.version.split()[0],
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "simglucose": "0.2.11",
        },
        "device": args.device,
        "resumed_from": args.resume,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[t1d] Lauf: {run_dir}")
    print(f"[t1d] obs={args.obs}  Patient={args.patient}  Reward={t1d_env.REWARD_VERSION}  gamma={args.gamma}")

    # Verschiedene Seeds sehen verschiedene Szenarienfolgen, bleiben aber
    # vollstaendig im Trainingsbereich [0, HELDOUT_SEED_BASE).
    train_offset = (args.seed * 137) % max(1, t1d_env.HELDOUT_SEED_BASE // args.n_envs)
    venv = t1d_env.make_vec_env(args.patient, args.obs, n_envs=args.n_envs,
                                days=args.days, seed_base=train_offset,
                                n_scenarios=args.n_scenarios,
                                cgm_hist_len=args.cgm_hist_len)

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=args.device,
                         tensorboard_log=str(run_dir / "tensorboard"))
        # PPO.load uebernimmt den gespeicherten Zufallszustand. Fuer einen
        # echten zweiten Durchlauf muss der Seed nach dem Laden neu gesetzt
        # werden, sonst unterscheiden sich die Laeufe nur im Verzeichnisnamen.
        model.set_random_seed(args.seed)
        if args.reset_log_std is not None:
            with torch.no_grad():
                model.policy.log_std.data.fill_(args.reset_log_std)
            print(f"[t1d] Explorationsbreite neu gesetzt: log_std = {args.reset_log_std}")
    else:
        model = PPO(
            "MlpPolicy", venv, seed=args.seed, verbose=1, device=args.device,
            tensorboard_log=str(run_dir / "tensorboard"),
            policy_kwargs=dict(net_arch=dict(pi=[args.hidden] * 2, vf=[args.hidden] * 2),
                               log_std_init=args.log_std_init),
            **HYPERPARAMS,
        )

    # Evaluation laeuft auf dem disjunkten Held-out-Seed-Block, nie auf Trainingsszenarien.
    eval_env = t1d_env.make_vec_env(args.patient, args.obs, n_envs=1, days=args.days,
                                    seed_base=t1d_env.HELDOUT_SEED_BASE,
                                    n_scenarios=args.eval_episodes,
                                    cgm_hist_len=args.cgm_hist_len)

    callbacks = [
        EpisodeLogger(run_dir, verbose=1),
        CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                           save_path=str(run_dir / "checkpoints"), name_prefix="model"),
        EvalEpisodeLogger(
            str(run_dir), eval_env,
            n_eval_episodes=args.eval_episodes,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            log_path=str(run_dir / "eval"),
            best_model_save_path=str(run_dir / "checkpoints"),
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks,
                reset_num_timesteps=not bool(args.resume))
    model.save(run_dir / "checkpoints" / "model_final")
    print(f"[t1d] fertig. Artefakte in {run_dir}")


if __name__ == "__main__":
    main()

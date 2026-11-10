# runs/

Every logged result of the project. Each training run folder holds `config.json` (hyperparameters,
reward version, package versions), `episodes.csv` (one row per training episode) and `eval.csv`
(one row per evaluation episode). Trained PPO weights (`checkpoints/`) are not included (74 MB);
the scripts in `scripts/` regenerate them. The behavioural-cloning warm starts in `pretrained/` are
small and included.

| Folder | Content |
|---|---|
| `train/` | cohort controllers (one for the 20 training patients), privileged and CGM-only |
| `adolescent008` ... `child010`, `adult001` ... | personalised controllers, one per held-out patient |
| `cohort_seed0/`, `cohort_seed1/` | the 2x2 evaluation on the 10 held-out patients, two training seeds |
| `ablation_cgm_w60/`, `ablation_w60_seed0/1` | CGM-only agent with a 180 min (60 step) sensor window |
| `baselines/` | constant basal rate and clinical basal-bolus on the held-out cohort |
| `pretrained/` | behavioural-cloning warm starts and their validation metrics |
| `sensitivity/` | Morris screening results |

Discarded runs, kept on purpose because they document what went wrong:

| Folder | What it shows |
|---|---|
| `_negative_logstd15` | exploration too narrow: only the basal rate is adjusted, boluses never found |
| `_negative_logstd10` | boluses found, but 97 % of episodes end early |
| `_reward_v1` | reward v1 drives glucose into hypoglycaemia (TIR peaks at 92.0 %, then time below range rises to 12.7 %) |
| `_invalid_seedleak` | the 22 runs with overlapping train/test seeds |
| `_invalid_cohort/` | the first cohort evaluation, invalidated by the seed overlap |
| `_outdated_before_resetfix` | reference run from before a reset optimisation, no longer reproducible bit for bit |

`personalisiert_*` in the JSON files means "personalised".

The German report uses the original folder names: `_kontaminiert_seedleak` = `_invalid_seedleak`,
`_negativ_logstd10/15` = `_negative_logstd10/15`, `_veraltet_vor_resetfix` = `_outdated_before_resetfix`,
`cohort` = `_invalid_cohort`.

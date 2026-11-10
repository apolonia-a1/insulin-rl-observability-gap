#!/usr/bin/env python3
"""Builds all report figures from the logged run files.

Erzeugt alle Abbildungen der Projektarbeit aus den Laufdateien.

Grundsatz: jede Abbildung entsteht aus den
protokollierten CSV- und JSON-Dateien, nie aus einem Screenshot. Damit sind alle
Abbildungen ohne einen einzigen erneuten Trainingslauf reproduzierbar.

Aufruf (aus dem Projektverzeichnis):
    PYTHONPATH=src python src/make_figures.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (6.4, 3.8), "figure.dpi": 140, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.8,
})

# Zwei Signalfarben plus Neutrale, bewusst ohne Verlaeufe.
C = {
    "priv": "#1f4e79",     # privilegiert
    "cgm": "#c0504d",      # nur CGM
    "clin": "#2e7d32",     # klinische Referenz
    "none": "#8a8a8a",     # ohne Regelung
    "band": "#2e7d32",
}

made = []


def save(fig, name, caption):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=140)
    plt.close(fig)
    made.append((name, caption))
    print(f"  OK  {name}  - {caption}")


def _read_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def _eval_curve(run_dir):
    """(Schritte, mittlere TIR) aus eval.csv eines Laufs."""
    rows = _read_csv(os.path.join(run_dir, "eval.csv"))
    by = defaultdict(list)
    for r in rows:
        by[int(r["global_step"])].append(float(r["tir"]))
    steps = sorted(by)
    return np.array(steps), np.array([np.mean(by[s]) for s in steps])


def _find_run(obs_mode):
    cands = sorted(glob.glob(os.path.join(ROOT, "runs", "adult001", f"*_{obs_mode}_*")))
    return cands[-1] if cands else None


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# 1  Aktionsverteilung der klinischen Referenz: Begruendung des Aktionsraums
# --------------------------------------------------------------------------
try:
    print("[1] Aktionsverteilung des Basal-Bolus-Reglers")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import warnings, logging
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from t1d_env import T1DEnv, MAX_MULT
    from baselines import _Obs
    from simglucose.controller.basal_bolus_ctrller import BBController

    env = T1DEnv(patient_name="adult#001", obs_mode="cgm", days=1)
    ctrl = BBController()
    env.reset(seed=100000)
    st_time = env._core.sample_time
    cgm, meal = env._last_cgm, 0.0
    us, done = [], False
    while not done:
        act = ctrl.policy(_Obs(cgm), 0.0, False, patient_name="adult#001",
                          meal=meal, sample_time=st_time)
        u = act.basal + act.bolus
        us.append(u)
        _, _, t_, tr, info = env.step_insulin(u)
        cgm, meal = info["cgm"], info["meal"]
        done = t_ or tr
    us = np.array(us)
    basal = env.basal_ss

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    a1.hist(us / basal, bins=60, color=C["clin"], alpha=0.9)
    a1.set_yscale("log")
    a1.set_xlabel("Insulinrate als Vielfaches der Basalrate")
    a1.set_ylabel("Anzahl Schritte (log)")
    a1.set_title("Lineare Achse: alles bei 1, Boli weit rechts")
    a2.hist(np.log(us / basal) / np.log(MAX_MULT), bins=60, color=C["priv"], alpha=0.9)
    a2.set_yscale("log")
    a2.set_xlabel("Aktion $a$ im logarithmischen Aktionsraum")
    a2.set_ylabel("Anzahl Schritte (log)")
    a2.set_title("Logarithmisch: Basalrate und Boli getrennt aufgelöst")
    fig.suptitle("Warum der Aktionsraum logarithmisch ist (klinische Referenz, ein Tag)")
    save(fig, "fig_aktionsraum.pdf", "Aktionsverteilung der klinischen Referenztherapie")
except Exception:
    traceback.print_exc()

# --------------------------------------------------------------------------
# 2  Beobachtbarkeitsluecke: Plasma, subkutan, CGM
# --------------------------------------------------------------------------
try:
    print("[2] Beobachtbarkeitsluecke")
    env = T1DEnv(patient_name="adult#001", obs_mode="privileged", days=1)
    env.reset(seed=100007)
    Vg = env.Vg
    plasma, subq, cgm_t, meals = [], [], [], []
    done = False
    while not done:
        _, _, t_, tr, info = env.step_insulin(env.basal_ss)
        plasma.append(float(env._core.patient.state[3]) / Vg)
        subq.append(info["bg"]); cgm_t.append(info["cgm"]); meals.append(info["meal"])
        done = t_ or tr
    tmin = np.arange(len(plasma)) * 3 / 60.0
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.axhspan(70, 180, color=C["band"], alpha=0.10, label="Zielbereich 70–180")
    ax.plot(tmin, plasma, color=C["priv"], label="Plasmaglukose (privilegiert)")
    ax.plot(tmin, subq, color="#7a7a7a", lw=1.2, ls="--", label="subkutane Glukose")
    ax.plot(tmin, cgm_t, color=C["cgm"], lw=1.1, alpha=0.85, label="CGM-Messwert (beobachtbar)")
    mt = np.array(tmin)[np.array(meals) > 0]
    for m in mt:
        ax.axvline(m, color="#333333", lw=0.6, alpha=0.35)
    ax.set_xlabel("Zeit (h)"); ax.set_ylabel("Glukose (mg/dL)")
    ax.set_xlim(0, tmin[-1]); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Was der Agent sieht: 12 min Verzögerung und korreliertes Rauschen")
    save(fig, "fig_beobachtbarkeit.pdf", "Plasma, subkutane Glukose und CGM ueber einen Tag")
except Exception:
    traceback.print_exc()

# --------------------------------------------------------------------------
# 3  Explorationsreichweite: warum beide Extreme scheitern
# --------------------------------------------------------------------------
try:
    print("[3] Explorationsreichweite")
    from math import erf, sqrt
    ls = np.array([-1.5, -1.2, -1.0, -0.8, -0.5, 0.0])
    sd = np.exp(ls)
    a_bolus = 0.897
    hits = np.array([(1 - 0.5 * (1 + erf((a_bolus / s) / sqrt(2)))) * 480 for s in sd])
    fig, ax = plt.subplots()
    ax.semilogy(ls, np.maximum(hits, 1e-3), "o-", color=C["priv"])
    ax.axhline(3.5, color=C["clin"], ls="--", lw=1.2, label="etwa 3,5 Mahlzeiten je Tag")
    ax.axhline(0.014, color=C["cgm"], ls=":", lw=1.2, label="log_std -1,5: praktisch nie")
    ax.set_xlabel(r"$\log \sigma$ der Startpolitik")
    ax.set_ylabel("erwartete Boli je Episode (log)")
    ax.set_title("Explorationsbreite bestimmt, ob ein Bolus je gezogen wird")
    ax.legend(fontsize=8)
    save(fig, "fig_exploration.pdf", "Erwartete Bolus-Ziehungen je Episode")
except Exception:
    traceback.print_exc()

# --------------------------------------------------------------------------
# 4  Imitationsguete: die Wahrnehmungsluecke als eine Zahl
# --------------------------------------------------------------------------
try:
    print("[4] Imitationsguete")
    info = {}
    for o in ("privileged", "cgm"):
        p = os.path.join(ROOT, "runs", "pretrained", f"bc_{o}_adult001_info.json")
        if os.path.exists(p):
            info[o] = _load_json(p)
    if len(info) == 2:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.4))
        labels = ["privilegiert", "nur CGM"]
        corr = [info["privileged"]["val_corr"], info["cgm"]["val_corr"]]
        mse = [info["privileged"]["val_mse"], info["cgm"]["val_mse"]]
        a1.bar(labels, corr, color=[C["priv"], C["cgm"]], width=0.55)
        for i, v in enumerate(corr):
            a1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
        a1.set_ylim(0, 1.08); a1.set_ylabel("Korrelation mit dem Lehrer")
        a1.set_title("Imitationsgüte")
        a2.bar(labels, mse, color=[C["priv"], C["cgm"]], width=0.55)
        a2.set_yscale("log"); a2.set_ylabel("Validierungsfehler (MSE, log)")
        for i, v in enumerate(mse):
            a2.text(i, v * 1.25, f"{v:.5f}", ha="center", fontsize=9)
        a2.set_title(f"Faktor {mse[1]/mse[0]:.0f} zwischen den Fehlern")
        fig.suptitle("Gleiche Daten, gleiches Netz, gleiches Training: nur die Beobachtung unterscheidet sich")
        save(fig, "fig_imitation.pdf", "Imitationsguete privilegiert gegen nur CGM")
except Exception:
    traceback.print_exc()

# --------------------------------------------------------------------------
# 5  Morris-Sensitivitaet
# --------------------------------------------------------------------------
try:
    print("[7] Morris-Sensitivitaet")
    cands = sorted(glob.glob(os.path.join(ROOT, "runs", "sensitivity", "morris_*.json")))
    if cands:
        d = _load_json(cands[-1])
        res = d["results"]
        names = [r["name"] for r in res][::-1]
        mus = [r["mu_star"] for r in res][::-1]
        sig = [r["sigma"] for r in res][::-1]
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        y = np.arange(len(names))
        ax.barh(y, mus, color=C["priv"], alpha=0.9, xerr=sig, ecolor="#999999", capsize=2)
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_xlabel(r"$\mu^*$ (mittlerer elementarer Effekt auf die Time in Range, Prozentpunkte)")
        ax.set_title(f"Morris-Screening, Regler {d['controller']}, Variation $\\pm$ {int(d['rel_range']*100)} %")
        save(fig, "fig_sensitivitaet.pdf", "Morris-Screening der Patientenparameter")
    else:
        print("  (noch keine Sensitivitaetsdaten)")
except Exception:
    traceback.print_exc()

# --------------------------------------------------------------------------
# 6  Das 2x2: Information mal Personalisierung
# --------------------------------------------------------------------------
try:
    print("[8] 2x2 Information mal Personalisierung")
    seeds = sorted(glob.glob(os.path.join(ROOT, "runs", "cohort_seed*", "cohort_summary.json")))
    if seeds:
        per = [_load_json(p_)["arms"] for p_ in seeds]
        def cell(k):
            return [a[k]["overall"]["tir"] for a in per if k in a]
        labels = ["generalistisch\n(20 Patienten)", "personalisiert\n(je Patient)"]
        pv = [cell(f"{k}_privileged") for k in ("generalist", "personalisiert")]
        cv = [cell(f"{k}_cgm") for k in ("generalist", "personalisiert")]
        priv = [float(np.mean(v)) for v in pv]
        cgm = [float(np.mean(v)) for v in cv]
        e_pv = [[abs(np.mean(v) - min(v)) for v in pv], [abs(max(v) - np.mean(v)) for v in pv]]
        e_cv = [[abs(np.mean(v) - min(v)) for v in cv], [abs(max(v) - np.mean(v)) for v in cv]]
        if all(len(v) > 0 for v in pv + cv):
            x = np.arange(2); w = 0.34
            fig, ax = plt.subplots(figsize=(6.6, 3.9))
            b1 = ax.bar(x - w/2, priv, w, color=C["priv"], label="privilegiert",
                        yerr=e_pv, capsize=3, ecolor="#333333")
            b2 = ax.bar(x + w/2, cgm, w, color=C["cgm"], label="nur CGM",
                        yerr=e_cv, capsize=3, ecolor="#333333")
            for bars in (b1, b2):
                for b in bars:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                            f"{b.get_height():.1f}".replace(".", ","), ha="center", fontsize=9)
            # Informationsluecke als Spanne annotieren
            for i in range(2):
                ax.annotate("", xy=(i - w/2, priv[i]), xytext=(i + w/2, cgm[i]),
                            arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.0))
                ax.text(i, (priv[i] + cgm[i]) / 2, f"  {priv[i]-cgm[i]:.1f}".replace(".", ","),
                        fontsize=9, color="#333333", va="center")
            ax.axhline(70, color="#444444", ls=":", lw=1.1)
            ax.text(1.42, 71.5, "Konsensziel 70 %", fontsize=8, ha="right")
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_ylabel("Held-out Time in Range (%)"); ax.set_ylim(0, 105)
            ax.legend(fontsize=9, loc="upper left")
            ax.set_title(f"Mittel aus {len(seeds)} Trainingsdurchläufen, Balken zeigen die Spanne")
            save(fig, "fig_2x2.pdf", "Information mal Personalisierung, gemittelt ueber Seeds")
    else:
        print("  (noch keine Kohortendaten)")
except Exception:
    traceback.print_exc()

# --------------------------------------------------------------------------
# 7  Aufschluesselung nach Patientengruppe
# --------------------------------------------------------------------------
try:
    print("[9] Aufschluesselung nach Gruppe")
    seedfiles = sorted(glob.glob(os.path.join(ROOT, "runs", "cohort_seed*", "cohort_summary.json")))
    bl = os.path.join(ROOT, "runs", "baselines", "summary_heldout_heldout.json")
    if seedfiles:
        _per = [_load_json(p_)["arms"] for p_ in seedfiles]
        d = {}
        for k in _per[0]:
            d[k] = {"by_group": {g: {"tir": float(np.mean([a[k]["by_group"][g]["tir"] for a in _per
                                                          if g in a[k]["by_group"]]))}
                                for g in ("adolescent", "adult", "child")}}
        groups = ["adolescent", "adult", "child"]
        names = ["Jugendliche", "Erwachsene", "Kinder"]
        series = []
        if os.path.exists(bl):
            b = _load_json(bl)["controllers"]["bb"]["by_group"]
            series.append(("Basal-Bolus (klinisch)", [b[g]["tir"] for g in groups], C["clin"], "//"))
        for key, lab, col, hatch in (
            ("personalisiert_privileged", "privilegiert, personalisiert", C["priv"], None),
            ("generalist_privileged", "privilegiert, generalistisch", C["priv"], ".."),
            ("personalisiert_cgm", "nur CGM, personalisiert", C["cgm"], None),
            ("generalist_cgm", "nur CGM, generalistisch", C["cgm"], ".."),
        ):
            bg = d.get(key, {}).get("by_group")
            if bg:
                series.append((lab, [bg[g]["tir"] for g in groups], col, hatch))
        if series:
            x = np.arange(len(groups)); w = 0.8 / len(series)
            fig, ax = plt.subplots(figsize=(7.6, 3.9))
            for i, (lab, vals, col, hatch) in enumerate(series):
                ax.bar(x + (i - (len(series)-1)/2) * w, vals, w, label=lab,
                       color=col, hatch=hatch, edgecolor="white", linewidth=0.4)
            ax.axhline(70, color="#444444", ls=":", lw=1.1)
            ax.set_xticks(x); ax.set_xticklabels(names)
            ax.set_ylabel("Held-out Time in Range (%)"); ax.set_ylim(0, 105)
            ax.legend(fontsize=7.5, ncol=2, loc="lower left")
            ax.set_title("Kinder sind die schwierigste Gruppe, auch für die klinische Referenz")
            save(fig, "fig_gruppen.pdf", "Time in Range je Patientengruppe")
except Exception:
    traceback.print_exc()

print(f"\n=== {len(made)} Abbildungen in {FIG} ===")
for n, c in made:
    print(f"  {n}")

#!/usr/bin/env python3
"""Builds the README images in docs/img/ from the logged run files (English labels).

Same rule as src/make_figures.py: every figure comes from logged data or a seeded
re-simulation, never from a screenshot.

Run from the project root:
    PYTHONPATH=src python tools/readme_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
OUT = os.path.join(ROOT, "docs", "img")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.8,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
C = {"priv": "#1f4e79", "cgm": "#c0504d", "clin": "#2e7d32", "none": "#8a8a8a", "band": "#2e7d32"}


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {name}  {os.path.getsize(path) / 1024:.0f} KB")


def observability_gap():
    """One uncontrolled day: plasma glucose, subcutaneous glucose and the CGM reading."""
    from t1d_env import T1DEnv

    env = T1DEnv(patient_name="adult#001", obs_mode="privileged", days=1)
    env.reset(seed=100007)
    plasma, subq, cgm, meals = [], [], [], []
    done = False
    while not done:
        _, _, term, trunc, info = env.step_insulin(env.basal_ss)
        plasma.append(float(env._core.patient.state[3]) / env.Vg)
        subq.append(info["bg"])
        cgm.append(info["cgm"])
        meals.append(info["meal"])
        done = term or trunc
    t = np.arange(len(plasma)) * 3 / 60.0

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhspan(70, 180, color=C["band"], alpha=0.10, label="Target range 70-180 mg/dL")
    ax.plot(t, plasma, color=C["priv"], label="Plasma glucose (simulator only)")
    ax.plot(t, subq, color="#7a7a7a", lw=1.2, ls="--", label="Subcutaneous glucose")
    ax.plot(t, cgm, color=C["cgm"], lw=1.1, alpha=0.85, label="CGM reading (what a device sees)")
    for m in t[np.array(meals) > 0]:
        ax.axvline(m, color="#333333", lw=0.6, alpha=0.35)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_xlim(0, t[-1])
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.text(0.99, 0.03, "constant basal rate, no controller", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#666666")
    save(fig, "observability_gap.png")


def result_2x2():
    """Held-out TIR: what the controller sees x one controller vs. one per patient."""
    arms = [load(f"runs/cohort_seed{s}/cohort_summary.json")["arms"] for s in (0, 1)]
    keys = {("generalist", "privileged"): (71.0, 73.3), ("generalist", "cgm"): (42.4, 37.4),
            ("personalisiert", "privileged"): (74.8, 72.4), ("personalisiert", "cgm"): (36.7, 38.3)}
    runs = {}
    for (who, obs), expected in keys.items():
        vals = [a[f"{who}_{obs}"]["overall"]["tir"] for a in arms]
        # the README quotes these numbers: fail loudly if the files say something else
        assert all(abs(v - e) < 0.06 for v, e in zip(vals, expected)), (who, obs, vals)
        runs[(who, obs)] = vals

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    groups = ["generalist", "personalisiert"]
    labels = ["One controller\n(20 training patients)", "One controller\nper patient"]
    x = np.arange(len(groups))
    w = 0.34
    for off, obs, col, name in ((-w / 2, "privileged", C["priv"], "Full simulator state (privileged)"),
                                (w / 2, "cgm", C["cgm"], "CGM only")):
        means = [np.mean(runs[(g, obs)]) for g in groups]
        ax.bar(x + off, means, w, color=col, label=name)
        for xi, g, m in zip(x + off, groups, means):
            ax.scatter([xi] * 2, runs[(g, obs)], color="black", s=12, zorder=3)
            ax.text(xi, m + 3, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    for xi, g in zip(x, groups):
        gap = np.mean(runs[(g, "privileged")]) - np.mean(runs[(g, "cgm")])
        ax.text(xi + w + 0.04, np.mean(runs[(g, "cgm")]) + 4, f"gap {gap:.1f}",
                fontsize=9, color="#333333", va="bottom")
    ax.axhline(70, color="#333333", lw=0.8, ls=":")
    ax.text(1.72, 71.5, "Consensus target 70 %", fontsize=8, color="#333333", ha="right")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, 1.75)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Held-out Time in Range (%)")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "result_2x2.png")


def morris():
    """Morris screening of 10 patient parameters, clinical controller, adult#001."""
    d = load("runs/sensitivity/morris_bb_adult001.json")
    english = {
        "kp1": "endogenous glucose production", "kp3": "insulin effect on glucose production",
        "Km0": "Michaelis constant of glucose uptake", "Vmx": "insulin-dependent glucose uptake",
        "kmax": "max. gastric emptying rate", "kd": "subcutaneous insulin dissociation",
        "ka1": "subcutaneous insulin absorption", "kabs": "gut absorption rate",
        "ksc": "plasma-to-subcutaneous glucose diffusion", "u2ss": "steady-state insulin need",
    }
    res = sorted(d["results"], key=lambda r: r["mu_star"])
    names = [f"{r['name']}  {english.get(r['name'], '')}" for r in res]
    mu = [r["mu_star"] for r in res]
    sd = [r["sigma"] for r in res]
    cols = [C["priv"] if r["name"] == "kp1" else "#9a9a9a" for r in res]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y = np.arange(len(res))
    ax.barh(y, mu, color=cols, xerr=sd, error_kw={"ecolor": "#bbbbbb", "lw": 0.8})
    ax.set_yticks(y, names, fontsize=8)
    for yi, r in zip(y, res):
        if r["name"] == "ksc":
            ax.text(r["mu_star"] + 1, yi, f"sensor lag: {r['mu_star']:.2f}", va="center", fontsize=8)
    ax.set_xlabel("mu* (mean absolute effect on Time in Range, points)")
    ax.set_title("Morris screening, clinical controller, adult#001, ±20 %")
    ax.grid(axis="y", alpha=0)
    save(fig, "morris.png")


if __name__ == "__main__":
    failed = []
    for f in (observability_gap, result_2x2, morris):
        try:
            f()
        except Exception as exc:  # report every figure, then fail
            failed.append(f.__name__)
            print(f"  FAILED {f.__name__}: {exc!r}")
    sys.exit(1 if failed else 0)

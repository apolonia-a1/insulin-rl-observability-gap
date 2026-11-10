"""Time in Range metrics (Battelino 2019) and risk index (Kovatchev 1997).

Klinische Bewertungsmetriken fuer die Blutzuckerregelung.

Alle Schwellen folgen dem internationalen Konsens zu Time in Range
(Battelino et al. 2019, Diabetes Care 42(8):1593-1603). Die Zielwerte fuer
Typ-1-Diabetes stehen als TARGETS zur Verfuegung, damit Auswertung und Bericht
dieselben Zahlen verwenden.

Der Risikoindex (LBGI/HBGI/RI) stammt aus der symmetrisierten Blutzuckerskala
nach Kovatchev et al. 1997 und wird von simglucose selbst verwendet.
"""

from __future__ import annotations

import numpy as np

# --- Konsens-Schwellen in mg/dL (Battelino et al. 2019, Tabelle 2) ---
TIR_LOW = 70.0        # untere Grenze des Zielbereichs
TIR_HIGH = 180.0      # obere Grenze des Zielbereichs
TBR2_HIGH = 54.0      # unterhalb: schwere Hypoglykaemie (Level 2)
TAR2_LOW = 250.0      # oberhalb: schwere Hyperglykaemie (Level 2)

# --- Zielwerte fuer Typ-1-Diabetes in Prozent der Zeit (Battelino et al. 2019, Tabelle 3) ---
TARGETS = {
    "tir": (">", 70.0),        # > 70 % im Bereich 70-180
    "tbr_total": ("<", 4.0),   # < 4 % unter 70
    "tbr_level2": ("<", 1.0),  # < 1 % unter 54
    "tar_total": ("<", 25.0),  # < 25 % ueber 180
    "tar_level2": ("<", 5.0),  # < 5 % ueber 250
    "cv": ("<", 36.0),         # Variationskoeffizient < 36 %
}


def risk(bg: float) -> tuple[float, float, float]:
    """Risikoindex eines einzelnen Blutzuckerwerts: (LBGI, HBGI, RI).

    Identisch zu simglucose.analysis.risk.risk, hier dupliziert, damit die
    Auswertung unabhaengig von der Simulationsbibliothek nutzbar bleibt.
    """
    if bg <= 20.0:
        return 100.0, 0.0, 100.0
    if bg >= 600.0:
        return 0.0, 100.0, 100.0
    u = 1.509 * (np.log(bg) ** 1.084 - 5.381)
    ri = 10.0 * u * u
    return (ri if u <= 0 else 0.0), (ri if u >= 0 else 0.0), ri


def risk_index(bg: np.ndarray) -> tuple[float, float, float]:
    """Mittlere (LBGI, HBGI, RI) ueber eine Blutzucker-Zeitreihe."""
    bg = np.asarray(bg, dtype=float)
    if bg.size == 0:
        return 0.0, 0.0, 0.0
    r = np.array([risk(v) for v in bg])
    return float(r[:, 0].mean()), float(r[:, 1].mean()), float(r[:, 2].mean())


def glycemic_metrics(bg: np.ndarray) -> dict[str, float]:
    """Vollstaendiger Metriksatz einer Blutzucker-Zeitreihe (mg/dL).

    Die Prozentwerte beziehen sich auf den Anteil der Messpunkte. Bei
    aequidistanter Abtastung entspricht das dem Zeitanteil.
    """
    bg = np.asarray(bg, dtype=float)
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        return {k: float("nan") for k in (
            "mean_bg", "std_bg", "cv", "tir", "tbr_total", "tbr_level1",
            "tbr_level2", "tar_total", "tar_level1", "tar_level2",
            "lbgi", "hbgi", "risk_index", "min_bg", "max_bg", "n")}

    n = bg.size
    pct = lambda mask: 100.0 * float(np.count_nonzero(mask)) / n
    mean_bg = float(bg.mean())
    std_bg = float(bg.std(ddof=0))
    lbgi, hbgi, ri = risk_index(bg)

    return {
        "n": float(n),
        "mean_bg": mean_bg,
        "std_bg": std_bg,
        "cv": 100.0 * std_bg / mean_bg if mean_bg > 0 else float("nan"),
        "tir": pct((bg >= TIR_LOW) & (bg <= TIR_HIGH)),
        "tbr_total": pct(bg < TIR_LOW),
        "tbr_level1": pct((bg >= TBR2_HIGH) & (bg < TIR_LOW)),
        "tbr_level2": pct(bg < TBR2_HIGH),
        "tar_total": pct(bg > TIR_HIGH),
        "tar_level1": pct((bg > TIR_HIGH) & (bg <= TAR2_LOW)),
        "tar_level2": pct(bg > TAR2_LOW),
        "lbgi": lbgi,
        "hbgi": hbgi,
        "risk_index": ri,
        "min_bg": float(bg.min()),
        "max_bg": float(bg.max()),
    }


def meets_targets(m: dict[str, float]) -> dict[str, bool]:
    """Prueft die Konsens-Zielwerte punktweise. Basis der Zielerreichungstabelle."""
    out = {}
    for key, (op, value) in TARGETS.items():
        got = m.get(key, float("nan"))
        out[key] = bool(got > value) if op == ">" else bool(got < value)
    return out


def summarize(m: dict[str, float]) -> str:
    """Einzeilige Zusammenfassung fuer Logs und Konsolenausgaben."""
    return (f"TIR {m['tir']:.1f}% | TBR {m['tbr_total']:.1f}% (L2 {m['tbr_level2']:.2f}%) | "
            f"TAR {m['tar_total']:.1f}% (L2 {m['tar_level2']:.1f}%) | "
            f"CV {m['cv']:.1f}% | mittl. BG {m['mean_bg']:.0f} mg/dL | RI {m['risk_index']:.2f}")

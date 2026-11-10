"""Gymnasium environment for insulin control on the UVA/Padova model (simglucose ODE core).

Gymnasium-Umgebung fuer die Insulinregelung auf dem UVA/Padova-Modell.

Die Umgebung setzt direkt auf dem ODE-Kern von simglucose auf, nicht auf dessen
mitgelieferter Gym-Umgebung. Zwei Gruende:

1. `simglucose.envs` importiert das seit 2017 nicht mehr gepflegte `gym`
   (Pin `gym==0.9.4`), das unter Python 3.10 nicht mehr baut.
2. Die mitgelieferte Beobachtung ist ein einzelner CGM-Skalar ohne Verlauf und
   der Aktionsraum ist `Box(0, max_basal)` mit `max_basal = 30 U/min`, also rund
   das 1400-fache der physiologischen Basalrate. Beides ist fuer RL unbrauchbar.

Kernpunkt der Arbeit ist der Vergleich zweier Beobachtungsraeume auf identischer
Dynamik:

* `privileged` - der Agent liest den vollstaendigen 13-dimensionalen ODE-Zustand,
  den echten Plasmaglukosewert und angekuendigte Mahlzeiten. Nur im Simulator
  verfuegbar.
* `cgm` - der Agent liest ausschliesslich den verrauschten, verzoegerten
  CGM-Verlauf, seine eigene Insulinhistorie und die Tageszeit. Das entspricht
  dem, was ein reales Closed-Loop-System sieht.
"""

from __future__ import annotations

import sys
import types
import warnings
from datetime import datetime

import numpy as np

# simglucose nutzt pkg_resources, um seine Parameter-CSVs zu finden. Das ist in
# aktuellen setuptools-Versionen abgekuendigt und erzeugt pro Import eine
# Warnung. Der Aufruf selbst funktioniert (setuptools ist auf <81 gepinnt).
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# ---------------------------------------------------------------------------
# simglucose/__init__.py registriert eine veraltete Gym-Umgebung und importiert
# dafuer `gym`. Diese Umgebung wird hier nie verwendet. Der Stub erfuellt den
# Import, ohne das abgekuendigte Paket zu installieren oder site-packages zu
# veraendern.
# ---------------------------------------------------------------------------
if "gym" not in sys.modules:
    _reg = types.ModuleType("gym.envs.registration")
    _reg.register = lambda *a, **k: None
    _envs = types.ModuleType("gym.envs")
    _envs.registration = _reg
    _gym = types.ModuleType("gym")
    _gym.envs = _envs
    # Stable-Baselines3 liest beim Speichern eines Modells `gym.__version__` fuer
    # seinen System-Report. Der Wert ist bewusst als Stub gekennzeichnet, damit in
    # system_info.txt kein echtes gym vorgetaeuscht wird.
    _gym.__version__ = "0-stub-simglucose-shim"
    sys.modules.update({"gym": _gym, "gym.envs": _envs, "gym.envs.registration": _reg})

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from simglucose.actuator.pump import InsulinPump  # noqa: E402
from simglucose.controller.base import Action  # noqa: E402
from simglucose.patient.t1dpatient import T1DPatient  # noqa: E402
from simglucose.sensor.cgm import CGMSensor  # noqa: E402
from simglucose.simulation.env import T1DSimEnv as _SimCore  # noqa: E402
from simglucose.simulation.scenario_gen import RandomScenario  # noqa: E402

from metrics import TBR2_HIGH, TIR_LOW, glycemic_metrics, risk  # noqa: E402

# --- Versionierung der Belohnungsfunktion; jeder Lauf schreibt sie in config.json ---
#
# v1: r = -W_RISK * RI(BG). Nur der Risikoindex nach Kovatchev.
#     VERWORFEN. Der Index bewertet milde Unterzuckerung GUENSTIGER als maessige
#     Ueberzuckerung: BG 60 kostet 1,36, BG 250 kostet 2,24. Ein Agent, der diesen
#     Reward maximiert, verschiebt den Blutzucker folgerichtig nach unten. Gemessen
#     im Lauf des privilegierten Agenten (runs/_reward_v1, eval.csv): TIR steigt
#     bis 92,0 % bei 150000 Schritten und faellt danach auf 83,2 %, waehrend der
#     Anteil unterhalb des Zielbereichs auf 12,7 % steigt. Klinisch ist das die falsche Rangfolge, denn
#     Unterzuckerung wirkt akut (Krampfanfall, Bewusstlosigkeit), Ueberzuckerung
#     chronisch. Der Konsens toleriert 25 % oberhalb, aber nur 4 % unterhalb des
#     Zielbereichs.
#
# v2: zusaetzlich eine explizite, asymmetrische Strafe unterhalb des Zielbereichs,
#     an den Konsensgrenzen ausgerichtet.
REWARD_VERSION = "v2"

# Belohnungsgewichte
W_RISK = 0.1           # Skalierung des Risikoindex (RI liegt typisch bei 0 bis 20)
MAX_RISK = 100.0       # Obergrenze des Risikoindex nach Kovatchev
W_HYPO = 10.0          # lineare Rampe unterhalb 70 mg/dL (Untergrenze des Zielbereichs)
W_SEVERE = 2.0         # zusaetzliche Pauschale unterhalb 54 mg/dL (Hypoglykaemie Level 2)

# Abbruchstrafe: Der Simulator beendet die Episode, wenn der Blutzucker den
# ueberlebbaren Bereich verlaesst. Eine FLACHE Strafe waere hier ein
# Konstruktionsfehler. Rechnet man die Rueckgaben durch, kostet anhaltende
# Hyperglykaemie ueber 480 Schritte rund -619, ein Abbruch nach 100 Schritten
# mit flacher Strafe -100 dagegen nur rund -300. Der Agent haette also einen
# Anreiz, die Episode absichtlich zu beenden.
#
# Stattdessen wird der Abbruch mit dem MAXIMALEN Risiko fuer die verbleibende
# Episodendauer belastet. Sterben ist damit exakt so teuer wie die schlechteste
# denkbare Regelung bis zum Episodenende, nie billiger. Der Anreiz verschwindet,
# ohne dass eine willkuerliche Konstante gewaehlt werden muss.

# --- Aktionsraum ---
# Die Aktion ist ein Exponent auf die patienteneigene Basalrate:
#     u = basal_ss * MAX_MULT ** a,   a in [-1, 1]
# a = 0 liefert exakt die stationaere Basalrate des Patienten, a = 1 das
# MAX_MULT-fache, a = -1 praktisch null. Die logarithmische Skala ist eine
# direkte Antwort auf die gemessene Schiefe der klinischen Referenztherapie:
# der Basal-Bolus-Regler gibt in 99 % der Schritte exakt die Basalrate und im
# Maximum das 124- bis 159-fache davon. Ein linearer Aktionsraum verschwendet
# seine Aufloesung genau dort, wo fast alle Entscheidungen liegen.
MAX_MULT = 160.0

# --- Beobachtung ---
HIST_LEN = 12          # 12 Schritte a 3 min = 36 min Verlauf
ANNOUNCE_WINDOW = 30   # Vorlaufzeit der Mahlzeitenankuendigung in Minuten

SENSOR = "Dexcom"      # Abtastzeit 3 min
PUMP = "Insulet"

# Seed-Bereiche: Trainings- und Held-out-Szenarien sind disjunkt.
HELDOUT_SEED_BASE = 100_000

# --- Kohorte ---
# simglucose liefert 30 virtuelle Patienten: je zehn Jugendliche, Erwachsene und
# Kinder. Die Aufteilung trennt Patienten, nicht nur Szenarien: die zehn
# Held-out-Patienten werden vom generalistischen Regler NIE trainiert. Die
# Aufteilung ist ueber alle drei Gruppen geschichtet, damit die Held-out-Menge
# nicht versehentlich nur leicht regelbare Physiologien enthaelt.
ALL_PATIENTS = [f"{g}#{i:03d}" for g in ("adolescent", "adult", "child")
                for i in range(1, 11)]
HELDOUT_PATIENTS = [
    "adolescent#008", "adolescent#009", "adolescent#010",
    "adult#008", "adult#009", "adult#010",
    "child#007", "child#008", "child#009", "child#010",
]
TRAIN_PATIENTS = [p for p in ALL_PATIENTS if p not in HELDOUT_PATIENTS]


def resolve_patients(spec: str) -> list[str]:
    """Loest eine Patientenangabe auf: Name, Komma-Liste oder Kohortenschluessel."""
    key = spec.strip().lower()
    if key in ("train", "trainsplit"):
        return list(TRAIN_PATIENTS)
    if key in ("heldout", "holdout"):
        return list(HELDOUT_PATIENTS)
    if key == "all":
        return list(ALL_PATIENTS)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in ALL_PATIENTS]
    if unknown:
        raise ValueError(f"unbekannte Patienten: {unknown}")
    return names


class _FastParams:
    """Patientenparameter mit direktem Attributzugriff statt pandas-Lookup.

    Die rechte Seite des ODE-Systems (`T1DPatient.model`) liest rund 30
    Parameter als Attribute einer pandas-Series. Jeder dieser Zugriffe laeuft
    ueber `Series.__getattr__` und eine Indexsuche. Da der Loeser `model` je
    Umgebungsschritt etwa 23-mal auswertet, entfielen im Profiling rund 70 %
    der Laufzeit auf diese Lookups (336000 pandas-Zugriffe pro 300 Schritte).

    Diese Huelle kopiert alle Felder einmal in echte Attribute. Das Ergebnis ist
    numerisch identisch, denn es werden dieselben Werte gelesen; nur der Weg
    dorthin ist kuerzer. `.iloc` bleibt fuer `T1DPatient.reset` erhalten.
    """

    __slots__ = ("_series", "Name", "__dict__")

    def __init__(self, series):
        self._series = series
        for key, value in series.items():
            if isinstance(key, str) and key.isidentifier():
                try:
                    setattr(self, key, float(value))
                except (TypeError, ValueError):
                    setattr(self, key, value)
        self.Name = series.Name

    @property
    def iloc(self):
        return self._series.iloc


# Parametertabellen werden einmal je Prozess gelesen. Die `withName`-Methoden von
# simglucose lesen ihre CSV bei JEDEM Aufruf neu. Da eine Episode bei jedem Reset
# Patient, Sensor und Pumpe neu erzeugt, waren das drei pandas-Ladevorgaenge pro
# Episode, im Profiling 39 % der Reset-Zeit. Das faellt bei kurzen Episoden
# doppelt ins Gewicht: `SubprocVecEnv` ist synchron, ein resettender Worker haelt
# alle anderen an der Barriere auf.
_PARAM_CACHE: dict[tuple[str, str], object] = {}


def _patient_params(name: str) -> "_FastParams":
    key = ("patient", name)
    if key not in _PARAM_CACHE:
        _PARAM_CACHE[key] = _FastParams(T1DPatient.withName(name)._params)
    return _PARAM_CACHE[key]


def _sensor_params(name: str):
    key = ("sensor", name)
    if key not in _PARAM_CACHE:
        _PARAM_CACHE[key] = CGMSensor.withName(name)._params
    return _PARAM_CACHE[key]


def _pump_params(name: str):
    key = ("pump", name)
    if key not in _PARAM_CACHE:
        _PARAM_CACHE[key] = InsulinPump.withName(name)._params
    return _PARAM_CACHE[key]


def _params_with(name: str, overrides: dict[str, float] | None):
    """Parametersatz mit veraenderten Einzelwerten (fuer die Sensitivitaetsanalyse)."""
    base = _patient_params(name)
    if not overrides:
        return base
    import copy
    p = copy.copy(base)
    p.__dict__ = dict(base.__dict__)
    for key, value in overrides.items():
        if not hasattr(base, key):
            raise KeyError(f"unbekannter Patientenparameter: {key}")
        setattr(p, key, float(value))
    return p


def _make_patient(name: str, seed: int, random_init_bg: bool = True,
                  overrides: dict[str, float] | None = None) -> T1DPatient:
    """Patient aus zwischengespeicherten Parametern.

    Die geteilte `_FastParams`-Instanz ist unbedenklich: `T1DPatient` liest sie
    nur. `reset` kopiert den Startzustand ueber `np.copy`, veraendert also die
    Vorlage nicht.
    """
    return T1DPatient(_params_with(name, overrides), random_init_bg=random_init_bg, seed=seed)


def _state_scale(x0: np.ndarray) -> np.ndarray:
    """Normierungsvektor fuer den 13-dimensionalen ODE-Zustand.

    Die Zustaende unterscheiden sich um Groessenordnungen. Als Massstab dient
    der patienteneigene Startzustand, sodass jede Komponente im Gleichgewicht
    ungefaehr bei 1 liegt. Die drei Magen-Darm-Zustaende starten bei null und
    bekommen daher einen festen Massstab in der Groessenordnung einer Mahlzeit
    (50 g Kohlenhydrate entsprechen 50000 mg).
    """
    scale = np.abs(np.asarray(x0, dtype=float)).copy()
    scale[0:3] = 50_000.0
    scale[scale < 1e-6] = 1.0
    return scale


class T1DEnv(gym.Env):
    """Ein simulierter Tag Insulinregelung fuer einen virtuellen Patienten."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        patient_name: str = "adult#001",
        obs_mode: str = "cgm",
        days: int = 1,
        seed_base: int = 0,
        n_scenarios: int | None = None,
        announce_window: int = ANNOUNCE_WINDOW,
        hist_len: int = HIST_LEN,
        cgm_hist_len: int | None = None,
        param_overrides: dict | None = None,
    ):
        super().__init__()
        if obs_mode not in ("cgm", "privileged"):
            raise ValueError(f"obs_mode muss 'cgm' oder 'privileged' sein, war {obs_mode!r}")
        # `patient_name` darf eine Liste sein. Dann wird je Episode ein Patient
        # gezogen, und es entsteht EIN Regler fuer die Kohorte statt einer je
        # Patient. Moeglich ist das nur, weil der Aktionsraum auf die
        # patienteneigene Basalrate normiert ist (siehe MAX_MULT): dieselbe
        # Aktion bedeutet fuer Kind und Erwachsenen dieselbe relative Dosis.
        if isinstance(patient_name, str):
            self.patients = resolve_patients(patient_name)
        else:
            self.patients = list(patient_name)
        self.patient_name = self.patients[0]
        self.obs_mode = obs_mode
        self.days = days
        self.seed_base = seed_base
        self.n_scenarios = n_scenarios
        self.announce_window = announce_window
        self.hist_len = hist_len
        # Laenge des CGM-Fensters, getrennt von der Insulinhistorie. Ohne diese
        # Trennung wuerde ein laengeres Sensorfenster auch die Insulinhistorie
        # verlaengern und damit den privilegierten Beobachtungsraum veraendern.
        # Die Ablation zur Fenstergroesse waere dann kein Ein-Faktor-Versuch
        # mehr, weil sich die Vergleichsbedingung mitbewegt.
        self.cgm_hist_len = hist_len if cgm_hist_len is None else cgm_hist_len
        self.param_overrides = param_overrides

        # Patientenkonstanten. Bei mehreren Patienten werden sie je Episode neu
        # gesetzt; hier dienen sie zur Definition der Raeume.
        self._set_patient_constants(self.patient_name)
        self.max_basal = float(_pump_params(PUMP)["max_basal"])

        self.max_steps = int(days * 24 * 60 / 3)   # Sensorabtastung 3 min

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._obs_dim(),), dtype=np.float32
        )

        self._episode_counter = 0
        self._core = None

    def _set_patient_constants(self, name: str) -> None:
        """Setzt die patientenabhaengigen Groessen. Der Aktionsraum ist auf
        `basal_ss` normiert, deshalb muessen diese Werte je Episode zum gerade
        gezogenen Patienten passen."""
        pp = _params_with(name, self.param_overrides)
        self.basal_ss = float(pp.u2ss * pp.BW / 6000.0)
        self.Vg = float(pp.Vg)
        self._x0 = np.asarray(pp.iloc[2:15].values, dtype=float)
        self._scale = _state_scale(self._x0)

    # -- Beobachtungsraum ---------------------------------------------------

    def _obs_dim(self) -> int:
        if self.obs_mode == "privileged":
            # 13 ODE-Zustaende + Plasmaglukose + CGM + Mahlzeit jetzt
            # + angekuendigte Mahlzeit + Insulinhistorie + 2 Tageszeit
            return 13 + 1 + 1 + 1 + 1 + self.hist_len + 2
        # CGM-Verlauf + Insulinhistorie + 2 Tageszeit
        return self.cgm_hist_len + self.hist_len + 2

    def _build_obs(self) -> np.ndarray:
        tod = self._minutes_of_day() / 1440.0
        tod_feat = [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)]

        if self.obs_mode == "privileged":
            x = np.asarray(self._core.patient.state, dtype=float)
            plasma = x[3] / self.Vg
            obs = np.concatenate([
                x / self._scale,                     # 13 normierte ODE-Zustaende
                [(plasma - 140.0) / 100.0],          # echte Plasmaglukose
                [(self._last_cgm - 140.0) / 100.0],  # Sensorwert zum Vergleich
                [self._last_meal / 50.0],            # aktuell aufgenommene Kohlenhydrate
                [self._upcoming_meal() / 50.0],      # angekuendigte Mahlzeit
                self._ins_hist,
                tod_feat,
            ])
        else:
            obs = np.concatenate([
                (np.asarray(self._cgm_hist) - 140.0) / 100.0,
                self._ins_hist,
                tod_feat,
            ])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # -- Mahlzeiten ---------------------------------------------------------

    def _minutes_of_day(self) -> float:
        t = self._core.time
        return t.hour * 60.0 + t.minute + t.second / 60.0

    def _upcoming_meal(self) -> float:
        """Kohlenhydrate einer Mahlzeit, die innerhalb des Ankuendigungsfensters beginnt.

        Liest den Mahlzeitenplan des Szenarios direkt aus, statt
        `scenario.get_action` in die Zukunft aufzurufen. Dieser Aufruf hat einen
        Seiteneffekt (er erzeugt um Mitternacht einen neuen Tagesplan) und
        wuerde den Simulationszustand veraendern.
        """
        plan = getattr(self._core.scenario, "scenario", None)
        if not plan:
            return 0.0
        now = self._minutes_of_day()
        total = 0.0
        for t_meal, amount in zip(plan["meal"]["time"], plan["meal"]["amount"]):
            if now <= t_meal <= now + self.announce_window:
                total += float(amount)
        return total

    # -- Aktion -------------------------------------------------------------

    def _action_to_insulin(self, a: float) -> float:
        """Exponentieller Aktionsraum, siehe MAX_MULT."""
        u = self.basal_ss * (MAX_MULT ** float(np.clip(a, -1.0, 1.0)))
        return float(np.clip(u, 0.0, self.max_basal))

    # -- Belohnung ----------------------------------------------------------

    def _reward(self, bg: float, terminated: bool) -> tuple[float, dict]:
        """Belohnung v2: negativer Risikoindex, asymmetrische Hypoglykaemie-Strafe und Abbruchstrafe.

        Der Risikoindex nach Kovatchev ist auf der symmetrisierten
        Blutzuckerskala definiert und bestraft Unter- und Ueberzuckerung bereits
        asymmetrisch in klinisch begruendeter Weise. v1 nutzte nur den Risikoindex;
        v2 ergaenzt r_hypo (siehe REWARD_VERSION).
        """
        _, _, ri = risk(bg)
        r_risk = -W_RISK * ri

        # Asymmetrische Hypoglykaemie-Strafe (v2). Der Risikoindex allein stellt
        # die klinische Rangfolge falsch dar, siehe Kommentar bei REWARD_VERSION.
        r_hypo = 0.0
        if bg < TIR_LOW:
            r_hypo = -W_HYPO * (TIR_LOW - bg) / TIR_LOW
            if bg < TBR2_HIGH:
                r_hypo -= W_SEVERE

        r_term = 0.0
        if terminated:
            remaining = max(0, self.max_steps - self._t)
            r_term = -W_RISK * MAX_RISK * remaining
        return r_risk + r_hypo + r_term, {"r_risk": r_risk, "r_hypo": r_hypo, "r_term": r_term}

    # -- Gymnasium-API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Ein Episoden-Seed steuert Mahlzeitenplan, Sensorrauschen und
        # Startblutzucker gemeinsam, damit Episoden exakt reproduzierbar sind.
        if seed is not None:
            ep_seed = int(seed)
        else:
            idx = self._episode_counter
            if self.n_scenarios:
                idx = idx % self.n_scenarios
            ep_seed = self.seed_base + idx
        self._episode_counter += 1
        self._ep_seed = ep_seed

        rng = np.random.default_rng(ep_seed)
        start_hour = int(rng.integers(0, 24))
        start_time = datetime(2018, 1, 1, start_hour, 0, 0)

        # Patient je Episode aus der zugewiesenen Menge ziehen. Bei nur einem
        # Patienten ist das ein Nulloperation, sodass Einzelpatienten-Laeufe
        # bitgleich zu vorher bleiben.
        if len(self.patients) > 1:
            self.patient_name = self.patients[int(rng.integers(0, len(self.patients)))]
            self._set_patient_constants(self.patient_name)

        patient = _make_patient(self.patient_name, ep_seed, random_init_bg=True,
                                overrides=self.param_overrides)
        sensor = CGMSensor(_sensor_params(SENSOR), seed=ep_seed + 1)
        pump = InsulinPump(_pump_params(PUMP))
        scenario = RandomScenario(start_time=start_time, seed=ep_seed + 2)
        self._core = _SimCore(patient, sensor, pump, scenario)

        # `_SimCore.__init__` initialisiert bereits alle Verlaeufe und misst den
        # ersten CGM-Wert. Ein zusaetzliches `_core.reset()` waere redundant und
        # wuerde Patient, Sensor und Szenario ein zweites Mal aufbauen. Da das
        # Szenario dabei ueber `scipy.stats.truncnorm` neu gewuerfelt wird, war
        # das im Profiling der teuerste Einzelposten des Resets. Die
        # Reproduzierbarkeit bleibt erhalten: alle Komponenten wurden gerade mit
        # ihrem Seed frisch konstruiert.
        cgm0 = float(self._core.CGM_hist[0])
        self._last_cgm = cgm0
        self._last_meal = 0.0
        self._cgm_hist = [cgm0] * self.cgm_hist_len
        self._ins_hist = np.zeros(self.hist_len, dtype=float)  # Aktion 0 = Basalrate
        self._t = 0

        # Verlaufsspeicher fuer die Episodenauswertung
        self._bg_trace, self._cgm_trace, self._ins_trace, self._meal_trace = [], [], [], []
        self._rew_components = {"r_risk": 0.0, "r_hypo": 0.0, "r_term": 0.0}
        self._reward_total = 0.0

        return self._build_obs(), {"episode_seed": ep_seed}

    def step(self, action):
        a = float(np.asarray(action).ravel()[0])
        return self._advance(self._action_to_insulin(a), a)

    def step_insulin(self, u_per_min: float):
        """Wie `step`, aber mit direkt vorgegebener Insulinrate in U/min.

        Damit durchlaufen klinische Referenzregler (Basal-Bolus, PID) exakt
        dieselbe Episodenlogik, dieselben Seeds und dieselbe Bilanzierung wie
        der RL-Agent. Nur so ist der Vergleich der Endzahlen fair.
        """
        u = float(np.clip(u_per_min, 0.0, self.max_basal))
        # Aequivalente Aktion nur fuer die Historie, damit die Beobachtung konsistent bleibt.
        a = float(np.clip(np.log(max(u, 1e-12) / self.basal_ss) / np.log(MAX_MULT), -1.0, 1.0))
        return self._advance(u, a)

    def _advance(self, u: float, a: float):
        step = self._core.step(Action(basal=u, bolus=0.0))
        self._t += 1

        bg = float(step.info["bg"])          # subkutane Glukose, die Wahrheit fuer die Bewertung
        cgm = float(step.observation.CGM)    # Sensorwert, das was der cgm-Agent sieht
        meal = float(step.info["meal"])

        self._last_cgm, self._last_meal = cgm, meal
        self._cgm_hist = self._cgm_hist[1:] + [cgm]
        self._ins_hist = np.concatenate([self._ins_hist[1:], [np.clip(a, -1.0, 1.0)]])

        self._bg_trace.append(bg)
        self._cgm_trace.append(cgm)
        self._ins_trace.append(u)
        self._meal_trace.append(meal)

        terminated = bool(step.done)         # Simulator meldet nicht ueberlebbaren Zustand
        truncated = self._t >= self.max_steps

        reward, comps = self._reward(bg, terminated)
        for k, v in comps.items():
            self._rew_components[k] += v
        self._reward_total += reward

        info = {
            "bg": bg, "cgm": cgm, "insulin": u, "meal": meal,
            "action": a, "risk": float(step.info["risk"]),
        }
        if terminated or truncated:
            info["episode_stats"] = self._episode_stats(terminated)

        return self._build_obs(), reward, terminated, truncated, info

    def _episode_stats(self, terminated: bool) -> dict:
        """Vollstaendige Episodenbilanz. Eine Zeile pro Episode in episodes.csv."""
        bg = np.asarray(self._bg_trace, dtype=float)
        m = glycemic_metrics(bg)
        ins = np.asarray(self._ins_trace, dtype=float)
        return {
            "episode_seed": self._ep_seed,
            "patient": self.patient_name,
            "obs_mode": self.obs_mode,
            "outcome": "terminated" if terminated else "completed",
            "steps": int(self._t),
            "reward_total": float(self._reward_total),
            "r_risk": float(self._rew_components["r_risk"]),
            "r_hypo": float(self._rew_components["r_hypo"]),
            "r_term": float(self._rew_components["r_term"]),
            "tir": m["tir"], "tbr_total": m["tbr_total"], "tbr_level1": m["tbr_level1"],
            "tbr_level2": m["tbr_level2"], "tar_total": m["tar_total"],
            "tar_level1": m["tar_level1"], "tar_level2": m["tar_level2"],
            "mean_bg": m["mean_bg"], "std_bg": m["std_bg"], "cv": m["cv"],
            "min_bg": m["min_bg"], "max_bg": m["max_bg"],
            "lbgi": m["lbgi"], "hbgi": m["hbgi"], "risk_index": m["risk_index"],
            "total_insulin_U": float(ins.sum() * 3.0),   # U/min ueber 3-min-Schritte
            "total_cho_g": float(np.sum(self._meal_trace) * 3.0),
        }

    def close(self):
        self._core = None


def make_env(patient_name="adult#001", obs_mode="cgm", days=1, seed_base=0,
             n_scenarios=None, **kwargs):
    """Factory fuer SubprocVecEnv (muss auf Modulebene liegen, damit sie picklebar ist)."""
    def _init():
        return T1DEnv(patient_name=patient_name, obs_mode=obs_mode, days=days,
                      seed_base=seed_base, n_scenarios=n_scenarios, **kwargs)
    return _init


def train_seed_blocks(seed_base: int, n_envs: int) -> int:
    """Blockgroesse je Worker, so dass alle Trainings-Seeds in [0, HELDOUT_SEED_BASE) liegen."""
    block = (HELDOUT_SEED_BASE - seed_base) // max(1, n_envs)
    assert seed_base + n_envs * block <= HELDOUT_SEED_BASE, (
        "Trainings-Seeds duerfen den Held-out-Bereich nicht erreichen")
    return block


def make_vec_env(patient_name="adult#001", obs_mode="cgm", n_envs=8, days=1,
                 seed_base=0, n_scenarios=None, **kwargs):
    """Vektorisierte Umgebung. Jeder Worker bekommt einen disjunkten Seed-Block.

    Die Blockgroesse darf den Held-out-Bereich nicht erreichen. Mit einem festen
    Block von 10000 und 24 Arbeitsprozessen reichten die Trainings-Seeds bis
    239999 und ueberlappten damit den Auswertungsbereich ab 100000: die
    personalisierten Regler trainierten auf 19 der 20 Episoden, auf denen sie
    anschliessend bewertet wurden. Die Trainingsbloecke fuellen deshalb genau
    das Intervall [0, HELDOUT_SEED_BASE), und eine Zusicherung haelt das fest.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if seed_base < HELDOUT_SEED_BASE:                 # Trainingsumgebung
        block = train_seed_blocks(seed_base, n_envs)
    else:                                             # Auswertungsumgebung
        block = 10_000

    fns = [make_env(patient_name, obs_mode, days, seed_base + i * block, n_scenarios, **kwargs)
           for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)

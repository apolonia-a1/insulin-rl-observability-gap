"""Time in Range metrics on a hand-made trace (boundaries 54 / 70 / 180 / 250 mg/dL)."""
import math

import numpy as np

from metrics import glycemic_metrics


def test_range_boundaries():
    # 10 points: 2 below 54, 1 in [54, 70), 4 in [70, 180], 2 in (180, 250], 1 above 250
    bg = np.array([40, 53, 60, 70, 100, 150, 180, 200, 250, 300], dtype=float)
    m = glycemic_metrics(bg)
    assert m["n"] == 10
    assert m["tir"] == 40.0          # 70, 100, 150, 180 (both limits inclusive)
    assert m["tbr_level2"] == 20.0   # 40, 53
    assert m["tbr_level1"] == 10.0   # 60
    assert m["tbr_total"] == 30.0
    assert m["tar_level1"] == 20.0   # 200, 250
    assert m["tar_level2"] == 10.0   # 300
    assert m["tar_total"] == 30.0
    assert math.isclose(m["tir"] + m["tbr_total"] + m["tar_total"], 100.0)


def test_nan_values_are_ignored():
    m = glycemic_metrics(np.array([100.0, np.nan, 120.0]))
    assert m["n"] == 2
    assert m["tir"] == 100.0


def test_empty_trace_gives_nan():
    m = glycemic_metrics(np.array([]))
    assert math.isnan(m["tir"])

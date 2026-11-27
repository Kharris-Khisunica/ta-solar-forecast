import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clearsky import clip, ineichen_hourly, kc  # noqa: E402


def test_kc_basic_and_night_is_nan():
    out = kc([500, 0, 100], [1000, 0, 200], [30, 95, 60])
    assert out[0] == 0.5
    assert np.isnan(out[1])          # malam -> NaN, bukan 0
    assert out[2] == 0.5


def test_kc_zero_clearsky_and_high_zenith_are_nan():
    out = kc([10, 10], [0, 100], [40, 86])
    assert np.isnan(out).all()


def test_kc_propagates_missing_ghi():
    assert np.isnan(kc([np.nan], [800], [20])[0])


def test_clip_bounds():
    out = clip([-50, 500, 1500, 30], [1000, 1000, 1000, 0], k_max=1.1)
    assert list(out) == [0.0, 500.0, 1100.0, 0.0]   # malam (GHI_cs=0) -> 0


def test_ineichen_hourly_uses_same_aggregation_rule():
    idx = pd.date_range("2020-03-20 00:00", "2020-03-20 11:50", freq="10min", tz="UTC")
    h = ineichen_hourly(idx, -6.8899, 107.61, 770, delta_min=1)
    assert h.index[0] == pd.Timestamp("2020-03-20 01:00", tz="UTC")   # period-ending
    assert len(h) == 12
    assert h["ineichen_ghi"].max() > 800                             # siang tropis ~equinox
    assert h["ineichen_ghi"].idxmax().hour in (5, 6)                 # puncak ~05 UTC (12 WIB)


def test_kmax_for_zenith_bins_and_clip_with_array():
    from clearsky import kmax_for_zenith
    edges, vals = [0, 40, 75, 85], [1.05, 1.15, 1.70]
    k = kmax_for_zenith([10, 40, 41, 75, 80, 90], edges, vals)
    assert list(k[:5]) == [1.05, 1.05, 1.15, 1.15, 1.70]   # batas kanan inklusif
    assert np.isnan(k[5])                                  # di luar rentang
    out = clip([1200, 1200], [1000, 1000], np.array([1.05, 1.15]))
    assert list(out) == [1050.0, 1150.0]

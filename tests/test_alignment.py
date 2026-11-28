"""T09.5 — uji penyelarasan waktu sebagai pytest, termasuk tes NEGATIF: data yang sengaja digeser
1 jam harus membuat uji gagal. Dipakai untuk NSRDB vs clear-sky (T09.3) dan GFS vs observasi (T18.6)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import clearsky as C  # noqa: E402
import gfs_process as G  # noqa: E402


@pytest.fixture(scope="module")
def series():
    """GHI clear-sky Ineichen per jam (lokasi ITB, period-ending) + observasi sintetis berawan."""
    raw = pd.date_range("2020-01-01 00:01", "2020-03-31 23:51", freq="10min", tz="UTC")
    cs = C.ineichen_hourly(raw, -6.8899, 107.6100, 770.0)["ineichen_ghi"]
    rng = np.random.default_rng(0)
    day_kc = pd.Series(rng.uniform(0.3, 1.0, cs.index.normalize().nunique()), index=cs.index.normalize().unique())
    obs = cs * day_kc.reindex(cs.index.normalize()).to_numpy() * rng.uniform(0.85, 1.05, len(cs))
    mask = cs > 50
    return cs, obs, mask


def test_aligned_series_pass(series):
    cs, obs, mask = series
    assert G.assert_aligned(G.alignment_by_shift(cs, obs, mask)) == 0


@pytest.mark.parametrize("shift_h", [1, -1])
def test_shifted_series_fail(series, shift_h):
    cs, obs, mask = series
    shifted = obs.shift(shift_h)                                          # label salah 1 jam
    with pytest.raises(AssertionError, match="bukan 0"):
        G.assert_aligned(G.alignment_by_shift(cs, shifted, mask))

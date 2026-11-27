"""Tes kebocoran fitur (T12.6) — memakai data per jam sintetis, tidak butuh data asli."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402

CFG = {
    "data": {"years": [2016, 2017], "split": {"train": [2016], "val": [2017], "test": []}},
    "physics": {"zenith_max_deg": 85},
    "forecast": {"issue_time_utc": "05:00"},
}


@pytest.fixture(scope="module")
def hourly():
    idx = pd.date_range("2016-01-01 01:00", "2018-01-01 00:00", freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    wib_h = (idx.hour + 7) % 24
    zen = np.where((wib_h >= 6) & (wib_h <= 18), 30 + 8 * np.abs(wib_h - 12), 120).astype(float)
    cs = np.where(zen < 90, 1000 * np.cos(np.deg2rad(zen)), 0.0)
    kc = np.where(zen < 85, rng.uniform(0.2, 1.0, len(idx)), np.nan)
    return pd.DataFrame({"solar_zenith_angle": zen, "ghi_cs": cs, "kc": kc, "ghi": np.nan_to_num(kc) * cs,
                         "interp_frac": 0.0, "air_temperature": 25.0}, index=idx)


@pytest.fixture(scope="module")
def table(hourly):
    return F.build_feature_table(hourly, CFG)


def test_24_rows_per_day_and_time_mapping(table):
    assert (table.groupby("target_day").size() == 24).all()
    r = table[(table["target_day"] == "2016-06-10") & (table["target_hour_wib"] == 12)].iloc[0]
    assert r["target_time_utc"] == pd.Timestamp("2016-06-10 05:00", tz="UTC")   # 12.00 WIB = 05 UTC
    assert r["issue_time_utc"] == pd.Timestamp("2016-06-09 05:00", tz="UTC")    # 12.00 WIB D-1


def test_no_leakage(table):
    F.assert_no_leakage(table)


def test_same_hour_lag_rule(table, hourly):
    day = table[table["target_day"] == "2016-06-10"].set_index("target_hour_wib")
    # Jam 10 WIB: D-1 jam 10 sudah teramati (≤ 12.00 WIB D-1) → dari D-1
    assert day.loc[10, "kc_same_hour_lag"] == hourly["kc"].loc["2016-06-09 03:00+00:00"]
    assert day.loc[10, "same_hour_from_d2"] == 0
    # Jam 15 WIB: D-1 jam 15 belum teramati → dari D-2
    assert day.loc[15, "kc_same_hour_lag"] == hourly["kc"].loc["2016-06-08 08:00+00:00"]
    assert day.loc[15, "same_hour_from_d2"] == 1


def test_negative_leaky_feature_is_caught(table):
    """Tes negatif: fitur 'jam yang sama D-1' tanpa aturan #8 HARUS terdeteksi bocor."""
    leaky = table.copy()
    leaky["ftm__kc_same_hour_naive"] = leaky["target_time_utc"] - pd.Timedelta(days=1)
    with pytest.raises(AssertionError, match="KEBOCORAN"):
        F.assert_no_leakage(leaky)


def test_negative_target_hour_value_is_caught(table):
    leaky = table.copy()
    leaky["ftm__ghi_target_hour"] = leaky["target_time_utc"]
    rep = F.leakage_report(leaky).set_index("fitur")
    assert rep.loc["ghi_target_hour", "baris_bocor"] == len(leaky)


def test_climatology_uses_train_years_only(hourly):
    h2 = hourly.copy()
    h2.loc[h2.index.year == 2017, "kc"] = 5.0          # ubah tahun validasi secara ekstrem
    t1 = F.build_feature_table(hourly, CFG)
    t2 = F.build_feature_table(h2, CFG)
    assert np.allclose(t1["kc_clim"], t2["kc_clim"], equal_nan=True)


def test_split_by_target_day(table):
    assert set(table.loc[table["target_day"].dt.year == 2016, "split"]) == {"train"}
    assert set(table.loc[table["target_day"].dt.year == 2017, "split"]) == {"val"}

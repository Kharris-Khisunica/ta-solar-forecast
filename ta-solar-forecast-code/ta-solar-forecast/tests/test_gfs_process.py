"""Tes pemrosesan GFS (T18.3, T18.5, T18.6) dengan data & GRIB sintetis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402
import gfs_process as G  # noqa: E402

INIT = pd.Timestamp("2020-01-01 00:00", tz="UTC")
LEADS = [18, 21, 24, 27, 30, 33, 36, 39]
TRUE_BLOCKS = {(12, 15): 0.0, (15, 18): 0.0, (18, 21): 0.0, (21, 24): 40.0, (24, 27): 520.0, (27, 30): 860.0,
               (30, 33): 600.0, (33, 36): 90.0, (36, 39): 0.0}


def gfs_style_avgs(blocks):
    """Buat rata-rata gaya GFS (reset 6 jam) dari blok 3 jam yang diketahui."""
    rows = []
    for L in LEADS:
        if L % 6 == 3:
            s, v = L - 3, blocks[(L - 3, L)]
        else:
            s, v = L - 6, (blocks[(L - 6, L - 3)] + blocks[(L - 3, L)]) / 2
        rows.append({"init": INIT, "name": "dswrf_sfc", "step_type": "avg", "start_h": s, "end_h": L, "v_point": v})
    return pd.DataFrame(rows)


def test_deaverage_recovers_true_3h_blocks():
    out = G.deaverage(gfs_style_avgs(TRUE_BLOCKS)).set_index("block_end_h")["value"]
    for (s, e), v in TRUE_BLOCKS.items():
        if s >= 18:                                  # blok 15–18 butuh lead 15 (tidak diunduh)
            assert out.loc[e] == pytest.approx(v), (s, e)
    assert 18 not in out.index


def test_deaverage_skips_block_when_lead_missing():
    long = gfs_style_avgs(TRUE_BLOCKS)
    long = long[long["end_h"] != 27]                 # seperti 20160222 f027 yang hilang
    out = G.deaverage(long).set_index("block_end_h")
    assert 27 not in out.index and 30 not in out.index   # 27 hilang, 30 butuh 27
    assert out.loc[33, "value"] == pytest.approx(600.0)


def test_blocks_to_hourly_labels_period_ending():
    b = pd.DataFrame({"init": [INIT], "name": ["x"], "block_start_h": [24], "block_end_h": [27], "value": [1.0]})
    h = G.blocks_to_hourly(b).sort_values("label")
    assert list(h["label"]) == [INIT + pd.Timedelta(hours=k) for k in (25, 26, 27)]


def test_to_hourly_kc_route_preserves_block_mean():
    labels = pd.date_range(INIT + pd.Timedelta(hours=19), INIT + pd.Timedelta(hours=39), freq="1h")
    cs = pd.Series(np.linspace(100, 900, len(labels)), index=labels)
    hr = G.to_hourly(gfs_style_avgs(TRUE_BLOCKS), cs)
    blk = hr.loc[INIT].loc[INIT + pd.Timedelta(hours=28): INIT + pd.Timedelta(hours=30), "gfs_ghi_nwp"]
    assert blk.mean() == pytest.approx(860.0)        # rata-rata 3 jam tetap sama dengan blok GFS
    assert blk.is_monotonic_increasing               # bentuk mengikuti clear-sky
    k = hr.loc[INIT].loc[INIT + pd.Timedelta(hours=28), "gfs_kc_nwp"]
    assert k == pytest.approx(860.0 / cs.loc[INIT + pd.Timedelta(hours=28): INIT + pd.Timedelta(hours=30)].mean())


def test_instant_linear_interpolation():
    long = pd.DataFrame({"init": INIT, "name": "tmp_2m", "step_type": "instant", "start_h": [18, 21],
                         "end_h": [18, 21], "v_point": [290.0, 296.0]})
    h = G.instant_to_hourly(long).set_index("label")["value"]
    assert h.loc[INIT + pd.Timedelta(hours=19)] == pytest.approx(292.0)


def _write_grib(path, param, start, end, avg, val):
    import eccodes as ec
    h = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ec.codes_set_long(h, "Ni", 5); ec.codes_set_long(h, "Nj", 7)
    for k, v in dict(latitudeOfFirstGridPointInDegrees=-6.0, longitudeOfFirstGridPointInDegrees=107.0,
                     latitudeOfLastGridPointInDegrees=-7.5, longitudeOfLastGridPointInDegrees=108.0,
                     iDirectionIncrementInDegrees=0.25, jDirectionIncrementInDegrees=0.25).items():
        ec.codes_set(h, k, v)
    ec.codes_set(h, "dataDate", 20200101); ec.codes_set(h, "dataTime", 0)
    d, c, n, tl, lv = param
    ec.codes_set(h, "discipline", d); ec.codes_set(h, "parameterCategory", c); ec.codes_set(h, "parameterNumber", n)
    if avg:
        ec.codes_set(h, "productDefinitionTemplateNumber", 8)
        ec.codes_set(h, "typeOfStatisticalProcessing", 0)
        ec.codes_set(h, "forecastTime", start); ec.codes_set(h, "lengthOfTimeRange", end - start)
    else:
        ec.codes_set(h, "forecastTime", end)
    ec.codes_set(h, "typeOfFirstFixedSurface", tl)
    if lv is not None:
        ec.codes_set(h, "scaledValueOfFirstFixedSurface", lv); ec.codes_set(h, "scaleFactorOfFirstFixedSurface", 0)
    ec.codes_set_values(h, np.full(35, float(val)) + np.arange(35))
    with open(path, "ab") as f:
        ec.codes_write(h, f)
    ec.codes_release(h)


def test_extract_from_grib(tmp_path):
    pytest.importorskip("eccodes")
    p = tmp_path / "gfs.0p25.2020010100.f018.grib2"
    _write_grib(p, (0, 4, 7, 1, None), 12, 18, True, 300)
    _write_grib(p, (0, 6, 1, 10, None), 12, 18, True, 60)
    _write_grib(p, (0, 6, 1, 214, None), 12, 18, True, 20)
    _write_grib(p, (0, 0, 0, 103, 2), 18, 18, False, 295)
    _write_grib(p, (0, 1, 3, 200, None), 18, 18, False, 50)
    other = tmp_path / "gfs.0p25.2020010106.f018.grib2"      # run 06 → harus diabaikan
    _write_grib(other, (0, 0, 0, 103, 2), 18, 18, False, 999)
    files = G.run00_files(tmp_path)
    assert [f.name for f in files] == [p.name]
    long = G.extract_files(files, -6.8899, 107.61, progress_every=0)
    names = set(long["name"])
    assert names == {"dswrf_sfc", "tcdc_atm", "tcdc_low", "tmp_2m", "pwat_atm1"}
    r = long.set_index("name").loc["dswrf_sfc"]
    assert (r["grid_lat"], r["grid_lon"]) == (-7.0, 107.5)     # titik 0,25° terdekat dari ITB
    assert r["n_3x3"] == 9 and r["step_type"] == "avg" and (r["start_h"], r["end_h"]) == (12, 18)
    # nilai = val + indeks titik; titik (-7.0, 107.5) = baris 4, kolom 2 → indeks 22 (grid 7×5, utara→selatan)
    assert r["v_point"] == pytest.approx(300 + 22)
    assert r["v_3x3"] == pytest.approx(300 + 22)               # rata-rata 3×3 simetris = nilai tengah


def test_merge_and_leakage_latency():
    cfg = {"data": {"years": [2020], "split": {"train": [], "val": [], "test": [2020]}},
           "physics": {"zenith_max_deg": 85}, "forecast": {"issue_time_utc": "05:00"}}
    idx = pd.date_range("2020-01-01 01:00", "2021-01-01 00:00", freq="1h", tz="UTC")
    hourly = pd.DataFrame({"solar_zenith_angle": 50.0, "ghi_cs": 500.0, "kc": 0.7, "ghi": 350.0,
                           "interp_frac": 0.0}, index=idx)
    table = F.build_feature_table(hourly, cfg)
    labels = INIT + pd.to_timedelta(np.arange(19, 40), unit="h")
    gh = pd.DataFrame({"gfs_ghi_nwp": 1.0}, index=pd.MultiIndex.from_arrays([[INIT] * len(labels), labels],
                                                                            names=["init", "label"]))
    ok = G.merge_into_table(table, gh, latency_h=4.5)
    F.assert_no_leakage(ok)                          # 00 UTC + 4,5 j = 04.30 UTC ≤ 05.00 UTC
    row = ok[(ok["target_day"] == "2020-01-02") & (ok["target_hour_wib"] == 12)].iloc[0]
    assert row["gfs_ghi_nwp"] == 1.0 and row["gfs_init"] == INIT
    late = G.merge_into_table(table, gh, latency_h=6)   # data GFS baru tersedia 06 UTC → bocor
    with pytest.raises(AssertionError, match="KEBOCORAN"):
        F.assert_no_leakage(late)


def test_outputs_have_datetime_dtypes_even_from_object_input():
    """Regresi: di sebagian versi pandas kolom init bisa jadi object → kunci merge tidak cocok."""
    long = gfs_style_avgs(TRUE_BLOCKS)
    long["init"] = long["init"].astype(object)
    blk = G.deaverage(long)
    assert isinstance(blk["init"].dtype, pd.DatetimeTZDtype)
    assert pd.DatetimeIndex(blk.groupby("init").size().index).year[0] == 2020
    labels = pd.date_range(INIT + pd.Timedelta(hours=19), INIT + pd.Timedelta(hours=39), freq="1h")
    hr = G.to_hourly(long, pd.Series(500.0, index=labels))
    assert isinstance(hr.index.get_level_values(0).dtype, pd.DatetimeTZDtype)
    assert isinstance(hr.index.get_level_values(1).dtype, pd.DatetimeTZDtype)


def test_ncep_local_dswrf_code_and_cache_normalization(tmp_path):
    pytest.importorskip("eccodes")
    assert G.param_name(0, 4, 192, 1, 0) == "dswrf_sfc"
    p = tmp_path / "gfs.0p25.2020010100.f018.grib2"
    _write_grib(p, (0, 4, 192, 1, None), 12, 18, True, 300)      # kode lokal NCEP
    long = G.extract_files(G.run00_files(tmp_path), -6.8899, 107.61, progress_every=0)
    assert set(long["name"]) == {"dswrf_sfc"}
    old = pd.DataFrame({"name": ["d0c4n192_sfc", "tcdc_low", "d9c9n9_sfc"]})   # cache versi lama
    assert list(G.normalize_names(old)["name"]) == ["dswrf_sfc", "tcdc_low", "d9c9n9_sfc"]


def test_to_hourly_raises_clear_error_without_dswrf():
    long = gfs_style_avgs(TRUE_BLOCKS).assign(name="d0c4n999_sfc")
    with pytest.raises(ValueError, match="DSWRF tidak ditemukan"):
        G.to_hourly(long, pd.Series(dtype=float))

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as E  # noqa: E402
import models as M  # noqa: E402

KZ = {"edges": [0, 40, 60, 70, 75, 85], "values": [1.04, 1.06, 1.11, 1.17, 1.71]}


def test_metrics_known_values():
    m = E.metrics([100, 200, 300], [110, 190, 330])
    assert m["MBE"] == pytest.approx(10.0)
    assert m["MAE"] == pytest.approx(50 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert m["nRMSE_%"] == pytest.approx(100 * m["RMSE"] / 200)


def test_violation_rate_before_clipping():
    v = E.violation_rate([-5, 500, 1200, 10], [1000, 1000, 1000, 0], 1.1)
    assert v["pct_negatif"] == 25 and v["pct_di_atas_kmax"] == 50   # 1200 > 1100; 10 > 0 di malam


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    zen = rng.uniform(10, 95, n)
    cs = np.where(zen < 90, 1000 * np.cos(np.deg2rad(zen)), 0.0)
    kc = rng.uniform(0.2, 1.0, n)
    df = pd.DataFrame({"zenith": zen, "ghi_cs": cs, "y_kc": np.where(zen < 85, kc, np.nan), "y_ghi": kc * cs,
                       "is_day": zen < 85, "eval_valid": zen < 85, "kc_clim": 0.6, "kc_same_hour_lag": kc[::-1],
                       "gfs_ghi_nwp": cs * np.clip(kc + 0.1, 0, None), "gfs_kc_nwp": np.clip(kc + 0.1, 0, None),
                       "cos_zenith": np.cos(np.deg2rad(zen)), "hour_sin": rng.normal(size=n),
                       "hour_cos": rng.normal(size=n), "doy_sin": 0.0, "doy_cos": 1.0, "ghi_extra": 1360.0,
                       "target_hour_wib": rng.integers(6, 18, n),
                       "target_day": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 365, n), "D")})
    return df


def test_finalize_clips_and_zeroes_night():
    df = _toy()
    raw = df["ghi_cs"].to_numpy() * 2 - 50
    p = M.finalize(raw, df, KZ, do_clip=True)
    k = M.kmax_array(df, KZ)
    assert (p >= 0).all() and (p <= k * df["ghi_cs"] + 1e-9).all()
    assert (p[df["ghi_cs"] <= 0] == 0).all()


def test_baselines_and_mos_beat_nothing_and_are_zero_at_night():
    df = _toy()
    b = M.baselines(df, df["is_day"])
    assert set(b) == {"pred_clim", "pred_persist", "pred_gfs_raw", "pred_mos"}
    assert (b.loc[df["ghi_cs"] <= 0] == 0).all().all()
    t = E.evaluate(pd.concat([df, b], axis=1), list(b))
    assert t.loc[("valid", "pred_mos"), "RMSE"] < t.loc[("valid", "pred_gfs_raw"), "RMSE"]   # MOS menghapus bias +0,1


def test_skill_and_stratify_shapes():
    df = _toy()
    b = M.baselines(df, df["is_day"])
    d = pd.concat([df, b], axis=1)
    t = E.add_skill(E.evaluate(d, list(b)))
    assert t.loc[("valid", "pred_clim"), "skill_vs_clim"] == pytest.approx(0.0)
    s = E.stratify(d, list(b), "musim")
    assert set(s.index) <= {"DJF", "MAM", "JJA", "SON"} and "n" in s


def test_fit_predict_xgb_kc_target_returns_ghi_scale():
    pytest.importorskip("xgboost")
    df = _toy(600)
    feats = ["cos_zenith", "gfs_kc_nwp", "hour_sin"]
    p = M.fit_predict("xgb", {"n_estimators": 50, "max_depth": 3}, df, feats, df["is_day"], "kc", seed=0)
    assert p.shape == (len(df),)
    assert np.nanmax(p) <= 1.5 * df["ghi_cs"].max()


def _days(n_days=120, seed=0, edge=0.0):
    rng = np.random.default_rng(seed)
    days = np.repeat(pd.date_range("2020-01-01", periods=n_days), 8)
    y = rng.uniform(100, 900, len(days))
    return pd.DataFrame({"target_day": days, "y_ghi": y,
                         "a": y + rng.normal(0, 50, len(y)), "b": y + rng.normal(edge, 50 + 40 * (edge > 0), len(y))})


def test_dm_identical_and_clear_difference():
    d = _days()
    se = E.daily_sq_error(d, ["a", "b"], pd.Series(True, index=d.index))
    same = E.diebold_mariano(se["a"], se["a"])
    assert same["p"] == 1.0
    d2 = _days(edge=1.0)
    se2 = E.daily_sq_error(d2, ["a", "b"], pd.Series(True, index=d2.index))
    r = E.diebold_mariano(se2["a"], se2["b"])
    assert r["DM"] < 0 and r["p"] < 0.01          # a jelas lebih akurat


def test_block_bootstrap_ci_contains_point_estimate():
    d = _days(edge=1.0).assign(pred_clim=lambda x: x["b"], pred_gfs_raw=lambda x: x["b"])
    ci = E.block_bootstrap(d, ["a", "pred_clim", "pred_gfs_raw"], pd.Series(True, index=d.index), n_boot=300)
    point = np.sqrt(((d["a"] - d["y_ghi"]) ** 2).mean())
    assert ci.loc["RMSE_a", "ci_low"] <= point <= ci.loc["RMSE_a", "ci_high"]
    assert ci.loc["skill_a_vs_pred_clim", "ci_low"] > 0


def test_save_load_models_roundtrip(tmp_path):
    pytest.importorskip("xgboost")
    df = _toy(300)
    feats = ["cos_zenith", "gfs_kc_nwp", "hour_sin"]
    for kind, p in (("xgb", {"n_estimators": 20, "max_depth": 3}), ("rf", {"n_estimators": 20, "max_depth": 5})):
        m = M.fit_model(kind, p, df, feats, df["is_day"], "kc", seed=0)
        M.save_model(m, kind, tmp_path / kind)
        m2 = M.load_model(kind, tmp_path / kind)
        assert np.allclose(m.predict(df[feats]), m2.predict(df[feats]), atol=1e-5)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import xai as X  # noqa: E402


def test_apriori_file_covers_frozen_features():
    fmap, order = X.load_apriori(ROOT / "configs" / "apriori_order.yaml")
    assert list(order.values) == list(range(1, len(order) + 1))
    import models as M
    frozen = M.GEOM + M.PHYS + M.LAG + M.NWP
    missing = [f for f in frozen if f not in fmap]
    assert not missing, f"Fitur beku tanpa kelompok a priori: {missing}"


def test_group_abs_and_rank():
    attr = pd.DataFrame({"a1": [1, -1], "a2": [2, 0], "b": [-0.5, 0.5]})
    g = X.group_abs(attr, {"a1": "A", "a2": "A", "b": "B"})
    assert list(g["A"]) == [3, 1] and list(g["B"]) == [0.5, 0.5]
    assert X.rank_groups(g.mean()).to_dict() == {"A": 1.0, "B": 2.0}


def test_spearman_perfect_and_reversed():
    ap = pd.Series({"g1": 1, "g2": 2, "g3": 3, "g4": 4})
    good = pd.DataFrame(np.tile([4.0, 3.0, 2.0, 1.0], (50, 1)), columns=list(ap.index))
    assert X.spearman_vs_apriori(good, ap, n_boot=50)["rho"] == pytest.approx(1.0)
    assert X.spearman_vs_apriori(good[good.columns[::-1]].set_axis(good.columns, axis=1), ap, n_boot=50)["rho"] == pytest.approx(-1.0)


def test_jaccard():
    a = pd.Series({"x": 3, "y": 2, "z": 1}); b = pd.Series({"x": 3, "z": 2, "y": 1})
    assert X.jaccard_topk(a, b, 1) == 1.0 and X.jaccard_topk(a, b, 2) == pytest.approx(1 / 3)


def test_stratified_sample_size_and_reproducible():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y_kc": rng.uniform(0, 1.1, 2000),
                       "target_day": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 365, 2000), "D")})
    s1, s2 = X.stratified_sample(df, 200, 1), X.stratified_sample(df, 200, 1)
    assert len(s1) == 200 and s1.equals(s2) and s1.is_unique


def test_xgb_shap_additivity():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    Xd = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = 2 * Xd["a"] - Xd["b"] + rng.normal(0, .1, 300)
    m = xgb.XGBRegressor(n_estimators=50, max_depth=3).fit(Xd, y)
    sv = X.xgb_shap(m, Xd)
    full = m.get_booster().predict(xgb.DMatrix(Xd), pred_contribs=True)
    assert np.allclose(sv.sum(axis=1) + full[:, -1], m.predict(Xd), atol=1e-4)
    assert sv.abs().mean().idxmax() == "a"


def test_group_lime_recovers_additive_contributions():
    contrib = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0]])           # 2 sampel × 3 kelompok
    fn = lambda Z: Z.astype(float) @ contrib.T + 7.0                    # noqa: E731
    coef = X.group_lime(fn, 3, n_samples=400, seed=1)
    assert np.allclose(coef, contrib, atol=1e-2)

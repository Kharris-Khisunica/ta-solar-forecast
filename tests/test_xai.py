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


# ---------------------------------------------------------------- Shapley kelompok (dipindah dari lstm.py)
def test_coalition_index_matches_order():
    S = X.coalitions(4)
    assert (X.coalition_index(S) == np.arange(16)).all()


def test_shapley_from_values_matches_value_fn_route():
    rng = np.random.default_rng(0)
    contrib = rng.normal(size=(5, 4))
    fn = lambda S: S.astype(float) @ contrib.T                             # noqa: E731  model aditif
    phi, V = X.exact_group_shapley(fn, list("abcd"), return_values=True)
    assert np.allclose(phi.to_numpy(), contrib)
    assert np.allclose(X.shapley_from_values(V, 4), contrib)


class _Lin:
    """Model linear kecil dengan interaksi, berlaku seperti model sklearn (predict DataFrame)."""
    def predict(self, d):
        a = np.asarray(d, float)
        return a[:, 0] + 2 * a[:, 1] - a[:, 2] + 0.5 * a[:, 0] * a[:, 2]


def test_tabular_bg_with_single_mean_row_equals_point_baseline():
    rng = np.random.default_rng(1)
    Xd = pd.DataFrame(rng.normal(size=(6, 3)), columns=["f1", "f2", "f3"])
    fmap = {"f1": "g1", "f2": "g2", "f3": "g1"}; groups = ["g1", "g2"]
    bgmean = pd.Series([0.3, -0.2, 1.0], index=Xd.columns)
    a = X.exact_group_shapley(X.tabular_value_fn(_Lin(), Xd, bgmean, fmap, groups), groups)
    bg = np.broadcast_to(bgmean.to_numpy(), (6, 1, 3)).copy()
    b = X.exact_group_shapley(X.tabular_value_fn_bg(_Lin(), Xd, bg, fmap, groups), groups)
    assert np.allclose(a, b)


def test_tabular_bg_distribution_is_efficient():
    rng = np.random.default_rng(2)
    Xd = pd.DataFrame(rng.normal(size=(4, 3)), columns=["f1", "f2", "f3"])
    fmap = {"f1": "g1", "f2": "g2", "f3": "g3"}; groups = ["g1", "g2", "g3"]
    bg = rng.normal(size=(4, 7, 3))
    fn = X.tabular_value_fn_bg(_Lin(), Xd, bg, fmap, groups)
    phi, V = X.exact_group_shapley(fn, groups, return_values=True)
    full = _Lin().predict(Xd)
    empty = _Lin().predict(bg.reshape(-1, 3)).reshape(4, 7).mean(1)
    assert np.allclose(V[-1], full) and np.allclose(V[0], empty)
    assert np.allclose(phi.sum(axis=1), full - empty)                     # efisiensi

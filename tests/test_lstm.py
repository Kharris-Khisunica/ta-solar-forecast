import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402
import lstm as L  # noqa: E402

CFG = {"data": {"years": [2016, 2017], "split": {"train": [2016], "val": [2017], "test": []}},
       "physics": {"zenith_max_deg": 85}, "forecast": {"issue_time_utc": "05:00"}}


@pytest.fixture(scope="module")
def data():
    idx = pd.date_range("2016-01-01 01:00", "2018-01-01 00:00", freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    wib = (idx.hour + 7) % 24
    zen = np.where((wib >= 6) & (wib <= 18), 30 + 8 * np.abs(wib - 12), 120).astype(float)
    cs = np.where(zen < 90, 1000 * np.cos(np.deg2rad(zen)), 0.0)
    kc = np.where(zen < 85, rng.uniform(0.2, 1.0, len(idx)), np.nan)
    hourly = pd.DataFrame({"solar_zenith_angle": zen, "ghi_cs": cs, "kc": kc, "ghi": np.nan_to_num(kc) * cs,
                           "interp_frac": 0.0, "temperature": 25.0, "relative_humidity": 80.0}, index=idx)
    table = F.build_feature_table(hourly, CFG)
    table["gfs_kc_nwp_3x3"] = table["y_kc"].fillna(0) + rng.normal(0, 0.05, len(table))   # sinyal kuat
    return hourly, table


def test_shapes_and_no_leakage(data):
    hourly, table = data
    A = L.build_arrays(hourly, table)
    nd = table["target_day"].nunique()
    assert A["xs"].shape[:2] == (nd, L.SEQ_LEN) and A["xt"].shape[:2] == (nd, 24)
    assert (A["seq_last_label"] == A["issue"]).all()                 # sekuens berakhir tepat di penerbitan
    assert "gfs_kc_nwp_3x3" in A["tgt_ch"]


def test_normalizer_uses_train_only(data):
    hourly, table = data
    A = L.build_arrays(hourly, table)
    tr = np.where(A["split"] == "train")[0]
    A2 = {**A, "xs": A["xs"].copy()}
    A2["xs"][A["split"] == "val"] += 100.0                            # ubah validasi ekstrem
    n1, n2 = L.Normalizer().fit(A, tr), L.Normalizer().fit(A2, tr)
    assert np.allclose(n1.ms, n2.ms) and np.allclose(n1.ss, n2.ss)


def test_exact_group_shapley_additive_and_efficient():
    groups = ["a", "b", "c"]
    contrib = np.array([[1.0, 2.0, -0.5], [0.5, 0.0, 3.0]])             # 2 sampel
    fn = lambda S: S.astype(float) @ contrib.T                            # noqa: E731  model aditif
    phi = L.exact_group_shapley(fn, groups)
    assert np.allclose(phi.to_numpy(), contrib)                           # Shapley = kontribusi aditif
    full, empty = fn(np.ones((1, 3), bool))[0], fn(np.zeros((1, 3), bool))[0]
    assert np.allclose(phi.sum(axis=1), full - empty)                     # efisiensi


def test_lstm_learns_signal_quickly(data):
    pytest.importorskip("torch")
    hourly, table = data
    A = L.build_arrays(hourly, table)
    tr, va = np.where(A["split"] == "train")[0], np.where(A["split"] == "val")[0]
    xs, xt = L.Normalizer().fit(A, tr).transform(A)
    p = {"hidden": 16, "layers": 1, "dropout": 0.0, "lr": 3e-3, "batch": 32}
    net, hist = L.train(p, xs, xt, A["y"], A["w_train"], A["w_eval"], tr, va, seed=0, max_epochs=25, patience=5)
    pred = L.predict(net, xs, xt)
    w = A["w_eval"][va]
    mse = (w * (pred[va] - A["y"][va]) ** 2).sum() / w.sum()
    base = (w * (A["y"][va] - (A["y"][tr] * A["w_train"][tr]).sum() / A["w_train"][tr].sum()) ** 2).sum() / w.sum()
    assert mse < 0.5 * base                                               # jauh lebih baik dari rata-rata
    net2, _ = L.train(p, xs, xt, A["y"], A["w_train"], A["w_eval"], tr, va, seed=0, max_epochs=3, patience=5)
    net3, _ = L.train(p, xs, xt, A["y"], A["w_train"], A["w_eval"], tr, va, seed=0, max_epochs=3, patience=5)
    assert np.allclose(L.predict(net2, xs, xt), L.predict(net3, xs, xt))  # reprodusibel dengan seed sama


def test_save_load_roundtrip(data, tmp_path):
    torch = pytest.importorskip("torch")
    hourly, table = data
    A = L.build_arrays(hourly, table)
    tr, va = np.where(A["split"] == "train")[0], np.where(A["split"] == "val")[0]
    xs, xt = L.Normalizer().fit(A, tr).transform(A)
    p = {"hidden": 8, "layers": 1, "dropout": 0.0, "lr": 1e-3, "batch": 32}
    net, _ = L.train(p, xs, xt, A["y"], A["w_train"], A["w_eval"], tr, va, seed=0, max_epochs=2)
    torch.save(net.state_dict(), tmp_path / "m.pt")
    net2 = L.load_net(tmp_path / "m.pt", p, xs.shape[2], xt.shape[2])
    assert np.allclose(L.predict(net, xs, xt), L.predict(net2, xs, xt))
    groups = ["lag_observasi", "geometri_waktu", "termodinamika", "nwp_radiasi", "klimatologi"]
    seq_g = [L.SEQ_GROUPS[c] for c in A["seq_ch"]]; tgt_g = [L.TGT_GROUPS[c] for c in A["tgt_ch"]]
    fn = L.lstm_value_fn_days(net, xs[:3], xt[:3], seq_g, tgt_g, groups)
    v = fn(np.ones((1, len(groups)), bool))
    assert v.shape == (1, 3 * 24) and np.allclose(v[0], L.predict(net, xs[:3], xt[:3]).ravel())

"""XAI: TreeSHAP & LIME, agregasi per kelompok fitur, kesesuaian dengan urutan a priori, stabilitas (T38–T42).

Atribusi dihitung pada keluaran model (target k_c). Kepentingan kelompok = jumlah |atribusi| fitur
anggotanya, dirata-rata atas sampel. Kesesuaian fisis = Spearman ρ antara peringkat kelompok model
dan urutan a priori (configs/apriori_order.yaml), dengan CI bootstrap atas sampel.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def load_apriori(path) -> tuple[dict, pd.Series]:
    """→ (peta fitur→kelompok, urutan a priori per kelompok [1 = terpenting])."""
    cfg = yaml.safe_load(open(path))
    fmap, order = {}, {}
    for g in cfg["kelompok"]:
        order[g["nama"]] = int(g["urutan"])
        for f in g["fitur"]:
            fmap[f] = g["nama"]
    return fmap, pd.Series(order).sort_values()


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.Index:
    """Sampel jam siang valid, terstratifikasi musim × kondisi langit (3 bin k_c), ±seimbang (T38)."""
    sky = pd.cut(df["y_kc"], [-np.inf, 0.4, 0.8, np.inf], labels=["berawan", "sebagian", "cerah"])
    strata = df["target_day"].dt.month.map(SEASON).astype(str) + "|" + sky.astype(str)
    groups = list(df.groupby(strata).groups.items())
    per = max(1, n // len(groups))
    rng = np.random.default_rng(seed)
    picked = []
    for _, idx in groups:
        idx = np.asarray(idx)
        picked += list(rng.choice(idx, size=min(per, len(idx)), replace=False))
    rest = np.setdiff1d(df.index.to_numpy(), picked)
    if len(picked) < n:
        picked += list(rng.choice(rest, size=n - len(picked), replace=False))
    return pd.Index(sorted(picked[:n]))


def xgb_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP eksak untuk XGBoost via pred_contribs (tanpa paket shap). Kolom terakhir (bias) dibuang."""
    import xgboost as xgb
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns)


def rf_shap(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP untuk RF di dalam pipeline (imputer → RF); atribusi pada fitur setelah imputasi."""
    import shap
    imp, rf = pipeline[0], pipeline[-1]
    Xi = pd.DataFrame(imp.transform(X), index=X.index, columns=X.columns)
    vals = shap.TreeExplainer(rf).shap_values(Xi, check_additivity=False)
    return pd.DataFrame(vals, index=X.index, columns=X.columns)


def lime_attrib(predict_fn, X_train: pd.DataFrame, X: pd.DataFrame, seed: int, num_samples: int = 1000) -> pd.DataFrame:
    """LIME (regresi) untuk tiap baris X; mengembalikan bobot per fitur (semua fitur)."""
    from lime.lime_tabular import LimeTabularExplainer
    med = X_train.median()
    Xt = X_train.fillna(med).to_numpy()
    ex = LimeTabularExplainer(Xt, mode="regression", feature_names=list(X.columns),
                              discretize_continuous=False, random_state=seed)
    f = lambda a: predict_fn(pd.DataFrame(a, columns=X.columns))  # noqa: E731
    rows = []
    for _, r in X.fillna(med).iterrows():
        e = ex.explain_instance(r.to_numpy(), f, num_features=X.shape[1], num_samples=num_samples)
        w = dict(e.as_map()[1])
        rows.append([w.get(i, 0.0) for i in range(X.shape[1])])
    return pd.DataFrame(rows, index=X.index, columns=X.columns)


def group_abs(attr: pd.DataFrame, fmap: dict) -> pd.DataFrame:
    """|atribusi| dijumlahkan per kelompok → (sampel × kelompok). Fitur tanpa kelompok diabaikan."""
    a = attr.abs()
    cols = [c for c in a.columns if c in fmap]
    return a[cols].T.groupby(lambda c: fmap[c]).sum().T


def rank_groups(group_imp: pd.Series) -> pd.Series:
    """Peringkat 1 = kepentingan terbesar."""
    return group_imp.rank(ascending=False, method="average")


def spearman_vs_apriori(group_abs_df: pd.DataFrame, apriori: pd.Series, n_boot: int = 1000, seed: int = 0) -> dict:
    groups = [g for g in apriori.index if g in group_abs_df.columns]
    ap = apriori[groups].rank()
    rho = spearmanr(rank_groups(group_abs_df[groups].mean()), ap).statistic
    rng = np.random.default_rng(seed)
    n = len(group_abs_df)
    boots = []
    for _ in range(n_boot):
        s = group_abs_df.iloc[rng.integers(0, n, n)][groups].mean()
        boots.append(spearmanr(rank_groups(s), ap).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"rho": float(rho), "ci_low": float(lo), "ci_high": float(hi), "n_kelompok": len(groups)}


def jaccard_topk(imp_a: pd.Series, imp_b: pd.Series, k: int) -> float:
    a = set(imp_a.sort_values(ascending=False).index[:k])
    b = set(imp_b.sort_values(ascending=False).index[:k])
    return len(a & b) / len(a | b)


def bootstrap_rank_std(group_abs_df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.Series:
    """Simpangan baku peringkat tiap kelompok atas resampel bootstrap (stabilitas SHAP)."""
    rng = np.random.default_rng(seed)
    n = len(group_abs_df)
    ranks = [rank_groups(group_abs_df.iloc[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
    return pd.DataFrame(ranks).std()


def group_lime(value_fn, M: int, n_samples: int = 1000, seed: int = 0, kernel_width: float | None = None,
               ridge: float = 1e-3) -> np.ndarray:
    """LIME tingkat kelompok: representasi terinterpretasi z ∈ {0,1}^M (kelompok hadir/absen, seperti
    superpixel pada gambar). value_fn(z [n, M] bool) → [n, S]. Kernel eksponensial pada jarak ke z = 1.
    Mengembalikan koefisien regresi linear berbobot (S × M)."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=(n_samples, M)).astype(bool)
    z[0] = True
    y = np.asarray(value_fn(z))                                   # n × S
    kw = 0.75 * np.sqrt(M) if kernel_width is None else kernel_width
    d = np.sqrt((~z).sum(axis=1))
    w = np.exp(-(d ** 2) / kw ** 2)
    Xd = np.hstack([np.ones((n_samples, 1)), z.astype(float)])
    A = Xd.T @ (w[:, None] * Xd) + ridge * np.eye(M + 1)
    B = Xd.T @ (w[:, None] * y)
    coef = np.linalg.solve(A, B)                                  # (M+1) × S
    return coef[1:].T


# ---------------------------------------------------------------- Shapley kelompok eksak (dipindah dari lstm.py)
def coalitions(M: int) -> np.ndarray:
    """Semua 2^M koalisi (bool, 2^M × M), urutan itertools.product: kelompok pertama paling lambat berubah."""
    return np.array(list(itertools.product([0, 1], repeat=M)), dtype=bool)


def coalition_index(present) -> np.ndarray:
    """Indeks baris koalisi di coalitions(M): Σ s_g · 2^(M−1−g). present: bool [..., M]."""
    present = np.asarray(present, dtype=bool)
    M = present.shape[-1]
    return present.astype(np.int64) @ (1 << np.arange(M - 1, -1, -1))


def shapley_from_values(V: np.ndarray, M: int) -> np.ndarray:
    """Nilai Shapley eksak dari matriks nilai koalisi V [2^M, S] (urutan coalitions(M)) → φ [S, M]."""
    V = np.asarray(V, float)
    subsets = coalitions(M)
    fact = [math.factorial(k) for k in range(M + 1)]
    phi = np.zeros((V.shape[1], M))
    for g in range(M):
        absent = ~subsets[:, g]
        s = subsets[absent]
        k = s.sum(axis=1)
        w = np.array([fact[kk] * fact[M - kk - 1] / fact[M] for kk in k])
        with_g = s.copy(); with_g[:, g] = True
        phi[:, g] = (w[:, None] * (V[coalition_index(with_g)] - V[coalition_index(s)])).sum(axis=0)
    return phi


def exact_group_shapley(value_fn, groups: list[str], return_values: bool = False):
    """value_fn(present: bool [2^M, M]) → array [2^M, n_sampel] nilai model tiap koalisi.
    Mengembalikan φ (n_sampel × M). Sifat efisiensi: Σφ = v(semua) − v(kosong).
    return_values=True → juga mengembalikan V (untuk uji faithfulness tanpa menghitung ulang)."""
    M = len(groups)
    V = np.asarray(value_fn(coalitions(M)))
    phi = pd.DataFrame(shapley_from_values(V, M), columns=groups)
    return (phi, V) if return_values else phi


def tabular_value_fn(model, X: pd.DataFrame, background: pd.Series, fmap: dict, groups):
    """Koalisi → prediksi model tabular; fitur kelompok absen diganti rata-rata data latih (satu titik)."""
    cols_by_g = {g: [c for c in X.columns if fmap.get(c) == g] for g in groups}

    def fn(subsets):
        res = []
        for s in subsets:
            Xm = X.copy()
            for gi, g in enumerate(groups):
                if not s[gi]:
                    Xm[cols_by_g[g]] = background[cols_by_g[g]].to_numpy()
            res.append(model.predict(Xm))
        return np.array(res)
    return fn


def tabular_value_fn_bg(model, X: pd.DataFrame, bg: np.ndarray, fmap: dict, groups):
    """Seperti tabular_value_fn, tetapi latar = DISTRIBUSI: bg [n_sampel, K, n_fitur] (kolom = X.columns).
    Nilai koalisi = rata-rata prediksi atas K baris latar (Shapley intervensional)."""
    n, K, p = bg.shape
    assert n == len(X) and p == X.shape[1]
    Xr = np.repeat(X.to_numpy(float), K, axis=0)                  # baris i*K + k = sampel i, latar k
    Br = bg.reshape(n * K, p).astype(float)
    gidx = np.array([groups.index(fmap[c]) if fmap.get(c) in groups else -1 for c in X.columns])

    def fn(subsets):
        res = []
        for s in subsets:
            keep = np.array([True if g < 0 else bool(s[g]) for g in gidx])
            Xm = pd.DataFrame(np.where(keep[None, :], Xr, Br), columns=X.columns)
            res.append(np.asarray(model.predict(Xm), float).reshape(n, K).mean(axis=1))
        return np.array(res)
    return fn

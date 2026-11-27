"""Metrik, skill score, laju pelanggaran fisis, dan stratifikasi (T13, T29, T35).

Semua metrik dihitung pada GHI (W/m²) di jam siang. Dua mask (keputusan #10):
  - "valid": jam siang dengan < 2/6 sampel NSRDB hasil interpolasi (evaluasi utama)
  - "all"  : semua jam siang (uji sensitivitas)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) == 0:
        return {"n": 0, "RMSE": np.nan, "nRMSE_%": np.nan, "MAE": np.nan, "MBE": np.nan, "R2": np.nan}
    e = yhat - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "RMSE": rmse, "nRMSE_%": 100 * rmse / float(y.mean()),
            "MAE": float(np.mean(np.abs(e))), "MBE": float(np.mean(e)),
            "R2": 1 - float(np.sum(e ** 2)) / ss_tot if ss_tot > 0 else np.nan}


def masks(df: pd.DataFrame) -> dict:
    base = df["is_day"] & df["y_ghi"].notna()
    return {"valid": base & df["eval_valid"], "all": base}


def evaluate(df: pd.DataFrame, pred_cols, mask_names=("valid", "all")) -> pd.DataFrame:
    """Tabel metrik: baris = (mask, model). Semua model dinilai pada baris yang SAMA
    (baris di mana semua prediksi tersedia), supaya perbandingan adil."""
    ms = masks(df)
    rows = []
    for mname in mask_names:
        m = ms[mname] & df[list(pred_cols)].notna().all(axis=1)
        for c in pred_cols:
            rows.append({"mask": mname, "model": c, **metrics(df.loc[m, "y_ghi"], df.loc[m, c])})
    return pd.DataFrame(rows).set_index(["mask", "model"])


def add_skill(tab: pd.DataFrame, refs=("pred_clim", "pred_persist", "pred_gfs_raw")) -> pd.DataFrame:
    """Skill score RMSE = 1 − RMSE_model / RMSE_acuan (per mask)."""
    tab = tab.copy()
    for ref in refs:
        if ref not in tab.index.get_level_values("model"):
            continue
        r = tab.xs(ref, level="model")["RMSE"]
        tab[f"skill_vs_{ref.replace('pred_', '')}"] = 1 - tab["RMSE"] / tab.index.get_level_values("mask").map(r).to_numpy()
    return tab


def violation_rate(pred_ghi, ghi_cs, k_max) -> dict:
    """% nilai negatif dan % nilai > k_max·GHI_cs, SEBELUM clipping, untuk semua jam (malam: GHI_cs=0)."""
    p, cs = np.asarray(pred_ghi, float), np.asarray(ghi_cs, float)
    k = np.broadcast_to(np.asarray(k_max, float), p.shape)
    ok = np.isfinite(p) & np.isfinite(cs)
    p, cs, k = p[ok], cs[ok], k[ok]
    neg = p < -1e-9
    over = p > np.nan_to_num(k, nan=np.nanmax(k) if np.isfinite(k).any() else 0) * np.maximum(cs, 0) + 1e-9
    return {"pct_negatif": 100 * neg.mean(), "pct_di_atas_kmax": 100 * over.mean(),
            "pct_total": 100 * (neg | over).mean(), "n": int(len(p))}


SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def stratify(df: pd.DataFrame, pred_cols, by: str, mask_name: str = "valid") -> pd.DataFrame:
    """RMSE per strata: by ∈ {'jam', 'musim', 'kc_bin'}."""
    m = masks(df)[mask_name] & df[list(pred_cols)].notna().all(axis=1)
    d = df[m].copy()
    if by == "jam":
        key = d["target_hour_wib"]
    elif by == "musim":
        key = d["target_day"].dt.month.map(SEASON)
    elif by == "kc_bin":
        key = pd.cut(d["y_kc"], [-np.inf, 0.4, 0.8, np.inf], labels=["berawan (<0,4)", "sebagian (0,4–0,8)", "cerah (>0,8)"])
    else:
        raise ValueError(by)
    se = pd.DataFrame({c: (d[c] - d["y_ghi"]) ** 2 for c in pred_cols})
    res = np.sqrt(se.groupby(key, observed=True).mean())
    res["n"] = d.groupby(key, observed=True).size()
    return res


# ---------------------------------------------------------------- signifikansi (T35.5)
def daily_sq_error(df: pd.DataFrame, pred_cols, mask: pd.Series) -> pd.DataFrame:
    """Rata-rata galat kuadrat per hari target (mengurangi autokorelasi antar-jam)."""
    m = mask & df[list(pred_cols)].notna().all(axis=1)
    d = df[m]
    se = pd.DataFrame({c: (d[c] - d["y_ghi"]) ** 2 for c in pred_cols})
    return se.groupby(d["target_day"]).mean()


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, max_lag: int | None = None) -> dict:
    """DM dengan varians HAC Newey–West pada selisih loss harian d = a − b.
    stat < 0 → A lebih akurat. p dua sisi (normal)."""
    from scipy.stats import norm
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    L = int(np.floor(n ** (1 / 3))) if max_lag is None else max_lag
    dc = d - d.mean()
    var = np.sum(dc ** 2) / n
    for k in range(1, L + 1):
        var += 2 * (1 - k / (L + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    if var <= 0:
        return {"n_hari": n, "mean_diff": float(d.mean()), "DM": 0.0, "p": 1.0}
    stat = d.mean() / np.sqrt(var / n)
    return {"n_hari": n, "mean_diff": float(d.mean()), "DM": float(stat), "p": float(2 * norm.sf(abs(stat)))}


def block_bootstrap(df: pd.DataFrame, pred_cols, mask: pd.Series, refs=("pred_clim", "pred_gfs_raw"),
                    n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """CI 95% RMSE & skill dengan bootstrap blok per HARI (hari diambil ulang dengan pengembalian)."""
    m = mask & df[list(pred_cols)].notna().all(axis=1)
    d = df[m]
    se = pd.DataFrame({c: (d[c] - d["y_ghi"]) ** 2 for c in pred_cols})
    day_sum, day_n = se.groupby(d["target_day"]).sum(), d.groupby("target_day").size()
    rng = np.random.default_rng(seed)
    nd = len(day_n)
    rec = {f"RMSE_{c}": [] for c in pred_cols}
    for r in refs:
        for c in pred_cols:
            rec[f"skill_{c}_vs_{r}"] = []
    for _ in range(n_boot):
        pick = rng.integers(0, nd, nd)
        rmse = np.sqrt(day_sum.iloc[pick].sum() / day_n.iloc[pick].sum())
        for c in pred_cols:
            rec[f"RMSE_{c}"].append(rmse[c])
        for r in refs:
            for c in pred_cols:
                rec[f"skill_{c}_vs_{r}"].append(1 - rmse[c] / rmse[r])
    out = pd.DataFrame({k: np.percentile(v, [2.5, 50, 97.5]) for k, v in rec.items()}, index=["ci_low", "median", "ci_high"]).T
    return out

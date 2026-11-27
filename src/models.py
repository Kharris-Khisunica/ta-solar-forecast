"""Baseline, model ML, physical consistency, dan matriks ablasi (T13, T25, T29–T32).

Ablasi 2×2×2 (keputusan rencana): target {GHI, k_c} × fitur fisika {tanpa, dengan} × clipping {tanpa, dengan}.
Konfigurasi utama: target k_c + fitur fisika + clipping k_max(zenit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from clearsky import clip, kmax_for_zenith

# ---------------------------------------------------------------- daftar fitur (TK2: dibekukan di sini)
GEOM = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
PHYS = ["ghi_cs", "ghi_extra", "cos_zenith", "kc_clim", "gfs_kc_nwp", "gfs_kc_nwp_3x3"]   # fitur "fisika"
LAG = ["kc_same_hour_lag", "same_hour_from_d2", "kc_mean_d1_to_issue", "kc_mean_d1_morning", "kc_mean_d2",
       "kc_at_issue", "temperature_at_issue", "relative_humidity_at_issue", "dew_point_at_issue",
       "aerosol_optical_depth_at_issue"]
NWP = ["gfs_ghi_nwp", "gfs_ghi_nwp_3x3", "gfs_tcdc_atm", "gfs_tcdc_low", "gfs_tcdc_mid", "gfs_tcdc_high",
       "gfs_tcdc_atm_3x3", "gfs_tmp_2m", "gfs_rh_2m", "gfs_pwat_atm1", "gfs_cape_sfc"]


def feature_list(df: pd.DataFrame, with_phys: bool = True) -> list[str]:
    cols = GEOM + LAG + NWP + (PHYS if with_phys else [])
    return [c for c in cols if c in df.columns]


def kmax_array(df: pd.DataFrame, kz: dict) -> np.ndarray:
    k = kmax_for_zenith(df["zenith"], kz["edges"], kz["values"])
    return np.where(np.isfinite(k), k, max(kz["values"]))


def finalize(pred_ghi, df: pd.DataFrame, kz: dict, do_clip: bool) -> np.ndarray:
    """Malam (GHI_cs = 0) → 0; opsional clipping ke [0, k_max(z)·GHI_cs]."""
    p = np.asarray(pred_ghi, float).copy()
    cs = df["ghi_cs"].to_numpy()
    if do_clip:
        p = clip(p, cs, kmax_array(df, kz))
    p[cs <= 0] = 0.0 if do_clip else p[cs <= 0]
    return p


# ---------------------------------------------------------------- baseline
def baselines(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Klimatologi, persistence (aturan #8), GFS mentah, MOS regresi linear k_c. Tanpa clipping."""
    from sklearn.linear_model import LinearRegression
    out = pd.DataFrame(index=df.index)
    cs = df["ghi_cs"].fillna(0)
    out["pred_clim"] = df["kc_clim"].fillna(0) * cs
    out["pred_persist"] = df["kc_same_hour_lag"] * cs
    out["pred_gfs_raw"] = df["gfs_ghi_nwp"]
    mos_x = ["gfs_kc_nwp", "cos_zenith", "hour_sin", "hour_cos"]
    tr = train_mask & df[mos_x + ["y_kc"]].notna().all(axis=1)
    if tr.sum() < 100:
        raise ValueError(f"MOS: hanya {int(tr.sum())} baris latih dengan GFS lengkap — cek features_all (notebook 06)")
    lr = LinearRegression().fit(df.loc[tr, mos_x], df.loc[tr, "y_kc"])
    ok = df[mos_x].notna().all(axis=1)
    out["pred_mos"] = np.nan
    out.loc[ok, "pred_mos"] = lr.predict(df.loc[ok, mos_x]) * cs[ok]
    for c in out:
        out.loc[df["ghi_cs"].fillna(0) <= 0, c] = 0.0
    return out


# ---------------------------------------------------------------- model ML
def make_model(kind: str, params: dict, seed: int):
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**params, random_state=seed, n_jobs=-1, tree_method="hist")
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(**params, random_state=seed, n_jobs=-1))
    raise ValueError(kind)


def fit_model(kind, params, df, feats, train_mask, target: str, seed: int):
    """Latih pada jam siang data latih; kembalikan model terlatih."""
    ycol = "y_kc" if target == "kc" else "y_ghi"
    tr = train_mask & df[ycol].notna()
    return make_model(kind, params, seed).fit(df.loc[tr, feats], df.loc[tr, ycol])


def fit_predict(kind, params, df, feats, train_mask, target: str, seed: int) -> np.ndarray:
    """Latih pada jam siang data latih; kembalikan prediksi GHI mentah (sebelum clipping) untuk semua baris."""
    m = fit_model(kind, params, df, feats, train_mask, target, seed)
    yhat = m.predict(df[feats])
    return yhat * df["ghi_cs"].fillna(0).to_numpy() if target == "kc" else yhat


def search_space(trial, kind: str) -> dict:
    if kind == "xgb":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True)}
    return {"n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0)}


def tune(kind, df, feats, train_mask, val_mask, target, kz, seed, n_trials):
    """Optuna, metrik = RMSE GHI di validasi (mask valid) SETELAH clipping (konfigurasi utama)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yv = df.loc[val_mask, "y_ghi"].to_numpy()

    def obj(trial):
        p = search_space(trial, kind)
        pred = finalize(fit_predict(kind, p, df, feats, train_mask, target, seed), df, kz, do_clip=True)
        return float(np.sqrt(np.nanmean((pred[val_mask.to_numpy()] - yv) ** 2)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials)
    return study.best_params, study.trials_dataframe()


def ablation(df, train_mask, kind, params, kz, seed) -> pd.DataFrame:
    """8 konfigurasi; kolom prediksi GHI bernama abl__<target>__<fisika>__<clip>, plus versi mentah."""
    out = pd.DataFrame(index=df.index)
    for target in ("ghi", "kc"):
        for phys in (False, True):
            raw = fit_predict(kind, params, df, feature_list(df, phys), train_mask, target, seed)
            name = f"{target}__{'fis' if phys else 'nofis'}"
            out[f"raw__{name}"] = raw
            for do_clip in (False, True):
                out[f"abl__{name}__{'clip' if do_clip else 'noclip'}"] = finalize(raw, df, kz, do_clip)
    return out


# ---------------------------------------------------------------- simpan / muat
def save_model(model, kind: str, path_noext) -> str:
    if kind == "xgb":
        p = f"{path_noext}.json"; model.save_model(p)
    else:
        import joblib
        p = f"{path_noext}.joblib"; joblib.dump(model, p)
    return p


def load_model(kind: str, path_noext):
    if kind == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(); m.load_model(f"{path_noext}.json"); return m
    import joblib
    return joblib.load(f"{path_noext}.joblib")

"""Tabel fitur (issue_date, target_hour) + audit kebocoran (T12, dipakai T13/T19/T24).

Konvensi
--------
- Data per jam berlabel period-ending UTC (notebook 03): label L mewakili interval (L-1h, L].
- Hari target D (tanggal WIB), jam target h = 1..24 WIB period-ending: interval ((h-1):00, h:00] WIB hari D.
  Label UTC jam target = D 00:00 WIB + h jam - 7 jam.
- Waktu penerbitan = D-1 12:00 WIB = D-1 05:00 UTC (configs/config.yaml: forecast.issue_time_utc).
  Data berlabel <= waktu penerbitan dianggap tersedia (latensi satelit diabaikan; ditulis sebagai asumsi).
- Keputusan #8 (definisi lag anti-kebocoran):
  * k_c jam yang sama: dari D-1 bila label jam itu <= penerbitan, selain itu dari D-2.
  * rata-rata k_c D-1 hanya s.d. penerbitan; rata-rata k_c D-2 penuh.
- Setiap fitur berbasis data mencatat kolom `ftm__<fitur>` (feature_time_max) = label data terakhir yang dipakai.
  Fitur deterministik (geometri, clear-sky Ineichen, klimatologi data latih) tidak punya ftm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WIB = pd.Timedelta(hours=7)
H = pd.Timedelta(hours=1)
FTM_PREFIX = "ftm__"


def wib_day_hour(labels_utc: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Untuk label period-ending UTC → (tanggal WIB hari jam itu, jam WIB period-ending 1..24)."""
    t = pd.DatetimeIndex(labels_utc).tz_convert("UTC") + WIB - pd.Timedelta(minutes=1)
    day = t.tz_localize(None).normalize()
    hour_end = t.hour.to_numpy() + 1
    return pd.DatetimeIndex(day), hour_end


def target_index(first_day: str, last_day: str, issue_hour_wib: int = 12) -> pd.DataFrame:
    """Kerangka baris: satu baris per (hari target D, jam 1..24 WIB)."""
    days = pd.date_range(first_day, last_day, freq="D")
    d = np.repeat(days.to_numpy(), 24)
    h = np.tile(np.arange(1, 25), len(days))
    df = pd.DataFrame({"target_day": pd.DatetimeIndex(d), "target_hour_wib": h})
    df["target_time_utc"] = (df["target_day"] + pd.to_timedelta(df["target_hour_wib"], unit="h")
                             - WIB).dt.tz_localize("UTC")
    df["issue_time_utc"] = (df["target_day"] - pd.Timedelta(days=1) + pd.Timedelta(hours=issue_hour_wib)
                            - WIB).dt.tz_localize("UTC")
    return df


def _lookup(series: pd.Series, labels) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(labels)).to_numpy()


def add_geometry(df: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Fitur deterministik jam target: zenit, GHI_cs (Ineichen), radiasi ekstraterestrial, siklus jam & tahun."""
    import pvlib
    t = df["target_time_utc"]
    df["zenith"] = _lookup(hourly["solar_zenith_angle"], t)
    df["cos_zenith"] = np.cos(np.deg2rad(df["zenith"]))
    df["ghi_cs"] = _lookup(hourly["ghi_cs"], t)
    df["ghi_extra"] = pvlib.irradiance.get_extra_radiation(pd.DatetimeIndex(t)).to_numpy()
    df["hour_sin"] = np.sin(2 * np.pi * df["target_hour_wib"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["target_hour_wib"] / 24)
    doy = df["target_day"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_targets(df: pd.DataFrame, hourly: pd.DataFrame, zenith_max: float, interp_max: float = 1 / 6) -> pd.DataFrame:
    """Target dan mask evaluasi (BUKAN fitur)."""
    t = df["target_time_utc"]
    df["y_ghi"] = _lookup(hourly["ghi"], t)
    df["y_kc"] = _lookup(hourly["kc"], t)
    df["is_day"] = df["zenith"] < zenith_max
    df["interp_frac"] = _lookup(hourly["interp_frac"], t)
    df["eval_valid"] = df["is_day"] & (df["interp_frac"] <= interp_max + 1e-9) & df["y_ghi"].notna()
    return df


def add_lag_features(df: pd.DataFrame, hourly: pd.DataFrame,
                     obs_vars=("air_temperature", "temperature", "relative_humidity", "dew_point",
                               "aerosol_optical_depth", "aod")) -> pd.DataFrame:
    """Fitur observasi NSRDB yang tersedia saat penerbitan (keputusan #8), dengan ftm per fitur."""
    kc = hourly["kc"]
    t, issue = df["target_time_utc"], df["issue_time_utc"]

    # 1) k_c jam yang sama: D-1 bila sudah teramati saat penerbitan, selain itu D-2
    lag1 = t - pd.Timedelta(days=1)
    same_lbl = lag1.where(lag1 <= issue, t - pd.Timedelta(days=2))
    df["kc_same_hour_lag"] = _lookup(kc, same_lbl)
    df[FTM_PREFIX + "kc_same_hour_lag"] = same_lbl
    df["same_hour_from_d2"] = (lag1 > issue).astype(int)        # penanda dari mana nilainya diambil

    # 2) Agregat harian per tanggal WIB
    wday, whour = wib_day_hour(hourly.index)
    tmp = pd.DataFrame({"kc": kc.to_numpy(), "day": wday, "hour": whour}, index=hourly.index)
    d1_upto_issue = tmp[tmp["hour"] <= 12].groupby("day")["kc"].mean()          # s.d. 12.00 WIB
    morning = tmp[(tmp["hour"] >= 8) & (tmp["hour"] <= 12)].groupby("day")["kc"].mean()
    full_day = tmp.groupby("day")["kc"].mean()

    d1 = df["target_day"] - pd.Timedelta(days=1)
    d2 = df["target_day"] - pd.Timedelta(days=2)
    df["kc_mean_d1_to_issue"] = d1_upto_issue.reindex(d1).to_numpy()
    df[FTM_PREFIX + "kc_mean_d1_to_issue"] = issue
    df["kc_mean_d1_morning"] = morning.reindex(d1).to_numpy()
    df[FTM_PREFIX + "kc_mean_d1_morning"] = issue
    df["kc_mean_d2"] = full_day.reindex(d2).to_numpy()
    df[FTM_PREFIX + "kc_mean_d2"] = (d2 + pd.Timedelta(days=1) - WIB).dt.tz_localize("UTC")  # 24.00 WIB D-2

    # 3) Observasi terakhir sebelum penerbitan (jam yang berakhir 12.00 WIB D-1)
    df["kc_at_issue"] = _lookup(kc, issue)
    df[FTM_PREFIX + "kc_at_issue"] = issue
    for v in obs_vars:
        if v in hourly:
            df[f"{v}_at_issue"] = _lookup(hourly[v], issue)
            df[FTM_PREFIX + f"{v}_at_issue"] = issue
    return df


def add_climatology(df: pd.DataFrame, hourly: pd.DataFrame, train_years, zenith_max: float,
                    interp_max: float = 1 / 6) -> pd.DataFrame:
    """k_c klimatologi (bulan × jam WIB) dari data latih saja → deterministik untuk semua baris."""
    wday, whour = wib_day_hour(hourly.index)
    ok = (hourly["solar_zenith_angle"] < zenith_max) & (hourly["interp_frac"] <= interp_max + 1e-9)
    ok &= pd.Series(wday.year, index=hourly.index).isin(train_years)
    tr = pd.DataFrame({"kc": hourly["kc"], "m": wday.month, "h": whour}, index=hourly.index)[ok.to_numpy()]
    clim = tr.groupby(["m", "h"])["kc"].mean()
    key = pd.MultiIndex.from_arrays([df["target_day"].dt.month, df["target_hour_wib"]])
    df["kc_clim"] = clim.reindex(key).to_numpy()
    return df


def ftm_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FTM_PREFIX)]


def leakage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per fitur berbasis data: jumlah baris dengan feature_time_max > issue_time."""
    rows = []
    for c in ftm_columns(df):
        ftm = pd.to_datetime(df[c], utc=True)
        bad = (ftm > df["issue_time_utc"])
        rows.append({"fitur": c[len(FTM_PREFIX):], "baris_bocor": int(bad.sum()),
                     "maks_selisih_jam": float(((ftm - df["issue_time_utc"]) / H).max())})
    return pd.DataFrame(rows)


def assert_no_leakage(df: pd.DataFrame) -> None:
    rep = leakage_report(df)
    assert len(rep) > 0, "Tidak ada kolom ftm__ — audit kebocoran tidak bisa dijalankan"
    bad = rep[rep["baris_bocor"] > 0]
    assert bad.empty, f"KEBOCORAN: {bad.to_dict('records')}"


def build_feature_table(hourly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    years = cfg["data"]["years"]
    zmax = cfg["physics"]["zenith_max_deg"]
    issue_h_wib = (int(cfg["forecast"]["issue_time_utc"].split(":")[0]) + 7) % 24
    df = target_index(f"{years[0]}-01-01", f"{years[-1]}-12-31", issue_h_wib)
    df = add_geometry(df, hourly)
    df = add_targets(df, hourly, zmax)
    df = add_lag_features(df, hourly)
    df = add_climatology(df, hourly, cfg["data"]["split"]["train"], zmax)
    split = {y: name for name, ys in cfg["data"]["split"].items() for y in ys}
    df["split"] = df["target_day"].dt.year.map(split)
    return df

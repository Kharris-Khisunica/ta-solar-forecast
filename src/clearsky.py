"""Clear-sky, indeks kecerahan k_c, dan clipping fisis (T10, T11, dipakai ulang di T29).

Semua agregasi per jam memakai aturan yang SAMA dengan notebook 03:
waktu efektif = timestamp NSRDB + delta_min, lalu resample('1h', closed='right', label='right').
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOURLY_RULE = dict(rule="1h", closed="right", label="right")


def kc(ghi, ghi_cs, zenith, zenith_max: float = 85.0):
    """k_c = GHI / GHI_cs untuk zenit < zenith_max dan GHI_cs > 0; selain itu NaN (bukan 0)."""
    ghi, ghi_cs, zenith = (np.asarray(a, dtype=float) for a in (ghi, ghi_cs, zenith))
    ok = (zenith < zenith_max) & (ghi_cs > 0) & np.isfinite(ghi) & np.isfinite(ghi_cs)
    out = np.full(ghi.shape, np.nan)
    np.divide(ghi, ghi_cs, out=out, where=ok)
    return out


def clip(ghi_pred, ghi_cs, k_max: float):
    """Clip prediksi GHI ke [0, k_max · GHI_cs]. Malam (GHI_cs = 0) → 0."""
    ghi_pred, ghi_cs = np.asarray(ghi_pred, dtype=float), np.asarray(ghi_cs, dtype=float)
    upper = k_max * np.maximum(ghi_cs, 0.0)
    return np.clip(ghi_pred, 0.0, upper)


def ineichen_hourly(raw_index: pd.DatetimeIndex, lat: float, lon: float, altitude: float,
                    delta_min: int = 0) -> pd.DataFrame:
    """Ineichen-Perez (pvlib, Linke turbidity klimatologi bulanan) dihitung pada waktu efektif
    setiap sampel 10 menit, lalu diagregasi per jam dengan aturan yang sama seperti GHI observasi."""
    import pvlib

    t_eff = pd.DatetimeIndex(raw_index) + pd.Timedelta(minutes=delta_min)
    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=altitude)
    cs = loc.get_clearsky(t_eff, model="ineichen")
    return cs.resample(**HOURLY_RULE).mean().add_prefix("ineichen_")


def kmax_for_zenith(zenith, edges, values):
    """k_max bergantung zenit: values[i] berlaku untuk zenit di (edges[i], edges[i+1]].
    Zenit di luar rentang -> NaN (tidak ada batas atas yang didefinisikan; malam ditangani clip via GHI_cs=0)."""
    zenith = np.asarray(zenith, dtype=float)
    edges, values = np.asarray(edges, dtype=float), np.asarray(values, dtype=float)
    assert len(values) == len(edges) - 1
    idx = np.searchsorted(edges, zenith, side="left") - 1
    out = np.full(zenith.shape, np.nan)
    ok = (idx >= 0) & (idx < len(values)) & np.isfinite(zenith)
    out[ok] = values[idx[ok]]
    return out

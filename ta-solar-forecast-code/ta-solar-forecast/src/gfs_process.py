"""Pemrosesan GFS 0,25° (NCAR d084001) menjadi fitur per jam (T18, T24).

Alur
----
1. extract_files(): baca setiap pesan GRIB2 (run 00 UTC) dengan eccodes → tabel panjang
   (init, start_h, end_h, stepType, nama, nilai titik terdekat, rata-rata 3×3).
   Parameter dikenali dari kode numerik GRIB2 (discipline/kategori/nomor + jenis level),
   BUKAN dari shortName (shortName berbeda antar versi eccodes).
2. deaverage(): rata-rata GFS direset tiap 6 jam → blok 3 jam:
       (e-3, e] langsung bila tersedia;  bila tidak: 2·avg(e-6, e) − avg(e-6, e-3).
3. to_hourly(): blok 3 jam → per jam (period-ending).
     - DSWRF via k_c: k_c_blok = DSWRF_blok / mean(GHI_cs 3 jam), lalu GHI_nwp(jam) = k_c_blok · GHI_cs(jam).
     - Tutupan awan (avg): nilai blok dipakai untuk ketiga jamnya.
     - Variabel instan: interpolasi linear antar waktu valid 3-jaman.
4. merge_into_table(): gabung ke tabel (hari target D, jam) memakai run 00 UTC D-1,
   dengan ftm = init + latensi GFS (config forecast.gfs_latency_h).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

H = pd.Timedelta(hours=1)

# Kode WMO standar + kode lokal NCEP (GFS menyimpan DSWRF sebagai 0/4/192)
PARAM_NAMES = {(0, 4, 7): "dswrf", (0, 4, 192): "dswrf", (0, 6, 1): "tcdc", (0, 0, 0): "tmp", (0, 1, 1): "rh",
               (0, 1, 3): "pwat", (0, 7, 6): "cape"}
LEVEL_NAMES = {1: "sfc", 10: "atm", 200: "atm1", 214: "low", 224: "mid", 234: "high", 211: "bl"}


def normalize_names(long: pd.DataFrame) -> pd.DataFrame:
    """Petakan ulang nama generik (dari cache lama) ke nama standar, mis. d0c4n192_sfc → dswrf_sfc."""
    import re
    def fix(n):
        m = re.fullmatch(r"d(\d+)c(\d+)n(\d+)_(.+)", n)
        if not m:
            return n
        base = PARAM_NAMES.get((int(m[1]), int(m[2]), int(m[3])))
        return f"{base}_{m[4]}" if base else n
    long = long.copy()
    long["name"] = long["name"].map(fix)
    return long


def param_name(disc, cat, num, ltype, level) -> str:
    base = PARAM_NAMES.get((disc, cat, num), f"d{disc}c{cat}n{num}")
    lv = f"{level}m" if ltype == 103 else LEVEL_NAMES.get(ltype, f"t{ltype}l{level}")
    return f"{base}_{lv}"


# --------------------------------------------------------------------------- 1. ekstraksi
def _read_messages(path: Path, lat: float, lon: float, half_deg: float = 0.25):
    import eccodes as ec
    out = []
    lon360 = lon % 360
    with open(path, "rb") as fh:
        while (h := ec.codes_grib_new_from_file(fh)) is not None:
            try:
                g = lambda k: ec.codes_get(h, k)  # noqa: E731
                gl = lambda k: ec.codes_get_long(h, k)  # noqa: E731  (kode numerik, bukan label teks)
                lats = ec.codes_get_array(h, "latitudes")
                lons = ec.codes_get_array(h, "longitudes") % 360
                vals = ec.codes_get_values(h).astype(float)
                if gl("bitmapPresent"):
                    vals[vals == g("missingValue")] = np.nan
                d2 = (lats - lat) ** 2 + (lons - lon360) ** 2
                i0 = int(np.argmin(d2))
                near = (np.abs(lats - lats[i0]) <= half_deg + 1e-6) & (np.abs(lons - lons[i0]) <= half_deg + 1e-6)
                date, time = gl("dataDate"), gl("dataTime")
                out.append({
                    "init": pd.Timestamp(f"{date:08d}{time:04d}", tz="UTC"),
                    "start_h": gl("startStep"), "end_h": gl("endStep"),
                    "step_type": str(g("stepType")),
                    "name": param_name(gl("discipline"), gl("parameterCategory"), gl("parameterNumber"),
                                       gl("typeOfFirstFixedSurface"), gl("level")),
                    "v_point": float(vals[i0]), "v_3x3": float(np.nanmean(vals[near])),
                    "grid_lat": float(lats[i0]), "grid_lon": float(lons[i0]), "n_3x3": int(near.sum()),
                })
            finally:
                ec.codes_release(h)
    return out


def run00_files(year_dir: Path) -> list[Path]:
    """File GRIB run 00 UTC: gfs.0p25.YYYYMMDD00.fFFF.grib2."""
    return sorted(p for p in Path(year_dir).rglob("gfs.0p25.*.grib2") if p.name.split(".")[2].endswith("00"))


def extract_files(files, lat: float, lon: float, progress_every: int = 500) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(files, 1):
        rows += _read_messages(p, lat, lon)
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(files)} file")
    df = pd.DataFrame(rows)
    df["init"] = pd.to_datetime(df["init"], utc=True)
    return df.drop_duplicates(["init", "name", "step_type", "start_h", "end_h"])


# --------------------------------------------------------------------------- 2. de-averaging
def deaverage(long: pd.DataFrame, value_col: str = "v_point") -> pd.DataFrame:
    """Rata-rata (stepType avg) → blok 3 jam (block_start_h, block_end_h, value). Nilai negatif kecil → 0."""
    avg = long[long["step_type"] == "avg"]
    out = []
    for (init, name), g in avg.groupby(["init", "name"]):
        d = {(int(s), int(e)): v for s, e, v in zip(g["start_h"], g["end_h"], g[value_col])}
        ends = sorted({e for _, e in d})
        for e in ends:
            if (e - 3, e) in d:
                v = d[(e - 3, e)]
            elif (e - 6, e) in d and (e - 6, e - 3) in d:
                v = 2 * d[(e - 6, e)] - d[(e - 6, e - 3)]
            else:
                continue                      # blok tidak bisa diturunkan (mis. lead hilang)
            out.append({"init": init, "name": name, "block_start_h": e - 3, "block_end_h": e, "value": v})
    res = pd.DataFrame(out, columns=["init", "name", "block_start_h", "block_end_h", "value"])
    res["init"] = pd.to_datetime(res["init"], utc=True)          # pastikan tipe tanggal (beda versi pandas)
    if len(res):
        res.loc[res["name"].str.startswith("dswrf"), "value"] = res.loc[res["name"].str.startswith("dswrf"), "value"].clip(lower=0)
        res.loc[res["name"].str.startswith("tcdc"), "value"] = res.loc[res["name"].str.startswith("tcdc"), "value"].clip(0, 100)
    return res


# --------------------------------------------------------------------------- 3. per jam
def blocks_to_hourly(blocks: pd.DataFrame) -> pd.DataFrame:
    """Setiap blok (s, e] → tiga label jam period-ending: init+s+1 .. init+e."""
    reps = []
    for k in (1, 2, 3):
        b = blocks.copy()
        b["label"] = b["init"] + pd.to_timedelta(b["block_start_h"] + k, unit="h")
        reps.append(b)
    return pd.concat(reps, ignore_index=True)


def instant_to_hourly(long: pd.DataFrame, value_col: str = "v_point") -> pd.DataFrame:
    """Variabel instan 3-jaman → per jam via interpolasi linear di dalam rentang lead yang ada."""
    inst = long[long["step_type"] == "instant"]
    out = []
    for (init, name), g in inst.groupby(["init", "name"]):
        g = g.sort_values("end_h")
        hrs = np.arange(g["end_h"].min(), g["end_h"].max() + 1)
        v = np.interp(hrs, g["end_h"].to_numpy(), g[value_col].to_numpy())
        out.append(pd.DataFrame({"init": init, "name": name, "label": init + pd.to_timedelta(hrs, unit="h"), "value": v}))
    if not out:
        return pd.DataFrame(columns=["init", "name", "label", "value"])
    res = pd.concat(out, ignore_index=True)
    res["init"] = pd.to_datetime(res["init"], utc=True)
    res["label"] = pd.to_datetime(res["label"], utc=True)
    return res


def to_hourly(long: pd.DataFrame, ghi_cs: pd.Series, value_col: str = "v_point", cs_min: float = 10.0,
              suffix: str = "") -> pd.DataFrame:
    """Tabel lebar per (init, label): GHI_nwp & k_c_nwp (dari DSWRF), tutupan awan, variabel instan."""
    if not long["name"].str.startswith("dswrf").any():
        raise ValueError("DSWRF tidak ditemukan. Nama parameter yang ada: "
                         f"{sorted(long['name'].unique())} — tambahkan kodenya ke PARAM_NAMES")
    blocks = deaverage(long, value_col)
    hb = blocks_to_hourly(blocks)
    hb["cs"] = ghi_cs.reindex(pd.DatetimeIndex(hb["label"])).to_numpy()
    hb["block_id"] = hb["init"].astype(str) + "|" + hb["name"] + "|" + hb["block_end_h"].astype(str)
    hb["cs_block_mean"] = hb.groupby("block_id")["cs"].transform("mean")

    wide = []
    sw = hb[hb["name"].str.startswith("dswrf")].copy()
    if len(sw):
        kc = np.where(sw["cs_block_mean"] >= cs_min, sw["value"] / sw["cs_block_mean"], np.nan)
        sw["kc_nwp"] = kc
        sw["ghi_nwp"] = np.where(np.isfinite(kc), kc * sw["cs"], sw["value"])
        sw["dswrf_block"] = sw["value"]
        wide.append(sw.set_index(["init", "label"])[["ghi_nwp", "kc_nwp", "dswrf_block"]])
    other = hb[~hb["name"].str.startswith("dswrf")]
    if len(other):
        wide.append(other.pivot_table(index=["init", "label"], columns="name", values="value"))
    inst = instant_to_hourly(long, value_col)
    if len(inst):
        wide.append(inst.pivot_table(index=["init", "label"], columns="name", values="value"))
    res = pd.concat(wide, axis=1).sort_index()
    res.index = pd.MultiIndex.from_arrays([pd.to_datetime(res.index.get_level_values(0), utc=True),
                                           pd.to_datetime(res.index.get_level_values(1), utc=True)],
                                          names=["init", "label"])
    res.columns = [f"gfs_{c}{suffix}" for c in res.columns]
    return res


# --------------------------------------------------------------------------- 4. gabung ke tabel fitur
def merge_into_table(table: pd.DataFrame, gfs_hourly: pd.DataFrame, latency_h: float, run_hour: int = 0) -> pd.DataFrame:
    """Baris (D, jam) memakai run `run_hour` UTC D-1. ftm GFS = init + latensi."""
    t = table.copy()
    t["gfs_init"] = (t["target_day"] - pd.Timedelta(days=1) + pd.Timedelta(hours=run_hour)).dt.tz_localize("UTC")
    key = pd.MultiIndex.from_arrays([t["gfs_init"], t["target_time_utc"]], names=["init", "label"])
    g = gfs_hourly.reindex(key)
    for c in gfs_hourly.columns:
        t[c] = g[c].to_numpy()
        t["ftm__" + c] = t["gfs_init"] + pd.Timedelta(hours=latency_h)
    return t


def alignment_by_shift(ghi_nwp_by_label: pd.Series, ghi_obs: pd.Series, mask: pd.Series, shifts=range(-2, 3)) -> pd.Series:
    """Korelasi GHI_nwp vs GHI observasi untuk pergeseran k jam (jam siang). Maksimum harus di 0."""
    idx = mask[mask].index
    return pd.Series({k: ghi_nwp_by_label.reindex(idx).corr(ghi_obs.shift(k).reindex(idx)) for k in shifts}, name="r")

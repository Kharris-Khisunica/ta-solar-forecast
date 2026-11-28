"""LSTM encoder–decoder untuk ramalan k_c 24 jam (T34) + Shapley eksak tingkat kelompok (T40).

Encoder : 48 jam observasi NSRDB sampai waktu penerbitan (label ≤ 12.00 WIB D-1).
Decoder : 24 jam hari target D, input = fitur hari D yang tersedia saat penerbitan
          (GFS run 00 UTC D-1, klimatologi data latih, geometri deterministik).
Keluaran: k_c per jam; loss hanya pada jam siang. Normalisasi di-fit pada hari latih saja.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

H = pd.Timedelta(hours=1)
SEQ_LEN = 48

SEQ_GROUPS = {"kc_f": "lag_observasi", "kc_ok": "lag_observasi", "cos_zen": "geometri_waktu",
              "hour_sin": "geometri_waktu", "hour_cos": "geometri_waktu",
              "temperature": "termodinamika", "relative_humidity": "termodinamika",
              "aerosol_optical_depth": "aerosol", "aod": "aerosol"}
TGT_GROUPS = {"gfs_kc_nwp_3x3": "nwp_radiasi", "gfs_kc_nwp": "nwp_radiasi",
              "gfs_tcdc_atm": "nwp_awan", "gfs_tcdc_low": "nwp_awan", "gfs_tcdc_mid": "nwp_awan",
              "gfs_tcdc_high": "nwp_awan", "kc_clim": "klimatologi",
              "gfs_pwat_atm1": "termodinamika", "gfs_rh_2m": "termodinamika", "gfs_cape_sfc": "termodinamika",
              "gfs_tmp_2m": "termodinamika", "cos_zenith": "geometri_waktu", "hour_sin": "geometri_waktu",
              "hour_cos": "geometri_waktu", "doy_sin": "geometri_waktu", "doy_cos": "geometri_waktu"}


# ------------------------------------------------------------------ data
def build_arrays(hourly: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Susun tensor per hari target. Mengembalikan dict berisi array & metadata."""
    tab = table.sort_values(["target_day", "target_hour_wib"])
    days = pd.DatetimeIndex(tab["target_day"].drop_duplicates())
    nd = len(days)
    assert len(tab) == nd * 24, "tabel harus 24 baris per hari"

    # --- sekuens observasi
    obs = pd.DataFrame(index=hourly.index)
    kc = hourly["kc"]
    obs["kc_f"] = kc.fillna(0.0)
    obs["kc_ok"] = kc.notna().astype(float)
    obs["cos_zen"] = np.clip(np.cos(np.deg2rad(hourly["solar_zenith_angle"])), 0, None)
    wib_h = (hourly.index.hour + 7) % 24
    obs["hour_sin"] = np.sin(2 * np.pi * wib_h / 24)
    obs["hour_cos"] = np.cos(2 * np.pi * wib_h / 24)
    for c in ("temperature", "relative_humidity", "aerosol_optical_depth", "aod"):
        if c in hourly:
            obs[c] = hourly[c]
    seq_ch = list(obs.columns)
    issue = pd.DatetimeIndex(tab.groupby("target_day")["issue_time_utc"].first().loc[days])
    offs = pd.to_timedelta(np.arange(-(SEQ_LEN - 1), 1), unit="h")
    base = issue.tz_convert("UTC").tz_localize(None).to_numpy()
    labels = pd.DatetimeIndex((base[:, None] + offs.to_numpy()[None, :]).ravel()).tz_localize("UTC")
    xs = obs.reindex(labels).to_numpy(float).reshape(nd, SEQ_LEN, len(seq_ch))

    # --- blok hari target
    tgt_ch = [c for c in TGT_GROUPS if c in tab.columns]
    t = tab[tgt_ch].copy()
    if "cos_zenith" in t:
        t["cos_zenith"] = t["cos_zenith"].clip(lower=0)
    xt = t.to_numpy(float).reshape(nd, 24, len(tgt_ch))

    y = tab["y_kc"].to_numpy(float).reshape(nd, 24)
    w_train = (tab["is_day"] & tab["y_kc"].notna()).to_numpy().reshape(nd, 24).astype(float)
    w_eval = (tab["eval_valid"] & tab["y_kc"].notna()).to_numpy().reshape(nd, 24).astype(float)
    split = tab.groupby("target_day")["split"].first().loc[days].to_numpy()
    return {"xs": xs, "xt": xt, "y": np.nan_to_num(y), "w_train": w_train, "w_eval": w_eval,
            "days": days, "split": split, "seq_ch": seq_ch, "tgt_ch": tgt_ch,
            "seq_last_label": labels[SEQ_LEN - 1::SEQ_LEN], "issue": issue,
            "row_index": tab.index.to_numpy().reshape(nd, 24)}


class Normalizer:
    """z-score per kanal, statistik dari hari LATIH saja; NaN → 0 (= rata-rata latih) setelah normalisasi."""

    def fit(self, A: dict, train_days: np.ndarray):
        self.ms = np.nanmean(A["xs"][train_days].reshape(-1, A["xs"].shape[2]), axis=0)
        self.ss = np.nanstd(A["xs"][train_days].reshape(-1, A["xs"].shape[2]), axis=0) + 1e-6
        self.mt = np.nanmean(A["xt"][train_days].reshape(-1, A["xt"].shape[2]), axis=0)
        self.st = np.nanstd(A["xt"][train_days].reshape(-1, A["xt"].shape[2]), axis=0) + 1e-6
        return self

    def save(self, path):
        np.savez(path, ms=self.ms, ss=self.ss, mt=self.mt, st=self.st)

    @classmethod
    def load(cls, path):
        z = np.load(path); n = cls()
        n.ms, n.ss, n.mt, n.st = z["ms"], z["ss"], z["mt"], z["st"]
        return n

    def transform(self, A: dict) -> tuple[np.ndarray, np.ndarray]:
        xs = np.nan_to_num((A["xs"] - self.ms) / self.ss)
        xt = np.nan_to_num((A["xt"] - self.mt) / self.st)
        return xs.astype(np.float32), xt.astype(np.float32)


# ------------------------------------------------------------------ model
def _torch():
    import torch
    return torch


def make_net(n_seq: int, n_tgt: int, hidden: int, layers: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            d = dropout if layers > 1 else 0.0
            self.enc = nn.LSTM(n_seq, hidden, layers, batch_first=True, dropout=d)
            self.dec = nn.LSTM(n_tgt + hidden, hidden, layers, batch_first=True, dropout=d)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, xs, xt):
            _, (h, c) = self.enc(xs)
            ctx = h[-1].unsqueeze(1).expand(-1, xt.shape[1], -1)
            out, _ = self.dec(torch.cat([xt, ctx], dim=2), (h, c))
            return self.head(self.drop(out)).squeeze(-1)

    return Seq2Seq()


def set_torch_seed(seed: int):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)   # A5: pilih algoritma deterministik bila ada
    torch.set_num_threads(max(1, torch.get_num_threads()))


def train(params: dict, xs, xt, y, w_tr, w_val, tr_idx, val_idx, seed: int, max_epochs: int = 150,
          patience: int = 12, verbose: bool = False):
    """Latih dengan early stopping pada loss validasi (jam siang valid). Kembalikan (net, riwayat)."""
    torch = _torch()
    set_torch_seed(seed)
    net = make_net(xs.shape[2], xt.shape[2], params["hidden"], params["layers"], params["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 0.0))
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    Xs, Xt, Y, Wtr, Wv = T(xs), T(xt), T(y), T(w_tr), T(w_val)
    tr = torch.as_tensor(tr_idx)
    va = torch.as_tensor(val_idx)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = math.inf, None, 0, []
    bs = int(params["batch"])
    for ep in range(max_epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = net(Xs[b], Xt[b])
            loss = (Wtr[b] * (pred - Y[b]) ** 2).sum() / Wtr[b].sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xs[va], Xt[va])
            vl = float(((Wv[va] * (pv - Y[va]) ** 2).sum() / Wv[va].sum().clamp(min=1)).item())
        hist.append(vl)
        if verbose:
            print(f"epoch {ep:3d} val {vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, hist


def predict(net, xs, xt, batch: int = 512) -> np.ndarray:
    torch = _torch()
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            out.append(net(torch.from_numpy(xs[i:i + batch]), torch.from_numpy(xt[i:i + batch])).numpy())
    return np.concatenate(out)


def search_space(trial) -> dict:
    return {"hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "layers": trial.suggest_int("layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch": trial.suggest_categorical("batch", [32, 64])}


# ------------------------------------------------------------------ Shapley eksak tingkat kelompok
# exact_group_shapley & tabular_value_fn dipindah ke xai.py (dipakai ketiga model); diekspor ulang di sini
# agar kode lama yang memanggil L.exact_group_shapley tetap jalan.
from xai import exact_group_shapley, tabular_value_fn  # noqa: E402,F401


def lstm_value_fn(net, xs, xt, hour_idx, seq_groups, tgt_groups, groups):
    """Koalisi → keluaran LSTM pada jam sampel; kelompok absen diganti 0 (= rata-rata latih ternormalisasi)."""
    seq_g = np.array([groups.index(g) if g in groups else -1 for g in seq_groups])
    tgt_g = np.array([groups.index(g) if g in groups else -1 for g in tgt_groups])

    def fn(subsets):
        res = []
        for s in subsets:
            ms = np.array([s[i] if i >= 0 else True for i in seq_g], dtype=np.float32)
            mt = np.array([s[i] if i >= 0 else True for i in tgt_g], dtype=np.float32)
            p = predict(net, (xs * ms).astype(np.float32), (xt * mt).astype(np.float32))
            res.append(p[np.arange(len(p)), hour_idx])
        return np.array(res)
    return fn


def tabular_value_fn(model, X: pd.DataFrame, background: pd.Series, fmap: dict, groups):
    """Koalisi → prediksi model tabular; fitur kelompok absen diganti rata-rata data latih."""
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


def load_net(path, params: dict, n_seq: int, n_tgt: int):
    """Muat LSTM tersimpan (models/lstm_best.pt) dengan hyperparameter yang sama."""
    torch = _torch()
    net = make_net(n_seq, n_tgt, params["hidden"], params["layers"], params["dropout"])
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def lstm_value_fn_days(net, xs, xt, seq_groups, tgt_groups, groups):
    """Seperti lstm_value_fn tetapi untuk SEMUA jam: keluaran [2^M, n_hari*24] (untuk peta per jam)."""
    seq_g = np.array([groups.index(g) if g in groups else -1 for g in seq_groups])
    tgt_g = np.array([groups.index(g) if g in groups else -1 for g in tgt_groups])

    def fn(subsets):
        res = []
        for s in subsets:
            ms = np.array([s[i] if i >= 0 else True for i in seq_g], dtype=np.float32)
            mt = np.array([s[i] if i >= 0 else True for i in tgt_g], dtype=np.float32)
            res.append(predict(net, (xs * ms).astype(np.float32), (xt * mt).astype(np.float32)).ravel())
        return np.array(res)
    return fn


def lstm_value_fn_bg(net, xs, xt, hour_idx, seq_groups, tgt_groups, groups, bg_xs, bg_xt):
    """Seperti lstm_value_fn, tetapi latar = DISTRIBUSI hari: bg_xs [K, 48, c_enc], bg_xt [K, 24, c_dec].
    Kanal kelompok absen diambil dari hari latar (utuh, sehingga kanal jam tetap selaras);
    nilai koalisi = rata-rata keluaran atas K hari latar."""
    n, K = len(xs), len(bg_xs)
    seq_g = np.array([groups.index(g) if g in groups else -1 for g in seq_groups])
    tgt_g = np.array([groups.index(g) if g in groups else -1 for g in tgt_groups])
    xs_r, xt_r = np.repeat(xs, K, axis=0), np.repeat(xt, K, axis=0)        # baris i*K + k
    bs_r, bt_r = np.tile(bg_xs, (n, 1, 1)), np.tile(bg_xt, (n, 1, 1))
    h_r = np.repeat(np.asarray(hour_idx), K)

    def fn(subsets):
        res = []
        for s in subsets:
            ms = np.array([True if i < 0 else bool(s[i]) for i in seq_g])
            mt = np.array([True if i < 0 else bool(s[i]) for i in tgt_g])
            p = predict(net, np.where(ms[None, None, :], xs_r, bs_r).astype(np.float32),
                        np.where(mt[None, None, :], xt_r, bt_r).astype(np.float32))
            res.append(p[np.arange(n * K), h_r].reshape(n, K).mean(axis=1))
        return np.array(res)
    return fn

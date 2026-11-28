"""Angka kunci untuk laporan (A7): satu sumber angka, dibaca dari results/*.csv — tidak menghitung ulang.

Pakai:  python -c "import sys; sys.path.insert(0,'src'); import report; report.write_key_numbers()"
        (dijalankan otomatis di akhir scripts/run_all.py)
Keluaran: results/key_numbers.json — setiap angka mencatat file sumbernya.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ML = ("rf", "xgb", "lstm")


def _r(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd) if isinstance(x, (float, np.floating)) else (int(x) if isinstance(x, (np.integer,)) else x)


def _git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() or None
    except FileNotFoundError:
        return None


def _find(name: str) -> Path | None:
    for p in (RES / name, RES / "past" / name):
        if p.exists():
            return p
    return None


def accuracy(res: Path = RES) -> dict:
    out = {}
    t = pd.read_csv(res / "test_main.csv").query("mask == 'valid'").set_index("model")
    v = pd.read_csv(res / "val_main.csv").query("mask == 'valid'").set_index("model")
    out["n_jam_uji_valid"] = int(t["n"].iloc[0])
    out["n_jam_val_valid"] = int(v["n"].iloc[0])
    out["uji_2020"] = {m.replace("pred_", ""): {k: _r(t.loc[m, k]) for k in ("RMSE", "nRMSE_%", "MAE", "MBE", "R2", "skill_vs_clim")}
                       for m in t.index}
    out["val_2019"] = {m.replace("pred_", ""): {k: _r(v.loc[m, k]) for k in ("RMSE", "MBE")} for m in v.index}
    ci = pd.read_csv(res / "test_bootstrap_ci.csv").query("mask == 'valid'").set_index("besaran")
    out["ci95_bootstrap_hari"] = {}
    for m in ML:
        if f"RMSE_pred_{m}" in ci.index:
            out["ci95_bootstrap_hari"][m] = {
                "RMSE": [_r(ci.loc[f"RMSE_pred_{m}", "ci_low"]), _r(ci.loc[f"RMSE_pred_{m}", "ci_high"])],
                "skill_vs_clim": [_r(ci.loc[f"skill_pred_{m}_vs_pred_clim", "ci_low"]),
                                  _r(ci.loc[f"skill_pred_{m}_vs_pred_clim", "ci_high"])]}
    dm = pd.read_csv(res / "test_dm.csv").query("mask == 'valid'")
    out["diebold_mariano_holm"] = {"n_perbandingan": int(len(dm)),
                                   "pasangan": {f"{a} vs {b}": {"p": _r(p, 4), "p_holm": _r(ph, 4), "signifikan_5%": bool(ph < 0.05)}
                                                for a, b, p, ph in dm[["model", "vs", "p", "p_holm"]].itertuples(index=False)}}
    vi = pd.read_csv(res / "test_violation.csv", index_col=0)
    out["pelanggaran_sebelum_clipping_%"] = {m: {"siang": _r(vi.loc[m, "siang_%"]), "semua_jam": _r(vi.loc[m, "semua_jam_%"])}
                                             for m in vi.index}
    return out


def ablation(res: Path = RES) -> dict:
    a = pd.read_csv(res / "test_ablation.csv").query("mask == 'valid' and clipping == 'clip'")
    d = pd.read_csv(res / "test_dm_ablation.csv").query("mask == 'valid'")
    return {"uji_2020_clip": {f"{r.target}__{r.fitur_fisika}": {"RMSE": _r(r.RMSE), "MBE": _r(r.MBE),
                                                                "pelanggaran_siang_%": _r(r["pelanggaran_siang_%"])}
                              for _, r in a.iterrows()},
            "dm_vs_utama_p_holm_min": _r(d["p_holm"].min(), 4)}


def xai(res: Path = RES) -> dict:
    out = {}
    imp = pd.read_csv(res / "xai_group_importance_all.csv", header=[0, 1], index_col=0)
    ap = imp[("a_priori", imp.columns[0][1])] if ("a_priori", "") not in imp.columns else imp[("a_priori", "")]
    out["urutan_a_priori"] = {g: int(ap[g]) for g in imp.index}
    out["peringkat_kelompok"] = {c: {g: int(imp.loc[g, ("peringkat", c)]) for g in imp.index}
                                 for c in imp["peringkat"].columns}
    out["porsi_kelompok"] = {c: {g: _r(imp.loc[g, ("porsi", c)]) for g in imp.index} for c in imp["porsi"].columns}
    ag = pd.read_csv(res / "xai_agreement_all.csv")
    sp = ag[ag["metode"].isin(["SHAP", "LIME"])]
    out["spearman_vs_a_priori"] = {f"{r.model}|{r.tingkat}|{r.metode}": {"rho": _r(r.rho), "ci": [_r(r.ci_low), _r(r.ci_high)]}
                                   for r in sp.itertuples(index=False)}
    jc = ag[ag["metode"].str.startswith("Jaccard")]
    out["jaccard_top3_shap_lime"] = {f"{r.model}|{r.tingkat}": _r(r.rho) for r in jc.itertuples(index=False)}
    fi = pd.read_csv(res / "xai_feature_importance.csv", index_col=0)
    out["fitur_teratas_shap"] = {m: {f: _r(fi.loc[f, m]) for f in fi[m].sort_values(ascending=False).index[:5]}
                                 for m in ("xgb", "rf") if m in fi}
    ge = res / "xai_ge_lstm_groups.csv"
    if ge.exists():
        g = pd.read_csv(ge, header=[0, 1], index_col=0)
        out["lstm_gradientexplainer_peringkat"] = {c: {k: int(g.loc[k, ("peringkat", c)]) for k in g.index}
                                                   for c in g["peringkat"].columns}
        ga = pd.read_csv(res / "xai_ge_lstm_agreement.csv")
        out["lstm_gradientexplainer_kesesuaian"] = ga.round(3).replace({np.nan: None}).to_dict("records")
    return out


def tuning(res: Path = RES) -> dict:
    out = {}
    for m in ("xgb", "rf", "lstm"):
        p = _find(f"optuna_{m}.csv")
        if p is not None:
            o = pd.read_csv(p)
            out[m] = {"n_trial": int(len(o)), "rmse_val_terbaik": _r(o["value"].min()), "rmse_val_terburuk": _r(o["value"].max()),
                      "sumber": str(p.relative_to(ROOT))}
    return out


def post_analysis(res: Path = RES) -> dict:
    """Ringkasan notebook 10 (pasca-hasil) bila sudah dijalankan."""
    out = {}
    p = res / "post_seed_summary.csv"
    if p.exists():
        out["variasi_seed"] = pd.read_csv(p, index_col=[0, 1]).round(3).reset_index().replace({np.nan: None}).to_dict("records")
    p = res / "post_shapley_baseline_rank.csv"
    if p.exists():
        out["sensitivitas_baseline_shapley"] = pd.read_csv(p, index_col=0).to_dict()
    p = res / "post_faithfulness.csv"
    if p.exists():
        out["faithfulness_deletion"] = pd.read_csv(p).round(4).replace({np.nan: None}).to_dict("records")
    return out


def key_numbers(res: Path = RES) -> dict:
    kn = {"_catatan": "Dibangkitkan otomatis dari results/*.csv oleh src/report.py. Jangan diedit manual.",
          "_dibuat": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "_git_head": _git_head()}
    for name, fn in (("akurasi", accuracy), ("ablasi", ablation), ("xai", xai), ("tuning", tuning),
                     ("pasca_hasil", post_analysis)):
        try:
            kn[name] = fn(res)
        except FileNotFoundError as e:
            kn[name] = {"_hilang": str(Path(e.filename).name) if e.filename else str(e)}
    return kn


def write_key_numbers(res: Path = RES) -> Path:
    out = res / "key_numbers.json"
    out.write_text(json.dumps(key_numbers(res), indent=2, ensure_ascii=False))
    return out

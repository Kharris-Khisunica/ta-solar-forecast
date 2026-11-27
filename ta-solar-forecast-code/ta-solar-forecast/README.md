# ta-solar-forecast

Peramalan iradiasi surya day-ahead (ML + XAI + physical consistency) — Kampus ITB Ganesha, data NSRDB Himawari 2016–2020 + GFS 0,25°.

## Setup

```bash
cp .env.example .env          # isi NLR_API_KEY, NLR_EMAIL, GDEX_TOKEN
docker build -t ta-solar-forecast .
docker run --rm -v "$PWD/data:/app/data" --env-file .env ta-solar-forecast   # menjalankan pytest
```

Tanpa Docker: `pip install -r requirements.txt && pytest -q`.

## Urutan notebook

| # | Notebook | Isi | Keluaran utama |
|---|---|---|---|
| 01 | gfs_fetch | Request & unduh GFS (GDEX) | data/raw/gfs/ |
| 02 | nsrdb_download | Unduh NSRDB Himawari | data/raw/nsrdb_*.csv |
| 03 | qc_aggregate | QC, konvensi timestamp, agregasi per jam | data/interim/nsrdb_hourly.parquet |
| 04 | clearsky_kc | Ineichen, k_c, k_max(zenit) | nsrdb_hourly_kc.parquet, config.yaml |
| 05 | features_nsrdb | Tabel fitur + audit kebocoran | data/processed/features_nsrdb.parquet |
| 06 | gfs_process | De-averaging, per jam, gabung GFS | data/processed/features_all.parquet |
| 07 | **train_models** | Baseline, RF, XGB, LSTM, ablasi (latih & simpan) | models/, predictions.parquet |
| 08 | **evaluate** | Semua metrik, DM, CI, ablasi, stratifikasi, diagnosis | results/test_*.csv, figures/ |
| 09 | **xai** | SHAP & LIME untuk RF, XGB, LSTM; kesesuaian a priori | results/xai_*.csv, figures/xai_* |

Notebook lama ada di `notebooks/arsip/` (tidak dipakai lagi).

## Struktur

```
configs/      config.yaml, gdex.yaml, apriori_order.yaml
data/         raw/ interim/ processed/   (tidak di-commit)
notebooks/    01–09 (lihat tabel di atas), arsip/
src/          utils, gfs_fetch, clearsky, features, gfs_process, evaluate, models, lstm, xai
tests/        pytest (53 tes)
models/       best_params.json, xgb_best.json, rf_best.joblib, lstm_best.pt, lstm_normalizer.npz
figures/ results/
decisions.md  log keputusan teknis (bahan Bab III)
```

Kebiasaan: commit + push setiap akhir hari.

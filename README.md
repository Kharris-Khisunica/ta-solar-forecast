# ta-solar-forecast

Peramalan iradiasi surya day-ahead (ML + XAI + physical consistency) — Kampus ITB Ganesha, data NSRDB Himawari 2016–2020 + GFS 0,25°.

## Setup

```bash
cp .env.example .env          # isi NLR_API_KEY, NLR_EMAIL, GDEX_TOKEN
docker build -t ta-solar-forecast .
docker run --rm -v "$PWD/data:/app/data" --env-file .env ta-solar-forecast   # menjalankan pytest
```

Tanpa Docker: `pip install -r requirements.txt && pytest -q tests`.

## Reproduksi satu perintah

```bash
python scripts/run_all.py                 # 03 → 09, berhenti bila ada cek yang gagal
python scripts/run_all.py --with-post     # + 10_post_analysis
python scripts/run_all.py --from 07 --to 08
```

Cek otomatis: sel error; 03 `T09.3 LULUS`; 04 `config.yaml` tidak berubah; 06 `T18.6 LULUS`;
07 `✓ identik dengan run sebelumnya`; 09 urutan a priori ter-commit. Keluaran:
`results/run_all_log.json` (status & file yang berubah), `results/environment.json` (versi paket vs
`requirements.txt`), `results/key_numbers.json` (angka kunci laporan, dari `src/report.py`).
01 & 02 tidak ikut karena mengakses server luar; di 01 ada sakelar `RUN_REMOTE = False`.

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
| 09 | **xai** | SHAP & LIME untuk RF, XGB, LSTM; kesesuaian a priori; §8 GradientExplainer LSTM | results/xai_*.csv, figures/xai_* |
| 10 | post_analysis | *Pasca-hasil:* variasi seed, sensitivitas baseline Shapley, XAI val 2019, faithfulness, bias | results/post_*.csv, figures/post_* |

Hasil run sebelum konsolidasi ada di `results/past/` (termasuk `optuna_*.csv` dan `test_run_log.json`).

## Struktur

```
configs/      config.yaml, gdex.yaml, apriori_order.yaml
data/         raw/ interim/ processed/   (tidak di-commit)
notebooks/    01–10 (lihat tabel di atas)
src/          utils, gfs_fetch, clearsky, features, gfs_process, evaluate, models, lstm, xai, report
scripts/      run_all.py
tests/        pytest (61 tes)
models/       best_params.json, xgb_best.json, rf_best.joblib, lstm_best.pt, lstm_normalizer.npz
figures/ results/
decisions.md  log keputusan teknis (bahan Bab III)
```

Kebiasaan: commit + push setiap akhir hari.

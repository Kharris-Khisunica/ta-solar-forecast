# ta-solar-forecast

Peramalan iradiasi surya day-ahead (ML + XAI + physical consistency) — Kampus ITB Ganesha, data NSRDB Himawari 2016–2020 + GFS 0,25°.

## Setup

```bash
cp .env.example .env          # isi NLR_API_KEY, NLR_EMAIL, GDEX_TOKEN
docker build -t ta-solar-forecast .
docker run --rm -v "$PWD/data:/app/data" --env-file .env ta-solar-forecast   # menjalankan pytest
```

Tanpa Docker: `pip install -r requirements.txt && pytest -q`.

## Struktur

```
configs/      config.yaml (lokasi, split, seed), gdex.yaml (request GFS)
data/         raw/ interim/ processed/   (tidak di-commit)
notebooks/    01_gfs_fetch.ipynb, 02_nsrdb_download.ipynb
src/          utils.py (load_config, set_seed), gfs_fetch.py, models/
tests/        pytest
figures/ results/ models/
decisions.md  log keputusan teknis (bahan Bab III)
```

Kebiasaan: commit + push setiap akhir hari.

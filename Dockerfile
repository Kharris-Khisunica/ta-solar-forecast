FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Paket eccodes di pip sudah membawa library ecCodes; build-essential untuk lime (sdist)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Data dan .env TIDAK ikut image (lihat .dockerignore); mount saat run:
#   docker run --rm -v "$PWD/data:/app/data" --env-file .env ta-solar-forecast
# Default: jalankan tes. Pipeline penuh (butuh data/ ter-mount):
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/.git:/app/.git" \
#       ta-solar-forecast python scripts/run_all.py --kernel python3
#   (.git perlu di-mount karena 09 memeriksa commit urutan a priori)
CMD ["pytest", "-q", "tests"]

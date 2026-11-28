"""Jalankan pipeline 03→09 (opsional 10) dari awal sampai akhir dan BERHENTI bila ada cek yang gagal (A1/T48).

Pakai (dari root repo, dengan environment yang sama dengan kernel notebook):
    python scripts/run_all.py                 # 03..09, notebook disimpan di tempat (seperti Run All)
    python scripts/run_all.py --with-post     # + 10_post_analysis
    python scripts/run_all.py --from 07 --to 08
    python scripts/run_all.py --kernel python3   # paksa kernel tertentu

01 & 02 (unduh data) sengaja TIDAK ikut: keduanya mengakses server luar.

Cek otomatis:
  • sel error → berhenti (pesan error ditampilkan)
  • 03: "T09.3 LULUS"   • 04: configs/config.yaml tidak berubah   • 06: "T18.6 LULUS"
  • 07: "✓ identik dengan run sebelumnya"   • 09: "✓ Urutan a priori ter-commit"
  • semua: tidak ada "GAGAL ✗", "BERBEDA", "BELUM di-commit", "W/m² ⚠", dsb. di keluaran
Keluaran: results/run_all_log.json, results/environment.json, results/key_numbers.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR, RES = ROOT / "notebooks", ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

ORDER = ["03_qc_aggregate", "04_clearsky_kc", "05_features_nsrdb", "06_gfs_process",
         "07_train_models", "08_evaluate", "09_xai"]
POST = "10_post_analysis"
REQUIRED = {"03": ["T09.3 LULUS"], "06": ["T18.6 LULUS"], "07": ["✓ identik dengan run sebelumnya"],
            "09": ["✓ Urutan a priori ter-commit"]}
FORBIDDEN = [r"GAGAL ✗", r"⚠ BERBEDA", r"BERBEDA ⚠", r"BELUM di-commit", r"W/m² ⚠",
             r"⚠ Ada fitur berkorelasi", r"⚠ Tidak ditemukan", r"tidak ada file run 00"]


def sh(*args) -> str:
    try:
        return subprocess.run(list(args), cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def environment() -> dict:
    """Versi Python & paket terpasang dibanding requirements.txt (A5/A6)."""
    pins, mismatch = {}, {}
    req = ROOT / "requirements.txt"
    for line in req.read_text().splitlines() if req.exists() else []:
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1)] = m.group(2)
    installed = {}
    for pkg, want in pins.items():
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            have = None
        installed[pkg] = have
        if have is None or have.split("+")[0] != want:
            mismatch[pkg] = {"requirements": want, "terpasang": have}
    return {"python": sys.version.split()[0], "platform": platform.platform(), "git_head": sh("git", "rev-parse", "HEAD"),
            "git_dirty": bool(sh("git", "status", "--porcelain")), "paket": installed, "beda_dari_requirements": mismatch,
            "waktu": time.strftime("%Y-%m-%d %H:%M:%S")}


def outputs_text(nb) -> str:
    parts = []
    for c in nb.cells:
        for o in c.get("outputs", []):
            if "text" in o:
                parts.append(o["text"])
            elif "data" in o and "text/plain" in o["data"]:
                parts.append(o["data"]["text/plain"])
    return "\n".join(parts)


def run_notebook(name: str, kernel: str | None, timeout: int | None) -> tuple[float, str]:
    import nbformat
    from nbclient import NotebookClient
    path = NB_DIR / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    kw = {"timeout": timeout, "resources": {"metadata": {"path": str(NB_DIR)}}, "allow_errors": False}
    if kernel:
        kw["kernel_name"] = kernel
    t0 = time.time()
    try:
        NotebookClient(nb, **kw).execute()
    finally:
        nbformat.write(nb, path)               # simpan juga bila gagal, agar error terlihat di notebook
    return time.time() - t0, outputs_text(nb)


def check(name: str, text: str) -> list[str]:
    key = name[:2]
    problems = [f"tidak ada penanda wajib '{m}'" for m in REQUIRED.get(key, []) if m not in text]
    for pat in FORBIDDEN:
        for line in text.splitlines():
            if re.search(pat, line):
                problems.append(f"keluaran memuat '{line.strip()[:120]}'")
    return problems


def compare_results(before: dict) -> dict:
    """File results/*.csv yang berubah + selisih numerik maksimum."""
    import numpy as np
    import pandas as pd
    changed = {}
    for p in sorted(RES.glob("*.csv")):
        old = before.get(p.name)
        if old is None:
            changed[p.name] = "baru"
        elif md5(p) != old["md5"]:
            try:
                a, b = pd.read_csv(old["copy"]), pd.read_csv(p)
                num = [c for c in a.select_dtypes("number").columns if c in b]
                d = float(np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))) if num and a.shape == b.shape else None
                changed[p.name] = f"berubah, selisih maks {d:.3g}" if d is not None else "berubah (bentuk tabel beda)"
            except Exception as e:  # noqa: BLE001
                changed[p.name] = f"berubah ({type(e).__name__})"
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="03")
    ap.add_argument("--to", dest="end", default="09")
    ap.add_argument("--with-post", action="store_true", help="ikut jalankan 10_post_analysis")
    ap.add_argument("--kernel", default=None, help="nama kernel Jupyter (default: dari metadata notebook)")
    ap.add_argument("--timeout", type=int, default=None, help="batas waktu per sel (detik)")
    ap.add_argument("--first-run", action="store_true", help="07 belum punya predictions.parquet lama")
    args = ap.parse_args()

    if sys.platform.startswith("win"):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    names = [n for n in ORDER if args.start <= n[:2] <= args.end] + ([POST] if args.with_post else [])
    if args.first_run:
        REQUIRED.pop("07", None)
    env = environment()
    (RES / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
    print(f"Python {env['python']} | git {env['git_head'][:10]}{' (ada perubahan belum di-commit)' if env['git_dirty'] else ''}")
    if env["beda_dari_requirements"]:
        print("⚠ Paket berbeda dari requirements.txt (bukan penghenti, tapi catat):")
        for k, v in env["beda_dari_requirements"].items():
            print(f"   {k}: requirements {v['requirements']} | terpasang {v['terpasang']}")

    import shutil
    import tempfile
    snap = Path(tempfile.mkdtemp(prefix="results_before_"))
    before = {}
    for p in RES.glob("*.csv"):
        shutil.copy(p, snap / p.name)
        before[p.name] = {"md5": md5(p), "copy": snap / p.name}
    cfg_before = (ROOT / "configs" / "config.yaml").read_text()

    log = {"mulai": time.strftime("%Y-%m-%d %H:%M:%S"), "notebook": [], "status": "berjalan"}
    for name in names:
        print(f"\n▶ {name} …", flush=True)
        try:
            dt, text = run_notebook(name, args.kernel, args.timeout)
        except Exception as e:  # noqa: BLE001  CellExecutionError, DeadKernelError, …
            log["notebook"].append({"nama": name, "status": "ERROR", "pesan": str(e)[-2000:]})
            log["status"] = f"berhenti di {name} (error sel)"
            print(f"✗ {name}: error saat eksekusi — lihat notebook.\n{str(e)[-1500:]}")
            break
        problems = check(name, text)
        if name.startswith("04") and (ROOT / "configs" / "config.yaml").read_text() != cfg_before:
            problems.append("configs/config.yaml berubah setelah 04 — cek `git diff configs/config.yaml`")
        log["notebook"].append({"nama": name, "detik": round(dt), "status": "OK" if not problems else "GAGAL CEK",
                                "masalah": problems})
        if problems:
            log["status"] = f"berhenti di {name} (cek gagal)"
            print(f"✗ {name} ({dt:.0f} dtk):"); [print("   -", p) for p in problems]
            break
        print(f"✓ {name} ({dt:.0f} dtk)")
    else:
        log["status"] = "selesai"

    log["results_berubah"] = compare_results(before)
    log["selesai"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RES / "run_all_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    try:
        import report
        print("\nAngka kunci →", report.write_key_numbers().relative_to(ROOT))
    except Exception as e:  # noqa: BLE001
        print("⚠ key_numbers gagal:", e)
    print("\nStatus:", log["status"])
    ch = {k: v for k, v in log["results_berubah"].items()}
    print("File results/*.csv yang berubah:", "tidak ada" if not ch else "")
    for k, v in ch.items():
        print(f"   {k}: {v}")
    sys.exit(0 if log["status"] == "selesai" else 1)


if __name__ == "__main__":
    main()

"""Request subset GFS 0.25° (NCAR GDEX d084001): cek, buat control file, submit, pantau, unduh.

Alur (T01.4, T07.1–T07.5, T17):
    python src/gfs_fetch.py check                 # subsetting_available + simpan template & metadata
    python src/gfs_fetch.py build                 # buat configs/gdex_test.ctl + configs/gdex_<tahun>.ctl
    python src/gfs_fetch.py submit test           # submit uji 1 bulan
    python src/gfs_fetch.py submit 2016 2017 ...  # atau: submit years  (semua tahun di config)
    python src/gfs_fetch.py status                # status semua request yang tercatat
    python src/gfs_fetch.py download test         # unduh + MANIFEST md5 bila sudah selesai

Token: GDEX_TOKEN di file .env (root repo). Tidak ada file token yang ditulis ke disk.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from gdex_api_client import gdex_client as gc

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "gdex.yaml"
CTL_DIR = ROOT / "configs"
META_DIR = ROOT / "data" / "raw" / "gfs" / "meta"
GFS_DIR = ROOT / "data" / "raw" / "gfs"
REQ_LOG = GFS_DIR / "requests.json"
DECISIONS = ROOT / "decisions.md"

# Kunci control file yang diisi oleh skrip; kunci lain dari template dipertahankan.
OVERRIDE_KEYS = ("dataset", "date", "param", "level", "product", "nlat", "slat", "wlon", "elon")


# ----------------------------------------------------------------------------- auth
def init_auth() -> None:
    """Pakai token dari .env, bukan gdex_token.txt bawaan klien."""
    load_dotenv(ROOT / ".env")
    token = os.environ.get("GDEX_TOKEN", "").strip()
    if not token:
        sys.exit("GDEX_TOKEN tidak ditemukan di .env")
    gc.get_authentication = lambda token_file=None: token  # noqa: E731


# ----------------------------------------------------------------------------- util
def load_cfg(path: Path = CFG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def find_key(obj, key):
    """Cari nilai kunci pertama di JSON bersarang (skema respons API tidak seragam)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def log_decision(text: str) -> None:
    with open(DECISIONS, "a") as fh:
        fh.write(f"- {date.today().isoformat()} | {text}\n")


def load_requests() -> dict:
    return json.loads(REQ_LOG.read_text()) if REQ_LOG.exists() else {}


def save_requests(reqs: dict) -> None:
    REQ_LOG.parent.mkdir(parents=True, exist_ok=True)
    REQ_LOG.write_text(json.dumps(reqs, indent=2))


# ----------------------------------------------------------------------------- isi control file
def product_strings(leads: list[int]) -> tuple[list[str], list[str]]:
    """Nama produk GFS per lead time.

    Instan : '18-hour Forecast'
    Rata-rata (reset tiap 6 jam): lead %6==3 -> '3-hour Average (initial+15 to initial+18)'
                                  lead %6==0 -> '6-hour Average (initial+12 to initial+18)'
    """
    inst, avg = [], []
    for L in leads:
        inst.append(f"{L}-hour Forecast")
        span = 3 if L % 6 == 3 else 6
        avg.append(f"{span}-hour Average (initial+{L - span} to initial+{L})")
    return inst, avg


def date_range(label: str, cfg: dict) -> tuple[datetime, datetime]:
    """Rentang tanggal INISIALISASI (run) untuk request.

    test   : seluruh bulan test_month.
    <tahun>: 31 Des tahun sebelumnya s.d. 30 Des tahun itu (run D-1 untuk target 1 Jan..31 Des).
    """
    h = int(cfg["run_hour_utc"])
    if label == "test":
        y, m = map(int, cfg["test_month"].split("-"))
        return datetime(y, m, 1, h), datetime(y, m, calendar.monthrange(y, m)[1], h)
    y = int(label)
    return datetime(y - 1, 12, 31, h), datetime(y, 12, 30, h)


def bbox(cfg: dict) -> dict:
    lat, lon, d = cfg["location"]["lat"], cfg["location"]["lon"], cfg["box_half_width_deg"]
    r = lambda v: f"{v:.2f}"  # noqa: E731
    return {"nlat": r(lat + d), "slat": r(lat - d), "wlon": r(lon - d), "elon": r(lon + d)}


def build_ctl(label: str, cfg: dict, template: dict | None = None) -> dict:
    inst_p, avg_p = product_strings(cfg["leads_h"])
    params = {**cfg["params"]["avg"], **cfg["params"]["inst"]}
    levels = []
    for lv in params.values():
        levels += [x for x in lv if x not in levels]
    start, end = date_range(label, cfg)
    ctl = dict(template or {})
    ctl.update({
        "dataset": cfg["dataset"],
        "date": f"{start:%Y%m%d%H%M}/to/{end:%Y%m%d%H%M}",
        "param": "/".join(params),
        "level": ";".join(levels),
        "product": "/".join(inst_p + avg_p),
        **bbox(cfg),
    })
    return ctl


def ctl_to_text(ctl: dict) -> str:
    return "\n".join(f"{k}={v}" for k, v in ctl.items()) + "\n"


def validate_against_meta(ctl: dict, meta_text: str) -> list[str]:
    """Cek string param/produk/kode level muncul di metadata yang disimpan oleh `check`."""
    missing = []
    for p in ctl["param"].split("/"):
        if f'"{p}"' not in meta_text:
            missing.append(f"param {p}")
    for prod in ctl["product"].split("/"):
        if prod not in meta_text:
            missing.append(f"product '{prod}'")
    for lv in ctl["level"].split(";"):
        if lv.split(":")[0] not in meta_text:
            missing.append(f"level {lv}")
    return missing


def meta_text() -> str:
    files = sorted(META_DIR.glob("*.json")) + sorted(META_DIR.glob("*.ctl"))
    return "\n".join(f.read_text() for f in files)


# ----------------------------------------------------------------------------- perintah
def cmd_check(cfg: dict, _args) -> None:
    ds = cfg["dataset"]
    META_DIR.mkdir(parents=True, exist_ok=True)

    summary = gc.get_summary(ds)
    (META_DIR / f"{ds}_summary.json").write_text(json.dumps(summary, indent=2))
    sub = find_key(summary, "subsetting_available")
    if sub is None:
        psum = gc.get_param_summary(ds)
        sub = find_key(psum, "subsetting_available")
    print(f"subsetting_available = {sub}")

    for name, fn in (("paramsummary", gc.get_param_summary), ("metadata", gc.get_metadata)):
        (META_DIR / f"{ds}_{name}.json").write_text(json.dumps(fn(ds), indent=2))

    tmpl = gc.get_control_file_template(ds)
    tmpl_txt = find_key(tmpl, "template") or ""
    (META_DIR / f"{ds}_template.ctl").write_text(tmpl_txt)
    print(f"Template & metadata disimpan di {META_DIR.relative_to(ROOT)}/")
    print("---- template ----\n" + tmpl_txt)

    log_decision(f"GDEX {ds}: subsetting_available = {sub} (dicek {now_utc()}) [GO kriteria #1]")
    if sub is not True:
        print("⚠ Kriteria GO #1 TIDAK terpenuhi.")


def cmd_build(cfg: dict, args) -> None:
    tpath = META_DIR / f"{cfg['dataset']}_template.ctl"
    template = gc.read_control_file(str(tpath)) if tpath.exists() and tpath.stat().st_size else {}
    template = {k: v for k, v in template.items() if k not in OVERRIDE_KEYS}
    mtext = meta_text()
    if not mtext:
        print("⚠ Metadata belum ada; jalankan `check` dulu agar isi control file bisa divalidasi.")

    labels = ["test"] + [str(y) for y in cfg["years"]]
    for label in labels:
        ctl = build_ctl(label, cfg, template)
        out = CTL_DIR / f"gdex_{label}.ctl"
        out.write_text(ctl_to_text(ctl))
        print(f"✓ {out.relative_to(ROOT)}  date={ctl['date']}")

    if mtext:
        missing = validate_against_meta(build_ctl("test", cfg, template), mtext)
        if missing:
            print("\n⚠ Tidak ditemukan di metadata (perbaiki configs/gdex.yaml sebelum submit):")
            for m in missing:
                print("   -", m)
            print("  Lihat nama produk/level yang benar di data/raw/gfs/meta/.")
        else:
            print("\nSemua param, produk, dan level ditemukan di metadata.")


def cmd_submit(cfg: dict, args) -> None:
    labels = [str(y) for y in cfg["years"]] if args.labels == ["years"] else args.labels
    mtext = meta_text()
    reqs = load_requests()
    for label in labels:
        path = CTL_DIR / f"gdex_{label}.ctl"
        if not path.exists():
            sys.exit(f"{path.name} belum ada; jalankan `build` dulu.")
        if label in reqs and not args.resubmit:
            print(f"- {label}: sudah disubmit (ID {reqs[label]['request_id']}); pakai --resubmit untuk ulang.")
            continue
        ctl = gc.read_control_file(str(path))
        if mtext and not args.force:
            missing = validate_against_meta(ctl, mtext)
            if missing:
                sys.exit(f"{label}: {len(missing)} item tidak ada di metadata ({missing[:3]}...). "
                         "Perbaiki config atau pakai --force.")
        resp = gc.submit_json(ctl)
        rid = find_key(resp, "request_id") or find_key(resp, "request_index") or find_key(resp, "rindex")
        print(f"- {label}: {json.dumps(resp)[:300]}")
        if rid is None:
            print(f"  ⚠ ID request tidak terbaca otomatis; periksa respons di atas.")
        reqs[label] = {"request_id": rid, "submitted": now_utc(), "date": ctl["date"], "response": resp}
        save_requests(reqs)
        log_decision(f"GDEX request {label} disubmit, ID {rid} ({now_utc()})")


def cmd_status(cfg: dict, args) -> None:
    reqs = load_requests()
    if not reqs:
        print("Belum ada request tercatat."); return
    for label, r in reqs.items():
        resp = gc.get_status(r["request_id"])
        st = find_key(resp, "status") or find_key(resp, "request_status")
        r["last_status"], r["last_checked"] = st, now_utc()
        print(f"{label:>5}  ID {r['request_id']}  status={st}  (submit {r['submitted']})")
    save_requests(reqs)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_download(cfg: dict, args) -> None:
    reqs = load_requests()
    labels = [str(y) for y in cfg["years"]] if args.labels == ["years"] else args.labels
    for label in labels:
        if label not in reqs:
            print(f"- {label}: belum disubmit"); continue
        out = GFS_DIR / label
        out.mkdir(parents=True, exist_ok=True)
        resp = gc.download(str(reqs[label]["request_id"]), str(out) + os.sep)
        if not find_key(resp, "web_files"):
            print(f"- {label}: belum siap / tidak ada file ({json.dumps(resp)[:200]})"); continue
        files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "MANIFEST")
        lines = [f"# request {reqs[label]['request_id']} | date={reqs[label]['date']} | unduh {now_utc()}"]
        lines += [f"{md5(p)}  {p.name}  {p.stat().st_size}" for p in files]
        (out / "MANIFEST").write_text("\n".join(lines) + "\n")
        print(f"✓ {label}: {len(files)} file, MANIFEST ditulis")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check")
    sp.add_parser("build")
    s = sp.add_parser("submit"); s.add_argument("labels", nargs="+", help="test | years | 2016 ...")
    s.add_argument("--force", action="store_true"); s.add_argument("--resubmit", action="store_true")
    sp.add_parser("status")
    d = sp.add_parser("download"); d.add_argument("labels", nargs="+")
    args = ap.parse_args(argv)

    cfg = load_cfg()
    if args.cmd != "build":
        init_auth()
    {"check": cmd_check, "build": cmd_build, "submit": cmd_submit,
     "status": cmd_status, "download": cmd_download}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()

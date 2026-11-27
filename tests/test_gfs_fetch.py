"""Tes offline untuk src/gfs_fetch.py (API GDEX di-mock)."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gfs_fetch as gf  # noqa: E402

CFG = gf.load_cfg()


def test_product_strings_average_reset_6h():
    inst, avg = gf.product_strings([18, 21, 24, 39])
    assert inst == ["18-hour Forecast", "21-hour Forecast", "24-hour Forecast", "39-hour Forecast"]
    assert avg == [
        "6-hour Average (initial+12 to initial+18)",
        "3-hour Average (initial+18 to initial+21)",
        "6-hour Average (initial+18 to initial+24)",
        "3-hour Average (initial+36 to initial+39)",
    ]


def test_date_range_year_starts_dec31_previous_year():
    s, e = gf.date_range("2016", CFG)
    assert s == datetime(2015, 12, 31, 0) and e == datetime(2016, 12, 30, 0)


def test_date_range_test_month():
    s, e = gf.date_range("test", CFG)
    assert s == datetime(2020, 1, 1, 0) and e == datetime(2020, 1, 31, 0)


def test_bbox_half_degree():
    b = gf.bbox(CFG)
    assert float(b["nlat"]) - float(b["slat"]) == pytest.approx(1.0)
    assert float(b["elon"]) - float(b["wlon"]) == pytest.approx(1.0)


def test_build_ctl_keeps_template_keys_and_overrides():
    ctl = gf.build_ctl("2017", CFG, {"oformat": "grib2", "date": "LAMA"})
    assert ctl["oformat"] == "grib2"
    assert ctl["date"] == "201612310000/to/201712300000"
    assert ctl["param"].split("/") == ["DSWRF", "TCDC", "TMP", "RH", "PWAT", "CAPE"]
    assert len(ctl["product"].split("/")) == 16


def test_validate_flags_missing_items():
    ctl = gf.build_ctl("test", CFG)
    meta = '"DSWRF" "TCDC" SFC'
    missing = gf.validate_against_meta(ctl, meta)
    assert "param TMP" in missing
    assert any(m.startswith("product") for m in missing)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Arahkan semua path skrip ke folder sementara + mock klien GDEX."""
    (tmp_path / "configs").mkdir()
    monkeypatch.setattr(gf, "ROOT", tmp_path)
    monkeypatch.setattr(gf, "CTL_DIR", tmp_path / "configs")
    monkeypatch.setattr(gf, "META_DIR", tmp_path / "data/raw/gfs/meta")
    monkeypatch.setattr(gf, "GFS_DIR", tmp_path / "data/raw/gfs")
    monkeypatch.setattr(gf, "REQ_LOG", tmp_path / "data/raw/gfs/requests.json")
    monkeypatch.setattr(gf, "DECISIONS", tmp_path / "decisions.md")

    inst, avg = gf.product_strings(CFG["leads_h"])
    meta = {"data": {"params": ["DSWRF", "TCDC", "TMP", "RH", "PWAT", "CAPE"],
                     "products": inst + avg, "levels": ["SFC", "EATM", "LCY", "MCY", "HCY", "HTGL"]}}
    gc = gf.gc
    monkeypatch.setattr(gc, "get_summary", lambda ds: {"data": {"subsetting_available": True}})
    monkeypatch.setattr(gc, "get_param_summary", lambda ds: {"data": {"data": []}})
    monkeypatch.setattr(gc, "get_metadata", lambda ds: meta)
    monkeypatch.setattr(gc, "get_control_file_template",
                        lambda ds: {"data": {"template": "dataset=d084001\ndate=X\noformat=grib2\n"}})
    submitted = []
    monkeypatch.setattr(gc, "submit_json", lambda d: submitted.append(d) or {"data": {"request_id": 900000 + len(submitted)}})
    monkeypatch.setattr(gc, "get_status", lambda rid: {"data": [{"request_id": rid, "status": "Queued"}]})
    return tmp_path, submitted


def test_end_to_end_check_build_submit_status(sandbox, capsys):
    root, submitted = sandbox
    gf.cmd_check(CFG, None)
    assert "subsetting_available = True" in capsys.readouterr().out
    gf.cmd_build(CFG, None)
    out = capsys.readouterr().out
    assert "Semua param, produk, dan level ditemukan" in out
    assert (root / "configs/gdex_2020.ctl").exists()
    assert "oformat=grib2" in (root / "configs/gdex_test.ctl").read_text()

    class A: labels, force, resubmit = ["years"], False, False
    gf.cmd_submit(CFG, A)
    assert len(submitted) == 5
    reqs = gf.load_requests()
    assert reqs["2016"]["request_id"] == 900001

    gf.cmd_submit(CFG, A)  # tidak dobel
    assert len(submitted) == 5

    gf.cmd_status(CFG, None)
    assert "status=Queued" in capsys.readouterr().out
    log = (root / "decisions.md").read_text()
    assert "subsetting_available = True" in log and "request 2020 disubmit" in log

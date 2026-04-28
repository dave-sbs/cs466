"""Tests for scripts.alignment_report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def import_alignment_report():
    """Load the module dynamically since it's under scripts/."""
    import importlib.util
    import sys

    code_dir = Path(__file__).resolve().parent.parent
    mod_path = code_dir / "scripts" / "alignment_report.py"
    spec = importlib.util.spec_from_file_location("alignment_report", mod_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alignment_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_alignment(pg_raw: Path, gid: int, data: dict) -> None:
    pg_raw.mkdir(parents=True, exist_ok=True)
    (pg_raw / f"alignment_{gid}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def _write_ids_csv(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "gutenberg_id\n" + "\n".join(str(i) for i in ids) + "\n",
        encoding="utf-8",
    )


def test_summarise_ok_and_failed(tmp_path: Path, import_alignment_report):
    pg_raw = tmp_path / "pg_raw"
    _write_alignment(
        pg_raw,
        9825,
        {
            "status": "ok",
            "match_rate": 0.9995,
            "stanza_count": 12,
        },
    )
    _write_alignment(
        pg_raw,
        24449,
        {
            "status": "failed",
            "match_rate": 0.9362,
            "stanza_count": 0,
            "warning": "match_rate 0.9362 < 0.98; poem file not written.",
        },
    )

    rows = import_alignment_report.summarise([9825, 24449, 99999], pg_raw)
    assert len(rows) == 3
    by_id = {r.gutenberg_id: r for r in rows}
    assert by_id[9825].status == "ok"
    assert by_id[9825].stanza_count == 12
    assert by_id[24449].status == "failed"
    assert by_id[24449].match_rate == pytest.approx(0.9362, abs=1e-4)
    assert by_id[99999].status == "missing"


def test_write_report_produces_both_files(tmp_path: Path, import_alignment_report):
    pg_raw = tmp_path / "pg_raw"
    _write_alignment(
        pg_raw, 9825, {"status": "ok", "match_rate": 1.0, "stanza_count": 12}
    )

    out_prefix = tmp_path / "report"
    rows = import_alignment_report.summarise([9825], pg_raw)
    csv_path, json_path = import_alignment_report.write_report(rows, out_prefix)

    assert csv_path.exists() and csv_path.suffix == ".csv"
    assert json_path.exists() and json_path.suffix == ".json"
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["rows"][0]["status"] == "ok"


def test_main_cli_end_to_end(tmp_path: Path, import_alignment_report, capsys):
    ids_csv = tmp_path / "ids.csv"
    pg_raw = tmp_path / "pg_raw"
    _write_ids_csv(ids_csv, [9825, 24449])
    _write_alignment(
        pg_raw, 9825, {"status": "ok", "match_rate": 0.999, "stanza_count": 12}
    )
    _write_alignment(
        pg_raw,
        24449,
        {
            "status": "failed",
            "match_rate": 0.93,
            "stanza_count": 0,
            "warning": "below threshold",
        },
    )

    rc = import_alignment_report.main(
        [
            "--ids-csv",
            str(ids_csv),
            "--pg-raw",
            str(pg_raw),
            "--out",
            str(tmp_path / "out" / "report"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 ok" in out
    assert "1 failed" in out


def test_main_missing_ids_csv_returns_nonzero(tmp_path: Path, import_alignment_report):
    rc = import_alignment_report.main(
        [
            "--ids-csv",
            str(tmp_path / "missing.csv"),
            "--pg-raw",
            str(tmp_path),
            "--out",
            str(tmp_path / "rep"),
        ]
    )
    assert rc == 2

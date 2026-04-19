"""Tests for dream_preflight (checks module + CLI)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dream_preflight import run_preflight
from dream_preflight.__main__ import build_parser, main


def _stage(data_root: Path, llm_record: dict, manifest: dict, image_bytes: bytes = b"\xff\xd8") -> None:
    """Copy fixtures into the expected DATA_ROOT subtree."""
    (data_root / "exploration_output").mkdir(parents=True)
    (data_root / "exploration_output" / "llm_analysis.jsonl").write_text(
        json.dumps(llm_record) + "\n", encoding="utf-8"
    )

    manifest_dir = (
        data_root
        / "output"
        / "retrieval_results"
        / f"poem_{llm_record['gutenberg_id']}"
    )
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "retrieval_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    images_dir = data_root / "data" / "images"
    images_dir.mkdir(parents=True)
    for r in manifest["results"]:
        img_id = r["top_k"][0]["image_id"]
        (images_dir / img_id).write_bytes(image_bytes)


# ------------------ library-level tests ------------------


def test_preflight_ok(tmp_path: Path, minimal_llm_record, minimal_manifest_dict):
    _stage(tmp_path, minimal_llm_record, minimal_manifest_dict)
    report = run_preflight(gutenberg_id=9825, data_root=tmp_path)
    assert report.ok is True
    assert report.num_scenes == 2
    assert report.num_chunks == 2
    assert report.num_image_paths_checked == 2
    assert report.errors == []


def test_preflight_missing_data_root(tmp_path: Path):
    missing = tmp_path / "nope"
    report = run_preflight(gutenberg_id=9825, data_root=missing)
    assert report.ok is False
    assert any("data_root" in e for e in report.errors)


def test_preflight_missing_jsonl(tmp_path: Path, minimal_manifest_dict):
    tmp_path.mkdir(exist_ok=True)
    # Only stage manifest + images, no llm jsonl
    manifest_dir = tmp_path / "output" / "retrieval_results" / "poem_9825"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "retrieval_manifest.json").write_text(
        json.dumps(minimal_manifest_dict), encoding="utf-8"
    )
    (tmp_path / "data" / "images").mkdir(parents=True)
    for r in minimal_manifest_dict["results"]:
        (tmp_path / "data" / "images" / r["top_k"][0]["image_id"]).write_bytes(b"x")

    report = run_preflight(gutenberg_id=9825, data_root=tmp_path)
    assert report.ok is False
    assert any("llm_analysis.jsonl" in e for e in report.errors)


def test_preflight_mismatched_counts(tmp_path: Path, minimal_llm_record, minimal_manifest_dict):
    bad_record = dict(minimal_llm_record)
    bad_record["visual_scenes"] = minimal_llm_record["visual_scenes"][:1]
    _stage(tmp_path, bad_record, minimal_manifest_dict)

    report = run_preflight(gutenberg_id=9825, data_root=tmp_path)
    assert report.ok is False
    assert any("pairing failed" in e or "mismatch" in e for e in report.errors)


def test_preflight_missing_images(tmp_path: Path, minimal_llm_record, minimal_manifest_dict):
    _stage(tmp_path, minimal_llm_record, minimal_manifest_dict)
    # Remove one image
    for r in minimal_manifest_dict["results"][:1]:
        (tmp_path / "data" / "images" / r["top_k"][0]["image_id"]).unlink()

    report = run_preflight(gutenberg_id=9825, data_root=tmp_path)
    assert report.ok is False
    assert any("image not found" in e for e in report.errors)


def test_preflight_parse_error_record(tmp_path: Path, minimal_llm_record, minimal_manifest_dict):
    bad = dict(minimal_llm_record, llm_parse_error=True, error="boom")
    _stage(tmp_path, bad, minimal_manifest_dict)
    report = run_preflight(gutenberg_id=9825, data_root=tmp_path)
    assert report.ok is False
    assert any("validation" in e or "llm_parse_error" in e for e in report.errors)


# ------------------ CLI / --json mode ------------------


def test_cli_help_exits_zero(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_json_ok(tmp_path: Path, minimal_llm_record, minimal_manifest_dict, capsys):
    _stage(tmp_path, minimal_llm_record, minimal_manifest_dict)
    rc = main(
        [
            "--gutenberg-id",
            "9825",
            "--data-root",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["ok"] is True
    # Stable key ordering
    assert list(report.keys()) == sorted(report.keys())


def test_cli_plan_only_returns_frame_totals(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, capsys
):
    _stage(tmp_path, minimal_llm_record, minimal_manifest_dict)
    rc = main(
        [
            "--gutenberg-id",
            "9825",
            "--data-root",
            str(tmp_path),
            "--json",
            "--plan-only",
        ]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    # Plan estimate populated
    assert isinstance(report["plan_total_frames"], int)
    assert report["plan_total_frames"] > 0
    assert report["plan_duration_seconds"] > 0


def test_cli_json_failure_nonzero(tmp_path: Path, capsys):
    rc = main(
        [
            "--gutenberg-id",
            "9825",
            "--data-root",
            str(tmp_path / "nope"),
            "--json",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["ok"] is False
    assert report["errors"]

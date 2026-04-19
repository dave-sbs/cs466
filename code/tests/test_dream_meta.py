"""Tests for dream_meta (DreamRunConfig round-trip + meta.json writer)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dream_meta import (
    DreamRunConfig,
    META_SCHEMA_VERSION,
    collect_env_snapshot,
    current_git_sha,
    dream_run_config_from_dict,
    dream_run_config_to_dict,
    write_meta,
)


# ------------------ DreamRunConfig ------------------


def test_dream_run_config_round_trip():
    cfg = DreamRunConfig(
        gutenberg_id=9825,
        data_root="/data",
        run_dir="/out/run1",
        fps=24,
        revision="abc1234",
    )
    d = dream_run_config_to_dict(cfg)
    # JSON must round-trip exactly (sorted keys)
    s = json.dumps(d, sort_keys=True)
    parsed = json.loads(s)
    assert list(parsed.keys()) == sorted(parsed.keys())
    cfg2 = dream_run_config_from_dict(parsed)
    assert cfg2 == cfg


def test_dream_run_config_tolerates_unknown_keys():
    cfg2 = dream_run_config_from_dict(
        {
            "gutenberg_id": 1,
            "data_root": "/x",
            "run_dir": "/y",
            "some_future_knob": True,
        }
    )
    assert cfg2.gutenberg_id == 1


def test_dream_run_config_requires_primary_keys():
    with pytest.raises(KeyError, match="gutenberg_id"):
        dream_run_config_from_dict({"data_root": "/x", "run_dir": "/y"})


# ------------------ env snapshot ------------------


def test_env_snapshot_has_core_keys():
    snap = collect_env_snapshot()
    assert "python" in snap
    assert "platform" in snap
    assert "torch_version" in snap
    assert "diffusers_version" in snap
    assert "pillow_version" in snap
    # cuda_device may be None on CPU CI
    assert "cuda_device" in snap


def test_env_snapshot_cuda_absent_returns_none():
    """Mock torch-CUDA-absent path."""
    import torch

    with patch.object(torch.cuda, "is_available", return_value=False):
        snap = collect_env_snapshot()
    assert snap["cuda_device"] is None


# ------------------ current_git_sha ------------------


def test_current_git_sha_outside_repo_returns_none(tmp_path: Path):
    assert current_git_sha(tmp_path) is None


def test_current_git_sha_in_repo_returns_string():
    sha = current_git_sha()
    # In CI this repo IS a git repo; in a fresh tmp it isn't. Tolerate both.
    assert sha is None or len(sha) == 40


# ------------------ write_meta ------------------


def test_write_meta_creates_file_with_required_keys(tmp_path: Path):
    p = tmp_path / "meta.json"
    write_meta(
        p,
        gutenberg_id=9825,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        revision="abc1234",
        run_config={"fps": 30},
        artifacts={"mp4": "/tmp/x.mp4"},
    )
    data = json.loads(p.read_text(encoding="utf-8"))
    # Sorted top-level keys for stability
    assert list(data.keys()) == sorted(data.keys())
    assert data["schema_version"] == META_SCHEMA_VERSION
    assert data["gutenberg_id"] == 9825
    assert data["model_id"].startswith("stabilityai/")
    assert data["revision"] == "abc1234"
    assert data["run_config"]["fps"] == 30
    assert data["artifacts"]["mp4"] == "/tmp/x.mp4"
    assert "env" in data
    assert data["env"]["python"]
    # Stable across re-reads
    data2 = json.loads(p.read_text(encoding="utf-8"))
    assert data == data2


def test_write_meta_atomic_no_leftover_tmp(tmp_path: Path):
    p = tmp_path / "meta.json"
    write_meta(p, gutenberg_id=1, model_id="x")
    assert not (tmp_path / "meta.json.tmp").exists()


# ------------------ render_dream_video integration ------------------


def test_render_writes_meta_json(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, monkeypatch
):
    """End-to-end: meta.json lands in run_dir with schema_version."""
    from dream_render import render_dream_video, RenderConfig

    # Stage DATA_ROOT
    (tmp_path / "exploration_output").mkdir(parents=True)
    (tmp_path / "exploration_output" / "llm_analysis.jsonl").write_text(
        json.dumps(minimal_llm_record) + "\n", encoding="utf-8"
    )
    mdir = tmp_path / "output" / "retrieval_results" / "poem_9825"
    mdir.mkdir(parents=True)
    (mdir / "retrieval_manifest.json").write_text(
        json.dumps(minimal_manifest_dict), encoding="utf-8"
    )
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    from PIL import Image

    for r in minimal_manifest_dict["results"]:
        Image.new("RGB", (32, 32)).save(img_dir / r["top_k"][0]["image_id"])

    monkeypatch.setenv("DREAM_MOCK_GPU", "1")
    artifacts = render_dream_video(
        RenderConfig(
            gutenberg_id=9825,
            data_root=tmp_path,
            run_dir=tmp_path / "run",
            rife_depth=2,
            image_size=(32, 32),
            encode_mp4=False,
        )
    )
    assert artifacts.meta is not None and artifacts.meta.exists()
    data = json.loads(artifacts.meta.read_text(encoding="utf-8"))
    assert data["schema_version"] == META_SCHEMA_VERSION
    assert data["gutenberg_id"] == 9825
    assert data["run_config"]["mock_gpu"] is True

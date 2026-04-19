"""Tests for dream_render orchestration."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from dream_render import DreamArtifacts, RenderConfig, render_dream_video


def _stage_all(
    data_root: Path,
    llm_record: dict,
    manifest: dict,
    image_size: tuple[int, int] = (64, 64),
) -> None:
    """Stage LLM + manifest + init images matching the rest of the fixtures."""
    (data_root / "exploration_output").mkdir(parents=True)
    (data_root / "exploration_output" / "llm_analysis.jsonl").write_text(
        json.dumps(llm_record) + "\n", encoding="utf-8"
    )
    mdir = (
        data_root
        / "output"
        / "retrieval_results"
        / f"poem_{llm_record['gutenberg_id']}"
    )
    mdir.mkdir(parents=True)
    (mdir / "retrieval_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    img_dir = data_root / "data" / "images"
    img_dir.mkdir(parents=True)
    for r in manifest["results"]:
        img_id = r["top_k"][0]["image_id"]
        Image.new("RGB", image_size, color=(80, 120, 160)).save(img_dir / img_id)


def _cfg(tmp_path: Path, **overrides) -> RenderConfig:
    defaults = dict(
        gutenberg_id=9825,
        data_root=tmp_path,
        run_dir=tmp_path / "run",
        rife_depth=2,  # small for fast tests: 3 mids per transition
        fps=30,
        image_size=(64, 64),
        encode_mp4=False,
    )
    defaults.update(overrides)
    return RenderConfig(**defaults)


def test_render_mock_gpu_happy_path(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, monkeypatch
):
    _stage_all(tmp_path, minimal_llm_record, minimal_manifest_dict)
    monkeypatch.setenv("DREAM_MOCK_GPU", "1")

    artifacts = render_dream_video(_cfg(tmp_path))
    assert isinstance(artifacts, DreamArtifacts)
    assert artifacts.keyframe_manifest.exists()
    assert artifacts.frame_manifest.exists()
    assert artifacts.mp4 is None

    # 2 stanzas -> 2 keyframes
    kfs = sorted((tmp_path / "run" / "keyframes").glob("kf_*.png"))
    assert len(kfs) == 2

    # Frame count from plan:
    #   stanza_intensity(0, 2, arc [2,4,3]) -> 2  (opening)
    #   stanza_intensity(1, 2, arc)         -> 3  (closing)
    #   hold(75) + rife_depth=2 mids(3) + hold(60) = 138 frames
    frames = sorted((tmp_path / "run" / "frames").glob("frame_*.png"))
    assert len(frames) == 138


def test_render_keyframe_manifest_shape(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, monkeypatch
):
    _stage_all(tmp_path, minimal_llm_record, minimal_manifest_dict)
    monkeypatch.setenv("DREAM_MOCK_GPU", "1")
    artifacts = render_dream_video(_cfg(tmp_path))
    manifest = json.loads(artifacts.keyframe_manifest.read_text())
    assert manifest["gutenberg_id"] == 9825
    assert manifest["schema_version"] == "1.0"
    assert len(manifest["entries"]) == 2
    for entry in manifest["entries"]:
        assert "sha256" in entry and len(entry["sha256"]) == 64


def test_render_mock_produces_tiny_mp4_when_ffmpeg_available(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, monkeypatch
):
    import shutil

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    _stage_all(tmp_path, minimal_llm_record, minimal_manifest_dict)
    monkeypatch.setenv("DREAM_MOCK_GPU", "1")
    artifacts = render_dream_video(_cfg(tmp_path, encode_mp4=True))
    assert artifacts.mp4 is not None
    assert artifacts.mp4.exists()
    # Size < 5 MB for this tiny 64x64 video
    assert artifacts.mp4.stat().st_size < 5 * 1024 * 1024


def test_resume_skips_existing_keyframes(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, monkeypatch
):
    _stage_all(tmp_path, minimal_llm_record, minimal_manifest_dict)
    monkeypatch.setenv("DREAM_MOCK_GPU", "1")

    calls = {"n": 0}

    # Wrap generate_keyframe to count calls
    from dream_render import orchestrate as orch

    original = orch.generate_keyframe

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(orch, "generate_keyframe", counting)

    render_dream_video(_cfg(tmp_path))
    assert calls["n"] == 2
    # Second run should be a no-op for keyframes
    calls["n"] = 0
    render_dream_video(_cfg(tmp_path))
    assert calls["n"] == 0


def test_resume_force_regenerates(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, monkeypatch
):
    _stage_all(tmp_path, minimal_llm_record, minimal_manifest_dict)
    monkeypatch.setenv("DREAM_MOCK_GPU", "1")
    from dream_render import orchestrate as orch

    calls = {"n": 0}
    original = orch.generate_keyframe

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(orch, "generate_keyframe", counting)

    render_dream_video(_cfg(tmp_path))
    assert calls["n"] == 2
    calls["n"] = 0
    render_dream_video(_cfg(tmp_path, force=True))
    assert calls["n"] == 2


def test_missing_llm_record_raises(tmp_path: Path, minimal_manifest_dict):
    # Stage only manifest + images, no jsonl
    mdir = tmp_path / "output" / "retrieval_results" / "poem_9825"
    mdir.mkdir(parents=True)
    (mdir / "retrieval_manifest.json").write_text(
        json.dumps(minimal_manifest_dict), encoding="utf-8"
    )
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    for r in minimal_manifest_dict["results"]:
        Image.new("RGB", (64, 64)).save(img_dir / r["top_k"][0]["image_id"])
    # Empty jsonl file present
    (tmp_path / "exploration_output").mkdir()
    (tmp_path / "exploration_output" / "llm_analysis.jsonl").write_text("", encoding="utf-8")

    from dream_data import DreamDataError

    with pytest.raises(DreamDataError, match="no LLM record"):
        render_dream_video(_cfg(tmp_path))


def test_start_end_stanza_slice(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, monkeypatch
):
    _stage_all(tmp_path, minimal_llm_record, minimal_manifest_dict)
    monkeypatch.setenv("DREAM_MOCK_GPU", "1")
    artifacts = render_dream_video(
        _cfg(tmp_path, start_stanza=1, end_stanza=2)
    )
    kfs = sorted((tmp_path / "run" / "keyframes").glob("kf_*.png"))
    assert len(kfs) == 1
    assert kfs[0].name == "kf_001.png"


def test_determinism_mock_path(
    tmp_path: Path, minimal_llm_record, minimal_manifest_dict, monkeypatch
):
    """CPU mock path must produce identical sha256 on first keyframe."""
    from dream_sdxl import sha256_file

    _stage_all(tmp_path, minimal_llm_record, minimal_manifest_dict)
    monkeypatch.setenv("DREAM_MOCK_GPU", "1")

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    render_dream_video(_cfg(tmp_path, run_dir=run1))
    render_dream_video(_cfg(tmp_path, run_dir=run2))

    assert (
        sha256_file(run1 / "keyframes" / "kf_000.png")
        == sha256_file(run2 / "keyframes" / "kf_000.png")
    )

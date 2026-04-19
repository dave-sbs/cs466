"""Tests for dream_sdxl (factory / providers / manifest / safety)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from dream_sdxl import (
    DEFAULT_SAFETY_POLICY,
    KEYFRAME_MANIFEST_SCHEMA_VERSION,
    KeyframeEntry,
    KeyframeManifest,
    KeyframeProvider,
    MockKeyframeProvider,
    generate_keyframe,
    load_keyframe_manifest,
    load_sdxl_img2img_pipe,
    sha256_file,
    write_keyframe_manifest,
)


# ---------------- factory ----------------


def test_load_sdxl_mock_returns_none():
    assert load_sdxl_img2img_pipe(mock=True) is None


def test_load_sdxl_rejects_bad_dtype():
    with pytest.raises(ValueError, match="dtype"):
        load_sdxl_img2img_pipe(mock=False, dtype="int8")


def test_load_sdxl_factory_applies_memory_flags():
    """Mock the underlying diffusers import and assert flags are applied."""
    fake_pipe = MagicMock()
    fake_pipe.to.return_value = fake_pipe

    with patch.dict(
        "sys.modules",
        {"diffusers": MagicMock(StableDiffusionXLImg2ImgPipeline=MagicMock(
            from_pretrained=MagicMock(return_value=fake_pipe)
        ))},
    ):
        pipe = load_sdxl_img2img_pipe(
            device="cpu",
            dtype="float32",
            enable_attention_slicing=True,
            enable_vae_tiling=True,
            enable_model_cpu_offload=False,
            mock=False,
        )

    assert pipe is fake_pipe
    fake_pipe.enable_attention_slicing.assert_called_once()
    fake_pipe.enable_vae_tiling.assert_called_once()
    fake_pipe.enable_model_cpu_offload.assert_not_called()


# ---------------- generate_keyframe contract ----------------


def test_generate_keyframe_passes_kwargs():
    calls = {}

    def provider(**kwargs):
        calls.update(kwargs)
        return Image.new("RGB", (8, 8))

    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    out = generate_keyframe(
        provider,
        prompt="p",
        init_image=img,
        strength=0.7,
        seed=42,
        negative_prompt="bad",
        prompt_2="extra",
    )
    assert isinstance(out, Image.Image)
    assert calls["prompt"] == "p"
    assert calls["strength"] == 0.7
    assert calls["seed"] == 42
    assert calls["negative_prompt"] == "bad"
    assert calls["prompt_2"] == "extra"


def test_generate_keyframe_writes_png(tmp_path: Path):
    provider = MockKeyframeProvider()
    out_path = tmp_path / "kf.png"
    img = Image.new("RGB", (16, 16), color=(1, 2, 3))
    generate_keyframe(
        provider,
        prompt="p",
        init_image=img,
        strength=0.6,
        seed=7,
        output_path=out_path,
    )
    assert out_path.exists()
    with Image.open(out_path) as saved:
        assert saved.size == (16, 16)


def test_mock_provider_deterministic():
    a = Image.new("RGB", (8, 8), color=(100, 100, 100))
    m = MockKeyframeProvider()
    out1 = m(prompt="x", init_image=a, strength=0.7, seed=123)
    out2 = m(prompt="x", init_image=a, strength=0.7, seed=123)
    assert list(out1.getdata()) == list(out2.getdata())


def test_mock_provider_varies_with_seed():
    a = Image.new("RGB", (8, 8), color=(100, 100, 100))
    m = MockKeyframeProvider()
    out1 = m(prompt="x", init_image=a, strength=0.7, seed=1)
    out2 = m(prompt="x", init_image=a, strength=0.7, seed=2)
    assert list(out1.getdata()) != list(out2.getdata())


def test_mock_provider_is_interpolator_shape():
    assert isinstance(MockKeyframeProvider(), KeyframeProvider)


# ---------------- manifest ----------------


def test_sha256_file_deterministic(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert sha256_file(p) == sha256_file(p)
    assert sha256_file(p) == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_manifest_round_trip(tmp_path: Path):
    kf = tmp_path / "kf.png"
    Image.new("RGB", (4, 4)).save(kf, format="PNG")
    digest = sha256_file(kf)

    entry = KeyframeEntry(
        stanza_index=0,
        image_id="00001.jpg",
        prompt="a quiet meadow",
        prompt_2="dreamlike",
        negative_prompt="text, watermark",
        strength=0.65,
        seed=123456,
        num_inference_steps=30,
        guidance_scale=7.0,
        output_path=str(kf),
        sha256=digest,
    )
    manifest = KeyframeManifest(
        gutenberg_id=9825,
        entries=[entry, dataclass_inc(entry, stanza_index=1)],
        revision="abc1234",
    )
    p = tmp_path / "keyframe_manifest.json"
    write_keyframe_manifest(p, manifest)

    assert p.exists()
    loaded = load_keyframe_manifest(p)
    assert loaded.gutenberg_id == 9825
    assert loaded.schema_version == KEYFRAME_MANIFEST_SCHEMA_VERSION
    assert loaded.revision == "abc1234"
    assert len(loaded.entries) == 2
    assert loaded.entries[0].sha256 == digest


def test_manifest_rejects_wrong_schema(tmp_path: Path):
    bad = {
        "gutenberg_id": 1,
        "schema_version": "99.0",
        "entries": [],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_keyframe_manifest(p)


def test_manifest_rejects_missing_keys(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_keyframe_manifest(p)


def test_manifest_atomic_write_leaves_no_tmp(tmp_path: Path):
    manifest = KeyframeManifest(gutenberg_id=1)
    p = tmp_path / "m.json"
    write_keyframe_manifest(p, manifest)
    assert p.exists()
    assert not (tmp_path / "m.json.tmp").exists()


# ---------------- safety ----------------


def test_default_safety_policy_enabled():
    assert DEFAULT_SAFETY_POLICY.enable_safety_checker is True
    assert DEFAULT_SAFETY_POLICY.nsfw_replacement in ("skip", "blur", "error")


# ---------------- helpers ----------------


def dataclass_inc(entry: KeyframeEntry, **overrides) -> KeyframeEntry:
    import dataclasses

    return dataclasses.replace(entry, **overrides)

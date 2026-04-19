"""Tests for dream_interp (Interpolator / Crossfade / RIFE stub)."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dream_interp import (
    CrossfadeInterpolator,
    Interpolator,
    InterpolatorError,
    RifeInterpolator,
    assert_within_frame_cap,
)


def _solid(color=(0, 0, 0)):
    return Image.new("RGB", (32, 24), color=color)


# ------------------ assert_within_frame_cap ------------------


def test_frame_cap_allows_default():
    assert_within_frame_cap(16)  # no raise


def test_frame_cap_raises_over_limit():
    with pytest.raises(InterpolatorError, match="cap"):
        assert_within_frame_cap(1000, cap=256)


# ------------------ CrossfadeInterpolator ------------------


def test_crossfade_produces_2pow_depth_minus_1_frames(tmp_path: Path):
    interp = CrossfadeInterpolator()
    out = interp(_solid((0, 0, 0)), _solid((255, 255, 255)), tmp_path, depth=4)
    assert len(out) == 15  # 2**4 - 1
    assert all(p.exists() for p in out)


def test_crossfade_monotonic_mean_rgb(tmp_path: Path):
    a = _solid((0, 0, 0))
    b = _solid((255, 255, 255))
    interp = CrossfadeInterpolator()
    paths = interp(a, b, tmp_path, depth=3)

    means = [sum(Image.open(p).getpixel((0, 0))) / 3 for p in paths]
    # strictly increasing from near 0 toward near 255
    for prev, cur in zip(means, means[1:]):
        assert cur > prev, (prev, cur)
    assert means[0] > 0
    assert means[-1] < 255  # never includes endpoint


def test_crossfade_mismatched_sizes_raises(tmp_path: Path):
    a = Image.new("RGB", (32, 24))
    b = Image.new("RGB", (16, 16))
    with pytest.raises(InterpolatorError, match="sizes differ"):
        CrossfadeInterpolator()(a, b, tmp_path, depth=2)


def test_crossfade_converts_mode_on_b(tmp_path: Path):
    a = Image.new("RGB", (16, 16), color=(128, 128, 128))
    b = Image.new("L", (16, 16), color=200)  # grayscale
    paths = CrossfadeInterpolator()(a, b, tmp_path, depth=1)
    assert len(paths) == 1
    # Output should be RGB (matches a)
    with Image.open(paths[0]) as img:
        assert img.mode == "RGB"


def test_crossfade_conforms_to_protocol():
    interp = CrossfadeInterpolator()
    assert isinstance(interp, Interpolator)


def test_crossfade_respects_max_cap(tmp_path: Path):
    # depth 10 would produce 1023 frames, > 256 cap
    with pytest.raises(InterpolatorError, match="cap"):
        CrossfadeInterpolator()(_solid(), _solid(), tmp_path, depth=10)


# ------------------ RifeInterpolator stub ------------------


def test_rife_stub_raises_when_no_root(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("RIFE_ROOT", raising=False)
    interp = RifeInterpolator()
    with pytest.raises(InterpolatorError, match="RIFE_ROOT"):
        interp(_solid(), _solid(), tmp_path, depth=2)


def test_rife_stub_raises_when_root_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RIFE_ROOT", str(tmp_path / "does_not_exist"))
    interp = RifeInterpolator()
    with pytest.raises(InterpolatorError, match="does not exist"):
        interp(_solid(), _solid(), tmp_path, depth=2)


def test_rife_stub_raises_not_implemented_when_root_exists(
    monkeypatch, tmp_path: Path
):
    fake_root = tmp_path / "rife"
    fake_root.mkdir()
    monkeypatch.setenv("RIFE_ROOT", str(fake_root))
    interp = RifeInterpolator()
    with pytest.raises(InterpolatorError, match="not implemented"):
        interp(_solid(), _solid(), tmp_path / "out", depth=2)


def test_rife_stub_frame_cap_before_root_check(tmp_path: Path):
    """Bad depth should fail fast, before touching RIFE_ROOT."""
    interp = RifeInterpolator(rife_root="/nope")
    with pytest.raises(InterpolatorError, match="cap"):
        interp(_solid(), _solid(), tmp_path, depth=20)


def test_rife_stub_conforms_to_protocol():
    assert isinstance(RifeInterpolator(), Interpolator)

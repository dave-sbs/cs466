"""Tests for dream_frames.ken_burns."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dream_frames import apply_ken_burns, ken_burns_frame


def _solid(w: int = 64, h: int = 48, color=(200, 100, 50)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


def test_first_frame_is_identity_copy():
    src = _solid()
    out = ken_burns_frame(src, 0, total_frames=10)
    assert out.size == src.size
    # Center pixel sanity — with zoom=1.0 and solid color, the pixel
    # should equal the source pixel.
    assert out.getpixel((0, 0)) == src.getpixel((0, 0))


def test_last_frame_same_size_but_resampled():
    src = _solid()
    out = ken_burns_frame(src, total_frames=10, frame_idx=9, max_zoom=1.2)
    assert out.size == src.size


def test_single_frame_returns_identity():
    src = _solid()
    out = ken_burns_frame(src, 0, total_frames=1, max_zoom=1.5)
    assert out.size == src.size
    assert out.getpixel((0, 0)) == src.getpixel((0, 0))


def test_rejects_out_of_range_frame_idx():
    with pytest.raises(ValueError):
        ken_burns_frame(_solid(), -1, 10)
    with pytest.raises(ValueError):
        ken_burns_frame(_solid(), 10, 10)


def test_rejects_zero_total_frames():
    with pytest.raises(ValueError):
        ken_burns_frame(_solid(), 0, 0)


def test_rejects_max_zoom_less_than_one():
    with pytest.raises(ValueError):
        ken_burns_frame(_solid(), 0, 10, max_zoom=0.9)


def test_apply_ken_burns_generator_yields_n_frames(tmp_path: Path):
    frames = list(apply_ken_burns(_solid(), total_frames=5, max_zoom=1.1))
    assert len(frames) == 5
    for f in frames:
        assert f.size == (64, 48)


def test_hold_strip_written_to_disk(tmp_path: Path):
    """Sprint demo check: write 90 hold PNGs and count them."""
    src = _solid(32, 32)
    for i, f in enumerate(apply_ken_burns(src, total_frames=90, max_zoom=1.05)):
        f.save(tmp_path / f"frame_{i:05d}.png", format="PNG")
    files = sorted(tmp_path.glob("frame_*.png"))
    assert len(files) == 90

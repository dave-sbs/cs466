"""Unit tests for RifeInterpolator recursion using a stub model.

These tests do not require Practical-RIFE to be installed and do not require
CUDA. They monkeypatch the model loader so we can validate ordering, frame
count, and basic monotonicity properties.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from dream_interp import RifeInterpolator


def _solid(color=(0, 0, 0), size=(32, 32)) -> Image.Image:
    return Image.new("RGB", size, color=color)


class _StubModel:
    def inference(self, a, b):
        # a/b are torch tensors [1, 3, H, W] in [0, 1].
        return (a + b) / 2.0


def test_rife_recursion_writes_2pow_depth_minus_1_pngs(tmp_path: Path, monkeypatch):
    interp = RifeInterpolator(rife_root=tmp_path)  # root is ignored by patched methods

    monkeypatch.setattr(interp, "_validate_install", lambda root: None)
    monkeypatch.setattr(interp, "_load_model", lambda root: _StubModel())

    out_dir = tmp_path / "out"
    paths = interp(_solid((0, 0, 0)), _solid((255, 255, 255)), out_dir, depth=3)
    assert len(paths) == 7  # 2**3 - 1
    assert [p.name for p in paths] == [f"mid_{i:04d}.png" for i in range(7)]
    assert all(p.exists() for p in paths)


def test_rife_recursion_is_monotonic_mean_rgb(tmp_path: Path, monkeypatch):
    interp = RifeInterpolator(rife_root=tmp_path)
    monkeypatch.setattr(interp, "_validate_install", lambda root: None)
    monkeypatch.setattr(interp, "_load_model", lambda root: _StubModel())

    out_dir = tmp_path / "out2"
    paths = interp(_solid((0, 0, 0)), _solid((255, 255, 255)), out_dir, depth=3)

    means = [sum(Image.open(p).getpixel((0, 0))) / 3 for p in paths]
    for prev, cur in zip(means, means[1:]):
        assert cur > prev, (prev, cur)
    assert means[0] > 0
    assert means[-1] < 255


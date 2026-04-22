"""GPU-gated integration test for RifeInterpolator.

Requires:
- CUDA-enabled torch
- Practical-RIFE checked out at $RIFE_ROOT with weights under train_log/
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from dream_interp import RifeInterpolator


@pytest.mark.gpu
def test_rife_integration_produces_expected_frame_count(tmp_path: Path):
    try:
        import torch  # noqa: WPS433
    except Exception:
        pytest.skip("torch not installed")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not os.environ.get("RIFE_ROOT"):
        pytest.skip("RIFE_ROOT not set")

    a = Image.new("RGB", (512, 512), color=(255, 0, 0))
    b = Image.new("RGB", (512, 512), color=(0, 0, 255))

    out_dir = tmp_path / "rife"
    paths = RifeInterpolator()(a, b, out_dir, depth=2)
    assert len(paths) == 3  # 2**2 - 1
    assert all(p.exists() for p in paths)

    # Basic sanity: midpoint shouldn't be pure red or pure blue.
    mid = Image.open(paths[1])
    r, g, bl = mid.getpixel((0, 0))
    assert 0 < r < 255
    assert 0 <= g <= 255
    assert 0 < bl < 255


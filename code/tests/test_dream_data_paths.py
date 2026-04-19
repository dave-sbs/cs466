"""Tests for dream_data.paths."""
from __future__ import annotations

from pathlib import Path

import pytest

from dream_data import resolve_top1_image_path


def test_resolve_flat_basename():
    p = resolve_top1_image_path("/root", "00879.jpg")
    assert p == Path("/root/data/images/00879.jpg")


def test_resolve_nested_id():
    p = resolve_top1_image_path("/root", "unsplash/xyz.jpg")
    assert p == Path("/root/data/images/unsplash/xyz.jpg")


def test_resolve_pathlib_data_root(tmp_path: Path):
    p = resolve_top1_image_path(tmp_path, "00001.jpg")
    assert p == tmp_path / "data" / "images" / "00001.jpg"


def test_resolve_empty_image_id_raises():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_top1_image_path("/root", "")


def test_resolve_absolute_image_id_raises():
    with pytest.raises(ValueError, match="relative"):
        resolve_top1_image_path("/root", "/abs/path.jpg")

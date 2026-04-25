from __future__ import annotations

from PIL import Image

from dream_wan.flf2v import aspect_ratio_resize, center_crop_resize


def test_aspect_ratio_resize_rounds_to_mod_value():
    img = Image.new("RGB", (123, 456), color=(10, 20, 30))
    out, h, w = aspect_ratio_resize(img, max_area=720 * 1280, mod_value=32)
    assert out.size == (w, h)
    assert h % 32 == 0
    assert w % 32 == 0
    assert h > 0 and w > 0


def test_center_crop_resize_matches_target():
    img = Image.new("RGB", (300, 100), color=(0, 0, 0))
    out = center_crop_resize(img, height=128, width=256)
    assert out.size == (256, 128)


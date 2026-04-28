from __future__ import annotations

import pytest

from dream_wan import WanFlf2vConfig


def test_wan_config_defaults_smoke():
    cfg = WanFlf2vConfig()
    assert cfg.model_id
    assert cfg.max_area > 0
    assert cfg.num_frames > 0
    assert cfg.fps > 0
    assert cfg.guidance_scale > 0


@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16"])
def test_wan_config_allows_dtype(dtype: str):
    WanFlf2vConfig(torch_dtype=dtype)


def test_wan_config_rejects_bad_dtype():
    with pytest.raises(ValueError, match="torch_dtype"):
        WanFlf2vConfig(torch_dtype="fp8")  # type: ignore[arg-type]


def test_wan_config_rejects_bad_num_frames():
    with pytest.raises(ValueError, match="num_frames"):
        WanFlf2vConfig(num_frames=0)


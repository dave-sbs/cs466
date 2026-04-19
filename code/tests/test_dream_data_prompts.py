"""Tests for dream_data.prompts."""
from __future__ import annotations

import pytest

from dream_data import (
    DEFAULT_STYLE_PROMPT_2,
    DEFAULT_STYLE_TAIL,
    build_sdxl_prompt,
)
from dream_data.prompts import DEFAULT_NEGATIVE_PROMPT


def test_build_prompt_basic(minimal_llm_record):
    scene = minimal_llm_record["visual_scenes"][0]
    prompt, prompt_2 = build_sdxl_prompt(scene)

    assert "meadow at dawn" in prompt
    assert "amber, slate color palette" in prompt
    assert "dawn lighting" in prompt
    assert DEFAULT_STYLE_TAIL in prompt
    assert prompt_2 == DEFAULT_STYLE_PROMPT_2


def test_build_prompt_no_python_list_literal(minimal_llm_record):
    """Regression: must not stringify `dominant_colors` as a Python list."""
    scene = minimal_llm_record["visual_scenes"][0]
    prompt, _ = build_sdxl_prompt(scene)
    assert "[" not in prompt
    assert "']" not in prompt


def test_build_prompt_skips_unspecified_time_of_day():
    scene = {
        "scene_description": "soft river bend",
        "dominant_colors": ["blue"],
        "time_of_day": "unspecified",
    }
    prompt, _ = build_sdxl_prompt(scene, style_tail="", style_prompt_2=None)
    assert "lighting" not in prompt
    assert prompt == "soft river bend, blue color palette"


def test_build_prompt_empty_colors_ok():
    scene = {
        "scene_description": "pale fog",
        "dominant_colors": [],
        "time_of_day": "dawn",
    }
    prompt, _ = build_sdxl_prompt(scene, style_tail="", style_prompt_2=None)
    assert "color palette" not in prompt
    assert prompt == "pale fog, dawn lighting"


def test_build_prompt_missing_description_raises():
    with pytest.raises(ValueError, match="scene_description"):
        build_sdxl_prompt({"scene_description": "", "dominant_colors": ["x"]})


def test_build_prompt_wrong_colors_type_raises():
    with pytest.raises(TypeError, match="dominant_colors"):
        build_sdxl_prompt(
            {"scene_description": "x", "dominant_colors": "red,blue"}
        )


def test_build_prompt_snapshot(minimal_llm_record):
    """Stable snapshot: lock the string format so iteration is deliberate."""
    scene = minimal_llm_record["visual_scenes"][0]
    prompt, prompt_2 = build_sdxl_prompt(scene)
    expected = (
        "A quiet meadow at dawn with pale mist, "
        "amber, slate color palette, "
        "dawn lighting, "
        f"{DEFAULT_STYLE_TAIL}"
    )
    assert prompt == expected
    assert prompt_2 == DEFAULT_STYLE_PROMPT_2


def test_negative_prompt_constant_contains_key_exclusions():
    assert "watermark" in DEFAULT_NEGATIVE_PROMPT
    assert "ui elements" in DEFAULT_NEGATIVE_PROMPT

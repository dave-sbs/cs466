"""Tests for dream_data.validate."""
from __future__ import annotations

import pytest

from dream_data import DreamDataError, validate_llm_record


def test_validate_ok(minimal_llm_record):
    validate_llm_record(minimal_llm_record)  # does not raise


def test_validate_accepts_extra_keys(minimal_llm_record):
    rec = dict(minimal_llm_record, schema_version="1.0", extra_junk=[1, 2])
    validate_llm_record(rec)


def test_validate_rejects_non_dict():
    with pytest.raises(DreamDataError, match="must be a dict"):
        validate_llm_record([1, 2, 3])  # type: ignore[arg-type]


def test_validate_parse_error(minimal_llm_record):
    rec = dict(minimal_llm_record, llm_parse_error=True, error="bad")
    with pytest.raises(DreamDataError, match="llm_parse_error"):
        validate_llm_record(rec)


def test_validate_empty_scenes(minimal_llm_record):
    rec = dict(minimal_llm_record, visual_scenes=[])
    with pytest.raises(DreamDataError, match="visual_scenes"):
        validate_llm_record(rec)


def test_validate_missing_scene_key(minimal_llm_record):
    rec = dict(
        minimal_llm_record,
        visual_scenes=[{"stanza_index": 0, "scene_description": "x"}],
    )
    with pytest.raises(DreamDataError, match="dominant_colors"):
        validate_llm_record(rec)


def test_validate_wrong_colors_type(minimal_llm_record):
    rec = dict(minimal_llm_record)
    rec["visual_scenes"] = [
        {
            "stanza_index": 0,
            "scene_description": "x",
            "dominant_colors": "red",
            "time_of_day": "dawn",
        }
    ]
    rec["mood_arc"] = [{"intensity": 1}, {"intensity": 1}, {"intensity": 1}]
    with pytest.raises(DreamDataError, match="dominant_colors"):
        validate_llm_record(rec)


def test_validate_mood_arc_wrong_length(minimal_llm_record):
    rec = dict(minimal_llm_record, mood_arc=[{"intensity": 1}] * 2)
    with pytest.raises(DreamDataError, match="3 entries"):
        validate_llm_record(rec)


def test_validate_mood_intensity_out_of_range(minimal_llm_record):
    rec = dict(minimal_llm_record)
    rec["mood_arc"] = [
        {"intensity": 0},
        {"intensity": 3},
        {"intensity": 2},
    ]
    with pytest.raises(DreamDataError, match="1..5"):
        validate_llm_record(rec)


def test_validate_mood_intensity_wrong_type(minimal_llm_record):
    rec = dict(minimal_llm_record)
    rec["mood_arc"] = [
        {"intensity": "hot"},
        {"intensity": 3},
        {"intensity": 2},
    ]
    with pytest.raises(DreamDataError, match="intensity"):
        validate_llm_record(rec)

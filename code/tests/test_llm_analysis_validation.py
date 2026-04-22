from __future__ import annotations

import pytest

from llm_analysis import PoemAnalysis


def _base_record(num_chunks: int) -> dict:
    return {
        "num_chunks": num_chunks,
        "is_poem": True,
        "content_type": "poem",
        "content_type_rationale": "A poem.",
        "title": "T",
        "author": "A",
        "genre": "lyric",
        "is_collection": False,
        "themes": ["winter"],
        "primary_theme": "winter",
        "visual_scenes": [
            {
                "stanza_index": i,
                "scene_description": f"Scene {i}",
                "dominant_colors": ["blue"],
                "time_of_day": "night",
                "season": "winter",
            }
            for i in range(num_chunks)
        ],
        "mood_arc": [
            {"position": "opening", "mood": "calm", "intensity": 2},
            {"position": "middle", "mood": "tense", "intensity": 3},
            {"position": "closing", "mood": "calm", "intensity": 2},
        ],
        "overall_mood": "calm",
        "nature_categories": ["snow"],
        "primary_nature_setting": "snow",
        "language": "English",
        "ocr_artifacts_detected": False,
        "has_non_poem_content": False,
        "non_poem_content_types": [],
        "visualization_suitability": 5,
        "visualization_rationale": "Good visuals.",
        "most_visual_stanzas": [0],
        "notable_lines": ["x"],
    }


def test_poem_analysis_accepts_matching_scene_count_and_indices():
    record = _base_record(num_chunks=3)
    parsed = PoemAnalysis.model_validate(record)
    assert parsed.num_chunks == 3
    assert len(parsed.visual_scenes) == 3


def test_poem_analysis_rejects_scene_count_mismatch():
    record = _base_record(num_chunks=3)
    record["visual_scenes"] = record["visual_scenes"][:2]
    with pytest.raises(ValueError, match="visual_scenes length must equal num_chunks"):
        PoemAnalysis.model_validate(record)


def test_poem_analysis_rejects_non_sequential_stanza_index():
    record = _base_record(num_chunks=3)
    record["visual_scenes"][0]["stanza_index"] = 5
    with pytest.raises(ValueError, match="stanza_index must be sequential"):
        PoemAnalysis.model_validate(record)


"""Shared pytest fixtures for the dream pipeline tests.

Keep this file minimal: only fixtures used by multiple test modules.
No heavy imports (torch, diffusers) here — see module-level fixtures for those.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


# Make `code/` importable so tests can `from dream_data import ...` when
# pytest is invoked from either the repo root or `code/`.
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


# Guard against OpenMP conflicts on macOS CI (matches pattern used in clip_pipeline).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to the committed fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_gutenberg_id() -> int:
    """The showcase poem used end-to-end by the dream pipeline."""
    return 9825


@pytest.fixture
def minimal_manifest_dict() -> dict:
    """In-memory minimal retrieval manifest (2 chunks, top-1 only)."""
    return {
        "poem_name": "poem_9825",
        "gutenberg_id": 9825,
        "num_chunks": 2,
        "top_k": 1,
        "results": [
            {
                "chunk_index": 0,
                "query_text": "stanza zero / line two",
                "top_k": [
                    {
                        "rank": 1,
                        "score": 0.25,
                        "image_id": "00001.jpg",
                        "output_file": "chunk000_rank1_00001.jpg",
                    }
                ],
            },
            {
                "chunk_index": 1,
                "query_text": "stanza one / line two",
                "top_k": [
                    {
                        "rank": 1,
                        "score": 0.23,
                        "image_id": "00002.jpg",
                        "output_file": "chunk001_rank1_00002.jpg",
                    }
                ],
            },
        ],
    }


@pytest.fixture
def minimal_llm_record() -> dict:
    """In-memory minimal LLM analysis record matching minimal_manifest_dict."""
    return {
        "gutenberg_id": 9825,
        "is_poem": True,
        "content_type": "poem",
        "llm_parse_error": False,
        "title": "Fixture poem",
        "author": "Test",
        "genre": "lyric",
        "is_collection": False,
        "themes": ["fixture"],
        "primary_theme": "fixture",
        "visual_scenes": [
            {
                "stanza_index": 0,
                "scene_description": "A quiet meadow at dawn with pale mist",
                "dominant_colors": ["amber", "slate"],
                "time_of_day": "dawn",
                "season": "spring",
            },
            {
                "stanza_index": 1,
                "scene_description": "A storm gathering over dark pines at dusk",
                "dominant_colors": ["indigo", "charcoal"],
                "time_of_day": "dusk",
                "season": "autumn",
            },
        ],
        "mood_arc": [
            {"position": "opening", "mood": "calm", "intensity": 2},
            {"position": "middle", "mood": "tense", "intensity": 4},
            {"position": "closing", "mood": "release", "intensity": 3},
        ],
        "overall_mood": "contemplative",
        "nature_categories": ["meadow", "forest"],
        "primary_nature_setting": "meadow",
        "language": "English",
        "ocr_artifacts_detected": False,
        "has_non_poem_content": False,
        "non_poem_content_types": [],
        "visualization_suitability": 5,
        "visualization_rationale": "Clear scenes and arc",
        "most_visual_stanzas": [0, 1],
        "notable_lines": ["line one", "line two"],
    }


@pytest.fixture
def write_jsonl(tmp_path: Path):
    """Helper: write a list of records as JSONL and return the path."""

    def _write(records: list[dict], name: str = "llm_analysis.jsonl") -> Path:
        p = tmp_path / name
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return p

    return _write

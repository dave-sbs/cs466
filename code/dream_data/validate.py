"""Shape validation for LLM JSONL records used by the dream pipeline.

This is intentionally lighter than ``llm_analysis.PoemAnalysis`` (which
uses pydantic). We only validate what the dream pipeline actually reads
— so upstream schema drift on non-dream fields does not block video
generation.
"""
from __future__ import annotations

from typing import Any

from . import DreamDataError


REQUIRED_SCENE_KEYS = (
    "stanza_index",
    "scene_description",
    "dominant_colors",
    "time_of_day",
)
REQUIRED_MOOD_KEYS = ("intensity",)


def validate_llm_record(record: dict[str, Any]) -> None:
    """Raise ``DreamDataError`` if ``record`` is missing fields required for video.

    Checks performed (in order):

    1. ``llm_parse_error`` is falsy.
    2. ``visual_scenes`` is a non-empty list.
    3. Each scene has keys in :data:`REQUIRED_SCENE_KEYS`, with
       ``dominant_colors`` a list.
    4. ``mood_arc`` is a list of exactly 3 entries, each with integer
       ``intensity`` in 1..5.

    An optional ``schema_version`` field is allowed and ignored.
    """
    if not isinstance(record, dict):
        raise DreamDataError(
            f"record must be a dict, got {type(record).__name__}"
        )

    if record.get("llm_parse_error"):
        raise DreamDataError(
            f"record has llm_parse_error=True "
            f"(error={record.get('error', 'unknown')!r})"
        )

    scenes = record.get("visual_scenes")
    if not isinstance(scenes, list) or len(scenes) == 0:
        raise DreamDataError(
            "record.visual_scenes must be a non-empty list"
        )
    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise DreamDataError(
                f"visual_scenes[{i}] must be a dict, got {type(scene).__name__}"
            )
        for k in REQUIRED_SCENE_KEYS:
            if k not in scene:
                raise DreamDataError(
                    f"visual_scenes[{i}] missing required key {k!r}"
                )
        if not isinstance(scene["dominant_colors"], list):
            raise DreamDataError(
                f"visual_scenes[{i}].dominant_colors must be a list"
            )

    mood = record.get("mood_arc")
    if not isinstance(mood, list) or len(mood) != 3:
        raise DreamDataError(
            "record.mood_arc must be a list of exactly 3 entries"
        )
    for i, m in enumerate(mood):
        if not isinstance(m, dict):
            raise DreamDataError(
                f"mood_arc[{i}] must be a dict, got {type(m).__name__}"
            )
        for k in REQUIRED_MOOD_KEYS:
            if k not in m:
                raise DreamDataError(
                    f"mood_arc[{i}] missing required key {k!r}"
                )
        intensity = m["intensity"]
        if not isinstance(intensity, int) or not (1 <= intensity <= 5):
            raise DreamDataError(
                f"mood_arc[{i}].intensity must be an int in 1..5, "
                f"got {intensity!r}"
            )

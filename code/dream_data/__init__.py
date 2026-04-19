"""dream_data — pure-Python helpers for the dream pipeline data contract.

This package encodes the join between:
  - LLM analysis JSONL (produced by llm_analysis.py)
  - CLIP retrieval manifest JSON (produced by clip_pipeline.py retrieve)
  - Source image paths under data/images/

All helpers are importable without torch/diffusers/ffmpeg. GPU and binary
integrations live in separate modules.
"""
from __future__ import annotations


class DreamDataError(ValueError):
    """Raised when LLM analysis and retrieval manifest disagree or are malformed."""


from .loaders import load_last_llm_record  # noqa: E402
from .manifest import (  # noqa: E402
    load_retrieval_manifest,
    pair_scenes_with_chunks,
    sort_manifest_results,
)
from .paths import resolve_top1_image_path  # noqa: E402
from .prompts import (  # noqa: E402
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_STYLE_PROMPT_2,
    DEFAULT_STYLE_TAIL,
    build_sdxl_prompt,
)
from .mood import mood_to_strength, stanza_intensity, stanza_seed  # noqa: E402

__all__ = [
    "DreamDataError",
    "load_last_llm_record",
    "load_retrieval_manifest",
    "pair_scenes_with_chunks",
    "sort_manifest_results",
    "resolve_top1_image_path",
    "build_sdxl_prompt",
    "DEFAULT_STYLE_TAIL",
    "DEFAULT_STYLE_PROMPT_2",
    "DEFAULT_NEGATIVE_PROMPT",
    "stanza_intensity",
    "mood_to_strength",
    "stanza_seed",
]

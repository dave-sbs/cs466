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


__all__ = ["DreamDataError"]

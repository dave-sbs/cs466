"""dream_wan — Wan 2.1 video generation helpers.

This package is intentionally standalone for prototype experimentation.
It must remain importable on CPU-only environments (no eager torch/diffusers).
"""

from __future__ import annotations

from .flf2v import WanFlf2vConfig, generate_wan_transition, load_wan_flf2v_pipeline

__all__ = [
    "WanFlf2vConfig",
    "load_wan_flf2v_pipeline",
    "generate_wan_transition",
]


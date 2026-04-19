"""dream_interp — frame interpolators between SDXL keyframes.

Two implementations:

- CrossfadeInterpolator: always available; linear alpha blend in RGB.
  Deterministic and fast; used for CI and as a MVP fallback.
- RifeInterpolator: thin stub that defers heavy imports to ``__call__``
  and raises a clear error if ``RIFE_ROOT`` is not configured.

Both conform to :class:`Interpolator` so the orchestration module can
swap them behind a single protocol.
"""
from __future__ import annotations

from .base import (
    DEFAULT_MAX_FRAMES_PER_CALL,
    Interpolator,
    InterpolatorError,
    assert_within_frame_cap,
)
from .crossfade import CrossfadeInterpolator
from .rife import RifeInterpolator

__all__ = [
    "Interpolator",
    "InterpolatorError",
    "CrossfadeInterpolator",
    "RifeInterpolator",
    "DEFAULT_MAX_FRAMES_PER_CALL",
    "assert_within_frame_cap",
]

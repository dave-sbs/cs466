"""dream_frames — CPU-only frame math for the dream pipeline.

- Ken Burns hold frames (PIL, deterministic).
- Hold duration tables from mood intensity.
- RIFE recursion-depth -> intermediate-frame count.
- Segment plan: ordered [hold_0, transition_0to1, hold_1, ...].

No torch / no ffmpeg / no RIFE imports. Pure math + PIL.
"""
from __future__ import annotations

from .ken_burns import apply_ken_burns, ken_burns_frame
from .plan import (
    Segment,
    SegmentPlan,
    build_segment_plan,
    hold_frame_count,
    rife_intermediate_count,
)

__all__ = [
    "ken_burns_frame",
    "apply_ken_burns",
    "Segment",
    "SegmentPlan",
    "build_segment_plan",
    "hold_frame_count",
    "rife_intermediate_count",
]

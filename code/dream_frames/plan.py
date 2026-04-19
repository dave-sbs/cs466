"""Segment plan: holds + transitions in render order.

Hold lengths come from mood intensity (at 30 fps). Transition lengths
come from the RIFE recursion depth: depth ``d`` produces ``2**d - 1``
intermediates between two keyframes, and the "transition span" is
``(intermediates + 2)`` to include both endpoints. We count the
endpoints once (as part of the hold they belong to) and only count
intermediates in the transition, so the grand total at playback is:

    total = sum(hold_frame_count) + sum(intermediates_per_transition)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Literal, Sequence


# Default hold counts at 30 fps.
# Calm stanzas linger longer to match the MVP's contemplative pacing.
_HOLD_TABLE = {
    1: 90,  # 3.0s — calmest
    2: 75,  # 2.5s
    3: 60,  # 2.0s — moderate
    4: 50,  # ~1.7s
    5: 45,  # 1.5s — intense
}


def hold_frame_count(intensity: int) -> int:
    """Map mood intensity 1..5 to a hold length in frames (at 30 fps).

    Raises ``ValueError`` for out-of-range intensity.
    """
    if intensity not in _HOLD_TABLE:
        raise ValueError(f"intensity must be 1..5, got {intensity!r}")
    return _HOLD_TABLE[intensity]


def rife_intermediate_count(depth: int) -> int:
    """Return the number of *intermediate* frames produced by recursion ``depth``.

    RIFE's 2x interpolator run recursively with depth ``d`` produces
    ``2**d - 1`` intermediate frames between a pair (excluding the two
    endpoints).

    ``depth`` must be >= 1.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth!r}")
    return (1 << depth) - 1


SegmentKind = Literal["hold", "transition"]


@dataclass(frozen=True)
class Segment:
    """One hold or transition segment in the render plan.

    ``stanza_to`` is None for hold segments, and is set for transitions.
    """

    kind: SegmentKind
    stanza_from: int
    frame_count: int
    stanza_to: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SegmentPlan:
    """Ordered list of segments + convenience totals."""

    fps: int
    rife_depth: int
    segments: tuple[Segment, ...]

    @property
    def total_frames(self) -> int:
        return sum(s.frame_count for s in self.segments)

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "rife_depth": self.rife_depth,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration_seconds,
            "segments": [s.to_dict() for s in self.segments],
        }


def build_segment_plan(
    intensities: Sequence[int],
    rife_depth: int = 4,
    fps: int = 30,
) -> SegmentPlan:
    """Build an ordered segment plan for ``len(intensities)`` keyframes.

    Pattern: ``hold_0, transition_0to1, hold_1, ..., transition_{N-2 to N-1}, hold_{N-1}``.

    For a single keyframe (``len(intensities) == 1``) the plan is just one hold.

    Raises ``ValueError`` if ``intensities`` is empty.
    """
    if not intensities:
        raise ValueError("intensities must be non-empty")
    for i, v in enumerate(intensities):
        if v not in _HOLD_TABLE:
            raise ValueError(
                f"intensities[{i}] must be in 1..5, got {v!r}"
            )

    mids = rife_intermediate_count(rife_depth)

    segs: list[Segment] = []
    for idx, intensity in enumerate(intensities):
        segs.append(
            Segment(
                kind="hold",
                stanza_from=idx,
                frame_count=hold_frame_count(intensity),
            )
        )
        if idx < len(intensities) - 1:
            segs.append(
                Segment(
                    kind="transition",
                    stanza_from=idx,
                    stanza_to=idx + 1,
                    frame_count=mids,
                )
            )

    return SegmentPlan(fps=fps, rife_depth=rife_depth, segments=tuple(segs))

"""Tests for dream_frames.plan: frame count math and segment plan shape."""
from __future__ import annotations

import json

import pytest

from dream_frames import (
    Segment,
    SegmentPlan,
    build_segment_plan,
    hold_frame_count,
    rife_intermediate_count,
)


# -------- hold_frame_count --------


@pytest.mark.parametrize(
    "intensity,frames",
    [(1, 90), (2, 75), (3, 60), (4, 50), (5, 45)],
)
def test_hold_frame_count_table(intensity: int, frames: int):
    assert hold_frame_count(intensity) == frames


def test_hold_frame_count_rejects_out_of_range():
    with pytest.raises(ValueError):
        hold_frame_count(0)
    with pytest.raises(ValueError):
        hold_frame_count(6)


def test_hold_order_calm_longer_than_intense():
    assert hold_frame_count(1) > hold_frame_count(5)


# -------- rife_intermediate_count --------


@pytest.mark.parametrize(
    "depth,mids",
    [(1, 1), (2, 3), (3, 7), (4, 15), (5, 31)],
)
def test_rife_intermediate_count(depth: int, mids: int):
    assert rife_intermediate_count(depth) == mids


def test_rife_intermediate_rejects_zero_depth():
    with pytest.raises(ValueError):
        rife_intermediate_count(0)


# -------- build_segment_plan --------


def test_plan_single_keyframe_is_one_hold():
    plan = build_segment_plan([3])
    assert len(plan.segments) == 1
    assert plan.segments[0].kind == "hold"
    assert plan.total_frames == hold_frame_count(3)


def test_plan_pattern_and_counts():
    intensities = [2, 4, 3]  # matches fixture mood_arc
    plan = build_segment_plan(intensities, rife_depth=4, fps=30)
    kinds = [s.kind for s in plan.segments]
    assert kinds == ["hold", "transition", "hold", "transition", "hold"]
    # Frame totals
    expected_holds = sum(hold_frame_count(i) for i in intensities)
    expected_trans = 2 * rife_intermediate_count(4)
    assert plan.total_frames == expected_holds + expected_trans


def test_plan_sum_invariant_twelve_stanzas():
    intensities = [1, 2, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2]
    plan = build_segment_plan(intensities, rife_depth=4, fps=30)
    # Manual sum from intensities + n-1 transitions of 15 frames
    expected = sum(hold_frame_count(i) for i in intensities) + 11 * 15
    assert plan.total_frames == expected
    # duration_seconds consistent
    assert plan.duration_seconds == pytest.approx(expected / 30)


def test_plan_rejects_empty():
    with pytest.raises(ValueError):
        build_segment_plan([])


def test_plan_to_dict_is_json_serializable():
    plan = build_segment_plan([1, 5], rife_depth=2)
    d = plan.to_dict()
    # Round-trip through json with sorted keys
    s = json.dumps(d, sort_keys=True)
    d2 = json.loads(s)
    assert d2["total_frames"] == plan.total_frames
    assert d2["rife_depth"] == 2
    assert d2["fps"] == 30


def test_plan_segment_stanza_indices():
    plan = build_segment_plan([3, 3, 3], rife_depth=1)
    # holds at stanza 0, 1, 2; transitions 0->1 and 1->2
    segs = plan.segments
    assert (segs[0].kind, segs[0].stanza_from) == ("hold", 0)
    assert (segs[1].kind, segs[1].stanza_from, segs[1].stanza_to) == (
        "transition",
        0,
        1,
    )
    assert (segs[2].kind, segs[2].stanza_from) == ("hold", 1)
    assert (segs[3].kind, segs[3].stanza_from, segs[3].stanza_to) == (
        "transition",
        1,
        2,
    )
    assert (segs[4].kind, segs[4].stanza_from) == ("hold", 2)


def test_plan_golden_snapshot():
    """Small golden snapshot to detect accidental schema drift."""
    plan = build_segment_plan([2, 4], rife_depth=2, fps=30)
    got = json.dumps(plan.to_dict(), sort_keys=True)
    expected = json.dumps(
        {
            "duration_seconds": (75 + 3 + 50) / 30,
            "fps": 30,
            "rife_depth": 2,
            "segments": [
                {"frame_count": 75, "kind": "hold", "stanza_from": 0, "stanza_to": None},
                {"frame_count": 3, "kind": "transition", "stanza_from": 0, "stanza_to": 1},
                {"frame_count": 50, "kind": "hold", "stanza_from": 1, "stanza_to": None},
            ],
            "total_frames": 128,
        },
        sort_keys=True,
    )
    assert got == expected

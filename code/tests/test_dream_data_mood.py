"""Tests for dream_data.mood: stanza_intensity, mood_to_strength, stanza_seed."""
from __future__ import annotations

import pytest

from dream_data import mood_to_strength, stanza_intensity, stanza_seed


ARC_2_4_3 = [
    {"position": "opening", "intensity": 2},
    {"position": "middle", "intensity": 4},
    {"position": "closing", "intensity": 3},
]


# ------------------ stanza_intensity ------------------


@pytest.mark.parametrize(
    "idx,n,expected",
    [
        (0, 5, 2),   # opening
        (1, 5, 3),   # half-way opening->middle
        (2, 5, 4),   # middle
        (3, 5, 4),   # half-way middle->closing: round(4*0.5 + 3*0.5) = 4 (banker's)
        (4, 5, 3),   # closing
    ],
)
def test_intensity_five_stanzas(idx: int, n: int, expected: int):
    assert stanza_intensity(idx, n, ARC_2_4_3) == expected


def test_intensity_single_stanza_returns_opening():
    assert stanza_intensity(0, 1, ARC_2_4_3) == 2


def test_intensity_clips_to_5():
    hot = [
        {"intensity": 5},
        {"intensity": 5},
        {"intensity": 5},
    ]
    assert stanza_intensity(0, 3, hot) == 5
    assert stanza_intensity(2, 3, hot) == 5


def test_intensity_requires_three_points():
    with pytest.raises(ValueError, match="exactly 3"):
        stanza_intensity(0, 5, ARC_2_4_3[:2])


def test_intensity_rejects_bad_index():
    with pytest.raises(ValueError):
        stanza_intensity(5, 5, ARC_2_4_3)
    with pytest.raises(ValueError):
        stanza_intensity(-1, 5, ARC_2_4_3)


def test_intensity_rejects_zero_stanzas():
    with pytest.raises(ValueError):
        stanza_intensity(0, 0, ARC_2_4_3)


# ------------------ mood_to_strength ------------------


@pytest.mark.parametrize(
    "intensity,strength",
    [(1, 0.55), (2, 0.60), (3, 0.65), (4, 0.68), (5, 0.70)],
)
def test_strength_table(intensity: int, strength: float):
    assert mood_to_strength(intensity) == pytest.approx(strength)


def test_strength_never_exceeds_0_70_cap():
    assert all(mood_to_strength(i) <= 0.70 for i in range(1, 6))


def test_strength_rejects_out_of_range():
    with pytest.raises(ValueError):
        mood_to_strength(0)
    with pytest.raises(ValueError):
        mood_to_strength(6)


# ------------------ stanza_seed ------------------


def test_seed_deterministic():
    assert stanza_seed(9825, 0) == stanza_seed(9825, 0)
    assert stanza_seed(9825, 1) != stanza_seed(9825, 0)


def test_seed_different_poems_differ():
    assert stanza_seed(9825, 0) != stanza_seed(24449, 0)


def test_seed_is_uint32():
    for gid in (0, 1, 9825, 10 ** 6):
        for si in range(0, 20):
            s = stanza_seed(gid, si)
            assert 0 <= s <= 0xFFFFFFFF


def test_seed_rejects_negative():
    with pytest.raises(ValueError):
        stanza_seed(-1, 0)
    with pytest.raises(ValueError):
        stanza_seed(0, -1)

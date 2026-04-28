"""Mood-arc interpolation, denoising-strength mapping, and seed derivation.

The LLM returns exactly three ``mood_arc`` entries (opening / middle /
closing). The dream pipeline needs a per-stanza intensity, a mapped
SDXL img2img ``strength``, and a deterministic seed. All three helpers
live here because they are trivially coupled and all feed SDXL.
"""
from __future__ import annotations

from typing import Sequence


# ------------------------------------------------------------------
# Strength mapping table (intensity 1..5 -> img2img denoising strength)
#
# Design choice (cited in brainstorm/dream-pipeline-v1.md): cap max
# strength at 0.70 so the retrieved image's palette survives even on
# intense stanzas. Going higher erases the "pigment" we specifically
# chose via CLIP retrieval.
# ------------------------------------------------------------------
_STRENGTH_TABLE = {
    1: 0.55,
    2: 0.60,
    3: 0.65,
    4: 0.68,
    5: 0.70,
}


def stanza_intensity(
    stanza_idx: int, num_stanzas: int, mood_arc: Sequence[dict]
) -> int:
    """Interpolate intensity 1..5 for one stanza given a 3-point mood arc.

    ``mood_arc`` must have exactly three entries with an ``intensity``
    integer field, ordered opening / middle / closing. The arc is mapped
    to a position ``t in [0, 1]`` and linearly interpolated in two halves
    (opening-middle for ``t <= 0.5``, middle-closing otherwise), then
    rounded to the nearest integer and clipped to 1..5.
    """
    if len(mood_arc) != 3:
        raise ValueError(
            f"mood_arc must have exactly 3 entries, got {len(mood_arc)}"
        )
    if num_stanzas <= 0:
        raise ValueError("num_stanzas must be positive")
    if not (0 <= stanza_idx < num_stanzas):
        raise ValueError(
            f"stanza_idx {stanza_idx} out of range for num_stanzas={num_stanzas}"
        )

    opening = int(mood_arc[0]["intensity"])
    middle = int(mood_arc[1]["intensity"])
    closing = int(mood_arc[2]["intensity"])

    t = stanza_idx / max(num_stanzas - 1, 1)
    if t <= 0.5:
        local_t = t / 0.5 if num_stanzas > 1 else 0.0
        value = opening * (1 - local_t) + middle * local_t
    else:
        local_t = (t - 0.5) / 0.5
        value = middle * (1 - local_t) + closing * local_t

    rounded = int(round(value))
    return max(1, min(5, rounded))


def mood_to_strength(intensity: int) -> float:
    """Map an intensity 1..5 to an SDXL img2img ``strength`` float.

    Mapping (caps max at 0.70 to preserve retrieved palette):

    ==========  =========
    intensity   strength
    ==========  =========
    1           0.55
    2           0.60
    3           0.65
    4           0.68
    5           0.70
    ==========  =========
    """
    if intensity not in _STRENGTH_TABLE:
        raise ValueError(
            f"intensity must be 1..5, got {intensity!r}"
        )
    return _STRENGTH_TABLE[intensity]


def stanza_seed(gutenberg_id: int, stanza_idx: int) -> int:
    """Derive a deterministic 32-bit seed from a poem ID and stanza index.

    Uses a mixing function that does not depend on Python's hash
    randomization — so the seed is stable across processes and
    interpreter versions.
    """
    if gutenberg_id < 0 or stanza_idx < 0:
        raise ValueError("gutenberg_id and stanza_idx must be non-negative")
    # Simple splitmix-style mix — stable across Python versions, unlike
    # the builtin hash() which varies with PYTHONHASHSEED.
    x = (int(gutenberg_id) * 0x9E3779B97F4A7C15 + int(stanza_idx)) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return x & 0xFFFFFFFF

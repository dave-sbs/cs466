"""CrossfadeInterpolator: linear alpha blend between two RGB PIL images.

Always-available baseline interpolator. Deterministic: given the same
inputs it always produces the same pixels (pure PIL, no GPU).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .base import Interpolator, InterpolatorError, assert_within_frame_cap
from dream_frames.plan import rife_intermediate_count


class CrossfadeInterpolator:
    """Blend ``a`` into ``b`` over ``2**depth - 1`` intermediate frames.

    Intermediate frame ``k`` (1 <= k <= N) uses alpha
    ``k / (N + 1)`` so the blend is uniform and never includes the
    endpoints.
    """

    def __call__(
        self,
        a: Image.Image,
        b: Image.Image,
        out_dir: Path,
        depth: int,
        prefix: str = "mid_",
    ) -> list[Path]:
        if a.size != b.size:
            raise InterpolatorError(
                f"image sizes differ: {a.size} vs {b.size}"
            )
        if a.mode != b.mode:
            # Convert b to match a; don't mutate caller's image
            b = b.convert(a.mode)
        n_mids = rife_intermediate_count(depth)
        assert_within_frame_cap(n_mids)

        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for k in range(1, n_mids + 1):
            alpha = k / (n_mids + 1)
            blended = Image.blend(a, b, alpha)
            p = out_dir / f"{prefix}{k:05d}.png"
            blended.save(p, format="PNG")
            written.append(p)
        return written


# ``CrossfadeInterpolator`` must satisfy the ``Interpolator`` protocol.
_proto_check: Interpolator = CrossfadeInterpolator()

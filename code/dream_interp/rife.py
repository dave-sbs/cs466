"""RifeInterpolator — Practical-RIFE bridge (stub).

Heavy imports (``torch``, ``sys.path.append(RIFE_ROOT)``) happen inside
``__call__`` so that importing ``dream_interp`` on CPU CI does not
require RIFE to be installed.

Set ``RIFE_ROOT`` to the directory containing Practical-RIFE's
``inference.py`` and model weights. See ``docs/rife_setup.md``.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from .base import Interpolator, InterpolatorError, assert_within_frame_cap
from dream_frames.plan import rife_intermediate_count


class RifeInterpolator:
    """Placeholder that raises a clear error unless RIFE is configured.

    The real implementation will:

    1. Ensure ``RIFE_ROOT`` is on ``sys.path`` (only once).
    2. Lazily load the RIFE model onto the current CUDA device.
    3. Recursively halve each pair with ``2**depth - 1`` intermediates.

    Until that work lands, this class documents the interface and raises
    so callers can fall back to :class:`CrossfadeInterpolator`.
    """

    def __init__(self, rife_root: str | Path | None = None) -> None:
        self.rife_root = Path(rife_root) if rife_root else None

    def _resolve_root(self) -> Path:
        root = self.rife_root or os.environ.get("RIFE_ROOT")
        if not root:
            raise InterpolatorError(
                "RIFE not configured: pass rife_root=... or set RIFE_ROOT "
                "env var. See code/docs/rife_setup.md."
            )
        root_p = Path(root)
        if not root_p.exists():
            raise InterpolatorError(
                f"RIFE_ROOT does not exist: {root_p}"
            )
        return root_p

    def __call__(
        self,
        a: Image.Image,
        b: Image.Image,
        out_dir: Path,
        depth: int,
        prefix: str = "mid_",
    ) -> list[Path]:
        # Fail fast on bad inputs before any heavy import.
        n_mids = rife_intermediate_count(depth)
        assert_within_frame_cap(n_mids)
        if a.size != b.size:
            raise InterpolatorError(
                f"image sizes differ: {a.size} vs {b.size}"
            )
        _ = self._resolve_root()
        raise InterpolatorError(
            "RifeInterpolator.__call__ not implemented in this build; "
            "use CrossfadeInterpolator or install Practical-RIFE per "
            "code/docs/rife_setup.md"
        )


_proto_check: Interpolator = RifeInterpolator()

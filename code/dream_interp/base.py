"""Interpolator protocol + shared guard for dream_interp."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image


DEFAULT_MAX_FRAMES_PER_CALL = 256


class InterpolatorError(RuntimeError):
    """Raised by any interpolator when it cannot satisfy the request."""


def assert_within_frame_cap(
    count: int, cap: int = DEFAULT_MAX_FRAMES_PER_CALL
) -> None:
    """Raise ``InterpolatorError`` if ``count`` exceeds ``cap``.

    This exists so individual interpolator implementations do not accidentally
    blow up disk or RAM with pathologically deep recursion values.
    """
    if count > cap:
        raise InterpolatorError(
            f"interpolator asked to produce {count} frames but cap is {cap}"
        )


@runtime_checkable
class Interpolator(Protocol):
    """Protocol: produce intermediate frames between two keyframes.

    Implementations must be callable with the signature below and return
    an ordered list of ``Path`` objects pointing to written PNG files.
    The list MUST NOT include the endpoints ``a`` and ``b`` — those
    belong to the hold segments around the transition.
    """

    def __call__(
        self,
        a: Image.Image,
        b: Image.Image,
        out_dir: Path,
        depth: int,
        prefix: str = "mid_",
    ) -> list[Path]:
        ...

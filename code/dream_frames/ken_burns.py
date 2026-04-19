"""Ken Burns center-zoom on PIL images.

The math matches the brainstorm design: at ``frame_idx=0`` the output
equals the input (scale 1.0); at ``frame_idx=total_frames-1`` it has
been zoomed to ``max_zoom`` and resampled back to the input's WxH.
Intermediate frames interpolate linearly.
"""
from __future__ import annotations

from PIL import Image


def ken_burns_frame(
    image: Image.Image,
    frame_idx: int,
    total_frames: int,
    max_zoom: float = 1.05,
) -> Image.Image:
    """Return a single Ken-Burns frame as a new PIL Image.

    Parameters
    ----------
    image : PIL.Image.Image
        Source RGB image. Copied internally; not mutated.
    frame_idx : int
        Zero-based frame index in ``[0, total_frames - 1]``.
    total_frames : int
        Total number of frames in the hold segment.
    max_zoom : float, default 1.05
        Final zoom factor at ``frame_idx = total_frames - 1``.

    Returns
    -------
    PIL.Image.Image
        Same WxH as input. RGB unless input was already another mode.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be >= 1")
    if frame_idx < 0 or frame_idx >= total_frames:
        raise ValueError(
            f"frame_idx {frame_idx} out of range [0, {total_frames - 1}]"
        )
    if max_zoom < 1.0:
        raise ValueError(f"max_zoom must be >= 1.0, got {max_zoom}")

    # Linear ramp. At frame_idx=0 -> scale=1.0 exactly (identity).
    if total_frames == 1:
        scale = 1.0
    else:
        scale = 1.0 + (max_zoom - 1.0) * (frame_idx / (total_frames - 1))

    w, h = image.size
    if scale <= 1.0:
        return image.copy()

    new_w = max(1, int(round(w / scale)))
    new_h = max(1, int(round(h / scale)))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def apply_ken_burns(
    image: Image.Image, total_frames: int, max_zoom: float = 1.05
):
    """Generator yielding Ken-Burns frames in order.

    Convenient for writing ``for i, f in enumerate(apply_ken_burns(...))``.
    """
    for i in range(total_frames):
        yield ken_burns_frame(image, i, total_frames, max_zoom=max_zoom)

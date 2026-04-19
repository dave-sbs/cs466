"""Path resolution for dream-pipeline inputs.

Keeps layout assumptions explicit and testable without touching disk.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath


IMAGES_SUBDIR = ("data", "images")


def resolve_top1_image_path(data_root: str | Path, image_id: str) -> Path:
    """Resolve a manifest ``image_id`` to a filesystem path under ``data_root``.

    ``image_id`` as emitted by ``clip_pipeline.retrieve`` is a flat basename
    (e.g. ``"00879.jpg"``), but may include subdirectories in future runs.
    Both cases are handled by treating ``image_id`` as a POSIX-style
    relative path and joining it under ``data_root/data/images/``.

    This helper does *not* check existence — callers (e.g. dream_preflight)
    are responsible for that to keep the helper side-effect-free.
    """
    if not image_id:
        raise ValueError("image_id must be a non-empty string")
    rel = PurePosixPath(image_id)
    if rel.is_absolute():
        raise ValueError(f"image_id must be relative, got {image_id!r}")
    return Path(data_root, *IMAGES_SUBDIR, *rel.parts)

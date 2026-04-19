"""Output artifacts emitted by one dream run."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DreamArtifacts:
    """File paths produced by a single ``render_dream_video`` call.

    Every path is absolute. Optional artifacts (mp4, meta) are ``None``
    if skipped (e.g. frames-only dry run).
    """

    run_dir: Path
    frames_dir: Path
    keyframes_dir: Path
    keyframe_manifest: Path
    frame_manifest: Path
    mp4: Path | None = None
    meta: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(self).items()
        }


@dataclass
class FrameManifest:
    """High-level frame boundaries (built from a SegmentPlan)."""

    total_frames: int
    fps: int
    segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "segments": self.segments,
            "total_frames": self.total_frames,
        }


def write_frame_manifest(path: str | Path, manifest: FrameManifest) -> Path:
    """Atomic write of the frame manifest JSON (sorted keys)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path

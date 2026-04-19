"""dream_render — orchestrate SDXL + interpolator + ffmpeg into a video.

Entry points:

- :class:`DreamArtifacts` — paths produced by one run.
- :func:`render_dream_video` — pure-Python orchestration that a notebook
  cell calls as one function.

Design:

- Reads the committed contract (LLM record + retrieval manifest),
  generates keyframes via a ``KeyframeProvider`` (real or mock),
  materializes hold + transition frames on disk, assembles into an
  MP4 with the ffmpeg wrapper.
- ``DREAM_MOCK_GPU=1`` forces the mock provider and skips any
  diffusers import — this is the CI path (S8-T4b).
- Resumability: keyframe_manifest.json tracks per-stanza ``done`` and
  sha256. Re-running with the same ``run_config`` skips stanzas whose
  PNG + hash match. ``force=True`` overrides.
"""
from __future__ import annotations

from .artifacts import DreamArtifacts, FrameManifest, write_frame_manifest
from .orchestrate import RenderConfig, render_dream_video

__all__ = [
    "DreamArtifacts",
    "FrameManifest",
    "RenderConfig",
    "render_dream_video",
    "write_frame_manifest",
]

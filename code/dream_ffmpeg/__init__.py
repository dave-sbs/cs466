"""dream_ffmpeg — subprocess wrappers around ``ffmpeg`` / ``ffprobe``.

All subprocess calls use ``shell=False`` and a timeout. Integration
tests in ``tests/test_dream_ffmpeg.py`` are marked ``@pytest.mark.ffmpeg``
and auto-skip when ffmpeg is not on the PATH.
"""
from __future__ import annotations

from .ffmpeg import (
    FFMPEG_DEFAULT_TIMEOUT_SECONDS,
    FfmpegError,
    ffmpeg_available,
    ffmpeg_render_frames_to_mp4,
    frame_filename,
    probe_duration_seconds,
)

__all__ = [
    "FFMPEG_DEFAULT_TIMEOUT_SECONDS",
    "FfmpegError",
    "ffmpeg_available",
    "ffmpeg_render_frames_to_mp4",
    "frame_filename",
    "probe_duration_seconds",
]

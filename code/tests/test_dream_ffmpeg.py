"""Tests for dream_ffmpeg.

Unit tests run in plain CI. Integration tests marked ``@pytest.mark.ffmpeg``
auto-skip when ffmpeg/ffprobe are not on the PATH or when
``SKIP_FFMPEG=1`` is set.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from dream_ffmpeg import (
    FfmpegError,
    ffmpeg_available,
    ffmpeg_render_frames_to_mp4,
    frame_filename,
    probe_duration_seconds,
)


# ------------------ unit tests (always run) ------------------


def test_frame_filename_pads():
    assert frame_filename(0) == "frame_00000.png"
    assert frame_filename(42) == "frame_00042.png"


def test_frame_filename_custom_width_and_ext():
    assert frame_filename(7, width=3, ext="jpg") == "frame_007.jpg"


def test_frame_filename_rejects_negative():
    with pytest.raises(ValueError):
        frame_filename(-1)


def test_ffmpeg_available_returns_tuple():
    found, msg = ffmpeg_available()
    assert isinstance(found, bool)
    assert isinstance(msg, str)


def test_ffmpeg_available_false_when_missing_binary():
    with patch("dream_ffmpeg.ffmpeg.shutil.which", return_value=None):
        found, msg = ffmpeg_available()
    assert found is False
    assert "PATH" in msg


def test_render_without_fade_out_and_total_frames_raises(tmp_path: Path):
    with patch("dream_ffmpeg.ffmpeg.shutil.which", return_value="/fake/ffmpeg"):
        with pytest.raises(FfmpegError, match="total_frames"):
            ffmpeg_render_frames_to_mp4(
                tmp_path,
                tmp_path / "out.mp4",
                fade_out_seconds=0.5,
                total_frames=None,
            )


def test_render_raises_when_ffmpeg_missing(tmp_path: Path):
    with patch("dream_ffmpeg.ffmpeg.shutil.which", return_value=None):
        with pytest.raises(FfmpegError, match="ffmpeg not on PATH"):
            ffmpeg_render_frames_to_mp4(
                tmp_path, tmp_path / "out.mp4", fade_out_seconds=0
            )


def test_probe_raises_when_ffprobe_missing(tmp_path: Path):
    with patch("dream_ffmpeg.ffmpeg.shutil.which", return_value=None):
        with pytest.raises(FfmpegError, match="ffprobe not on PATH"):
            probe_duration_seconds(tmp_path / "x.mp4")


# ------------------ integration tests (marker + auto-skip) ------------------


def _ffmpeg_ok() -> bool:
    if os.environ.get("SKIP_FFMPEG") == "1":
        return False
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.ffmpeg
@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg/ffprobe not on PATH")
def test_render_ten_solid_frames(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    img = Image.new("RGB", (64, 48), color=(10, 20, 30))
    for i in range(10):
        img.save(frames_dir / frame_filename(i))

    out = tmp_path / "out.mp4"
    ffmpeg_render_frames_to_mp4(
        frames_dir,
        out,
        fps=30,
        total_frames=10,
        fade_in_seconds=0.1,
        fade_out_seconds=0.1,
    )
    assert out.exists()
    duration = probe_duration_seconds(out)
    # 10 frames at 30 fps ~= 0.333s; allow generous tolerance for
    # container/timebase rounding.
    assert 0.25 < duration < 0.45, duration

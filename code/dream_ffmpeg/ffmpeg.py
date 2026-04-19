"""Thin subprocess wrappers around ffmpeg / ffprobe.

Design notes:

- Every subprocess call passes ``shell=False`` with an explicit
  ``timeout``. Caller can override the default.
- Errors raise ``FfmpegError`` with the (truncated) stderr attached so
  the CI log is useful.
- ``ffmpeg_render_frames_to_mp4`` uses a single ``fade in, fade out``
  video filter instead of prepending black frames — that keeps the
  frame sequence total count stable between the segment plan and the
  rendered video.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


FFMPEG_DEFAULT_TIMEOUT_SECONDS = 600  # 10 min, ample for small MVP videos


class FfmpegError(RuntimeError):
    """Raised when an ffmpeg / ffprobe invocation fails."""


def frame_filename(index: int, width: int = 5, ext: str = "png") -> str:
    """Canonical frame filename, zero-padded to ``width``.

    Example: ``frame_filename(42)`` -> ``'frame_00042.png'``.
    """
    if index < 0:
        raise ValueError(f"frame index must be >= 0, got {index}")
    return f"frame_{index:0{width}d}.{ext}"


def ffmpeg_available() -> tuple[bool, str]:
    """Return ``(found, version_line)`` for the ffmpeg binary on PATH.

    ``found=False`` with an explanatory message when the binary is
    missing. Does not verify codec support — use ``has_libx264`` helpers
    for that if you ever need to.
    """
    binary = shutil.which("ffmpeg")
    if binary is None:
        return False, "ffmpeg not on PATH"
    try:
        proc = subprocess.run(
            [binary, "-version"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg -version timed out at {binary}"
    except OSError as e:
        return False, f"ffmpeg -version OSError: {e}"
    first_line = (proc.stdout or proc.stderr).splitlines()[0] if (
        proc.stdout or proc.stderr
    ) else "unknown"
    return proc.returncode == 0, first_line


def ffmpeg_render_frames_to_mp4(
    frames_dir: str | Path,
    output_path: str | Path,
    *,
    fps: int = 30,
    pattern: str = "frame_%05d.png",
    fade_in_seconds: float = 0.5,
    fade_out_seconds: float = 0.5,
    total_frames: int | None = None,
    crf: int = 18,
    preset: str = "medium",
    pix_fmt: str = "yuv420p",
    timeout_seconds: float = FFMPEG_DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Encode ``frames_dir/<pattern>`` into an H.264 MP4 with fade filters.

    Raises ``FfmpegError`` on non-zero exit.

    ``total_frames`` is only used to compute the fade-out start time; if
    omitted the caller can still request a fade-in. When fade-out is
    requested without ``total_frames`` we raise early.
    """
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise FfmpegError("ffmpeg not on PATH")

    frames_dir = Path(frames_dir)
    output_path = Path(output_path)

    if fade_out_seconds > 0 and total_frames is None:
        raise FfmpegError(
            "fade_out_seconds > 0 requires total_frames so ffmpeg knows "
            "when the fade should start"
        )

    filters: list[str] = []
    if fade_in_seconds > 0:
        filters.append(f"fade=t=in:st=0:d={fade_in_seconds}")
    if fade_out_seconds > 0 and total_frames is not None:
        duration = total_frames / fps
        start = max(0.0, duration - fade_out_seconds)
        filters.append(f"fade=t=out:st={start}:d={fade_out_seconds}")

    args: list[str] = [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / pattern),
    ]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        pix_fmt,
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        proc = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise FfmpegError(
            f"ffmpeg timed out after {timeout_seconds}s"
        ) from e

    if proc.returncode != 0:
        raise FfmpegError(
            "ffmpeg failed "
            f"(rc={proc.returncode}): {proc.stderr.strip()[:2000]}"
        )
    if not output_path.exists():
        raise FfmpegError(f"ffmpeg reported success but no output at {output_path}")
    return output_path


def probe_duration_seconds(
    mp4_path: str | Path,
    *,
    timeout_seconds: float = 30,
) -> float:
    """Return the container duration of ``mp4_path`` in seconds.

    Uses ``ffprobe -show_entries format=duration``. Raises
    ``FfmpegError`` when ffprobe is missing, fails, or returns
    something un-parseable.
    """
    binary = shutil.which("ffprobe")
    if binary is None:
        raise FfmpegError("ffprobe not on PATH")

    args = [
        binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mp4_path),
    ]
    try:
        proc = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise FfmpegError("ffprobe timed out") from e
    if proc.returncode != 0:
        raise FfmpegError(
            f"ffprobe failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    raw = proc.stdout.strip()
    try:
        return float(raw)
    except ValueError as e:
        raise FfmpegError(f"ffprobe returned unparseable duration: {raw!r}") from e

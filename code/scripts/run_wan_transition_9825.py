"""Generate a single Wan FLF2V transition clip (prototype).

Intended for Colab experimentation. Requires diffusers + Wan weights.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _ensure_code_on_path() -> None:
    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_wan_transition_9825")
    p.add_argument(
        "--first",
        type=Path,
        required=True,
        help="Path to the first keyframe PNG (e.g. output/dream_runs/.../kf_000.png).",
    )
    p.add_argument(
        "--last",
        type=Path,
        required=True,
        help="Path to the last keyframe PNG (e.g. output/dream_runs/.../kf_001.png).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired transition motion.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output MP4 path.",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    )
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--guidance-scale", type=float, default=5.5)
    p.add_argument("--max-area", type=int, default=720 * 1280)
    p.add_argument("--no-cpu-offload", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    _ensure_code_on_path()
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    from dream_wan import WanFlf2vConfig, generate_wan_transition  # noqa: WPS433

    cfg = WanFlf2vConfig(
        model_id=args.model_id,
        max_area=args.max_area,
        num_frames=args.num_frames,
        fps=args.fps,
        guidance_scale=args.guidance_scale,
        enable_model_cpu_offload=not args.no_cpu_offload,
    )

    out = generate_wan_transition(
        first_image=args.first,
        last_image=args.last,
        prompt=args.prompt,
        output_path=args.output,
        cfg=cfg,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


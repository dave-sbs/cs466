"""Run the dream pipeline for poem 9825 from one command.

This script exists as a git-tracked alternative to ad-hoc notebook cells.
It is intentionally small and focuses on the single-poem prototype loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _ensure_code_on_path() -> None:
    """Allow running from repo root (so `code/` isn't automatically on sys.path)."""

    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_dream_9825")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("code"),
        help="Path to the repo's code/ directory (contains data/ output/ exploration_output/).",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("code/output/dream_runs/9825"),
        help="Run directory; will contain frames/, keyframes/, manifests, mp4.",
    )
    p.add_argument("--use-rife", action="store_true", help="Use Practical-RIFE for transitions.")
    p.add_argument(
        "--rife-root",
        type=Path,
        default=None,
        help="Override RIFE_ROOT env var (directory containing Practical-RIFE).",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--rife-depth", type=int, default=4)
    p.add_argument("--max-zoom", type=float, default=1.05)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30, dest="num_inference_steps")
    p.add_argument("--guidance", type=float, default=7.0, dest="guidance_scale")
    p.add_argument("--start-stanza", type=int, default=0)
    p.add_argument("--end-stanza", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-mp4", action="store_true", help="Skip ffmpeg mp4 encode.")
    p.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    p.add_argument("--revision", type=str, default=None)
    return p


def _log_cuda_info() -> None:
    # Torch import is intentionally inside this function so CPU environments can
    # still run --help / parse-only flows without importing torch.
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            logging.info(
                "CUDA available: device=%s capability=%s total_mem_gb=%.2f",
                name,
                cap,
                total_gb,
            )
        else:
            logging.info("CUDA not available (torch.cuda.is_available() == False)")
    except Exception as e:
        logging.info("CUDA probe skipped (%s)", e)


def main(argv: list[str] | None = None) -> int:
    _ensure_code_on_path()

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.rife_root is not None:
        os.environ["RIFE_ROOT"] = str(args.rife_root)

    _log_cuda_info()

    from dream_interp import CrossfadeInterpolator, RifeInterpolator  # noqa: WPS433
    from dream_render import RenderConfig, render_dream_video  # noqa: WPS433
    from dream_sdxl import (  # noqa: WPS433
        SdxlKeyframeProvider,
        load_sdxl_img2img_pipe,
    )

    cfg = RenderConfig(
        gutenberg_id=9825,
        data_root=args.data_root,
        run_dir=args.run_dir,
        fps=args.fps,
        rife_depth=args.rife_depth,
        max_zoom=args.max_zoom,
        image_size=(args.image_size, args.image_size),
        start_stanza=args.start_stanza,
        end_stanza=args.end_stanza,
        force=args.force,
        encode_mp4=not args.no_mp4,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        model_id=args.model_id,
        revision=args.revision,
    )

    def provider_factory():
        pipe = load_sdxl_img2img_pipe(
            model_id=cfg.model_id,
            device="cuda",
            dtype="float16",
            enable_attention_slicing=True,
            enable_vae_tiling=True,
            enable_model_cpu_offload=False,
            revision=cfg.revision,
        )
        return SdxlKeyframeProvider(pipe)

    interpolator = (
        RifeInterpolator()
        if args.use_rife
        else CrossfadeInterpolator()
    )

    artifacts = render_dream_video(
        cfg,
        provider_factory=provider_factory,
        interpolator=interpolator,
        mock_gpu=False,
    )

    print(artifacts)

    # Convenience: print the tail of the keyframe manifest so runs are self-describing.
    km_path = artifacts.keyframe_manifest
    try:
        payload = json.loads(km_path.read_text(encoding="utf-8"))
        entries = payload.get("entries") or []
        tail = entries[-1] if entries else {}
        print("keyframe_manifest.tail =", json.dumps(tail, indent=2, sort_keys=True))
    except Exception as e:
        logging.info("Unable to pretty-print keyframe manifest (%s)", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


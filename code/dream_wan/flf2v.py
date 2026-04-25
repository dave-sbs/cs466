"""Wan 2.1 First-Last-Frame-to-Video (FLF2V) wrapper.

This module provides:
- CPU-safe config + resize helpers (no torch/diffusers import at module import time)
- Lazy loader for the diffusers Wan FLF2V pipeline
- A single convenience function to generate a transition MP4
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class WanFlf2vConfig:
    model_id: str = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
    max_area: int = 720 * 1280
    num_frames: int = 81
    fps: int = 16
    guidance_scale: float = 5.5
    torch_dtype: str = "bfloat16"  # "float16" | "float32" | "bfloat16"
    enable_model_cpu_offload: bool = True

    def __post_init__(self):
        if self.max_area <= 0:
            raise ValueError("max_area must be positive")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be positive")
        if self.torch_dtype not in ("float16", "float32", "bfloat16"):
            raise ValueError("torch_dtype must be float16|float32|bfloat16")


def _mod_value_from_pipe(pipe: Any) -> int:
    # Mirrors diffusers docs for Wan: mod_value = vae_scale_factor_spatial * patch_size
    try:
        vae_scale = int(getattr(pipe, "vae_scale_factor_spatial"))
        patch_size = int(pipe.transformer.config.patch_size[1])
        return max(1, vae_scale * patch_size)
    except Exception:
        return 32


def aspect_ratio_resize(
    image: Image.Image, *, max_area: int, mod_value: int
) -> tuple[Image.Image, int, int]:
    """Resize while preserving aspect ratio and rounding to mod_value multiples."""
    if max_area <= 0:
        raise ValueError("max_area must be positive")
    if mod_value <= 0:
        raise ValueError("mod_value must be positive")

    w0, h0 = image.size
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid image size")

    aspect_ratio = h0 / w0
    # Solve for h*w = max_area with h/w = aspect_ratio
    import math

    height = int(round(math.sqrt(max_area * aspect_ratio)))
    width = int(round(math.sqrt(max_area / aspect_ratio)))

    height = max(mod_value, (height // mod_value) * mod_value)
    width = max(mod_value, (width // mod_value) * mod_value)

    resized = image.resize((width, height), Image.LANCZOS)
    return resized, height, width


def center_crop_resize(image: Image.Image, *, height: int, width: int) -> Image.Image:
    """Resize (cover) then center-crop to (width, height)."""
    if height <= 0 or width <= 0:
        raise ValueError("height/width must be positive")

    w0, h0 = image.size
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid image size")

    resize_ratio = max(width / w0, height / h0)
    new_w = int(round(w0 * resize_ratio))
    new_h = int(round(h0 * resize_ratio))
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    cropped = resized.crop((left, top, left + width, top + height))
    return cropped


def load_wan_flf2v_pipeline(cfg: WanFlf2vConfig):
    """Lazy-load the diffusers Wan FLF2V pipeline."""
    import torch  # noqa: WPS433
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: WPS433
    from transformers import CLIPVisionModel  # noqa: WPS433

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[cfg.torch_dtype]

    image_encoder = CLIPVisionModel.from_pretrained(
        cfg.model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        cfg.model_id, subfolder="vae", torch_dtype=torch.float32
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        cfg.model_id,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch_dtype,
    )
    pipe.to("cuda")

    if cfg.enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        # Frees VRAM during forward pass on large models (A100 recommended).
        pipe.enable_model_cpu_offload()

    return pipe


def _load_image(path_or_image: str | Path | Image.Image) -> Image.Image:
    if isinstance(path_or_image, Image.Image):
        return path_or_image.convert("RGB")
    p = Path(path_or_image)
    return Image.open(p).convert("RGB")


def generate_wan_transition(
    *,
    first_image: str | Path | Image.Image,
    last_image: str | Path | Image.Image,
    prompt: str,
    output_path: str | Path,
    cfg: WanFlf2vConfig | None = None,
    pipe: Any | None = None,
) -> Path:
    """Generate an MP4 video conditioned on first+last frames and a prompt."""
    if not str(prompt).strip():
        raise ValueError("prompt must be non-empty")

    cfg = cfg or WanFlf2vConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pipe is None:
        # Allow HF auth via env (do not print token).
        _ = os.environ.get("HF_TOKEN")
        pipe = load_wan_flf2v_pipeline(cfg)

    first = _load_image(first_image)
    last = _load_image(last_image)

    mod_value = _mod_value_from_pipe(pipe)
    first, height, width = aspect_ratio_resize(first, max_area=cfg.max_area, mod_value=mod_value)
    if last.size != first.size:
        last = center_crop_resize(last, height=height, width=width)

    from diffusers.utils import export_to_video  # noqa: WPS433

    out = pipe(
        image=first,
        last_image=last,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=cfg.num_frames,
        guidance_scale=cfg.guidance_scale,
    ).frames[0]

    export_to_video(out, str(output_path), fps=cfg.fps)
    return output_path


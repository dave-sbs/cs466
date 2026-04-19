"""SDXL img2img factory + keyframe generation adapters.

Design:

- :class:`KeyframeProvider` is a Protocol — anything callable with the
  documented signature works. This lets us mock SDXL in CI.
- :class:`SdxlKeyframeProvider` wraps a diffusers ``StableDiffusionXLImg2ImgPipeline``
  and exposes a stable call contract (no diffusers types in the public
  signature).
- :func:`load_sdxl_img2img_pipe` is the factory referenced by S7-T2. It
  lazily imports diffusers and applies memory flags. Returns ``None``
  in mock/CI contexts if the caller sets ``mock=True``.
- :func:`generate_keyframe` is the thin wrapper that records the kwargs
  (strength, seed, prompt, ...) we care about for the keyframe manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class KeyframeProvider(Protocol):
    """Generate a keyframe PIL image from a prompt + init image.

    Must not block on shared state: callers may run multiple providers
    in sequence.
    """

    def __call__(
        self,
        *,
        prompt: str,
        init_image: Image.Image,
        strength: float,
        seed: int,
        negative_prompt: str | None = None,
        prompt_2: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        ...


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def load_sdxl_img2img_pipe(
    *,
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "cuda",
    dtype: str = "float16",
    enable_attention_slicing: bool = True,
    enable_vae_tiling: bool = True,
    enable_model_cpu_offload: bool = False,
    revision: str | None = None,
    mock: bool = False,
) -> Any:
    """Return a diffusers img2img pipeline configured for VRAM-constrained GPUs.

    Lazy import: ``diffusers`` and ``torch`` are imported only on demand.
    This lets CI import this module without GPU deps installed.

    When ``mock=True`` returns ``None`` so orchestration code paths can
    exercise the factory call-site without requiring diffusers.
    """
    if mock:
        return None

    allowed_dtypes = {"float16", "float32", "bfloat16"}
    if dtype not in allowed_dtypes:
        raise ValueError(
            f"unsupported dtype {dtype!r}; must be one of {sorted(allowed_dtypes)}"
        )

    import torch  # noqa: WPS433
    from diffusers import StableDiffusionXLImg2ImgPipeline  # noqa: WPS433

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[dtype],
        "use_safetensors": True,
    }
    if revision:
        kwargs["revision"] = revision

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    if enable_vae_tiling:
        pipe.enable_vae_tiling()
    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    return pipe


# ----------------------------------------------------------------------
# Production provider wrapping a diffusers pipe
# ----------------------------------------------------------------------


@dataclass
class SdxlKeyframeProvider:
    """``KeyframeProvider`` backed by a loaded diffusers pipe."""

    pipe: Any  # StableDiffusionXLImg2ImgPipeline (loosely typed)

    def __call__(
        self,
        *,
        prompt: str,
        init_image: Image.Image,
        strength: float,
        seed: int,
        negative_prompt: str | None = None,
        prompt_2: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        import torch  # noqa: WPS433

        generator = torch.Generator(
            device=getattr(self.pipe, "device", "cuda")
        ).manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            image=init_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return out.images[0]


# ----------------------------------------------------------------------
# Mock provider — same shape, no GPU
# ----------------------------------------------------------------------


@dataclass
class MockKeyframeProvider:
    """Deterministic mock provider used by tests and ``DREAM_MOCK_GPU=1``.

    Ignores the prompt and returns ``init_image`` tinted by a color
    derived from ``seed`` so consecutive stanzas produce visibly
    distinct frames in the mock E2E test.
    """

    def __call__(
        self,
        *,
        prompt: str,
        init_image: Image.Image,
        strength: float,
        seed: int,
        negative_prompt: str | None = None,
        prompt_2: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        # Tint: cheap per-seed color so mock runs look different but
        # remain deterministic across processes.
        r = (seed & 0xFF)
        g = ((seed >> 8) & 0xFF)
        b = ((seed >> 16) & 0xFF)
        tint = Image.new("RGB", init_image.size, color=(r, g, b))
        return Image.blend(init_image.convert("RGB"), tint, alpha=0.5)


# ----------------------------------------------------------------------
# Thin generate wrapper that tests can patch
# ----------------------------------------------------------------------


def generate_keyframe(
    provider: KeyframeProvider,
    *,
    prompt: str,
    init_image: Image.Image,
    strength: float,
    seed: int,
    negative_prompt: str | None = None,
    prompt_2: str | None = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
    output_path: str | Path | None = None,
) -> Image.Image:
    """Call ``provider`` and optionally save the PNG to ``output_path``.

    The function exists so the orchestration layer can call one consistent
    signature whether ``provider`` is real SDXL, a mock, or a test double.
    """
    img = provider(
        prompt=prompt,
        init_image=init_image,
        strength=strength,
        seed=seed,
        negative_prompt=negative_prompt,
        prompt_2=prompt_2,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG")
    return img

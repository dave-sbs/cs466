"""SDXL prompt construction from LLM visual scene dicts.

Keeps prompt formatting in one place so iteration does not touch the
rest of the pipeline. Style constants live at module level and can be
overridden per call.
"""
from __future__ import annotations

from typing import Any, Sequence


DEFAULT_STYLE_TAIL = (
    "dreamlike digital art, ethereal, high detail"
)
DEFAULT_STYLE_PROMPT_2 = (
    "flowing organic textures, data-driven visualization, Refik Anadol style, 4k"
)

DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, signature, blurry, low quality, photograph, "
    "realistic face, human figure, sharp edges, geometric shapes, ui elements"
)


def _colors_str(dominant_colors: Sequence[str]) -> str:
    return ", ".join(str(c).strip() for c in dominant_colors if str(c).strip())


def build_sdxl_prompt(
    scene: dict[str, Any],
    style_tail: str = DEFAULT_STYLE_TAIL,
    style_prompt_2: str | None = DEFAULT_STYLE_PROMPT_2,
) -> tuple[str, str | None]:
    """Return ``(prompt, prompt_2)`` for an SDXL img2img call.

    ``prompt`` bundles the scene description, colors, time-of-day, and a
    short style tail — fits comfortably under the 77-token CLIP cap for
    typical poem stanzas.

    ``prompt_2`` (returned separately, not concatenated) carries longer
    style vocabulary so it does not crowd the primary prompt. SDXL
    natively supports a second prompt via its second text encoder.

    The scene dict is expected to follow the schema emitted by
    ``llm_analysis.VisualScene`` — see ``code/llm_analysis.py``.
    """
    desc = str(scene.get("scene_description") or "").strip()
    if not desc:
        raise ValueError("scene.scene_description must be a non-empty string")

    colors = scene.get("dominant_colors") or []
    if not isinstance(colors, (list, tuple)):
        raise TypeError(
            f"scene.dominant_colors must be list[str], got {type(colors).__name__}"
        )
    colors_s = _colors_str(colors)
    tod = str(scene.get("time_of_day") or "").strip()

    parts = [desc]
    if colors_s:
        parts.append(f"{colors_s} color palette")
    if tod and tod.lower() != "unspecified":
        parts.append(f"{tod} lighting")
    if style_tail.strip():
        parts.append(style_tail.strip())
    prompt = ", ".join(parts)

    return prompt, (style_prompt_2.strip() if style_prompt_2 else None)

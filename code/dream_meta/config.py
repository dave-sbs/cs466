"""DreamRunConfig — the canonical, JSON-serialisable run config.

Distinct from ``dream_render.RenderConfig`` (which is the in-memory
orchestration parameter bundle). ``DreamRunConfig`` is the *declarative*
record a user would write to disk, attach to a ticket, or diff against
another run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DreamRunConfig:
    """Declarative run config (distinct from RenderConfig).

    The MVP leaves only a handful of knobs exposed here; more can land
    as tickets without breaking the JSON schema because unknown keys
    are tolerated by :func:`dream_run_config_from_dict`.
    """

    gutenberg_id: int
    data_root: str
    run_dir: str
    fps: int = 30
    rife_depth: int = 4
    max_zoom: float = 1.05
    image_size: tuple[int, int] = (1024, 1024)
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    revision: str | None = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    fade_in_seconds: float = 0.5
    fade_out_seconds: float = 0.5
    start_stanza: int = 0
    end_stanza: int | None = None
    force: bool = False


def dream_run_config_to_dict(cfg: DreamRunConfig) -> dict[str, Any]:
    """Return a JSON-safe dict with sorted keys."""
    d = asdict(cfg)
    # tuples lose their tuple-ness through JSON; store as list.
    d["image_size"] = list(d["image_size"])
    return {k: d[k] for k in sorted(d.keys())}


def dream_run_config_from_dict(data: dict[str, Any]) -> DreamRunConfig:
    """Round-trip from :func:`dream_run_config_to_dict` output.

    Unknown keys are ignored (forward-compatibility).
    """
    allowed = {f.name for f in fields(DreamRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    if "image_size" in kwargs and isinstance(kwargs["image_size"], list):
        kwargs["image_size"] = tuple(kwargs["image_size"])
    if "gutenberg_id" not in kwargs:
        raise KeyError("gutenberg_id required")
    if "data_root" not in kwargs:
        raise KeyError("data_root required")
    if "run_dir" not in kwargs:
        raise KeyError("run_dir required")
    return DreamRunConfig(**kwargs)

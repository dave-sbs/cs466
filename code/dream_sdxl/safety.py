"""Safety policy for SDXL outputs.

The MVP renders nature/landscape content (poetry-driven); we keep
the safety checker ENABLED by default and gate any decision to
disable it behind an explicit flag passed at the orchestration layer.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyPolicy:
    """User-facing policy for SDXL generation.

    ``enable_safety_checker``: if True, the diffusers safety checker
    (NSFW filter) is left on. If False, the caller accepts responsibility.

    ``nsfw_replacement``: documentation string — what the pipeline does
    if the checker flags a frame. "skip" means the generator returns a
    blank frame and the orchestration logs a warning.
    """

    enable_safety_checker: bool = True
    nsfw_replacement: str = "skip"


DEFAULT_SAFETY_POLICY = SafetyPolicy()

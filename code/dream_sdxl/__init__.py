"""dream_sdxl — SDXL img2img factory, generation adapter, keyframe manifest.

Imports are lazy where possible so the package can be imported in CPU-only
CI without pulling in ``diffusers`` / ``torch`` CUDA init.

Public surface:

- :class:`KeyframeProvider` — protocol for "given a pair, yield a PIL
  image" (SDXL in real use; mock for tests / CI E2E).
- :class:`SdxlKeyframeProvider` — production implementation. Lazily
  imports ``diffusers`` in its constructor.
- :func:`load_sdxl_img2img_pipe` — factory that applies memory flags.
- :func:`generate_keyframe` — thin wrapper that records kwargs for the
  manifest and is easily mockable in tests.
- :mod:`dream_sdxl.manifest` — keyframe manifest schema + IO.
"""
from __future__ import annotations

from .keyframes import (
    KeyframeProvider,
    MockKeyframeProvider,
    SdxlKeyframeProvider,
    generate_keyframe,
    load_sdxl_img2img_pipe,
)
from .manifest import (
    KEYFRAME_MANIFEST_SCHEMA_VERSION,
    KeyframeEntry,
    KeyframeManifest,
    load_keyframe_manifest,
    sha256_file,
    write_keyframe_manifest,
)
from .safety import DEFAULT_SAFETY_POLICY, SafetyPolicy

__all__ = [
    "KeyframeProvider",
    "MockKeyframeProvider",
    "SdxlKeyframeProvider",
    "generate_keyframe",
    "load_sdxl_img2img_pipe",
    "KEYFRAME_MANIFEST_SCHEMA_VERSION",
    "KeyframeEntry",
    "KeyframeManifest",
    "load_keyframe_manifest",
    "write_keyframe_manifest",
    "sha256_file",
    "DEFAULT_SAFETY_POLICY",
    "SafetyPolicy",
]

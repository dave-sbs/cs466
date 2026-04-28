"""dream_meta — per-run metadata writer for the dream pipeline.

Writes ``meta.json`` next to keyframes/frames so any artifact can be
traced back to the exact config, git commit, and library versions.
"""
from __future__ import annotations

from .config import DreamRunConfig, dream_run_config_to_dict, dream_run_config_from_dict
from .meta import (
    META_SCHEMA_VERSION,
    MetaRecord,
    collect_env_snapshot,
    current_git_sha,
    write_meta,
)

__all__ = [
    "DreamRunConfig",
    "dream_run_config_to_dict",
    "dream_run_config_from_dict",
    "META_SCHEMA_VERSION",
    "MetaRecord",
    "collect_env_snapshot",
    "current_git_sha",
    "write_meta",
]

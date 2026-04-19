"""``meta.json`` writer.

Captures enough context to reproduce (or usefully debug) a dream run:

- git_sha (best effort)
- gutenberg_id, model_id, revision
- pip-installed versions for the libraries that actually moved pixels:
  torch, diffusers, accelerate, Pillow (plus python version)
- Optional CUDA device name if torch+CUDA are available
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


META_SCHEMA_VERSION = "1.0"


_TRACKED_PACKAGES = ("torch", "diffusers", "accelerate", "Pillow", "numpy")


@dataclass
class MetaRecord:
    schema_version: str = META_SCHEMA_VERSION
    git_sha: str | None = None
    gutenberg_id: int | None = None
    model_id: str | None = None
    revision: str | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in sorted(d.keys())}


def current_git_sha(cwd: str | Path | None = None) -> str | None:
    """Return the HEAD SHA of the repo containing ``cwd`` (or current dir).

    Returns ``None`` if the directory is not a git repo or git is missing.
    Never raises.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            shell=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _package_version(name: str) -> str | None:
    """Best-effort version lookup; survives missing packages."""
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def _cuda_device_name() -> str | None:
    """Return a CUDA device string if torch + CUDA are available, else None."""
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def collect_env_snapshot() -> dict[str, Any]:
    """Return a shallow snapshot of the runtime environment."""
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in _TRACKED_PACKAGES:
        snapshot[f"{pkg.lower()}_version"] = _package_version(pkg)
    cuda = _cuda_device_name()
    snapshot["cuda_device"] = cuda  # None when absent — makes mock path testable
    return snapshot


def write_meta(
    path: str | Path,
    *,
    gutenberg_id: int,
    model_id: str,
    revision: str | None = None,
    run_config: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
    git_cwd: str | Path | None = None,
) -> Path:
    """Atomically write a ``meta.json`` file at ``path``.

    Parameters
    ----------
    path : str or Path
        Output path.
    gutenberg_id, model_id, revision : primary identity fields.
    run_config : dict, optional
        Usually ``dream_run_config_to_dict(cfg)`` output.
    artifacts : dict, optional
        Named artifact paths (e.g. ``{"mp4": "...", "keyframes_dir": "..."}``).
    git_cwd : path, optional
        Directory from which to resolve the git HEAD (default: cwd).
    """
    record = MetaRecord(
        git_sha=current_git_sha(git_cwd),
        gutenberg_id=gutenberg_id,
        model_id=model_id,
        revision=revision,
        run_config=run_config or {},
        env=collect_env_snapshot(),
        artifacts=artifacts or {},
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(record.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path

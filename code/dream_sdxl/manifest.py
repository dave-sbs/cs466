"""Keyframe manifest schema + IO.

The manifest records exactly what went into each SDXL call so a run is
reproducible (see S10 for run metadata). Files are written atomically
via ``*.tmp`` + ``os.replace``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


KEYFRAME_MANIFEST_SCHEMA_VERSION = "1.0"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return hex digest of a file (streaming, 1 MiB chunks)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class KeyframeEntry:
    """One row in the keyframe manifest."""

    stanza_index: int
    image_id: str  # source retrieved image (flat basename, per clip_pipeline)
    prompt: str
    prompt_2: str | None
    negative_prompt: str | None
    strength: float
    seed: int
    num_inference_steps: int
    guidance_scale: float
    output_path: str  # POSIX path relative to manifest dir or absolute
    sha256: str
    done: bool = True


@dataclass
class KeyframeManifest:
    """Top-level manifest object."""

    gutenberg_id: int
    schema_version: str = KEYFRAME_MANIFEST_SCHEMA_VERSION
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    revision: str | None = None
    entries: list[KeyframeEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure stable key order for JSON output.
        d["entries"] = [
            {k: e[k] for k in sorted(e.keys())} for e in d["entries"]
        ]
        return {k: d[k] for k in sorted(d.keys())}


def write_keyframe_manifest(path: str | Path, manifest: KeyframeManifest) -> Path:
    """Write manifest to ``path`` atomically (*.tmp -> rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def load_keyframe_manifest(path: str | Path) -> KeyframeManifest:
    """Load and structurally validate a manifest JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    required_top = {"gutenberg_id", "schema_version", "entries"}
    missing = required_top - data.keys()
    if missing:
        raise ValueError(
            f"manifest {path} missing required keys: {sorted(missing)}"
        )
    if data["schema_version"] != KEYFRAME_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version "
            f"{data['schema_version']!r}; expected "
            f"{KEYFRAME_MANIFEST_SCHEMA_VERSION!r}"
        )
    entries = [KeyframeEntry(**e) for e in data["entries"]]
    return KeyframeManifest(
        gutenberg_id=int(data["gutenberg_id"]),
        schema_version=data["schema_version"],
        model_id=data.get("model_id", KeyframeManifest.__dataclass_fields__["model_id"].default),
        revision=data.get("revision"),
        entries=entries,
    )

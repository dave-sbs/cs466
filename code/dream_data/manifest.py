"""Retrieval manifest helpers.

The manifest is produced by ``clip_pipeline.retrieve`` and lives at
``output/retrieval_results/poem_<id>/retrieval_manifest.json``. See
``clip_pipeline.py`` for the writer side of this contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import DreamDataError


def load_retrieval_manifest(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate a retrieval manifest JSON file.

    Raises ``DreamDataError`` if the file is missing, not valid JSON,
    or missing the top-level ``results`` key.
    """
    p = Path(path)
    if not p.exists():
        raise DreamDataError(f"retrieval manifest not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DreamDataError(f"invalid JSON in manifest {p}: {e}") from e
    if not isinstance(data, dict):
        raise DreamDataError(f"manifest {p} is not a JSON object")
    if "results" not in data:
        raise DreamDataError(f"manifest {p} missing top-level 'results' key")
    if not isinstance(data["results"], list):
        raise DreamDataError(f"manifest {p} 'results' is not a list")
    return data


def sort_manifest_results(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a defensive copy of ``manifest['results']`` sorted by chunk_index.

    Items missing ``chunk_index`` sort to the end with ``float('inf')`` key
    rather than raising — the preflight CLI is expected to surface that.
    """
    results = list(manifest.get("results") or [])
    return sorted(results, key=lambda r: r.get("chunk_index", float("inf")))


def pair_scenes_with_chunks(
    llm_record: dict[str, Any], manifest: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Zip LLM visual_scenes with manifest results after asserting equal length.

    Returns a list of ``(scene, chunk_result)`` tuples in chunk_index order.

    Raises ``DreamDataError`` with both counts in the message if lengths differ.
    """
    scenes = list(llm_record.get("visual_scenes") or [])
    results = sort_manifest_results(manifest)
    if len(scenes) != len(results):
        raise DreamDataError(
            "visual_scenes/retrieval count mismatch: "
            f"{len(scenes)} scenes vs {len(results)} chunks "
            f"(gutenberg_id={llm_record.get('gutenberg_id')!r})"
        )
    return list(zip(scenes, results))

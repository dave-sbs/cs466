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
    """Pair LLM visual_scenes with manifest results by chunk index.

    Returns a list of ``(scene, chunk_result)`` tuples in ``chunk_index`` order.

    This joins by ``scene['stanza_index']`` and ``chunk['chunk_index']`` rather
    than zipping by position. That makes pairing robust to out-of-order lists and
    surfaces missing indices clearly.
    """
    scenes = list(llm_record.get("visual_scenes") or [])
    results = sort_manifest_results(manifest)
    scenes_by_idx = {s.get("stanza_index"): s for s in scenes}
    chunks_by_idx = {c.get("chunk_index"): c for c in results}
    missing = set(scenes_by_idx.keys()) ^ set(chunks_by_idx.keys())
    if missing:
        raise DreamDataError(
            "scene/chunk index mismatch: "
            f"{sorted(missing)} (gutenberg_id={llm_record.get('gutenberg_id')!r})"
        )
    return [(scenes_by_idx[i], chunks_by_idx[i]) for i in sorted(chunks_by_idx)]

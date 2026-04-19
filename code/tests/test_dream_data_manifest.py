"""Tests for dream_data.manifest helpers (load / sort / pair)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dream_data import (
    DreamDataError,
    load_retrieval_manifest,
    pair_scenes_with_chunks,
    sort_manifest_results,
)


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    p = tmp_path / "retrieval_manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


# ----- load_retrieval_manifest ------------------------------------------------


def test_load_manifest_success(tmp_path: Path, minimal_manifest_dict):
    p = _write_manifest(tmp_path, minimal_manifest_dict)
    loaded = load_retrieval_manifest(p)
    assert loaded["gutenberg_id"] == 9825
    assert len(loaded["results"]) == 2


def test_load_manifest_missing_file_raises(tmp_path: Path):
    with pytest.raises(DreamDataError, match="not found"):
        load_retrieval_manifest(tmp_path / "nope.json")


def test_load_manifest_invalid_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(DreamDataError, match="invalid JSON"):
        load_retrieval_manifest(p)


def test_load_manifest_missing_results_key_raises(tmp_path: Path):
    p = tmp_path / "no_results.json"
    p.write_text(json.dumps({"poem_name": "x"}), encoding="utf-8")
    with pytest.raises(DreamDataError, match="'results'"):
        load_retrieval_manifest(p)


def test_load_manifest_not_object_raises(tmp_path: Path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(DreamDataError, match="not a JSON object"):
        load_retrieval_manifest(p)


# ----- sort_manifest_results --------------------------------------------------


def test_sort_shuffled_manifest(minimal_manifest_dict):
    shuffled = dict(minimal_manifest_dict)
    shuffled["results"] = list(reversed(minimal_manifest_dict["results"]))
    out = sort_manifest_results(shuffled)
    assert [r["chunk_index"] for r in out] == [0, 1]


def test_sort_is_defensive_copy(minimal_manifest_dict):
    out = sort_manifest_results(minimal_manifest_dict)
    out.append({"chunk_index": 99})
    assert len(minimal_manifest_dict["results"]) == 2


# ----- pair_scenes_with_chunks -----------------------------------------------


def test_pair_success(minimal_llm_record, minimal_manifest_dict):
    pairs = pair_scenes_with_chunks(minimal_llm_record, minimal_manifest_dict)
    assert len(pairs) == 2
    scene0, chunk0 = pairs[0]
    assert scene0["stanza_index"] == 0
    assert chunk0["chunk_index"] == 0


def test_pair_length_mismatch_raises(minimal_llm_record, minimal_manifest_dict):
    short = dict(minimal_llm_record)
    short["visual_scenes"] = minimal_llm_record["visual_scenes"][:1]
    with pytest.raises(DreamDataError, match="1 scenes vs 2 chunks") as exc_info:
        pair_scenes_with_chunks(short, minimal_manifest_dict)
    assert "gutenberg_id=9825" in str(exc_info.value)


def test_pair_respects_chunk_order(minimal_llm_record, minimal_manifest_dict):
    shuffled = dict(minimal_manifest_dict)
    shuffled["results"] = list(reversed(minimal_manifest_dict["results"]))
    pairs = pair_scenes_with_chunks(minimal_llm_record, shuffled)
    assert [c["chunk_index"] for _s, c in pairs] == [0, 1]

"""Integration-style tests that load committed on-disk fixture files.

These prove that the minimal JSON/JSONL fixtures under
``code/tests/fixtures/dream/`` match the in-memory fixtures used by
the conftest, and that the full chain (load -> validate -> pair) works
end-to-end without any real data.
"""
from __future__ import annotations

from pathlib import Path

from dream_data import (
    load_last_llm_record,
    load_retrieval_manifest,
    pair_scenes_with_chunks,
    validate_llm_record,
)


def test_committed_fixtures_load_and_pair(fixtures_dir: Path):
    jsonl = fixtures_dir / "dream" / "minimal_llm.jsonl"
    manifest_path = fixtures_dir / "dream" / "minimal_manifest.json"

    record = load_last_llm_record(jsonl, gutenberg_id=9825)
    assert record is not None
    validate_llm_record(record)

    manifest = load_retrieval_manifest(manifest_path)
    pairs = pair_scenes_with_chunks(record, manifest)
    assert len(pairs) == 2
    assert pairs[0][0]["stanza_index"] == 0
    assert pairs[0][1]["chunk_index"] == 0
    assert pairs[0][1]["top_k"][0]["image_id"] == "00001.jpg"

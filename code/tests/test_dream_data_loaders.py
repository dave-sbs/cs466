"""Tests for dream_data.loaders."""
from __future__ import annotations

from pathlib import Path

import pytest

from dream_data import load_last_llm_record


def test_load_last_llm_record_returns_last_matching(write_jsonl, minimal_llm_record):
    first = dict(minimal_llm_record, title="old run")
    second = dict(minimal_llm_record, title="new run")
    unrelated = dict(minimal_llm_record, gutenberg_id=111, title="other poem")

    path = write_jsonl([first, unrelated, second])
    got = load_last_llm_record(path, gutenberg_id=9825)
    assert got is not None
    assert got["title"] == "new run"


def test_load_last_llm_record_missing_id_returns_none(write_jsonl, minimal_llm_record):
    path = write_jsonl([minimal_llm_record])
    assert load_last_llm_record(path, gutenberg_id=999999) is None


def test_load_last_llm_record_missing_file_returns_none(tmp_path: Path):
    missing = tmp_path / "does_not_exist.jsonl"
    assert load_last_llm_record(missing, gutenberg_id=9825) is None


def test_load_last_llm_record_skips_malformed_lines(tmp_path: Path, minimal_llm_record):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        "this is not json\n"
        + __import__("json").dumps(minimal_llm_record)
        + "\n{partial json}\n",
        encoding="utf-8",
    )
    rec = load_last_llm_record(p, gutenberg_id=9825)
    assert rec is not None
    assert rec["gutenberg_id"] == 9825

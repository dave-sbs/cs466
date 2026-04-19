"""Pure tests for clip_pipeline.split_into_stanzas and chunk_lines.

These cover the corner cases that caused silent degradation on poems
without blank-line stanza boundaries (e.g. 24449).
"""
from __future__ import annotations

import pytest

from clip_pipeline import chunk_lines, split_into_stanzas


# ------------------ split_into_stanzas ------------------


def test_split_two_stanzas():
    lines = [
        "first line",
        "second line",
        "",
        "third line",
        "fourth line",
    ]
    got = split_into_stanzas(lines)
    assert got == ["first line / second line", "third line / fourth line"]


def test_split_trailing_blank_lines_ok():
    lines = ["a", "b", "", "c", "", ""]
    assert split_into_stanzas(lines) == ["a / b", "c"]


def test_split_leading_blank_lines_ok():
    lines = ["", "", "a", "", "b"]
    assert split_into_stanzas(lines) == ["a", "b"]


def test_split_strips_each_line():
    lines = ["  first  ", "\tsecond\t", "", "third"]
    got = split_into_stanzas(lines)
    assert got[0] == "first / second"
    assert got[1] == "third"


def test_split_empty_input_returns_empty():
    assert split_into_stanzas([]) == []


def test_split_only_blanks_returns_empty():
    assert split_into_stanzas(["", "", ""]) == []


def test_split_no_blanks_returns_single_chunk():
    """Regression guard: confirms the caller must check for this case."""
    lines = ["a", "b", "c"]
    got = split_into_stanzas(lines)
    assert len(got) == 1
    assert got[0] == "a / b / c"


# ------------------ chunk_lines ------------------


def test_chunk_lines_basic():
    lines = ["a", "b", "c", "d", "e"]
    got = chunk_lines(lines, chunk_size=2)
    assert got == ["a / b", "c / d", "e"]


def test_chunk_lines_drops_blank_lines():
    lines = ["a", "", "b", "  ", "c"]
    got = chunk_lines(lines, chunk_size=2)
    assert got == ["a / b", "c"]


def test_chunk_lines_strip_whitespace():
    lines = ["  a ", "\tb"]
    got = chunk_lines(lines, chunk_size=2)
    assert got == ["a / b"]


def test_chunk_lines_empty_input():
    assert chunk_lines([], chunk_size=4) == []


# ------------------ threshold env ------------------


def test_match_threshold_env_override(monkeypatch):
    """PG_RAW_MIN_MATCH_RATE env var should be picked up on import."""
    import importlib
    import fetch_raw_gutenberg

    monkeypatch.setenv("PG_RAW_MIN_MATCH_RATE", "0.50")
    reloaded = importlib.reload(fetch_raw_gutenberg)
    try:
        assert reloaded.MATCH_THRESHOLD == pytest.approx(0.50)
    finally:
        monkeypatch.delenv("PG_RAW_MIN_MATCH_RATE", raising=False)
        importlib.reload(fetch_raw_gutenberg)


def test_match_threshold_default_0_98(monkeypatch):
    import importlib
    import fetch_raw_gutenberg

    monkeypatch.delenv("PG_RAW_MIN_MATCH_RATE", raising=False)
    reloaded = importlib.reload(fetch_raw_gutenberg)
    assert reloaded.MATCH_THRESHOLD == pytest.approx(0.98)
